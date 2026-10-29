"""Synchronous application boundary for the Device Scheduler authority.

This module composes environment-neutral coordinators with versioned ports.  It
owns request authorization and durable orchestration; adapters own host effects.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import math
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ReconciliationRequiredError,
    SchedulerError,
    UnsupportedCapabilityError,
    ValidationError,
)
from .models import (
    AllocationMode,
    BackendSpec,
    CapabilityPrecondition,
    EvictionGrant,
    ExecutionContext,
    ExecutionHandle,
    ExecutionPurpose,
    GPU_EXCLUSIVE_TAG,
    HandoffSpec,
    JobSpec,
    LeaseState,
    LeaseUsage,
    Principal,
    PolicyDecision as SchedulerPolicyDecision,
    Resource,
    ResourceAllocation,
    ResourceRequest,
    ResourceRequirement,
    ScheduleSpec,
)
from .observability import (
    bound_output,
    decision_snapshot,
    execution_evidence,
    explain_decision,
    health_report,
    replay_policy,
)
from .policy import WeightedFairPolicy
from .scheduling import (
    ScheduleCoordinator,
    dependencies_ready,
    event_ready,
    scheduled_job_ready,
    validate_dependency_graph,
)
from .validation import identifier


OPERATION_NAMES: dict[str, str] = {
    "resources.list": "resources.list",
    "resources.capabilities": "resources.capabilities",
    "resources.forecast": "resources.forecast",
    "access.inspect": "access.inspect",
    "leases.request": "leases.request",
    "leases.demand": "leases.demand",
    "leases.get": "leases.get",
    "leases.list": "leases.list",
    "leases.renew": "leases.renew",
    "leases.release": "leases.release",
    "jobs.submit": "jobs.submit",
    "jobs.get": "jobs.get",
    "jobs.list": "jobs.list",
    "jobs.wait": "jobs.wait",
    "jobs.logs": "jobs.logs",
    "jobs.control": "jobs.control",
    "backends.register": "backends.register",
    "backends.get": "backends.get",
    "backends.list": "backends.list",
    "backends.control": "backends.control",
    "backends.attach": "backends.attach",
    "handoffs.submit": "handoffs.submit",
    "handoffs.get": "handoffs.get",
    "handoffs.list": "handoffs.list",
    "handoffs.control": "handoffs.control",
    "schedules.put": "schedules.put",
    "schedules.get": "schedules.get",
    "schedules.list": "schedules.list",
    "schedules.control": "schedules.control",
    "events.publish": "events.publish",
    "policies.explain": "policies.explain",
    "policies.simulate": "policies.simulate",
    "admin.status": "admin.status",
    "admin.reconcile": "admin.reconcile",
    "admin.backup": "admin.backup",
    "history.compact": "history.compact",
    "history.status": "history.status",
}

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
_ACTIVE = frozenset({"queued", "blocked", "running", "unknown", "cancelling"})
_UTC = timezone.utc
_REVISION = "_revision"
_GPU_KINDS = frozenset({"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"})


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Trusted host configuration consumed by the application layer.

    Profiles and backend profiles are provisioned by the operator. Request
    payloads may refer to them but cannot define their command or configuration.
    """

    state_root: Path = Path("~/.local/state/device-scheduler")
    artifact_root: Path | None = None
    execution_root: Path | None = None
    profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    backend_profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    secret_refs: frozenset[str] = frozenset()
    event_sources: Mapping[str, Sequence[str]] = field(default_factory=dict)
    policy: Any | None = None
    default_lease_ttl_seconds: float = 3660.0
    fairness_window_seconds: float = 86400.0
    max_wait_seconds: float = 30.0
    max_output_bytes: int = 262144
    max_output_lines: int = 2000
    backup_root: Path | None = None
    autobench_history_owner_limit_bytes: int = 64 * 1024 * 1024
    autobench_history_owner_limit_rows: int = 10_000

    def __post_init__(self) -> None:
        for name in ("state_root", "artifact_root", "execution_root", "backup_root"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if not isinstance(self.default_lease_ttl_seconds, (int, float)) or isinstance(self.default_lease_ttl_seconds, bool) or not (0 < self.default_lease_ttl_seconds < float("inf")):
            raise ValueError("default_lease_ttl_seconds must be positive")
        if (
            not isinstance(self.fairness_window_seconds, (int, float))
            or isinstance(self.fairness_window_seconds, bool)
            or not (0 < self.fairness_window_seconds < float("inf"))
        ):
            raise ValueError("fairness_window_seconds must be finite and positive")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be non-negative")
        if self.max_output_bytes < 0 or self.max_output_lines < 1:
            raise ValueError("output bounds are invalid")
        if (
            isinstance(self.autobench_history_owner_limit_bytes, bool)
            or not isinstance(self.autobench_history_owner_limit_bytes, int)
            or self.autobench_history_owner_limit_bytes < 1
            or isinstance(self.autobench_history_owner_limit_rows, bool)
            or not isinstance(self.autobench_history_owner_limit_rows, int)
            or self.autobench_history_owner_limit_rows < 1
        ):
            raise ValueError("Autobench history owner limits must be positive integers")
        if not isinstance(self.profiles, Mapping) or any(not isinstance(value, Mapping) for value in self.profiles.values()):
            raise ValueError("profiles must map profile IDs to objects")
        if not isinstance(self.backend_profiles, Mapping) or any(not isinstance(value, Mapping) for value in self.backend_profiles.values()):
            raise ValueError("backend_profiles must map profile IDs to objects")
        if not isinstance(self.event_sources, Mapping):
            raise ValueError("event_sources must map principal IDs to source patterns")
        sources: dict[str, tuple[str, ...]] = {}
        for principal_id, patterns in self.event_sources.items():
            if not isinstance(principal_id, str) or not principal_id:
                raise ValueError("event source principal IDs must be non-empty strings")
            if isinstance(patterns, str) or not isinstance(patterns, Sequence) or any(not isinstance(item, str) or not item for item in patterns):
                raise ValueError("event source grants must be arrays of non-empty patterns")
            sources[principal_id] = tuple(patterns)
        if any(not isinstance(item, str) or not item for item in self.secret_refs):
            raise ValueError("secret_refs must contain non-empty strings")
        object.__setattr__(self, "profiles", dict(self.profiles))
        object.__setattr__(self, "backend_profiles", dict(self.backend_profiles))
        object.__setattr__(self, "secret_refs", frozenset(self.secret_refs))
        object.__setattr__(self, "event_sources", sources)

    @property
    def resolved_artifact_root(self) -> Path:
        return self.artifact_root or self.state_root / "artifacts"

    @property
    def resolved_execution_root(self) -> Path:
        return self.execution_root or self.state_root / "executions"


class _SystemClock:
    api_version = "1"

    @staticmethod
    def now() -> datetime:
        return datetime.now(_UTC)

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()


def _without_revision(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != _REVISION}


def _json_value(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, datetime):
        return value.astimezone(_UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(child) for child in value]
    return value


def _strip_internal(value: Any) -> Any:
    """Remove ledger-owned revision metadata from every public JSON result."""
    if isinstance(value, Mapping):
        return {str(key): _strip_internal(child) for key, child in value.items() if key != _REVISION}
    if isinstance(value, list):
        return [_strip_internal(child) for child in value]
    return value


def _state(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    return str(getattr(value, "value", value)).lower()


def _now(clock: Any) -> datetime:
    value = clock.now() if callable(getattr(clock, "now", None)) else clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("scheduler clock must return a timezone-aware datetime")
    return value.astimezone(_UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(_UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"{name} must be an ISO-8601 datetime") from exc
    else:
        raise ValidationError(f"{name} must be a timezone-aware datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware")
    return parsed.astimezone(_UTC)


def _optional_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return _parse_time(value, "timestamp")
    except ValidationError:
        return None


class Scheduler:
    """Synchronous application services for the single-host scheduler."""

    def __init__(
        self,
        ledger: Any,
        allocator: Any,
        executor: Any | None = None,
        resource_adapters: Sequence[Any] = (),
        backend_adapter: Any | None = None,
        continuation_adapter: Any | None = None,
        clock: Any | None = None,
        config: SchedulerConfig | Mapping[str, Any] | Any | None = None,
    ) -> None:
        self.ledger = ledger
        self.allocator = allocator
        self.executor = executor
        self.resource_adapters = tuple(resource_adapters)
        self.backend_adapter = backend_adapter
        self.continuation_adapter = continuation_adapter
        self.clock = clock or _SystemClock()
        self.config = _config(config)
        self.policy = self.config.policy or WeightedFairPolicy()
        self._lock = threading.RLock()
        # A scheduler tick may perform slow adapter I/O. Serialize ticks among
        # themselves without blocking short client commands on those effects.
        self._tick_lock = threading.Lock()
        self._inventory_lock = threading.Lock()
        self._schedule_coordinator: ScheduleCoordinator | None = None
        self._handoff_coordinator: Any | None = None
        self._inventory_warnings: list[dict[str, str]] = []

    def dispatch(self, operation: str, args: Mapping[str, Any], principal: Principal | Mapping[str, Any]) -> dict[str, Any]:
        """Authorize and execute one canonical operation, returning a JSON object."""
        if not isinstance(operation, str):
            raise ValidationError("operation must be a string")
        canonical = operation.replace("_", ".") if operation not in OPERATION_NAMES else operation
        if canonical not in OPERATION_NAMES:
            raise ValidationError(f"unknown scheduler operation {operation!r}")
        if not isinstance(args, Mapping):
            raise ValidationError("operation arguments must be an object")
        actor = principal if isinstance(principal, Principal) else Principal.from_dict(principal)
        self._authorize(actor, canonical)
        # Waiting only polls canonical state. It does not retain the process lock,
        # so the background tick can observe and complete the same job.
        lock = _NullContext() if canonical in {"jobs.wait", "resources.list", "leases.request", "resources.forecast"} or (canonical == "leases.demand" and args.get("action") == "wait") else self._lock
        try:
            with lock:
                result = self._dispatch(canonical, dict(args), actor)
        except SchedulerError:
            raise
        except Exception as exc:
            raise SchedulerError(
                f"{canonical} failed in {type(exc).__name__}",
                details={"operation": canonical, "cause": type(exc).__name__},
            ) from exc
        normalized = _strip_internal(_json_value(result))
        if not isinstance(normalized, dict):
            raise SchedulerError("scheduler operation returned a non-object result")
        return normalized

    def _dispatch(self, operation: str, args: dict[str, Any], actor: Principal) -> dict[str, Any]:
        if operation == "resources.list":
            warnings = self._discover_resources()
            status = getattr(self.allocator, "list_resource_status", None)
            resources = status() if callable(status) else self.ledger.list("resource")
            decorated = []
            for item in resources:
                record = dict(item)
                resource_record = record.get("resource", record)
                if isinstance(resource_record, Mapping) and isinstance(resource_record.get("resource_id"), str):
                    record["external_lease"] = self._external_resource_lease_capability(resource_record)
                decorated.append(_json_value(record))
            return {
                "resources": decorated,
                "inventory_warnings": warnings,
            }
        if operation == "resources.capabilities":
            return self._capabilities()
        if operation == "resources.forecast":
            return self._resource_forecast(args, actor)
        if operation == "access.inspect":
            return {
                "principal_id": actor.principal_id,
                "actions": sorted(actor.actions),
                "execution_roots": list(actor.execution_roots),
                "backend_ids": sorted(actor.backend_ids),
                "max_priority": actor.max_priority,
                "scope": "authenticated caller only",
            }
        if operation == "leases.request":
            return self._request_lease(args, actor)
        if operation == "leases.demand":
            return self._lease_demand(args, actor)
        if operation == "leases.get":
            lease = self._get_lease(_required(args, "lease_id"))
            self._require_owner(_plain_lease(lease), actor, owner_key="owner_id", any_owner_action="leases.read:any_owner")
            return _json_value(lease)
        if operation == "leases.list":
            owner = None if self._has(actor, "leases.read:any_owner") else actor.principal_id
            leases = self.allocator.list_leases(owner_id=owner, state=args.get("state"))
            return {"leases": [_json_value(item) for item in leases]}
        if operation == "leases.renew":
            return self._renew_lease(args, actor)
        if operation == "leases.release":
            return self._release_lease(args, actor)
        if operation == "jobs.submit":
            spec_raw = args.get("spec", args)
            owner = self._owner_from_args(args, actor, "jobs.submit:any_owner")
            spec = self._prepare_job_spec(spec_raw, owner, actor)
            key = _idempotency_key(args, spec.idempotency_key)
            if spec.retention_policy == "autobench":
                self._validate_autobench_submission_key(key, owner)
            lease_id = args.get("lease_id")
            if lease_id is not None and not isinstance(lease_id, str):
                raise ValidationError("lease_id must be a string")
            lease_fencing_token = args.get("fencing_token")
            if lease_id is not None:
                if isinstance(lease_fencing_token, bool) or not isinstance(lease_fencing_token, int):
                    raise ValidationError("fencing_token is required and must be an integer when binding a lease")
            elif lease_fencing_token is not None:
                raise ValidationError("fencing_token requires lease_id")
            return self.submit_job(
                spec.to_dict(), owner, key,
                lease_id=lease_id,
                lease_fencing_token=lease_fencing_token,
            )
        if operation == "jobs.get":
            job = self.get_job(_required(args, "job_id"))
            self._require_owner(job, actor, owner_key="owner", any_owner_action="jobs.read:any_owner")
            return self._job_detail(job)
        if operation == "jobs.list":
            owner = None if self._has(actor, "jobs.read:any_owner") else actor.principal_id
            return {"jobs": [self._job_detail(item) for item in self._list("jobs", owner=owner)]}
        if operation == "jobs.wait":
            return self._wait_job(args, actor)
        if operation == "jobs.logs":
            return self._job_logs(args, actor)
        if operation == "jobs.control":
            return self._control_job(args, actor)
        if operation == "backends.register":
            return self._register_backend(args, actor)
        if operation == "backends.get":
            backend = self._get("backend", _required(args, "backend_id"))
            self._require_owner(backend, actor, owner_key="owner", any_owner_action="backends.read:any_owner")
            return self._backend_detail(backend)
        if operation == "backends.list":
            owner = None if self._has(actor, "backends.read:any_owner") else actor.principal_id
            return {"backends": [self._backend_detail(item) for item in self._list("backend", owner=owner)]}
        if operation == "backends.control":
            return self._control_backend(args, actor)
        if operation == "backends.attach":
            return self._attach_backend(args, actor)
        if operation == "handoffs.submit":
            return self._submit_handoff(args, actor)
        if operation == "handoffs.get":
            handoff = self._handoffs().get(_required(args, "handoff_id"))
            self._require_owner(handoff, actor, owner_key="owner", any_owner_action="handoffs.read:any_owner")
            return _json_value(handoff)
        if operation == "handoffs.list":
            owner = None if self._has(actor, "handoffs.read:any_owner") else actor.principal_id
            return {"handoffs": [self._handoff_detail(item) for item in self._list("handoff", owner=owner)]}
        if operation == "handoffs.control":
            return self._control_handoff(args, actor)
        if operation == "schedules.put":
            return self._put_schedule(args, actor)
        if operation == "schedules.get":
            schedule = self._get("schedules", _required(args, "schedule_id"))
            self._require_owner(schedule, actor, owner_key="owner_id", any_owner_action="schedules.read:any_owner")
            return _json_value(schedule)
        if operation == "schedules.list":
            owner = None if self._has(actor, "schedules.read:any_owner") else actor.principal_id
            return {"schedules": [_json_value(item) for item in self._list("schedules", owner=owner)]}
        if operation == "schedules.control":
            return self._control_schedule(args, actor)
        if operation == "events.publish":
            source = _required(args, "source")
            event_id = _required(args, "event_id")
            payload = args.get("payload")
            if not isinstance(payload, Mapping):
                raise ValidationError("event payload must be an object")
            allowed_sources = self._source_grants(actor)
            if not any(fnmatch.fnmatchcase(source, pattern) for pattern in allowed_sources):
                raise PermissionDeniedError("event source is outside the principal's grant")
            return self._schedules().publish_event(
                source, event_id, payload, actor.principal_id, _idempotency_key(args)
            )
        if operation == "policies.explain":
            return self._explain_policy(args, actor)
        if operation == "policies.simulate":
            return self._simulate_policy(args)
        if operation == "admin.status":
            return self.status()
        if operation == "admin.reconcile":
            return self._reconcile_admin()
        if operation == "admin.backup":
            return self._backup(args)
        if operation == "history.compact":
            return self._compact_history(args, actor)
        if operation == "history.status":
            return self._history_status(actor)
        raise ValidationError(f"unknown scheduler operation {operation!r}")

    def tick(self) -> dict[str, Any]:
        """Reconcile durable effects, materialize schedules, and admit safe work."""
        with self._tick_lock:
            summary: dict[str, Any] = {
                "reconciled_jobs": [], "reconciled_backends": [], "reconciled_preemptions": [], "scheduled_jobs": [],
                "handoff": None, "started_jobs": [], "errors": [], "inventory_warnings": [],
            }
            summary["inventory_warnings"] = self._discover_resources()
            try:
                summary["reconciled_jobs"] = self._reconcile_jobs()
            except Exception as exc:  # keep the loop alive; unresolved records remain visible
                summary["errors"].append({"component": "job_reconciliation", "error": type(exc).__name__})
            try:
                summary["reconciled_preemptions"] = self._reconcile_preemptions()
            except Exception as exc:
                summary["errors"].append({"component": "preemption_reconciliation", "error": type(exc).__name__})
            try:
                summary["reconciled_lease_demands"] = self._reconcile_lease_demands()
            except Exception as exc:
                summary["errors"].append({"component": "lease_demand_reconciliation", "error": type(exc).__name__})
            try:
                summary["reconciled_backends"] = self._reconcile_backends()
            except Exception as exc:
                summary["errors"].append({"component": "backend_reconciliation", "error": type(exc).__name__})
            try:
                expired = self.allocator.expire_due()
                summary["expired_claims"] = list(expired)
                summary["reconciled_expired_leases"] = self._reconcile_expired_leases(expired)
            except Exception as exc:
                summary["errors"].append({"component": "lease_expiry", "error": type(exc).__name__})
            try:
                summary["scheduled_jobs"] = self._schedules().tick()
            except Exception as exc:
                summary["errors"].append({"component": "schedule_tick", "error": type(exc).__name__})
            if self._has_pending_handoffs():
                try:
                    summary["handoff"] = self._handoffs().tick()
                except Exception as exc:
                    summary["errors"].append({"component": "handoff_tick", "error": type(exc).__name__})
            if summary["errors"]:
                summary["admission_skipped"] = "pre_admission_error"
            else:
                try:
                    summary["started_jobs"] = self._admit_jobs()
                except Exception as exc:
                    summary["errors"].append({"component": "job_admission", "error": type(exc).__name__})
            summary["status"] = "degraded" if summary["errors"] or summary["inventory_warnings"] else "ok"
            return _json_value(summary)

    def status(self) -> dict[str, Any]:
        """Return a health summary without starting or stopping managed work."""
        operations = self.ledger.operations() if callable(getattr(self.ledger, "operations", None)) else []
        report = health_report(
            leases=[_plain_lease(item) for item in self.allocator.list_leases()],
            jobs=self.ledger.list("jobs"),
            operations=operations,
            backends=self.ledger.list("backend"),
        )
        report["adapters"] = self._capabilities()
        report["scheduler"] = {
            "policy_id": getattr(self.policy, "policy_id", "unknown"),
            "policy_version": getattr(self.policy, "version", "unknown"),
            "policy_configuration": self._policy_configuration_snapshot(),
        }
        return _json_value(report)

    def _policy_configuration_snapshot(self) -> dict[str, Any] | None:
        """Return operator-provided policy provenance, or built-in exact settings."""
        configured = getattr(self.policy, "configuration_snapshot", None)
        if isinstance(configured, Mapping):
            snapshot = _json_value(configured)
            return snapshot if isinstance(snapshot, dict) else None
        if isinstance(self.policy, WeightedFairPolicy):
            return {
                "api_version": "1",
                "policy_id": self.policy.policy_id,
                "version": self.policy.version,
                "source": {"kind": "builtin", "name": self.policy.policy_id},
                "parameters": {
                    "class_weights": dict(self.policy.class_weights),
                    "owner_weights": dict(self.policy.owner_weights),
                    "age_quantum_seconds": self.policy.age_quantum_seconds,
                    "max_age_boost": self.policy.max_age_boost,
                    "large_request_protection_seconds": self.policy.large_request_protection_seconds,
                    "fairness_penalty": self.policy.fairness_penalty,
                },
                "fairness_window_seconds": self.config.fairness_window_seconds,
            }
        return None

    def submit_job(
        self,
        spec: Mapping[str, Any],
        owner: str,
        key: str,
        reservation_id: str | None = None,
        *,
        lease_id: str | None = None,
        lease_fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Persist a job once, optionally binding it to a supervised client lease."""
        if not isinstance(spec, Mapping):
            raise ValidationError("job spec must be an object")
        normalized = dict(spec)
        normalized["owner_id"] = owner
        if normalized.get("retention_policy") == "autobench":
            self._validate_autobench_submission_key(key, owner)
            prior_key = normalized.get("idempotency_key")
            if prior_key is not None and prior_key != key:
                raise ConflictError("Autobench spec idempotency_key must match the submission key")
            # Keep the expiring replay namespace with the retained job so the
            # maintenance path can purge complete identities only after expiry.
            normalized["idempotency_key"] = key
        job_spec = JobSpec.from_dict(normalized)
        if lease_id is not None:
            if reservation_id is not None:
                raise ValidationError("a job cannot bind both a lease and a reservation")
            if lease_fencing_token is None:
                raise ValidationError("a supervised lease binding requires its fencing token")
            if job_spec.resources:
                raise ValidationError("a job bound to a lease inherits its allocation; omit job-level resources")
        elif lease_fencing_token is not None:
            raise ValidationError("lease_fencing_token requires lease_id")
        if lease_id is not None:
            bound_lease = _plain_lease(self._get_lease(lease_id))
            self._validate_supervised_lease_allocations(bound_lease)
        payload = {"spec": job_spec.to_dict(), "reservation_id": reservation_id}
        if lease_id is not None:
            payload.update(lease_id=lease_id, lease_fencing_token=lease_fencing_token)
        if not owner or not key:
            raise ValidationError("job owner and idempotency key are required")

        def create(tx: Any) -> dict[str, Any]:
            if job_spec.retention_policy == "autobench":
                budget_check = getattr(tx, "check_autobench_owner_budget", None)
                if not callable(budget_check):
                    raise UnsupportedCapabilityError(
                        "ledger does not support bounded Autobench history admission"
                    )
                budget_check(
                    owner,
                    additional_bytes=(
                        len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")) * 3
                        + job_spec.max_attempts * 4096
                        + 4096
                    ),
                    additional_rows=4 + job_spec.max_attempts * 10,
                    limit_bytes=self.config.autobench_history_owner_limit_bytes,
                    limit_rows=self.config.autobench_history_owner_limit_rows,
                )
            if lease_id is not None:
                lease = tx.get("lease", lease_id)
                if lease is None:
                    raise NotFoundError(f"lease {lease_id!r} does not exist")
                if lease.get("usage", LeaseUsage.EXTERNAL.value) != LeaseUsage.SUPERVISED.value:
                    raise UnsupportedCapabilityError("jobs can bind only to leases requested for supervised use")
                if lease.get("owner_id") != owner:
                    raise PermissionDeniedError("job owner must match the supervised lease owner")
                if lease.get("state") != LeaseState.ACTIVE.value:
                    raise ConflictError("supervised lease is not active")
                if int(lease.get("fencing_token", 0)) != lease_fencing_token:
                    raise ConflictError("supervised lease fencing token is stale")
                if _parse_time(lease.get("expires_at"), "lease expires_at") <= _now(self.clock):
                    raise ConflictError("supervised lease has expired")
                existing_jobs = [
                    item for item in tx.list("jobs")
                    if item.get("lease_id") == lease_id
                    and _state(item.get("state")) not in _TERMINAL
                ]
                if existing_jobs:
                    raise ConflictError(
                        "a supervised lease may have only one queued or running job at a time",
                        details={"active_job_ids": [str(item.get("id", item.get("job_id"))) for item in existing_jobs]},
                    )
            job_id = str(uuid.uuid4())
            validate_dependency_graph(tx, job_id, job_spec.dependencies)
            now = _now(self.clock)
            dependencies_ok = dependencies_ready(tx, job_spec.dependencies)
            event_ok = event_ready(tx, job_spec.event_trigger, created_at=now)
            state = "queued" if dependencies_ok and event_ok else "blocked"
            record = {
                "id": job_id,
                "job_id": job_id,
                "owner": owner,
                "owner_id": owner,
                "spec": job_spec.to_dict(),
                "state": state,
                "created_at": _iso(now),
                "updated_at": _iso(now),
                "attempt_id": None,
                "lease_id": lease_id,
                "lease_fencing_token": lease_fencing_token,
                "lease_binding": "supervised" if lease_id is not None else None,
                "reservation_id": reservation_id,
                "result": None,
                "cleanup_verified": False,
                "blocked_reason": None if state == "queued" else "dependency_or_event_pending",
            }
            saved = tx.put("jobs", job_id, record, None)
            tx.event("job.submitted", job_id, {
                "owner": owner,
                "state": state,
                "reservation_id": reservation_id,
                "lease_id": lease_id,
                "lease_binding": "supervised" if lease_id is not None else None,
            })
            return saved

        return self.ledger.idempotent(owner, key, "jobs.submit", payload, create)

    def _validate_autobench_submission_key(self, key: str, owner: str) -> None:
        """Require a bounded replay window for every retained Autobench job."""
        parts = key.split(":", 2)
        if len(parts) != 3 or parts[0] != "autobench-v1" or not parts[2] or len(key) > 256:
            raise ValidationError(
                "retention_policy=autobench requires idempotency_key "
                "autobench-v1:<expiry-unix-seconds>:<opaque-unique>"
            )
        try:
            expiry = int(parts[1])
        except ValueError as exc:
            raise ValidationError("Autobench idempotency key expiry must be Unix seconds") from exc
        if expiry <= 0:
            raise ValidationError("Autobench idempotency key expiry must be positive")
        now = int(_now(self.clock).timestamp())
        if expiry <= now:
            raise ConflictError(
                "Autobench submission idempotency window has expired",
                details={"expiry_unix_seconds": expiry, "now_unix_seconds": now},
            )
        if expiry > now + 30 * 24 * 60 * 60 + 60:
            raise ValidationError("Autobench idempotency expiry may be at most 30 days in the future")
        validate = getattr(self.ledger, "validate_autobench_idempotency_key", None)
        if not callable(validate):
            raise UnsupportedCapabilityError("ledger does not enforce expiring Autobench replay identities")
        validate(owner, key, now_unix_seconds=now)

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self._get("jobs", job_id)
        return job

    def cancel_job(self, job_id: str, reason: str = "caller") -> dict[str, Any]:
        """Trusted, idempotent cleanup callback used by job and handoff control."""
        job = self.get_job(job_id)
        state = _state(job.get("state"))
        if state in _TERMINAL:
            return self._job_detail(job)
        attempt_id = job.get("attempt_id")
        if not attempt_id:
            return self._update_job(job_id, {"state": "cancelled", "result": {"reason": reason}, "cleanup_verified": True})
        attempt = self.ledger.get("attempts", attempt_id)
        handle = _handle(attempt.get("execution_handle") if attempt else None)
        if self.executor is None or handle is None:
            self._mark_attempt_unknown(job_id, attempt_id, "execution handle unavailable during cancellation")
            return self.get_job(job_id)
        operation_id = f"cancel:{attempt_id}"
        current = self._ensure_operation(operation_id, "job.cancel", {"job_id": job_id, "attempt_id": attempt_id}, str(job.get("owner", "system")))
        if current.get("status") == "complete":
            receipt = current.get("receipt") or {}
        else:
            try:
                receipt = _call_cancel(self.executor, handle, operation_id, _timeout(job.get("spec"), 30.0))
            except Exception as exc:
                receipt = {"state": "unknown", "cleanup_verified": False, "error": type(exc).__name__}
            self._record_operation(operation_id, receipt, status="complete" if receipt.get("cleanup_verified") else "unknown")
        if receipt.get("cleanup_verified") is not True:
            self._mark_attempt_unknown(job_id, attempt_id, "executor did not verify cleanup after cancellation", receipt)
            return self.get_job(job_id)
        receipt = {**dict(receipt), "cancellation_reason": reason}
        self._store_attempt_result(job_id, attempt_id, receipt, state="cancelled", cleanup_verified=True)
        self._release_job_lease(self.get_job(job_id), receipt)
        return self.get_job(job_id)

    def _request_lease(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        self._discover_resources()
        raw = args.get("request", args)
        if not isinstance(raw, Mapping):
            raise ValidationError("lease request must be an object")
        data = dict(raw)
        requested_owner = data.get("owner_id", actor.principal_id)
        has_external_owner_grant = self._has(actor, "leases.request:any_owner")
        if requested_owner != actor.principal_id:
            if not has_external_owner_grant:
                raise PermissionDeniedError("external lease owners require leases.request:any_owner")
        data["owner_id"] = requested_owner
        data["idempotency_key"] = args.get("idempotency_key", data.get("idempotency_key"))
        if not data["idempotency_key"]:
            raise ValidationError("idempotency_key is required")
        if not data.get("request_id"):
            data["request_id"] = data["idempotency_key"]
        request = ResourceRequest.from_dict(data)
        existing = next(
            (
                item for item in self.ledger.operations()
                if item.get("owner") == request.owner_id
                and item.get("idempotency_key") == request.idempotency_key
            ),
            None,
        )
        if existing is not None:
            # Let the allocator perform canonical digest/operation conflict
            # checks. A completed grant is then read from current durable state;
            # do not allocate a second placement just to replay it.
            lease = self.allocator.request_lease(request)
            lease_record = _plain_lease(self._get_lease(str(lease.lease_id)))
            mode, _adapter, _resources = self._external_lease_reclamation_path(
                tuple(ResourceAllocation.from_dict(item) for item in lease_record.get("allocations", ())),
                usage=str(lease_record.get("usage", LeaseUsage.EXTERNAL.value)),
            )
            if mode is None:
                raise ReconciliationRequiredError("existing external lease no longer has a trusted reclamation adapter")
            if mode == "observe":
                evidence = self.ledger.get("lease_evidence", str(lease_record["lease_id"]))
                if (
                    not isinstance(evidence, Mapping)
                    or evidence.get("initial_fencing_token") is None
                    or not isinstance(evidence.get("baseline"), Mapping)
                ):
                    raise ReconciliationRequiredError("existing GPU lease lacks its persisted pre-grant reclamation baseline")
            return lease_record
        self._require_fresh_matching_inventory(request.requirements)
        preview = getattr(self.allocator, "preview_allocation", None)
        if not callable(preview):
            raise UnsupportedCapabilityError("allocator cannot verify external lease reclamation coverage before grant")
        try:
            preview_allocations = tuple(preview(request.requirements))
        except ConflictError:
            raise
        mode, adapter, _resources = self._external_lease_reclamation_path(
            preview_allocations, usage=request.usage.value
        )
        if mode is None or adapter is None:
            raise UnsupportedCapabilityError(
                "requested external lease has no trusted reclamation adapter; use a supervised job or managed backend"
            )
        lease = self.allocator.request_lease(request)
        lease_record = _plain_lease(lease)
        actual_mode, actual_adapter, descriptors = self._external_lease_reclamation_path(
            tuple(ResourceAllocation.from_dict(item) for item in lease_record.get("allocations", ())),
            usage=str(lease_record.get("usage", LeaseUsage.EXTERNAL.value)),
        )
        if actual_mode != mode or actual_adapter is not adapter:
            # The placement preview is advisory. If a concurrent allocator
            # changed the selected resource/source, never return an unsupported
            # external grant; retain it as unknown until an operator can prove
            # cleanup rather than freeing it optimistically.
            requested = self.allocator.request_release(
                str(lease_record["lease_id"]), int(lease_record["fencing_token"])
            )
            self.allocator.mark_unknown(
                str(lease_record["lease_id"]),
                int(requested["fencing_token"]),
                "external lease placement changed to a resource without verified reclamation coverage",
            )
            raise ReconciliationRequiredError("external lease placement changed; grant is held for reconciliation")
        if mode == "observe":
            self._capture_external_lease_baseline(lease_record, adapter, descriptors)
        return _json_value(lease)

    def _lease_demand(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        action = args.get("action", "request")
        if action not in {"request", "get", "wait", "cancel"}:
            raise ValidationError("leases.demand action must be request, get, wait, or cancel")
        if action in {"request", "wait"} and actor.max_priority < 70:
            raise PermissionDeniedError("interactive priority lease demand exceeds the caller's priority ceiling")
        demand_id = args.get("demand_id")
        if not isinstance(demand_id, str) or not demand_id.strip():
            raise ValidationError("demand_id is required")
        identifier(demand_id, "demand_id")
        key = f"{actor.principal_id}:{demand_id}"
        if action == "request":
            raw = args.get("request")
            if not isinstance(raw, Mapping):
                raise ValidationError("priority lease demand request must be an object")
            data = dict(raw)
            if data.get("owner_id", actor.principal_id) != actor.principal_id and not self._has(actor, "leases.demand:any_owner"):
                raise PermissionDeniedError("priority lease demand owner must match the authenticated principal")
            data["owner_id"] = data.get("owner_id", actor.principal_id)
            data["request_id"] = demand_id
            data["idempotency_key"] = demand_id
            data["usage"] = data.get("usage", LeaseUsage.EXTERNAL.value)
            request = ResourceRequest.from_dict(data)
            if request.owner_id != actor.principal_id and not self._has(actor, "leases.demand:any_owner"):
                raise PermissionDeniedError("priority lease demand owner must match the authenticated principal")
            record = self.ledger.get("lease_demands", key)
            if record is None:
                now = _now(self.clock)
                record = self.ledger.put("lease_demands", key, {
                    "demand_id": demand_id, "owner": request.owner_id,
                    "request": request.to_dict(), "state": "pending",
                    "created_at": _iso(now), "updated_at": _iso(now),
                    "expires_at": _iso(now + timedelta(hours=1)),
                })
            elif record.get("request") != request.to_dict():
                raise ConflictError("demand_id was already used for a different request")
            return self._process_lease_demand(key)
        record = self.ledger.get("lease_demands", key)
        if record is None:
            raise NotFoundError("priority lease demand does not exist")
        if record.get("owner") != actor.principal_id and not self._has(actor, "leases.demand:any_owner"):
            raise PermissionDeniedError("priority lease demand belongs to another principal")
        if action == "cancel":
            if record.get("state") == "granted":
                raise ConflictError("a granted priority lease demand cannot be cancelled; release its lease")
            return self._update("lease_demands", key, {"state": "cancelled", "updated_at": _iso(_now(self.clock))})
        if action == "get":
            return _without_revision(record)
        timeout = args.get("wait_seconds", 0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= 30:
            raise ValidationError("wait_seconds must be between 0 and 30")
        deadline = self._monotonic() + float(timeout)
        while True:
            record = self._process_lease_demand(key)
            if record.get("state") != "pending" or self._monotonic() >= deadline:
                return record
            time.sleep(min(0.1, max(0.0, deadline - self._monotonic())))

    def _process_lease_demand(self, key: str) -> dict[str, Any]:
        # Demand processing can cancel jobs and create leases. Serialize it
        # with cancellation and tick reconciliation so cancel can never be
        # overwritten by an in-flight grant.
        with self._lock:
            return self._process_lease_demand_locked(key)

    def _process_lease_demand_locked(self, key: str) -> dict[str, Any]:
        record = self.ledger.get("lease_demands", key)
        if record is None:
            raise NotFoundError("priority lease demand does not exist")
        state = str(record.get("state"))
        if state != "pending":
            return _without_revision(record)
        expires_at = _optional_time(record.get("expires_at"))
        if expires_at is None or expires_at <= _now(self.clock):
            return self._update("lease_demands", key, {"state": "expired", "updated_at": _iso(_now(self.clock))})
        request = ResourceRequest.from_dict(record["request"])
        try:
            lease = self._request_lease({"request": request.to_dict(), "idempotency_key": request.idempotency_key}, Principal(
                principal_id=request.owner_id, actions=frozenset({"leases.request", "leases.request:any_owner"}), max_priority=100
            ))
        except ConflictError:
            lease = None
        if lease is None:
            self._preempt_for_lease_demand(request, record)
            expires_at = _optional_time(record.get("expires_at"))
            if expires_at is None or expires_at <= _now(self.clock):
                return self._update("lease_demands", key, {"state": "expired", "updated_at": _iso(_now(self.clock))})
            try:
                lease = self._request_lease({"request": request.to_dict(), "idempotency_key": request.idempotency_key}, Principal(
                    principal_id=request.owner_id, actions=frozenset({"leases.request", "leases.request:any_owner"}), max_priority=100
                ))
            except ConflictError:
                lease = None
        if lease is None:
            return _without_revision(record)
        expires_at = _optional_time(record.get("expires_at"))
        if expires_at is None or expires_at <= _now(self.clock):
            lease_record = _plain_lease(lease)
            cleanup: Mapping[str, Any]
            try:
                cleanup = self._release_lease(
                    {"lease_id": lease_record["lease_id"], "fencing_token": lease_record["fencing_token"]},
                    Principal(
                        principal_id=request.owner_id,
                        actions=frozenset({"leases.release", "leases.manage:any_owner"}),
                        max_priority=100,
                    ),
                )
            except Exception as exc:
                cleanup = {"state": "unknown", "cleanup_verified": False, "error": type(exc).__name__}
            return self._update("lease_demands", key, {
                "state": "expired", "late_grant_cleanup": _json_value(cleanup),
                "updated_at": _iso(_now(self.clock)),
            })
        return self._update("lease_demands", key, {
            "state": "granted", "lease": _json_value(lease), "updated_at": _iso(_now(self.clock))
        })

    def _preempt_for_lease_demand(self, request: ResourceRequest, demand: Mapping[str, Any]) -> bool:
        can_allocate = getattr(self.allocator, "can_allocate", None)
        if not callable(can_allocate) or self.executor is None:
            return False
        try:
            available_resources_fn = getattr(self.allocator, "list_resources", None)
            raw_resources = available_resources_fn(available_only=False) if callable(available_resources_fn) else self.ledger.list("resource")
            resources_by_id = {
                item.resource_id: item if isinstance(item, Resource) else Resource.from_dict(item)
                for item in raw_resources
            }
        except Exception:
            return False
        candidates = [job for job in self.ledger.list("jobs") if _state(job.get("state")) == "running"]
        eligible: list[tuple[float, str, Mapping[str, Any], str, str, bool]] = []
        for job in candidates:
            spec = JobSpec.from_dict(job.get("spec", {}))
            if (
                spec.priority_class != "background" or not spec.flexible
                or not spec.interruption.cancellation_supported
                or job.get("reservation_id") or job.get("reservation_phase_id")
                or job.get("lease_binding") == "supervised" or job.get("lease_binding") == "protected"
            ):
                continue
            attempt_count = job.get("attempt_count", 0)
            restart = spec.preemption_action.value == "restart"
            if restart and (not spec.interruption.restart_safe or attempt_count >= spec.max_attempts):
                continue
            lease_id = job.get("lease_id")
            if not isinstance(lease_id, str):
                continue
            lease = _plain_lease(self._get_lease(lease_id))
            if lease.get("state") != LeaseState.ACTIVE.value or lease.get("owner_id") != job.get("owner"):
                continue
            contributes = False
            for allocation in lease.get("allocations", ()):
                resource = resources_by_id.get(str(allocation.get("resource_id")))
                if resource is None:
                    continue
                for requirement in request.requirements:
                    if not _requirement_matches_resource(requirement, resource):
                        continue
                    if requirement.mode == AllocationMode.EXCLUSIVE or set(requirement.quantities).intersection(allocation.get("quantities", {})):
                        contributes = True
                        break
                if contributes:
                    break
            if not contributes:
                continue
            attempt_id = job.get("attempt_id")
            attempt = self.ledger.get("attempts", str(attempt_id)) if attempt_id else None
            if not isinstance(attempt, Mapping) or attempt.get("state") != "running" or _handle(attempt.get("execution_handle")) is None:
                continue
            expected = spec.expected_duration_seconds
            eligible.append((float(expected) if expected is not None else float("inf"), str(job.get("created_at", "")), job, str(attempt_id), lease_id, restart))
        # Find a sufficient eligible set first. This supports atomic bundles
        # that require releasing multiple packets, while never cancelling a
        # victim unless the full set can satisfy the request.
        chosen: list[tuple[float, str, Mapping[str, Any], str, str, bool]] = []
        for candidate in sorted(eligible, key=lambda item: (item[0], item[1], str(item[2].get("job_id")))):
            chosen.append(candidate)
            try:
                if can_allocate(request.requirements, ignore_lease_ids=tuple(item[4] for item in chosen)):
                    break
            except Exception:
                return False
        else:
            return False
        if not chosen:
            return False
        for _expected, _created, job, attempt_id, lease_id, restart in chosen:
            # Re-read immediately before each effect; concurrent caller-owned
            # changes or reservations make this candidate ineligible.
            current = self.ledger.get("jobs", str(job.get("job_id"))) or {}
            if current.get("attempt_id") != attempt_id or current.get("lease_id") != lease_id or _state(current.get("state")) != "running":
                return False
            demand_id = str(demand.get("demand_id"))
            reason = f"priority_lease_demand:{demand_id}"
            cancelled = self.cancel_job(str(current.get("job_id")), reason=reason)
            updated = self.ledger.get("jobs", str(job.get("job_id"))) or {}
            if _state(cancelled.get("state")) != "cancelled" or cancelled.get("cleanup_verified") is not True:
                return False
            if updated.get("lease_id") not in (None, lease_id):
                return False
            self._discover_resources()
            if restart:
                self._resume_flexible_job(current, attempt_id, lease_id, demand_id)
        try:
            return bool(can_allocate(request.requirements))
        except Exception:
            return False

    def _resume_flexible_job(self, job: Mapping[str, Any], attempt_id: str, lease_id: str, demand_id: str) -> None:
        """Return a safely interrupted, explicitly restartable packet to queue."""
        now = _iso(_now(self.clock))
        job_id = str(job.get("job_id"))
        with self.ledger.transaction() as tx:
            current = tx.get("jobs", job_id)
            attempt = tx.get("attempts", attempt_id)
            if current is None or attempt is None or current.get("lease_id") != lease_id:
                raise ReconciliationRequiredError("interrupted flexible job identity changed")
            if _state(current.get("state")) != "cancelled" or current.get("cleanup_verified") is not True:
                raise ReconciliationRequiredError("interrupted flexible job cleanup was not verified")
            a = _without_revision(attempt)
            a["preemption"] = {"reason": f"priority_lease_demand:{demand_id}", "recorded_at": now}
            tx.put("attempts", attempt_id, a, attempt[_REVISION])
            updated = _without_revision(current)
            updated.update({"state": "queued", "lease_id": None, "lease_fencing_token": None,
                            "result": None, "cleanup_verified": False,
                            "blocked_reason": f"priority_lease_demand:{demand_id}", "updated_at": now})
            tx.put("jobs", job_id, updated, current[_REVISION])
            tx.event("job.preempted", job_id, {"reason": f"priority_lease_demand:{demand_id}", "attempt_id": attempt_id})

    def _compact_history(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        if not self._has(actor, "history.compact:autobench"):
            raise PermissionDeniedError("history compaction requires history.compact:autobench authorization")
        compact = getattr(self.ledger, "compact_autobench_history", None)
        if not callable(compact):
            raise UnsupportedCapabilityError("ledger does not support autobench history compaction")
        before = _parse_time(args.get("before", _iso(_now(self.clock))), "before")
        maximum = args.get("max_records", 1000)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 10000:
            raise ValidationError("max_records must be between 1 and 10000")
        owner = actor.principal_id
        result = compact(owner, before=before, max_records=maximum)
        owner_usage = getattr(self.ledger, "autobench_owner_usage", None)
        if callable(owner_usage):
            result["owner_usage"] = owner_usage(owner)
            result["owner_usage"]["configured_limit_bytes"] = self.config.autobench_history_owner_limit_bytes
            result["owner_usage"]["configured_limit_rows"] = self.config.autobench_history_owner_limit_rows
        usage = getattr(self.ledger, "storage_usage", None)
        if callable(usage):
            result["storage_usage"] = usage()
        return result

    def _history_status(self, actor: Principal) -> dict[str, Any]:
        if not self._has(actor, "history.status") and not self._has(actor, "history.compact:autobench"):
            raise PermissionDeniedError("history status requires history.status authorization")
        owner_usage = getattr(self.ledger, "autobench_owner_usage", None)
        storage_usage = getattr(self.ledger, "storage_usage", None)
        if not callable(owner_usage) or not callable(storage_usage):
            raise UnsupportedCapabilityError("ledger does not report bounded owner-scoped Autobench history usage")
        retention = self.ledger.get("autobench_retention", actor.principal_id) or {}
        return {
            "owner_id": actor.principal_id,
            "owner_usage": owner_usage(actor.principal_id),
            "retention_floor_expiry_unix_seconds": retention.get("floor_expiry_unix_seconds", 0),
            "configured_limits": {
                "bytes": self.config.autobench_history_owner_limit_bytes,
                "rows": self.config.autobench_history_owner_limit_rows,
            },
            "storage_usage": storage_usage(),
            "scope": "owner-scoped Autobench admission budget plus aggregate SQLite footprint",
        }

    def _resource_forecast(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        raw = args.get("request")
        if not isinstance(raw, Mapping):
            raise ValidationError("forecast request must be an object")
        request_data = dict(raw)
        owner = request_data.get("owner_id", actor.principal_id)
        if owner != actor.principal_id and not self._has(actor, "resources.forecast:any_owner"):
            raise PermissionDeniedError("forecast request owner must match the authenticated principal")
        request_data.setdefault("owner_id", actor.principal_id)
        request_data.setdefault("request_id", f"forecast:{actor.principal_id}")
        request_data.setdefault("lease_ttl_seconds", 60)
        request = ResourceRequest.from_dict(request_data)
        horizon = args.get("horizon_seconds", 600)
        limit = args.get("max_candidates", 64)
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) or not 1 <= horizon <= 600:
            raise ValidationError("horizon_seconds must be between 1 and 600")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
            raise ValidationError("max_candidates must be between 1 and 64")
        now = _now(self.clock)
        self._discover_resources()
        status_fn = getattr(self.allocator, "list_resource_status", None)
        resources = status_fn() if callable(status_fn) else []
        candidates: list[dict[str, Any]] = []
        current_allocations: list[dict[str, Any]] = []
        for lease in self.allocator.list_leases(state=None) if callable(getattr(self.allocator, "list_leases", None)) else []:
            lease_state = _state(getattr(lease, "state", None))
            lease_data = _json_value(lease)
            if lease_state in {LeaseState.ACTIVE.value, LeaseState.UNKNOWN.value, LeaseState.RECLAIMING.value, LeaseState.RELEASE_REQUESTED.value}:
                for allocation in lease_data.get("allocations", ()):
                    current_allocations.append({"resource_id": allocation.get("resource_id"), "quantities": allocation.get("quantities", {}), "lease_state": lease_state})
        active_job_release: dict[str, float] = {}
        for job in self.ledger.list("jobs"):
            if _state(job.get("state")) != "running" or not job.get("lease_id"):
                continue
            spec = JobSpec.from_dict(job.get("spec", {}))
            attempt = self.ledger.get("attempts", str(job.get("attempt_id"))) if job.get("attempt_id") else None
            lease = self.ledger.get("lease", str(job.get("lease_id")))
            started = _optional_time(attempt.get("started_at")) if isinstance(attempt, Mapping) else None
            if spec.expected_duration_seconds is None or started is None or not isinstance(lease, Mapping):
                continue
            # Only Scheduler-owned active packet leases with a measured start
            # timestamp are release estimates. Caller-owned, unknown and
            # protected claims remain unavailable regardless of expiry.
            if lease.get("owner_id") != job.get("owner") or lease.get("state") != LeaseState.ACTIVE.value or lease.get("reservation_id"):
                continue
            remaining = max(0.0, float(spec.expected_duration_seconds) - (now - started).total_seconds())
            for allocation in lease.get("allocations", ()):
                rid = str(allocation.get("resource_id"))
                active_job_release[rid] = min(active_job_release.get(rid, float("inf")), remaining)
        try:
            whole_bundle_fits = bool(self.allocator.preview_allocation(request.requirements))
        except ConflictError:
            whole_bundle_fits = False
        except (UnsupportedCapabilityError, AttributeError):
            whole_bundle_fits = False
        queue_count = sum(
            1 for job in self.ledger.list("jobs")
            if _state(job.get("state")) in {"queued", "blocked"}
            and _state(job.get("spec", {}).get("priority_class")) == "interactive"
        )
        pending_demand_count = sum(1 for demand in self.ledger.list("lease_demands") if demand.get("state") == "pending")
        known_schedules = [
            {
                "schedule_id": str(item.get("schedule_id", item.get("id", ""))),
                "state": _state(item.get("state")),
                "pending_occurrences": list(item.get("pending_occurrences", ()))[:4],
            }
            for item in self.ledger.list("schedules")
            if _state(item.get("state")) == "active"
        ][:limit]
        for requirement_index, requirement in enumerate(request.requirements):
            compatible = []
            for item in resources:
                resource_data = item.get("resource", item) if isinstance(item, Mapping) else {}
                if not isinstance(resource_data, Mapping) or resource_data.get("kind") != requirement.kind:
                    continue
                if item.get("available") is False:
                    continue
                resource = Resource.from_dict(resource_data)
                if any(resource.attributes.get(key) != value for key, value in requirement.selectors.items()):
                    continue
                if any(float(resource.capacities.get(key, 0)) < float(value) for key, value in requirement.quantities.items()):
                    continue
                compatible.append(resource)
            releases = [active_job_release[item.resource_id] for item in compatible if item.resource_id in active_job_release]
            earliest_offset = 0.0 if whole_bundle_fits else (min(releases) if releases else float(horizon))
            feasible_window = whole_bundle_fits or bool(releases)
            candidates.append({
                "candidate_id": f"requirement-{requirement_index}",
                "resource_ids": [item.resource_id for item in compatible],
                "earliest_start": _iso(now + timedelta(seconds=min(earliest_offset, float(horizon)))),
                "latest_start": _iso(now + timedelta(seconds=float(horizon))),
                "expected_release": _iso(now + timedelta(seconds=earliest_offset)) if releases and earliest_offset <= horizon else None,
                "uncertainty_seconds": float(horizon if not releases else min(float(horizon), max(1.0, earliest_offset * 0.5))),
                "feasible_now": whole_bundle_fits and bool(compatible),
                "blocking_reasons": [] if feasible_window else (["no_compatible_observed_capacity"] if not compatible else ["capacity_or_protected_claims"]),
            })
            if len(candidates) >= limit:
                break
        return {
            "generated_at": _iso(now), "horizon_seconds": float(horizon),
            "current_resources": [
                {"resource_id": item.resource_id, "host_id": item.host_id, "kind": item.kind,
                 "capacities": dict(item.capacities), "topology": dict(item.topology)}
                for item in [Resource.from_dict((status.get("resource", status))) for status in resources if isinstance(status, Mapping)]
            ][:limit],
            "current_allocations": current_allocations[:limit],
            "known_queued_interactive_jobs": queue_count,
            "known_pending_priority_demands": pending_demand_count,
            "known_schedules": known_schedules,
            "candidates": candidates[:limit], "advisory_only": True,
        }

    def _external_lease_reclamation_path(
        self, allocations: Sequence[ResourceAllocation], *, usage: str = LeaseUsage.EXTERNAL.value
    ) -> tuple[str | None, Any | None, tuple[Resource, ...]]:
        """Resolve a complete, single-adapter release contract for an external bundle."""
        if not allocations:
            return None, None, ()
        status_fn = getattr(self.allocator, "list_resource_status", None)
        if not callable(status_fn):
            return None, None, ()
        try:
            statuses = status_fn()
        except Exception:
            return None, None, ()
        by_id = {
            str(item["resource"]["resource_id"]): item
            for item in statuses
            if isinstance(item, Mapping) and isinstance(item.get("resource"), Mapping)
        }
        source_ids: set[str] = set()
        resources: list[Resource] = []
        for allocation in allocations:
            status = by_id.get(allocation.resource_id)
            if status is None or status.get("available") is False:
                return None, None, ()
            source = status.get("source_adapter_id")
            if not isinstance(source, str) or not source:
                return None, None, ()
            source_ids.add(source)
            try:
                resources.append(Resource.from_dict(status["resource"]))
            except ValidationError:
                return None, None, ()
        if len(source_ids) != 1:
            return None, None, ()
        source_id = next(iter(source_ids))
        adapter = next(
            (
                item for item in self.resource_adapters
                if str(getattr(item, "adapter_id", type(item).__name__)) == source_id
            ),
            None,
        )
        if adapter is None:
            return None, None, ()
        if usage == LeaseUsage.SUPERVISED.value:
            if not self._supports_supervised_lease(resources, source_id):
                return None, None, ()
            return "supervised", adapter, tuple(resources)
        if callable(getattr(adapter, "reclaim_lease", None)):
            return "reclaim", adapter, tuple(resources)
        gpu_only_exclusive = all(
            resource.kind.casefold() in _GPU_KINDS and allocation.mode == AllocationMode.EXCLUSIVE
            for resource, allocation in zip(resources, allocations, strict=True)
        )
        if (
            gpu_only_exclusive
            and callable(getattr(adapter, "capture_lease_baseline", None))
            and callable(getattr(adapter, "inspect_lease_release", None))
        ):
            return "observe", adapter, tuple(resources)
        return None, None, ()

    def _supports_supervised_lease(self, resources: Sequence[Resource], source_id: str) -> bool:
        """Require host inventory plus the production supervised-job adapter."""
        if source_id != "linux-host" or not resources:
            return False
        if any(
            resource.kind.casefold() not in {"cpu", "memory", "host_memory", "ram"}
            or "supervised-lease-job-only" not in resource.capabilities
            for resource in resources
        ):
            return False
        if self.executor is None or str(getattr(self.executor, "adapter_id", "")) != "systemd-user-unit":
            return False
        capabilities = getattr(self.executor, "capabilities", None)
        if not callable(capabilities):
            return False
        try:
            names = {
                str(item.get("name")) if isinstance(item, Mapping) else str(getattr(item, "name", ""))
                for item in capabilities()
            }
        except Exception:
            return False
        return {
            "process-tree-cgroup-supervision",
            "granted-resource-cgroup-limits",
            "supervised-lease-job-binding",
        }.issubset(names)

    def _validate_supervised_lease_allocations(self, lease: Mapping[str, Any]) -> None:
        allocations = tuple(ResourceAllocation.from_dict(item) for item in lease.get("allocations", ()))
        mode, _adapter, resources = self._external_lease_reclamation_path(
            allocations, usage=LeaseUsage.SUPERVISED.value
        )
        if mode != "supervised" or not resources:
            raise UnsupportedCapabilityError(
                "supervised leases require Linux CPU/memory resources and a systemd execution adapter"
            )

    def _capture_external_lease_baseline(
        self, lease: Mapping[str, Any], adapter: Any, resources: Sequence[Resource]
    ) -> None:
        lease_id = str(lease["lease_id"])
        prior = self.ledger.get("lease_evidence", lease_id)
        if prior is not None:
            if (
                prior.get("lease_id") == lease_id
                and prior.get("initial_fencing_token") == lease.get("fencing_token")
                and prior.get("resource_ids") == sorted(item.resource_id for item in resources)
                and isinstance(prior.get("baseline"), Mapping)
            ):
                return
            raise ConflictError("external lease has mismatched persisted reclamation evidence")
        operation_id = f"lease-baseline:{lease_id}:{lease['fencing_token']}:{uuid.uuid4()}"
        self._ensure_operation(
            operation_id,
            "lease.capture_baseline",
            {"lease_id": lease_id, "fencing_token": lease["fencing_token"], "resource_ids": sorted(item.resource_id for item in resources)},
            str(lease.get("owner_id", "scheduler")),
        )
        try:
            receipt = adapter.capture_lease_baseline(
                lease,
                resources,
                operation_id=operation_id,
            )
            baseline = dict(receipt) if isinstance(receipt, Mapping) else {}
        except Exception as exc:
            baseline = {"state": "unknown", "error": type(exc).__name__}
        valid = (
            baseline.get("state") == "observed"
            and baseline.get("operation_id") == operation_id
            and baseline.get("lease_id") == lease_id
            and baseline.get("fencing_token") == lease.get("fencing_token")
            and baseline.get("resource_ids") == sorted(item.resource_id for item in resources)
            and isinstance(baseline.get("baseline"), Mapping)
            and _optional_time(baseline.get("observed_at")) is not None
        )
        self._record_operation(operation_id, baseline, status="complete" if valid else "unknown")
        if not valid:
            current = _plain_lease(self._get_lease(lease_id))
            requested = self.allocator.request_release(lease_id, int(current["fencing_token"]))
            self.allocator.mark_unknown(
                lease_id,
                int(requested["fencing_token"]),
                "external GPU lease has no verified pre-grant reclamation baseline",
            )
            raise ReconciliationRequiredError("external GPU lease baseline was not verified; grant is held")
        with self.ledger.transaction() as tx:
            current_lease = tx.get("lease", lease_id)
            if (
                current_lease is None
                or current_lease.get("state") != LeaseState.ACTIVE.value
                or current_lease.get("fencing_token") != lease.get("fencing_token")
            ):
                raise ReconciliationRequiredError(
                    "external GPU lease changed while its pre-grant baseline was being captured"
                )
            current = tx.get("lease_evidence", lease_id)
            if current is not None:
                raise ConflictError("external lease reclamation baseline was concurrently recorded")
            tx.put(
                "lease_evidence",
                lease_id,
                {
                    "lease_id": lease_id,
                    "adapter_id": str(getattr(adapter, "adapter_id", type(adapter).__name__)),
                    "initial_fencing_token": int(lease["fencing_token"]),
                    "resource_ids": sorted(item.resource_id for item in resources),
                    "baseline": baseline,
                    "captured_at": baseline["observed_at"],
                },
                None,
            )

    def _renew_lease(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        lease_id = _required(args, "lease_id")
        lease = self._get_lease(lease_id)
        self._require_owner(_plain_lease(lease), actor, owner_key="owner_id", any_owner_action="leases.manage:any_owner")
        fencing = _fencing_arg(args, lease)
        ttl = args.get("ttl_seconds")
        if ttl is None:
            raise ValidationError("ttl_seconds is required")
        renewed = _json_value(self.allocator.renew_lease(lease_id, fencing, ttl))
        if renewed.get("usage", LeaseUsage.EXTERNAL.value) == LeaseUsage.SUPERVISED.value:
            # A queued job has not consumed the old fence yet; carry the renewed
            # fence forward so its eventual execution context uses the current
            # lease generation. Running attempts retain their launch fence.
            with self.ledger.transaction() as tx:
                for job in tx.list("jobs"):
                    if (
                        job.get("lease_id") == lease_id
                        and job.get("lease_binding") == "supervised"
                        and _state(job.get("state")) in {"queued", "blocked"}
                    ):
                        data = _without_revision(job)
                        data["lease_fencing_token"] = renewed["fencing_token"]
                        tx.put("jobs", str(job.get("id", job.get("job_id"))), data, job[_REVISION])
        return renewed

    def _release_lease(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        lease_id = _required(args, "lease_id")
        lease = self._get_lease(lease_id)
        self._require_owner(_plain_lease(lease), actor, owner_key="owner_id", any_owner_action="leases.manage:any_owner")
        token = _fencing_arg(args, lease)
        lease_record = _plain_lease(lease)
        if lease_record.get("state") == LeaseState.RELEASED.value:
            return lease_record
        linked_jobs = [item for item in self.ledger.list("jobs") if item.get("lease_id") == lease_id]
        supervised = lease_record.get("usage", LeaseUsage.EXTERNAL.value) == LeaseUsage.SUPERVISED.value
        if not supervised and any(
            not (_state(item.get("state")) in _TERMINAL and item.get("cleanup_verified") is True)
            for item in linked_jobs
        ):
            raise ConflictError("managed job resources are released only after verified job cleanup")
        linked_backend = next(
            (item for item in self.ledger.list("backend") if item.get("lease_id") == lease_id),
            None,
        )
        if linked_backend is not None:
            raise ConflictError("managed backend residency is protected by backend lifecycle and handoff operations")
        requested = self.allocator.request_release(lease_id, token)
        token = int(requested["fencing_token"])
        lease_record = _plain_lease(self._get_lease(lease_id))
        receipt: Mapping[str, Any] | None = None
        if supervised:
            receipt = self._reconcile_supervised_lease_jobs(
                lease_record, token, owner=actor.principal_id
            )
            if receipt is not None:
                operation_id = str(receipt.get("operation_id", ""))
                if not _receipt_covers(receipt, lease_record, fencing_token=token, operation_id=operation_id):
                    receipt = None
        elif linked_jobs and all(
            item.get("cleanup_verified") is True and _state(item.get("state")) in _TERMINAL
            for item in linked_jobs
        ):
            receipt = {
                "verified": True,
                "lease_id": lease_id,
                "fencing_token": token,
                "resource_ids": [item["resource_id"] for item in lease_record.get("allocations", [])],
                "evidence": {
                    "jobs": [
                        {"job_id": item.get("id", item.get("job_id")), "attempt_id": item.get("attempt_id")}
                        for item in linked_jobs
                    ]
                },
            }
        else:
            allocations = tuple(ResourceAllocation.from_dict(item) for item in lease_record.get("allocations", ()))
            mode, adapter, descriptors = self._external_lease_reclamation_path(
                allocations, usage=str(lease_record.get("usage", LeaseUsage.EXTERNAL.value))
            )
            if mode == "observe" and adapter is not None:
                candidate = self._observe_external_lease_release(
                    lease_record, adapter, descriptors, owner=actor.principal_id
                )
                if candidate is not None and _receipt_covers(candidate, lease_record, fencing_token=token):
                    receipt = candidate
            elif mode == "reclaim" and adapter is not None:
                reclaim = getattr(adapter, "reclaim_lease")
                operation_id = f"lease-release:{lease_id}:{token}"
                intent = self._ensure_operation(operation_id, "lease.release", {"lease_id": lease_id, "fencing_token": token}, actor.principal_id)
                if intent.get("status") == "complete":
                    candidate = intent.get("receipt") or {}
                elif not intent.get("created"):
                    # An earlier process may have performed the reclamation but
                    # failed before persisting its receipt. Do not repeat an
                    # ambiguous host effect.
                    candidate = None
                else:
                    try:
                        raw_candidate = reclaim(lease_record, operation_id=operation_id)
                        candidate = dict(raw_candidate) if isinstance(raw_candidate, Mapping) else {}
                    except Exception as exc:
                        candidate = {"verified": False, "error": type(exc).__name__}
                    self._record_operation(
                        operation_id,
                        candidate,
                        status="complete" if candidate.get("operation_id") == operation_id and candidate.get("verified") is True else "unknown",
                    )
                if candidate is not None and candidate.get("verified") is True and _receipt_covers(
                    candidate, lease_record, fencing_token=token, operation_id=operation_id
                ):
                    receipt = candidate
        if receipt is None:
            self.allocator.mark_unknown(lease_id, token, "no trusted supervised-job or adapter evidence has verified resource reclamation")
            return _plain_lease(self._get_lease(lease_id))
        current = self.allocator.begin_reclamation(lease_id, token)
        return _json_value(self.allocator.verify_release(lease_id, int(current["fencing_token"]), receipt))

    def _reconcile_supervised_lease_jobs(
        self,
        lease: Mapping[str, Any],
        fencing_token: int,
        *,
        owner: str,
    ) -> Mapping[str, Any] | None:
        """Cancel/reconcile only scheduler-created attempts before releasing a supervised lease."""
        lease_id = str(lease.get("lease_id", ""))
        resource_ids = sorted(str(item["resource_id"]) for item in lease.get("allocations", ()))
        jobs = [item for item in self.ledger.list("jobs") if item.get("lease_id") == lease_id]
        operation_id = f"lease-release:{lease_id}:{fencing_token}"
        intent = self._ensure_operation(
            operation_id,
            "lease.supervised_release",
            {
                "lease_id": lease_id,
                "fencing_token": fencing_token,
                "resource_ids": resource_ids,
                "job_ids": sorted(str(item.get("id", item.get("job_id"))) for item in jobs),
            },
            owner,
        )
        if intent.get("status") == "complete":
            receipt = intent.get("receipt") or {}
            return receipt if _receipt_covers(receipt, lease, fencing_token=fencing_token, operation_id=operation_id) else None

        for job in jobs:
            job_id = str(job.get("id", job.get("job_id", "")))
            current = self.get_job(job_id)
            if _state(current.get("state")) in _TERMINAL and current.get("cleanup_verified") is True:
                continue
            try:
                spec = JobSpec.from_dict(current.get("spec", {}))
            except ValidationError:
                self._record_operation(operation_id, {"state": "unknown", "reason": "bound job spec cannot be validated"}, status="unknown")
                return None
            if not spec.interruption.cancellation_supported:
                self._record_operation(
                    operation_id,
                    {"state": "unknown", "reason": "bound job does not permit cancellation", "job_id": job_id},
                    status="unknown",
                )
                return None
            try:
                observed = self.cancel_job(job_id, reason=f"lease_release:{operation_id}")
            except Exception as exc:
                observed = {"state": "unknown", "cleanup_verified": False, "error": type(exc).__name__}
            if _state(observed.get("state")) not in _TERMINAL or observed.get("cleanup_verified") is not True:
                self._record_operation(
                    operation_id,
                    {"state": "unknown", "reason": "attached job cleanup remains unverified", "job_id": job_id},
                    status="unknown",
                )
                return None

        # Check every job and attempt record; the high-level job flag alone is
        # not enough if a launch or cleanup attempt is still ambiguous.
        evidence: list[dict[str, Any]] = []
        for raw in jobs:
            job_id = str(raw.get("id", raw.get("job_id", "")))
            current = self.get_job(job_id)
            if _state(current.get("state")) not in _TERMINAL or current.get("cleanup_verified") is not True:
                self._record_operation(
                    operation_id,
                    {"state": "unknown", "reason": "attached job is not terminal with verified cleanup", "job_id": job_id},
                    status="unknown",
                )
                return None
            attempt_id = current.get("attempt_id")
            if attempt_id:
                attempt = self.ledger.get("attempts", str(attempt_id))
                if attempt is None or attempt.get("cleanup_verified") is not True or _state(attempt.get("state")) == "unknown":
                    self._record_operation(
                        operation_id,
                        {"state": "unknown", "reason": "attached attempt cleanup receipt is unavailable", "job_id": job_id},
                        status="unknown",
                    )
                    return None
            evidence.append({
                "job_id": job_id,
                "attempt_id": attempt_id,
                "state": _state(current.get("state")),
                "cleanup_verified": True,
            })

        receipt = {
            "verified": True,
            "lease_id": lease_id,
            "fencing_token": fencing_token,
            "resource_ids": resource_ids,
            "operation_id": operation_id,
            "evidence": {
                "mode": "scheduler-supervised-jobs-only",
                "jobs": evidence,
                "no_unmanaged_process_isolation_claim": True,
            },
        }
        self._record_operation(operation_id, receipt, status="complete")
        return receipt

    def _observe_external_lease_release(
        self,
        lease: Mapping[str, Any],
        adapter: Any,
        resources: Sequence[Resource],
        *,
        owner: str,
    ) -> Mapping[str, Any] | None:
        """Repeat a read-only reclamation observation after every release attempt."""
        lease_id = str(lease.get("lease_id", ""))
        fencing_token = lease.get("fencing_token")
        evidence = self.ledger.get("lease_evidence", lease_id)
        resource_ids = sorted(item.resource_id for item in resources)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("lease_id") != lease_id
            or evidence.get("resource_ids") != resource_ids
            or not isinstance(evidence.get("baseline"), Mapping)
            or evidence.get("adapter_id") != str(getattr(adapter, "adapter_id", type(adapter).__name__))
        ):
            return None
        operation_id = f"lease-release-observe:{lease_id}:{fencing_token}:{uuid.uuid4()}"
        self._ensure_operation(
            operation_id,
            "lease.observe_release",
            {"lease_id": lease_id, "fencing_token": fencing_token, "resource_ids": resource_ids},
            owner,
        )
        try:
            raw_receipt = adapter.inspect_lease_release(
                lease,
                evidence["baseline"],
                resources,
                operation_id=operation_id,
            )
            receipt = dict(raw_receipt) if isinstance(raw_receipt, Mapping) else {}
        except Exception as exc:
            receipt = {
                "state": "unknown",
                "verified": False,
                "lease_id": lease_id,
                "fencing_token": fencing_token,
                "resource_ids": [],
                "operation_id": operation_id,
                "error": type(exc).__name__,
            }
        if (
            receipt.get("operation_id") != operation_id
            or receipt.get("lease_id") != lease_id
            or receipt.get("fencing_token") != fencing_token
            or receipt.get("resource_ids") != resource_ids
        ):
            receipt = {
                **receipt,
                "state": "unknown",
                "verified": False,
                "reason": "release observation receipt identity or allocation coverage did not match its request",
            }
        self._record_operation(
            operation_id,
            receipt,
            status="complete" if receipt.get("operation_id") == operation_id else "unknown",
        )
        return receipt

    def _reconcile_expired_leases(self, expired_claim_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Reclaim expired external leases only through a matching trusted adapter.

        Expiry fences the lease holder but does not prove that its process has
        stopped. Managed jobs and resident backends have their own supervision
        contracts, so their claims remain held for those owners to reconcile.
        """
        outcomes: list[dict[str, Any]] = []
        for claim_id in expired_claim_ids:
            raw_lease = self.ledger.get("lease", str(claim_id))
            if raw_lease is None:
                # expire_due also returns expired reservation IDs.
                continue
            lease = _without_revision(raw_lease)
            lease_id = str(lease.get("lease_id", claim_id))
            state = _state(lease.get("state"))
            if state == LeaseState.RELEASED.value:
                continue

            linked_job = next(
                (item for item in self.ledger.list("jobs") if item.get("lease_id") == lease_id),
                None,
            )
            if lease.get("usage", LeaseUsage.EXTERNAL.value) == LeaseUsage.SUPERVISED.value:
                token = int(lease["fencing_token"])
                receipt = self._reconcile_supervised_lease_jobs(
                    lease, token, owner=str(lease.get("owner_id", "scheduler"))
                )
                if receipt is None or not _receipt_covers(receipt, lease, fencing_token=token, operation_id=str(receipt.get("operation_id", ""))):
                    self.allocator.mark_unknown(
                        lease_id, token, "expired supervised lease jobs are not all verified clean"
                    )
                    outcomes.append({
                        "lease_id": lease_id,
                        "state": LeaseState.UNKNOWN.value,
                        "reason": "supervised_job_cleanup_unverified",
                    })
                    continue
                try:
                    reclamation = self.allocator.begin_reclamation(lease_id, token)
                    released = self.allocator.verify_release(
                        lease_id, int(reclamation["fencing_token"]), receipt
                    )
                    outcomes.append({
                        "lease_id": lease_id,
                        "state": _state(released.get("state")),
                        "operation_id": receipt.get("operation_id"),
                    })
                except Exception as exc:
                    outcomes.append({"lease_id": lease_id, "state": "unknown", "error": type(exc).__name__})
                continue
            if linked_job is not None:
                if linked_job.get("cleanup_verified") is True and _state(linked_job.get("state")) in _TERMINAL:
                    try:
                        self._release_job_lease(linked_job, linked_job.get("result") or {})
                        outcomes.append({"lease_id": lease_id, "state": LeaseState.RELEASED.value, "owner": "job_cleanup"})
                    except Exception as exc:
                        outcomes.append({"lease_id": lease_id, "state": "unknown", "error": type(exc).__name__})
                else:
                    outcomes.append({"lease_id": lease_id, "state": state, "reason": "managed_job_reconciliation_required"})
                continue

            linked_backend = next(
                (item for item in self.ledger.list("backend") if item.get("lease_id") == lease_id),
                None,
            )
            if linked_backend is not None:
                outcomes.append({"lease_id": lease_id, "state": state, "reason": "managed_backend_residency_protected"})
                continue

            allocations = tuple(ResourceAllocation.from_dict(item) for item in lease.get("allocations", ()))
            mode, adapter, descriptors = self._external_lease_reclamation_path(
                allocations, usage=str(lease.get("usage", LeaseUsage.EXTERNAL.value))
            )
            if mode is None or adapter is None:
                outcomes.append({"lease_id": lease_id, "state": state, "reason": "no_matching_reclamation_adapter"})
                continue

            token = int(lease["fencing_token"])
            if mode == "observe":
                try:
                    receipt = self._observe_external_lease_release(
                        lease, adapter, descriptors, owner=str(lease.get("owner_id", "scheduler"))
                    )
                    if receipt is None or not _receipt_covers(receipt, lease, fencing_token=token):
                        self.allocator.mark_unknown(lease_id, token, "expired external lease release observation is not verified")
                        outcomes.append({"lease_id": lease_id, "state": LeaseState.UNKNOWN.value, "reason": "reclamation_unverified"})
                        continue
                    reclamation = self.allocator.begin_reclamation(lease_id, token)
                    released = self.allocator.verify_release(lease_id, int(reclamation["fencing_token"]), receipt)
                    outcomes.append({"lease_id": lease_id, "state": _state(released.get("state")), "operation_id": receipt.get("operation_id")})
                except Exception as exc:
                    outcomes.append({"lease_id": lease_id, "state": "unknown", "error": type(exc).__name__})
                continue
            operation_id = f"lease-release:{lease_id}:{token}"
            try:
                intent = self._ensure_operation(
                    operation_id,
                    "lease.release",
                    {"lease_id": lease_id, "fencing_token": token, "trigger": "expiry"},
                    str(lease.get("owner_id", "scheduler")),
                )
                if intent.get("status") == "complete":
                    receipt = intent.get("receipt") or {}
                elif not intent.get("created"):
                    # A prior adapter call may have completed without a durable
                    # receipt. Never repeat that ambiguous external effect.
                    self.allocator.mark_unknown(lease_id, token, "expired lease reclamation has an unresolved operation")
                    outcomes.append({"lease_id": lease_id, "state": LeaseState.UNKNOWN.value, "reason": "reclamation_operation_unresolved"})
                    continue
                elif mode == "reclaim":
                    reclaim = adapter.reclaim_lease
                    try:
                        raw_candidate = reclaim(lease, operation_id=operation_id)
                        candidate = dict(raw_candidate) if isinstance(raw_candidate, Mapping) else {}
                    except Exception as exc:
                        candidate = {"verified": False, "error": type(exc).__name__}
                    receipt = candidate
                    valid = _receipt_covers(receipt, lease, fencing_token=token, operation_id=operation_id)
                    self._record_operation(operation_id, receipt, status="complete" if valid else "unknown")
                    if not valid:
                        self.allocator.mark_unknown(lease_id, token, "expired lease reclamation was not verified")
                        outcomes.append({"lease_id": lease_id, "state": LeaseState.UNKNOWN.value, "reason": "reclamation_unverified"})
                        continue
                if not _receipt_covers(receipt, lease, fencing_token=token, operation_id=operation_id):
                    self.allocator.mark_unknown(lease_id, token, "persisted expired-lease receipt does not cover its allocation")
                    outcomes.append({"lease_id": lease_id, "state": LeaseState.UNKNOWN.value, "reason": "receipt_coverage_invalid"})
                    continue
                reclamation = self.allocator.begin_reclamation(lease_id, token)
                released = self.allocator.verify_release(lease_id, int(reclamation["fencing_token"]), receipt)
                outcomes.append({"lease_id": lease_id, "state": _state(released.get("state")), "operation_id": operation_id})
            except Exception as exc:
                outcomes.append({"lease_id": lease_id, "state": "unknown", "error": type(exc).__name__})
        return outcomes

    def _lease_reclamation_adapters(self, lease: Mapping[str, Any]) -> list[Any]:
        """Select one adapter with a complete trusted release contract."""
        allocations = tuple(ResourceAllocation.from_dict(item) for item in lease.get("allocations", ()))
        _mode, adapter, _resources = self._external_lease_reclamation_path(
            allocations, usage=str(lease.get("usage", LeaseUsage.EXTERNAL.value))
        )
        return [adapter] if adapter is not None else []

    def _prepare_job_spec(self, raw: Any, owner: str, actor: Principal | None) -> JobSpec:
        if not isinstance(raw, Mapping):
            if hasattr(raw, "to_dict"):
                raw = raw.to_dict()
            else:
                raise ValidationError("job spec must be an object")
        data = dict(raw)
        forbidden = {"backend_config", "backend_configuration", "pinned_configuration", "secret_values"} & set(data)
        if forbidden:
            raise ValidationError(f"job spec cannot embed trusted configuration fields: {', '.join(sorted(forbidden))}")
        requested_owner = data.get("owner_id", owner)
        if requested_owner != owner and not (actor is not None and self._has(actor, "jobs.submit:any_owner")):
            raise PermissionDeniedError("job owner must match the authenticated principal")
        data["owner_id"] = owner
        raw_tags = data.get("tags", ())
        if isinstance(raw_tags, str) or not isinstance(raw_tags, Sequence):
            raise ValidationError("tags must be an array of strings")
        tags = set(raw_tags)
        requirements: list[dict[str, Any]] = []
        for requirement in data.get("resources", ()):
            req_data = requirement.to_dict() if hasattr(requirement, "to_dict") else dict(requirement)
            if GPU_EXCLUSIVE_TAG in tags and str(req_data.get("kind", "")).casefold() in _GPU_KINDS:
                req_data["mode"] = AllocationMode.EXCLUSIVE.value
            requirements.append(req_data)
        data["resources"] = requirements
        if actor is not None:
            requested_priority = data.get("priority", 50)
            if isinstance(requested_priority, bool) or not isinstance(requested_priority, int):
                raise ValidationError("priority must be an integer")
            effective_priority = min(requested_priority, actor.max_priority)
            requested_class = data.get("priority_class", "normal")
            cap_class = "interactive" if actor.max_priority >= 80 else "normal" if actor.max_priority >= 30 else "background"
            rank = {"background": 0, "normal": 1, "interactive": 2}
            if requested_class not in rank:
                raise ValidationError("priority_class must be interactive, normal, or background")
            effective_class = min((requested_class, cap_class), key=lambda value: rank[value])
            data["priority"] = effective_priority
            data["priority_class"] = effective_class
        profile_id = data.get("profile_id")
        if profile_id is not None:
            profile = self.config.profiles.get(profile_id)
            if not isinstance(profile, Mapping):
                raise PermissionDeniedError(f"execution profile {profile_id!r} is not operator-registered")
            trusted_cwd = profile.get("working_directory")
            if data.get("working_directory") not in (None, trusted_cwd) and data.get("working_directory") != trusted_cwd:
                raise PermissionDeniedError("profile working directory is fixed by trusted configuration")
            if trusted_cwd is not None:
                data["working_directory"] = str(trusted_cwd)
            allowed_env = set(profile.get("allowed_environment", ()))
            requested_env = data.get("environment", {}) or {}
            if not isinstance(requested_env, Mapping) or set(requested_env) - allowed_env:
                raise PermissionDeniedError("profile environment overrides exceed the trusted allowlist")
            base_env = profile.get("environment", {})
            if not isinstance(base_env, Mapping):
                raise ValidationError("trusted profile environment must be an object")
            data["environment"] = {**dict(base_env), **dict(requested_env)}
            trusted_refs = set(profile.get("secret_refs", ()))
            requested_refs = set(data.get("secret_refs", ()))
            if requested_refs - trusted_refs:
                raise PermissionDeniedError("profile secret references are fixed by trusted configuration")
            data["secret_refs"] = sorted(trusted_refs)
            data["preconditions"] = self._profile_preconditions(
                data.get("preconditions", ()), profile.get("required_preconditions", ())
            )
        else:
            data["secret_refs"] = self._validate_secret_refs(data.get("secret_refs", ()))
        spec = JobSpec.from_dict(data)
        self._validate_working_directory(spec, actor)
        return spec

    @staticmethod
    def _profile_preconditions(
        requested: Any, required: Any
    ) -> list[dict[str, Any]]:
        """Add trusted profile conditions without allowing request-side weakening."""
        if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
            raise ValidationError("preconditions must be an array")
        if isinstance(required, (str, bytes)) or not isinstance(required, Sequence):
            raise ValidationError("trusted profile required_preconditions must be an array")
        requested_models = [
            item if isinstance(item, CapabilityPrecondition) else CapabilityPrecondition.from_dict(item)
            for item in requested
        ]
        required_models = [
            item if isinstance(item, CapabilityPrecondition) else CapabilityPrecondition.from_dict(item)
            for item in required
        ]
        by_name = {item.name: item for item in requested_models}
        for condition in required_models:
            provided = by_name.get(condition.name)
            if provided is not None and provided.parameters != condition.parameters:
                raise PermissionDeniedError(
                    f"profile required precondition {condition.name!r} cannot be overridden by a job request"
                )
            by_name[condition.name] = condition
        return [item.to_dict() for item in by_name.values()]

    def _validate_secret_refs(self, values: Sequence[str]) -> list[str]:
        refs = set(values)
        if refs - self.config.secret_refs:
            raise PermissionDeniedError("job secret references must be provisioned by the operator")
        return sorted(refs)

    def _validate_working_directory(self, spec: JobSpec, actor: Principal | None) -> Path:
        if not spec.working_directory:
            raise ValidationError("job working_directory is required")
        cwd = Path(spec.working_directory).expanduser().resolve()
        if actor is None:
            return cwd
        roots = [Path(root).expanduser().resolve() for root in actor.execution_roots]
        if not roots:
            raise PermissionDeniedError("principal has no execution root grant")
        if not any(_is_relative_to(cwd, root) for root in roots):
            raise PermissionDeniedError("working_directory is outside the principal's execution roots")
        return cwd

    def _control_job(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        job_id = _required(args, "job_id")
        job = self.get_job(job_id)
        self._require_owner(job, actor, owner_key="owner", any_owner_action="jobs.control:any_owner")
        action = _required(args, "action")
        if action == "cancel":
            return self.cancel_job(job_id, str(args.get("reason", "caller")))
        if action == "retry":
            return self._retry_job(job, actor, args)
        if action in {"suspend", "resume"}:
            raise UnsupportedCapabilityError(f"job {action} requires an adapter contract not available for this attempt")
        if action == "recover":
            return self._recover_job(job)
        raise ValidationError("job action must be cancel, retry, suspend, resume, or recover")

    def _retry_job(self, job: Mapping[str, Any], actor: Principal, args: Mapping[str, Any]) -> dict[str, Any]:
        if not (self._has(actor, "jobs.retry") or self._has(actor, "jobs.submit")):
            raise PermissionDeniedError("retry requires jobs.retry or jobs.submit authorization")
        spec_data = dict(job.get("spec") or {})
        if not (spec_data.get("interruption") or {}).get("restart_safe"):
            raise ConflictError("job is not declared safe to restart")
        current_attempts = int(job.get("attempt_count", 1))
        if current_attempts >= int(spec_data.get("max_attempts", 1)):
            raise ConflictError("job has reached max_attempts")
        if _state(job.get("state")) not in {"failed", "cancelled"}:
            raise ConflictError("only failed or cancelled jobs may be retried")
        key = _idempotency_key(args)
        spec = self._prepare_job_spec(spec_data, str(job.get("owner", actor.principal_id)), actor)
        return self.submit_job(spec.to_dict(), str(job.get("owner", actor.principal_id)), key)

    def _recover_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        if _state(job.get("state")) != "unknown":
            raise ConflictError("only unknown jobs need recovery")
        attempt_id = job.get("attempt_id")
        if not attempt_id or self.executor is None:
            raise UnsupportedCapabilityError("job recovery requires an execution adapter and attempt identity")
        self._reconcile_one(job)
        return self.get_job(str(job.get("id", job.get("job_id"))))

    def _wait_job(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        job_id = _required(args, "job_id")
        job = self.get_job(job_id)
        self._require_owner(job, actor, owner_key="owner", any_owner_action="jobs.read:any_owner")
        requested = args.get("timeout_seconds", 0)
        if isinstance(requested, bool) or not isinstance(requested, (int, float)) or requested < 0:
            raise ValidationError("timeout_seconds must be non-negative")
        timeout = min(float(requested), self.config.max_wait_seconds)
        target_state = set(args.get("terminal_states", _TERMINAL))
        deadline = self._monotonic() + timeout
        while _state(job.get("state")) not in target_state and self._monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - self._monotonic())))
            job = self.get_job(job_id)
        return {"job": self._job_detail(job), "timed_out": _state(job.get("state")) not in target_state}

    def _job_logs(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        job = self.get_job(_required(args, "job_id"))
        self._require_owner(job, actor, owner_key="owner", any_owner_action="jobs.read:any_owner")
        attempt_id = args.get("attempt_id", job.get("attempt_id"))
        if not attempt_id:
            return {"job_id": job.get("id"), "attempt_id": None, "stdout": bound_output("", max_bytes=self.config.max_output_bytes), "stderr": bound_output("", max_bytes=self.config.max_output_bytes)}
        attempt = self._get("attempts", attempt_id)
        if attempt.get("job_id") != job.get("id", job.get("job_id")):
            raise PermissionDeniedError("attempt does not belong to this job")
        result = dict(attempt.get("result") or {})
        logs = result.get("logs") if isinstance(result.get("logs"), Mapping) else {}
        stdout = result.get("stdout", logs.get("stdout", logs.get("text", "")))
        stderr = result.get("stderr", logs.get("stderr", ""))
        return {
            "job_id": job.get("id"),
            "attempt_id": attempt_id,
            "stdout": bound_output(stdout or "", max_bytes=self.config.max_output_bytes, max_lines=self.config.max_output_lines),
            "stderr": bound_output(stderr or "", max_bytes=self.config.max_output_bytes, max_lines=self.config.max_output_lines),
            "cleanup_verified": bool(attempt.get("cleanup_verified")),
        }

    def _register_backend(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        raw = args.get("spec", args)
        if not isinstance(raw, Mapping):
            raise ValidationError("backend spec must be an object")
        data = dict(raw)
        forbidden = {"configuration", "backend_config", "pinned_configuration", "argv", "command"} & set(data)
        if forbidden:
            raise ValidationError("backend launch configuration must come from an operator profile")
        spec = BackendSpec.from_dict({**data, "pinned_configuration": {}})
        self._require_backend_scope(actor, spec.backend_id)
        profile = self.config.backend_profiles.get(spec.profile_id)
        if not isinstance(profile, Mapping):
            raise PermissionDeniedError(f"backend profile {spec.profile_id!r} is not operator-registered")
        trusted_ref = profile.get("configuration_ref", spec.profile_id)
        if spec.configuration_ref != trusted_ref:
            raise PermissionDeniedError("backend configuration_ref must match its trusted profile")
        pinned = {"profile_version": profile.get("version")}
        if not pinned["profile_version"]:
            raise ValidationError("trusted backend profile must declare a version")
        spec = replace(spec, pinned_configuration=pinned)
        now = _iso(_now(self.clock))
        record = {
            "id": spec.backend_id,
            "backend_id": spec.backend_id,
            "owner": actor.principal_id,
            "spec": spec.to_dict(),
            "state": "stopped",
            "lease_id": None,
            "execution_handle": None,
            "registered_at": now,
            "updated_at": now,
            "observation": None,
        }
        key = _idempotency_key(args)
        payload = {"backend_id": spec.backend_id, "spec": spec.to_dict()}

        def create(tx: Any) -> dict[str, Any]:
            current = tx.get("backend", spec.backend_id)
            if current is not None and current.get("owner") != actor.principal_id and not self._has(actor, "backends.register:any_owner"):
                raise PermissionDeniedError("backend belongs to another owner")
            if current is not None:
                raise ConflictError("backend is already registered")
            saved = tx.put("backend", spec.backend_id, record, None)
            tx.event("backend.registered", spec.backend_id, {"owner": actor.principal_id, "profile_id": spec.profile_id})
            return saved

        return self.ledger.idempotent(actor.principal_id, key, "backends.register", payload, create)

    def _control_backend(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        backend_id = _required(args, "backend_id")
        self._require_backend_scope(actor, backend_id)
        record = self._get("backend", backend_id)
        self._require_owner(record, actor, owner_key="owner", any_owner_action="backends.control:any_owner")
        action = _required(args, "action")
        if action == "inspect":
            return self._backend_detail(record, refresh=True)
        if action == "start":
            if self.backend_adapter is None:
                raise UnsupportedCapabilityError("no managed backend adapter is configured")
            state = _state(record.get("state"))
            if state == "ready":
                return self._backend_detail(record, refresh=True)
            if state != "stopped":
                raise ConflictError(f"backend in {state!r} state cannot be started until it is reconciled")
            return self._start_backend(record, actor)
        if action in {"evict", "stop", "unload"}:
            raise PermissionDeniedError("managed backend eviction must occur inside an authorized handoff")
        raise ValidationError("backend action must be inspect or start")

    def _reconcile_backends(self) -> list[dict[str, Any]]:
        """Resolve persisted backend start effects without launching a backend."""
        if self.backend_adapter is None:
            return []
        updated: list[dict[str, Any]] = []
        operation_get = getattr(self.ledger, "operation", None)
        if not callable(operation_get):
            return updated
        for record in self.ledger.list("backend"):
            backend_id = str(record.get("backend_id", record.get("id", "")))
            if not backend_id or record.get("handoff_id"):
                continue
            operation_id = f"backend-start:{backend_id}"
            operation = operation_get(operation_id)
            state = _state(record.get("state"))
            if operation is None:
                if state in {"starting", "unknown"}:
                    raise ReconciliationRequiredError(
                        f"backend {backend_id!r} has no durable start operation"
                    )
                continue
            if state == "ready" and operation.get("status") == "complete":
                continue
            if operation.get("status") in {"failed"}:
                raise ReconciliationRequiredError(
                    f"backend {backend_id!r} has a failed start operation"
                )
            spec = BackendSpec.from_dict(record["spec"])
            try:
                observation = self.backend_adapter.inspect(spec)
            except Exception as exc:
                observation = {"state": "unknown", "error": type(exc).__name__}
            if not isinstance(observation, Mapping):
                observation = {"state": "unknown", "reason": "backend inspect returned a non-object"}
            if _state(observation.get("state")) not in {"running", "ready"}:
                if operation.get("status") not in {"complete", "failed"}:
                    self._record_operation(
                        operation_id,
                        {"state": "unknown", "observation": _json_value(observation)},
                        status="unknown",
                    )
                self._update("backend", backend_id, {
                    "state": "unknown",
                    "observation": _json_value(observation),
                    "updated_at": _iso(_now(self.clock)),
                })
                raise ReconciliationRequiredError(
                    f"backend {backend_id!r} start outcome remains unknown"
                )
            receipt = self._verify_and_reopen_backend(
                spec,
                operation_id,
                observation,
                admission_preparation=record.get("start_admission_receipt"),
            )
            if _state(receipt.get("state")) != "ready":
                if operation.get("status") not in {"complete", "failed"}:
                    self._record_operation(operation_id, receipt, status="unknown")
                self._update("backend", backend_id, {
                    "state": "unknown",
                    "observation": _json_value(receipt),
                    "updated_at": _iso(_now(self.clock)),
                })
                raise ReconciliationRequiredError(
                    f"backend {backend_id!r} could not be verified ready"
                )
            if operation.get("status") not in {"complete", "failed"}:
                self._record_operation(operation_id, receipt, status="complete")
            updated.append(self._update("backend", backend_id, {
                "state": "ready",
                "execution_handle": receipt.get("execution_handle", record.get("execution_handle")),
                "observation": _json_value(receipt),
                "updated_at": _iso(_now(self.clock)),
            }))
        return updated

    def _start_backend(self, record: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        spec = BackendSpec.from_dict(record["spec"])
        backend_id = spec.backend_id
        lease = None
        new_lease = False
        if record.get("lease_id"):
            lease = _plain_lease(self._get_lease(record["lease_id"]))
        elif spec.residency_requirements:
            lease = _json_value(self.allocator.request_lease(ResourceRequest(
                request_id=f"backend:{backend_id}", owner_id=str(record.get("owner", actor.principal_id)),
                requirements=spec.residency_requirements,
                lease_ttl_seconds=self.config.default_lease_ttl_seconds,
                idempotency_key=f"backend-start:{backend_id}",
            )))
            new_lease = True
        operation_id = f"backend-start:{backend_id}"
        try:
            with self.ledger.transaction() as tx:
                execution_context = self._backend_execution_context(
                    tx,
                    lease_id=str(lease["lease_id"]) if lease and lease.get("lease_id") else None,
                )
        except Exception:
            if new_lease and lease and lease.get("lease_id"):
                self._release_unstarted_job_lease(lease)
            raise
        start_payload = {
            "backend_id": backend_id,
            "spec": spec.to_dict(),
            "lease_id": lease.get("lease_id") if lease else None,
            "execution_context": execution_context.to_dict(),
        }
        intent = self._ensure_operation(operation_id, "backend.start", start_payload, actor.principal_id)
        if intent.get("status") == "complete":
            receipt = intent.get("receipt") or {}
        elif intent.get("created"):
            self._update("backend", backend_id, {
                "state": "starting",
                "lease_id": lease.get("lease_id") if lease else None,
                "start_operation_id": operation_id,
                "start_phase": "admission_prepare_intent",
                "updated_at": _iso(_now(self.clock)),
            })
            admission_preparation = self._prepare_initial_backend_admission(spec, operation_id, actor.principal_id)
            if not self._initial_admission_preparation_valid(spec, operation_id, admission_preparation):
                self._record_operation(
                    operation_id,
                    {"state": "unknown", "admission_preparation": _json_value(admission_preparation)},
                    status="unknown",
                )
                updated = self._update("backend", backend_id, {
                    "state": "unknown",
                    "lease_id": lease.get("lease_id") if lease else None,
                    "start_phase": "admission_unverified",
                    "observation": _json_value(admission_preparation),
                    "updated_at": _iso(_now(self.clock)),
                })
                return self._backend_detail(updated)
            self._update("backend", backend_id, {
                "state": "starting",
                "start_phase": "admission_closed",
                "start_admission_receipt": _json_value(admission_preparation),
                "updated_at": _iso(_now(self.clock)),
            })
            # Persist launch intent in the backend record before the adapter can
            # create an external process. Recovery inspects this identity and
            # never repeats an ambiguous launch.
            self._update("backend", backend_id, {
                "start_phase": "launch_intent",
                "updated_at": _iso(_now(self.clock)),
            })
            receipt = self._restore_backend(
                spec,
                operation_id,
                execution_context=execution_context,
                admission_preparation=admission_preparation,
            )
        else:
            # A previous process persisted this operation but left its outcome
            # unresolved. Inspect the deterministic backend identity; never
            # issue another launch merely because the receipt is missing.
            try:
                observation = self.backend_adapter.inspect(spec)
            except Exception as exc:
                observation = {"state": "unknown", "error": type(exc).__name__}
            if _state(observation.get("state")) in {"running", "ready"}:
                receipt = self._verify_and_reopen_backend(
                    spec,
                    operation_id,
                    observation,
                    admission_preparation=record.get("start_admission_receipt"),
                )
                if _state(receipt.get("state")) == "ready":
                    self._record_operation(operation_id, receipt, status="complete")
                else:
                    self._record_operation(operation_id, receipt, status="unknown")
            else:
                receipt = {"state": "unknown", "reason": "prior backend start requires explicit reconciliation", "observation": _json_value(observation)}
                if intent.get("status") == "intent":
                    self._record_operation(operation_id, receipt, status="unknown")
        if _state(receipt.get("state")) != "ready":
            updated = self._update("backend", backend_id, {
                "state": "unknown",
                "lease_id": lease.get("lease_id") if lease else None,
                "observation": dict(receipt),
                "updated_at": _iso(_now(self.clock)),
            })
            return self._backend_detail(updated)
        updated = self._update("backend", backend_id, {
            "state": "ready", "lease_id": lease.get("lease_id") if lease else record.get("lease_id"),
            "execution_handle": receipt.get("execution_handle"), "observation": dict(receipt),
            "start_phase": "ready",
            "updated_at": _iso(_now(self.clock)),
        })
        return self._backend_detail(updated)

    def _backend_execution_context(self, tx: Any, *, lease_id: str | None) -> ExecutionContext:
        if lease_id is None:
            return ExecutionContext(purpose=ExecutionPurpose.BACKEND)
        lease = tx.get("lease", lease_id)
        if lease is None or lease.get("state") != LeaseState.ACTIVE.value:
            raise ConflictError("backend residency lease is not active")
        if _parse_time(lease.get("expires_at"), "lease expires_at") <= _now(self.clock):
            raise ConflictError("backend residency lease has expired")
        return self._execution_context_from_claim(
            tx,
            lease.get("allocations", ()),
            claim_id=lease_id,
            fencing_token=int(lease["fencing_token"]),
            purpose=ExecutionPurpose.BACKEND,
        )

    def _initial_admission_preparation_valid(
        self,
        spec: BackendSpec,
        operation_id: str,
        receipt: Mapping[str, Any],
    ) -> bool:
        if receipt.get("backend_id") != spec.backend_id or receipt.get("operation_id") != operation_id:
            return False
        state = _state(receipt.get("state"))
        if state == "not_required":
            return (
                receipt.get("admission_supported") is False
                and receipt.get("eviction_supported") is False
                and receipt.get("admission_closed") is False
            )
        epoch = receipt.get("admission_epoch")
        valid_epoch = not isinstance(epoch, bool) and (
            isinstance(epoch, int) and epoch >= 0
            or isinstance(epoch, str) and bool(epoch.strip())
        )
        active_generations = receipt.get("active_generations")
        return (
            state in {"closed", "admission_closed"}
            and receipt.get("admission_closed") is True
            and valid_epoch
            and isinstance(active_generations, int)
            and not isinstance(active_generations, bool)
            and active_generations == 0
            and isinstance(receipt.get("active_generations_source"), str)
            and bool(receipt.get("active_generations_source"))
        )

    def _prepare_initial_backend_admission(
        self,
        spec: BackendSpec,
        operation_id: str,
        owner: str,
    ) -> dict[str, Any]:
        gate_operation_id = f"backend-start-admission:{spec.backend_id}"
        payload = {
            "backend_id": spec.backend_id,
            "backend_operation_id": operation_id,
            "spec": spec.to_dict(),
        }
        intent = self._ensure_operation(
            gate_operation_id,
            "backend.start.admission",
            payload,
            owner,
        )
        if intent.get("status") == "failed":
            return {"state": "blocked", "reason": "initial admission operation previously failed"}
        if intent.get("status") == "complete":
            return dict(intent.get("receipt") or {})
        try:
            prepare = getattr(self.backend_adapter, "prepare_initial_start", None)
            if not callable(prepare):
                raise UnsupportedCapabilityError("backend adapter lacks initial admission preparation")
            result = prepare(spec, operation_id=operation_id)
            receipt = dict(result) if isinstance(result, Mapping) else {"state": "unknown", "reason": "non-object admission receipt"}
        except Exception as exc:
            receipt = {
                "state": "unknown",
                "backend_id": spec.backend_id,
                "operation_id": operation_id,
                "reason": type(exc).__name__,
            }
        valid = self._initial_admission_preparation_valid(spec, operation_id, receipt)
        self._record_operation(gate_operation_id, receipt, status="complete" if valid else "unknown")
        return receipt

    def _restore_backend(
        self,
        spec: BackendSpec,
        operation_id: str,
        *,
        execution_context: ExecutionContext,
        admission_preparation: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            receipt = self.backend_adapter.restore(
                spec,
                operation_id=operation_id,
                execution_context=execution_context,
            )
            if not isinstance(receipt, Mapping):
                raise ValidationError("backend restore receipt must be an object")
            receipt = {**dict(receipt), "admission_preparation": _json_value(admission_preparation)}
            if _state(receipt.get("state")) == "ready":
                receipt = self._verify_and_reopen_backend(
                    spec,
                    operation_id,
                    receipt,
                    admission_preparation=admission_preparation,
                )
        except Exception as exc:
            receipt = {"state": "unknown", "error": type(exc).__name__}
        self._record_operation(operation_id, receipt, status="complete" if _state(receipt.get("state")) == "ready" else "unknown")
        return receipt

    def _verify_and_reopen_backend(
        self,
        spec: BackendSpec,
        operation_id: str,
        observation: Mapping[str, Any],
        *,
        admission_preparation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt = dict(observation)
        try:
            ready = self.backend_adapter.ready(
                spec,
                timeout_seconds=float(_profile_timeout(self.config, spec.profile_id)),
                operation_id=operation_id,
            )
            receipt["readiness"] = _json_value(ready)
            if ready.get("ready") is not True:
                receipt["state"] = _state(ready.get("state"), "unknown")
                return receipt
            if (
                admission_preparation is not None
                and _state(admission_preparation.get("state")) == "not_required"
                and self._initial_admission_preparation_valid(spec, operation_id, admission_preparation)
            ):
                receipt["admission"] = {
                    "state": "not_required",
                    "admission_open": False,
                    "reason": "registered backend profile has no FrameShift admission controller",
                }
                receipt["state"] = "ready"
                return receipt
            if (
                admission_preparation is None
                or not self._initial_admission_preparation_valid(spec, operation_id, admission_preparation)
            ):
                receipt["state"] = "unknown"
                receipt["admission_error"] = "initial admission closure is not verified for this start operation"
                return receipt
            reopened = self.backend_adapter.reopen(spec, operation_id=operation_id)
            receipt["admission"] = _json_value(reopened)
            opened_epoch_matches = reopened.get("admission_epoch") == admission_preparation.get("admission_epoch")
            if (
                opened_epoch_matches
                and (
                    reopened.get("opened") is True
                    or reopened.get("admission_open") is True
                    or reopened.get("state") in {"ready", "open"}
                )
            ):
                receipt["state"] = "ready"
            else:
                receipt["state"] = "unknown"
        except Exception as exc:
            receipt.update({"state": "unknown", "error": type(exc).__name__})
        return receipt

    def _backend_detail(self, record: Mapping[str, Any], *, refresh: bool = False) -> dict[str, Any]:
        view = _without_revision(record)
        if refresh and self.backend_adapter is not None:
            spec = BackendSpec.from_dict(record["spec"])
            observed = self.backend_adapter.inspect(spec)
            view["observation"] = _json_value(observed)
            observed_state = _state(observed.get("state"), _state(record.get("state")))
            view["observed_state"] = observed_state
        return _json_value(view)

    def _attach_backend(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        backend_id = _required(args, "backend_id")
        self._require_backend_scope(actor, backend_id)
        backend = self._get("backend", backend_id)
        self._require_owner(backend, actor, owner_key="owner", any_owner_action="backends.control:any_owner")
        if _state(backend.get("state")) != "ready":
            raise ConflictError("only a ready backend may accept a continuation consumer")
        continuation_ref = _required(args, "continuation_ref")
        if self.continuation_adapter is None:
            raise UnsupportedCapabilityError("no continuation adapter is configured")
        contract_id = _required(args, "contract_id")
        key = _idempotency_key(args)
        payload = {"backend_id": backend_id, "continuation_ref": continuation_ref, "contract_id": contract_id}
        operation_id = f"backend-attach:{backend_id}:{continuation_ref}"
        operation = self._ensure_operation(operation_id, "backend.attach_consumer", payload, actor.principal_id)
        if operation.get("status") == "complete":
            receipt = operation.get("receipt") or {}
        elif not operation.get("created"):
            raise ReconciliationRequiredError("consumer registration has an unresolved external operation")
        else:
            try:
                receipt = self.continuation_adapter.register_consumer(
                    continuation_ref, backend_id=backend_id, contract_id=contract_id
                )
            except Exception as exc:
                receipt = {"registered": False, "error": type(exc).__name__}
            self._record_operation(operation_id, receipt, status="complete" if receipt.get("registered") is True else "unknown")
        if receipt.get("registered") is not True:
            raise ReconciliationRequiredError("continuation adapter did not verify consumer registration")

        def attach(tx: Any) -> dict[str, Any]:
            current = tx.get("backend", backend_id)
            if current is None:
                raise NotFoundError(f"backend {backend_id!r} does not exist")
            consumers = list(current.get("attached_consumers", []))
            if continuation_ref in consumers:
                return current
            spec = dict(current["spec"])
            spec["attached_consumers"] = consumers + [continuation_ref]
            updated = _without_revision(current)
            updated["spec"] = spec
            updated["attached_consumers"] = consumers + [continuation_ref]
            updated["updated_at"] = _iso(_now(self.clock))
            saved = _put(tx, "backend", backend_id, updated, current)
            tx.event("backend.consumer_attached", backend_id, {"continuation_ref": continuation_ref, "owner": actor.principal_id})
            return saved

        return self.ledger.idempotent(actor.principal_id, key, "backends.attach", payload, attach)

    def _submit_handoff(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        if self.backend_adapter is None or self.continuation_adapter is None:
            raise UnsupportedCapabilityError("handoffs require managed-backend and continuation adapters")
        raw = args.get("spec", args)
        if not isinstance(raw, Mapping):
            raise ValidationError("handoff spec must be an object")
        data = dict(raw)
        owner = self._owner_from_args(data, actor, "handoffs.submit:any_owner")
        jobs_raw = data.get("jobs", ())
        if not isinstance(jobs_raw, Sequence) or isinstance(jobs_raw, (str, bytes)):
            raise ValidationError("handoff jobs must be an array")
        data["owner_id"] = owner
        data["jobs"] = [self._prepare_job_spec(item, owner, actor).to_dict() for item in jobs_raw]
        spec = HandoffSpec.from_dict(data)
        self._verify_eviction_grants(spec, actor)
        for backend_id in spec.backend_ids:
            self._require_backend_scope(actor, backend_id)
            self._get("backend", backend_id)
        key = _idempotency_key(args, data.get("idempotency_key"))
        return self._handoffs().submit(spec, owner, key)

    def _verify_eviction_grants(self, spec: HandoffSpec, actor: Principal) -> None:
        if not self._has(actor, "backends.evict"):
            raise PermissionDeniedError("handoff requires the backends.evict grant")
        now = _now(self.clock)
        grants: dict[str, EvictionGrant] = {grant.backend_id: grant for grant in spec.eviction_grants}
        for backend_id in spec.backend_ids:
            grant = grants.get(backend_id)
            if grant is None:
                raise PermissionDeniedError(f"backend {backend_id!r} has no eviction grant")
            if grant.granted_by != actor.principal_id and not self._has(actor, "backends.evict:any_grantor"):
                raise PermissionDeniedError("eviction grantor must match the authenticated principal")
            if grant.scope != f"backend:{backend_id}":
                raise PermissionDeniedError("eviction grant scope must name its exact backend")
            if grant.expires_at <= now:
                raise PermissionDeniedError("eviction grant has expired")

    def _control_handoff(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        handoff_id = _required(args, "handoff_id")
        handoff = self._handoffs().get(handoff_id)
        self._require_owner(handoff, actor, owner_key="owner", any_owner_action="handoffs.control:any_owner")
        action = _required(args, "action")
        if action == "cancel":
            return self._handoffs().cancel(handoff_id, str(args.get("reason", "caller")))
        if action == "recover":
            if not self._has(actor, "handoffs.recover"):
                raise PermissionDeniedError("handoff recovery requires handoffs.recover")
            return self._handoffs().recover(handoff_id)
        raise ValidationError("handoff action must be cancel or recover")

    def _put_schedule(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        raw = args.get("spec", args)
        if not isinstance(raw, Mapping):
            raise ValidationError("schedule spec must be an object")
        data = dict(raw)
        owner = self._owner_from_args(data, actor, "schedules.put:any_owner")
        job = self._prepare_job_spec(data.get("job"), owner, actor)
        data["owner_id"] = owner
        data["job"] = job.to_dict()
        spec = ScheduleSpec.from_dict(data)
        return self._schedules().put(spec, owner, _idempotency_key(args, spec.schedule_id))

    def _control_schedule(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        schedule_id = _required(args, "schedule_id")
        schedule = self._get("schedules", schedule_id)
        self._require_owner(schedule, actor, owner_key="owner_id", any_owner_action="schedules.control:any_owner")
        action = _required(args, "action")
        return self._schedules().control(schedule_id, action, str(schedule.get("owner_id")), _idempotency_key(args))

    def _explain_policy(self, args: Mapping[str, Any], actor: Principal) -> dict[str, Any]:
        explain = getattr(self.policy, "explain", None)
        if not callable(explain):
            raise UnsupportedCapabilityError("configured policy does not implement explain")
        decision_id = args.get("decision_id")
        if decision_id:
            snapshot = self._get("decisions", str(decision_id))
            subject_id = snapshot.get("subject_id")
            if subject_id:
                subject = self.ledger.get("jobs", str(subject_id))
                if subject is not None:
                    self._require_owner(subject, actor, owner_key="owner", any_owner_action="jobs.read:any_owner")
                elif not self._has(actor, "policies.explain:any_owner"):
                    raise PermissionDeniedError("decision subject is unavailable for owner verification")
            elif not self._has(actor, "policies.explain:any_owner"):
                raise PermissionDeniedError("decision is not associated with an inspectable owned job")
            return explain_decision(snapshot, now=_now(self.clock), max_observation_age_seconds=float(args.get("max_observation_age_seconds", 60)))
        job_id = args.get("job_id")
        if job_id:
            job = self.get_job(str(job_id))
            self._require_owner(job, actor, owner_key="owner", any_owner_action="jobs.read:any_owner")
            decision_id = job.get("decision_id")
            if decision_id:
                return explain_decision(self._get("decisions", decision_id), now=_now(self.clock))
            request = {"job_id": job.get("id"), "owner_id": job.get("owner"), "spec": job.get("spec"), "created_at": job.get("created_at")}
        elif isinstance(args.get("request"), Mapping):
            request = dict(args["request"])
            request.setdefault("request_id", request.get("job_id", "explain"))
            request.setdefault("owner_id", actor.principal_id)
        else:
            raise ValidationError("policies.explain requires decision_id, job_id, or request")
        decision = explain(request, now=_now(self.clock))
        observations = args.get("observations", ())
        snapshot = decision_snapshot(decision, observations=observations)
        return explain_decision(snapshot, now=_now(self.clock))

    def _simulate_policy(self, args: Mapping[str, Any]) -> dict[str, Any]:
        observations = args.get("observations")
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise ValidationError("observations must be an array of JSON objects")
        if any(not isinstance(item, Mapping) for item in observations):
            raise ValidationError("each policy observation must be an object")
        service_by_owner = args.get("service_by_owner")
        if service_by_owner is not None and not isinstance(service_by_owner, Mapping):
            raise ValidationError("service_by_owner must be an object mapping owner IDs to service values")
        return replay_policy(
            self.policy,
            observations,
            operation=str(args.get("operation", "admission")),
            now=_now(self.clock),
            service_by_owner=service_by_owner,
        )

    def _reconcile_admin(self) -> dict[str, Any]:
        jobs = self._reconcile_jobs()
        handoffs: list[dict[str, Any]] = []
        if self._has_pending_handoffs():
            # Coordinator tick advances existing durable handoffs only; it does
            # not infer a handoff from ordinary suspension or lease expiry.
            handoffs.append(_json_value(self._handoffs().tick()))
        return {"reconciled_jobs": jobs, "handoffs": handoffs, "status": self.status()}

    def _backup(self, args: Mapping[str, Any]) -> dict[str, Any]:
        backup = getattr(self.ledger, "backup", None)
        if not callable(backup):
            raise UnsupportedCapabilityError("ledger adapter does not support backups")
        requested = args.get("destination")
        if not isinstance(requested, str) or not requested:
            raise ValidationError("destination is required")
        destination = Path(requested).expanduser().resolve()
        root = self.config.backup_root or self.config.state_root
        if not _is_relative_to(destination, root):
            raise PermissionDeniedError("backup destination must be inside the configured backup root")
        path = backup(destination)
        return {"destination": str(path), "status": "complete"}

    def _admit_jobs(self) -> list[dict[str, Any]]:
        started: list[dict[str, Any]] = []
        candidates = []
        now = _now(self.clock)
        for job in self.ledger.list("jobs"):
            if _state(job.get("state")) not in {"queued", "blocked"}:
                continue
            spec = JobSpec.from_dict(job["spec"])
            if not scheduled_job_ready(self.ledger, job):
                self._update_job(
                    str(job.get("id", job.get("job_id"))),
                    {"state": "blocked", "blocked_reason": "schedule_occurrence_not_submitted"},
                )
                continue
            if spec.not_before is not None and spec.not_before > now:
                continue
            if spec.deadline is not None and spec.deadline <= now:
                self._update_job(str(job.get("id", job.get("job_id"))), {"state": "failed", "result": {"reason": "deadline_expired"}, "blocked_reason": None})
                continue
            if not dependencies_ready(self.ledger, spec.dependencies) or not event_ready(self.ledger, spec.event_trigger, created_at=job.get("created_at")):
                self._update_job(str(job.get("id", job.get("job_id"))), {"state": "blocked", "blocked_reason": "dependency_or_event_pending"})
                continue
            if self._blocked_by_priority_lease_demand(spec):
                self._update_job(str(job.get("id", job.get("job_id"))), {"state": "blocked", "blocked_reason": "priority_lease_demand_pending"})
                continue
            candidates.append(job)
        preemption_attempted = False
        while candidates:
            service = self._service_by_owner()
            policy_observations = self._policy_input_snapshot(candidates, now, service)
            decision = self.policy.choose_next(
                candidates,
                can_fit=lambda item: self._job_can_fit(item),
                safe_backfill=lambda candidate, protected: self._safe_backfill(candidate, protected),
                now=now,
                service_by_owner=service,
            )
            if decision is None:
                if not preemption_attempted:
                    preemption_attempted = True
                    if self._try_preemption(candidates, now=now, service_by_owner=service):
                        continue
                for job in candidates:
                    self._update_job(str(job.get("id", job.get("job_id"))), {"state": "blocked", "blocked_reason": "resource_request_not_currently_feasible"})
                break
            job = self._validate_policy_selection(decision, candidates)
            candidates.remove(job)
            try:
                record = self._start_job(job, decision, policy_observations=policy_observations)
                if record is not None:
                    started.append(record)
            except ConflictError as exc:
                # Preview and allocation are separated by time; the allocator's
                # transaction remains authoritative under cross-process races.
                self._update_job(str(job.get("id", job.get("job_id"))), {"state": "blocked", "blocked_reason": str(exc)[:500]})
        return started

    def _blocked_by_priority_lease_demand(self, spec: JobSpec) -> bool:
        if not spec.flexible or not spec.resources:
            return False
        for demand in self.ledger.list("lease_demands"):
            if demand.get("state") != "pending":
                continue
            try:
                request = ResourceRequest.from_dict(demand.get("request", {}))
            except ValidationError:
                # Corrupt/unknown demand state is treated conservatively.
                return True
            if not self._resource_requirements_may_conflict(spec.resources, request.requirements):
                continue
            return True
        return False

    def _resource_requirements_may_conflict(
        self, left: Sequence[ResourceRequirement], right: Sequence[ResourceRequirement]
    ) -> bool:
        try:
            resources_fn = getattr(self.allocator, "list_resources", None)
            raw = resources_fn(available_only=True) if callable(resources_fn) else self.ledger.list("resource")
            resources = [item if isinstance(item, Resource) else Resource.from_dict(item) for item in raw]
        except Exception:
            # Unknown topology means we cannot safely claim non-conflict.
            return True
        for a in left:
            for b in right:
                if a.kind != b.kind:
                    continue
                for resource in resources:
                    if not _requirement_matches_resource(a, resource) or not _requirement_matches_resource(b, resource):
                        continue
                    if a.mode == AllocationMode.EXCLUSIVE or b.mode == AllocationMode.EXCLUSIVE:
                        return True
                    if set(a.quantities).intersection(b.quantities):
                        return True
        return False

    def _validate_policy_selection(
        self, decision: Any, candidates: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Validate a policy proposal before it can trigger an external launch."""
        selected = _json_value(decision)
        if not isinstance(selected, Mapping):
            raise ValidationError("policy choose_next must return a PolicySelection object or None")
        required = {"request_id", "owner_id", "priority_class", "score", "reason_code", "explanation"}
        if required - set(selected):
            raise ValidationError("policy selection is missing required fields")
        request_id = selected.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValidationError("policy selection request_id must be a non-empty string")
        candidate = next(
            (item for item in candidates if str(item.get("id", item.get("job_id"))) == request_id),
            None,
        )
        if candidate is None:
            raise ValidationError("policy selected an unknown queued request")
        spec = JobSpec.from_dict(candidate["spec"])
        owner = str(candidate.get("owner", candidate.get("owner_id", spec.owner_id)))
        if selected.get("owner_id") != owner:
            raise ValidationError("policy selection owner does not match the queued job")
        if selected.get("priority_class") != spec.priority_class:
            raise ValidationError("policy selection priority class does not match the queued job")
        score = selected.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValidationError("policy selection score must be finite")
        if not isinstance(selected.get("reason_code"), str) or not selected["reason_code"]:
            raise ValidationError("policy selection reason_code must be a non-empty string")
        if not isinstance(selected.get("explanation"), str):
            raise ValidationError("policy selection explanation must be a string")
        # A policy chooses ordering; it cannot authorize work that is currently
        # infeasible. Rechecking here is advisory; the allocator transaction in
        # _start_job remains authoritative under races.
        if not self._job_can_fit(candidate):
            raise ValidationError("policy selected a request that is not currently feasible")
        return candidate

    def _policy_input_snapshot(
        self,
        candidates: Sequence[Mapping[str, Any]],
        now: datetime,
        service_by_owner: Mapping[str, float],
    ) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        for candidate in candidates:
            request = _without_revision(candidate)
            request_id = str(request.get("id", request.get("job_id", "")))
            request["request_id"] = request_id
            request["can_fit"] = self._job_can_fit(candidate)
            request["safe_backfill_for"] = [
                str(protected.get("id", protected.get("job_id", "")))
                for protected in candidates
                if protected is not candidate and self._safe_backfill(candidate, protected)
            ]
            requests.append(_json_value(request))
        return {
            "requests": requests,
            "now": _iso(now),
            "service_by_owner": dict(service_by_owner),
            "policy_configuration": self._policy_configuration_snapshot(),
        }

    def _job_can_fit(self, job: Mapping[str, Any]) -> bool:
        spec = JobSpec.from_dict(job["spec"])
        reservation_id = job.get("reservation_id")
        if reservation_id:
            try:
                self._require_live_reservation(str(reservation_id))
            except (ConflictError, NotFoundError):
                return False
            return True
        if not spec.resources:
            return True
        if not self._matching_inventory_is_fresh(spec.resources):
            return False
        can_allocate = getattr(self.allocator, "can_allocate", None)
        if callable(can_allocate):
            try:
                return bool(can_allocate(spec.resources))
            except Exception:
                return False
        return True

    def _safe_backfill(self, candidate: Mapping[str, Any], protected: Mapping[str, Any]) -> bool:
        """Prove that backfill cannot consume any resource needed by an aged job.

        This deliberately checks every resource matching the candidate's selectors,
        rather than trusting a preview placement that could change before the
        allocator transaction. Shared requests may overlap a resource only when
        their capacity keys are disjoint; exclusive requests also protect their
        conflict domains. Existing active reservations already count as claims,
        so starting work within one introduces no new resource footprint.
        """
        try:
            candidate_spec = JobSpec.from_dict(candidate["spec"])
            protected_spec = JobSpec.from_dict(protected["spec"])
            if candidate.get("reservation_id"):
                self._require_live_reservation(str(candidate["reservation_id"]))
                return True
            if not candidate_spec.resources or not protected_spec.resources:
                return True
            list_resources = getattr(self.allocator, "list_resources", None)
            raw_resources = (
                list_resources(available_only=True)
                if callable(list_resources)
                else [item for item in self.ledger.list("resource") if item.get("available", True)]
            )
            resources = [item if isinstance(item, Resource) else Resource.from_dict(item) for item in raw_resources]
        except (KeyError, ConflictError, NotFoundError, ValidationError):
            return False

        for candidate_req in candidate_spec.resources:
            for candidate_resource in resources:
                if not _requirement_matches_resource(candidate_req, candidate_resource):
                    continue
                candidate_exclusive = _exclusive_domains(candidate_req, candidate_resource)
                for protected_req in protected_spec.resources:
                    for protected_resource in resources:
                        if not _requirement_matches_resource(protected_req, protected_resource):
                            continue
                        protected_exclusive = _exclusive_domains(protected_req, protected_resource)
                        if (
                            candidate_exclusive.intersection(protected_resource.conflict_domains)
                            or protected_exclusive.intersection(candidate_resource.conflict_domains)
                            or candidate_exclusive.intersection(protected_exclusive)
                        ):
                            return False
                        if candidate_resource.resource_id != protected_resource.resource_id:
                            continue
                        if candidate_req.mode == AllocationMode.EXCLUSIVE or protected_req.mode == AllocationMode.EXCLUSIVE:
                            return False
                        if set(candidate_req.quantities).intersection(protected_req.quantities):
                            return False
        return True

    def _require_live_reservation(
        self, reservation_id: str, *, transaction: Any | None = None
    ) -> Mapping[str, Any]:
        record = (
            transaction.get("reservation", reservation_id)
            if transaction is not None
            else self.ledger.get("reservation", reservation_id)
        )
        if record is None:
            raise NotFoundError(f"reservation {reservation_id!r} does not exist")
        if record.get("state") != "active":
            raise ConflictError(
                f"reservation {reservation_id!r} is not active",
                details={"state": record.get("state")},
            )
        try:
            expires_at = _parse_time(record.get("expires_at"), "reservation expires_at")
        except ValidationError as exc:
            raise ConflictError("reservation expiry cannot be verified") from exc
        if expires_at <= _now(self.clock):
            raise ConflictError("reservation has expired")
        return record

    def _start_job(
        self,
        job: Mapping[str, Any],
        decision: Any,
        *,
        policy_observations: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.executor is None:
            raise UnsupportedCapabilityError("no execution adapter is configured")
        job_id = str(job.get("id", job.get("job_id")))
        current = self.get_job(job_id)
        if _state(current.get("state")) not in {"queued", "blocked"}:
            return None
        spec = JobSpec.from_dict(current["spec"])
        lease: Mapping[str, Any] | None = None
        bound_lease_id = current.get("lease_id") if current.get("lease_binding") == "supervised" else None
        reservation_id = current.get("reservation_id")
        phase_id = f"job-phase:{job_id}"
        if reservation_id:
            self._require_live_reservation(str(reservation_id))
        if bound_lease_id:
            lease = _plain_lease(self._get_lease(str(bound_lease_id)))
            if lease.get("usage", LeaseUsage.EXTERNAL.value) != LeaseUsage.SUPERVISED.value:
                raise ConflictError("bound lease is not declared for supervised use")
            if lease.get("state") != LeaseState.ACTIVE.value:
                raise ConflictError("bound supervised lease is no longer active")
            if int(lease.get("fencing_token", 0)) != int(current.get("lease_fencing_token") or 0):
                raise ConflictError(
                    "bound supervised lease is no longer at the submitted fencing token",
                    details={
                        "lease_fencing_token": lease.get("fencing_token"),
                        "job_fencing_token": current.get("lease_fencing_token"),
                    },
                )
            if spec.resources:
                raise ValidationError("a job bound to a lease inherits its allocation; omit job-level resources")
            self._validate_supervised_lease_allocations(lease)
        elif spec.resources:
            if reservation_id:
                phase = self.allocator.phase_lease(reservation_id, phase_id, spec.resources)
                lease = {"lease_id": None, "fencing_token": None, "reservation_phase_id": phase_id, "reservation_id": reservation_id, "allocations": phase.get("allocations", [])}
            else:
                request = ResourceRequest(
                    request_id=f"job:{job_id}",
                    owner_id=spec.owner_id,
                    requirements=spec.resources,
                    lease_ttl_seconds=max(float(spec.timeout_seconds) + 60.0, self.config.default_lease_ttl_seconds),
                    idempotency_key=f"job-lease:{job_id}",
                )
                lease = _json_value(self.allocator.request_lease(request))
        attempt_id = str(uuid.uuid4())
        operation_id = f"start:{attempt_id}"
        start_payload = {"job_id": job_id, "attempt_id": attempt_id, "spec": spec.to_dict(), "lease_id": lease.get("lease_id") if lease else None, "reservation_id": reservation_id}
        now = _iso(_now(self.clock))
        skip_start = False
        execution_context: ExecutionContext | None = None
        decision_id = str(getattr(decision, "decision_id", "") or uuid.uuid4())
        try:
            with self.ledger.transaction() as tx:
                current_tx = tx.get("jobs", job_id)
                if current_tx is None:
                    raise NotFoundError(f"job {job_id!r} does not exist")
                if _state(current_tx.get("state")) not in {"queued", "blocked"}:
                    skip_start = True
                elif not scheduled_job_ready(tx, current_tx):
                    skip_start = True
                    updated = _without_revision(current_tx)
                    updated.update({"state": "blocked", "blocked_reason": "schedule_occurrence_not_submitted", "updated_at": now})
                    tx.put("jobs", job_id, updated, current_tx[_REVISION])
                else:
                    execution_context = self._job_execution_context(
                        tx,
                        lease_id=str(lease["lease_id"]) if lease and lease.get("lease_id") else None,
                        reservation_id=str(reservation_id) if reservation_id else None,
                        phase_id=phase_id if reservation_id and spec.resources else None,
                    )
                    start_payload["execution_context"] = execution_context.to_dict()
                    policy_configuration = self._policy_configuration_snapshot()
                    decision_record = SchedulerPolicyDecision(
                        decision_id=decision_id,
                        policy_id=str(getattr(decision, "policy_id", getattr(self.policy, "policy_id", "unknown"))),
                        policy_version=str(getattr(decision, "policy_version", getattr(self.policy, "version", "unknown"))),
                        subject_id=job_id,
                        action="admit",
                        accepted=True,
                        reason=str(getattr(decision, "reason_code", "selected")),
                        selected_resource_ids=tuple(item.resource_id for item in execution_context.allocations),
                        observed_at=_now(self.clock),
                        facts={
                            "score": getattr(decision, "score", None),
                            "reason_code": getattr(decision, "reason_code", None),
                            "explanation": getattr(decision, "explanation", None),
                            "policy_configuration": policy_configuration,
                        },
                    )
                    decision_snapshot_record = decision_snapshot(
                        decision_record,
                        observations=policy_observations or {},
                    )
                    decision_snapshot_record["policy_configuration"] = policy_configuration
                    tx.put("decisions", decision_id, decision_snapshot_record, None)
                    attempt = {
                        "id": attempt_id,
                        "attempt_id": attempt_id,
                        "job_id": job_id,
                        "state": "starting",
                        "execution_handle": None,
                        "created_at": now,
                        "started_at": None,
                        "finished_at": None,
                        "result": {},
                        "error": {},
                        "cleanup_verified": False,
                        "execution_context": execution_context.to_dict(),
                        "operation_id": operation_id,
                    }
                    tx.put("attempts", attempt_id, attempt, None)
                    tx.operation_intent("job.start", start_payload, operation_id=operation_id, owner=str(current_tx.get("owner", spec.owner_id)))
                    updated = _without_revision(current_tx)
                    updated.update({
                        "state": "running", "attempt_id": attempt_id,
                        "attempt_count": int(updated.get("attempt_count", 0)) + 1,
                        "lease_id": lease.get("lease_id") if lease else None,
                        "lease_fencing_token": lease.get("fencing_token") if lease else None,
                        "reservation_phase_id": phase_id if reservation_id else None,
                        "blocked_reason": None, "decision_id": decision_id,
                        "updated_at": now,
                    })
                    tx.put("jobs", job_id, updated, current_tx[_REVISION])
                    if hasattr(tx, "event"):
                        tx.event("job.start_requested", job_id, {"attempt_id": attempt_id, "decision": _json_value(decision)})
        except Exception:
            if reservation_id and spec.resources:
                try:
                    self.allocator.release_phase(str(reservation_id), phase_id)
                except Exception:
                    pass
            elif lease and lease.get("lease_id"):
                self._release_unstarted_job_lease(lease)
            raise
        if skip_start:
            if reservation_id and spec.resources:
                self.allocator.release_phase(str(reservation_id), phase_id)
            elif lease and lease.get("lease_id"):
                self._release_unstarted_job_lease(lease)
            return None
        execution_root = self.config.resolved_execution_root
        try:
            if execution_context is None:
                raise ConflictError("execution context was not durably established before launch")
            if spec.preconditions:
                precondition_receipt = self._check_execution_preconditions(
                    spec,
                    execution_context,
                    attempt_id=attempt_id,
                    owner=spec.owner_id,
                )
                if precondition_receipt.get("state") != "verified":
                    self._block_attempt_on_precondition(
                        job_id,
                        attempt_id,
                        operation_id,
                        lease=lease,
                        reservation_id=str(reservation_id) if reservation_id else None,
                        phase_id=phase_id if reservation_id and spec.resources else None,
                        receipt=precondition_receipt,
                    )
                    return None
            launch_spec = replace(
                spec,
                environment={
                    **dict(spec.environment),
                    "DEVICE_SCHEDULER_JOB_ID": job_id,
                    "DEVICE_SCHEDULER_ATTEMPT_ID": attempt_id,
                },
            )
            handle = self.executor.start(
                launch_spec,
                attempt_id=attempt_id,
                execution_root=execution_root,
                execution_context=execution_context,
            )
            if not isinstance(handle, ExecutionHandle):
                handle = ExecutionHandle.from_dict(handle)
            self._store_start_receipt(job_id, attempt_id, operation_id, handle, state="running")
            return self.get_job(job_id)
        except Exception as exc:
            if getattr(exc, "execution_not_submitted", False) is True:
                rejection = {
                    "state": "failed",
                    "execution_started": False,
                    "reason": str(exc)[:500],
                }
                self._store_attempt_result(
                    job_id,
                    attempt_id,
                    rejection,
                    state="failed",
                    cleanup_verified=True,
                )
                self._release_job_lease(self.get_job(job_id), rejection)
                return self.get_job(job_id)
            uncertain_handle = getattr(exc, "execution_handle", None)
            try:
                if uncertain_handle is not None and not isinstance(uncertain_handle, ExecutionHandle):
                    uncertain_handle = ExecutionHandle.from_dict(uncertain_handle)
            except Exception:
                uncertain_handle = None
            self._store_start_receipt(
                job_id,
                attempt_id,
                operation_id,
                uncertain_handle,
                state="unknown",
                error={
                    "type": type(exc).__name__,
                    **(
                        {"diagnostic": str(exc.safe_diagnostic)[:500]}
                        if isinstance(getattr(exc, "safe_diagnostic", None), str)
                        else {}
                    ),
                },
            )
            return self.get_job(job_id)

    def _check_execution_preconditions(
        self,
        spec: JobSpec,
        execution_context: ExecutionContext,
        *,
        attempt_id: str,
        owner: str,
    ) -> dict[str, Any]:
        """Require trusted adapters to verify every requested condition."""
        conditions = tuple(spec.preconditions)
        operation_id = f"preconditions:{attempt_id}"
        payload = {
            "attempt_id": attempt_id,
            "execution_context": execution_context.to_dict(),
            "preconditions": [item.to_dict() for item in conditions],
        }
        intent = self._ensure_operation(operation_id, "job.execution_preconditions", payload, owner)
        if intent.get("status") in {"complete", "failed"}:
            prior = intent.get("receipt")
            if isinstance(prior, Mapping) and prior.get("state") == "verified":
                receipt = dict(prior)
                self._persist_precondition_evidence(attempt_id, receipt)
                return receipt
            return {
                "state": "unknown",
                "operation_id": operation_id,
                "reason": "prior precondition operation has a non-verified terminal receipt",
            }

        requested = {item.name for item in conditions}
        available_resource_ids = {item.resource_id for item in execution_context.allocations}
        seen: dict[str, list[dict[str, Any]]] = {name: [] for name in requested}
        adapter_errors: list[dict[str, str]] = []
        for adapter in self.resource_adapters:
            check = getattr(adapter, "check_execution_preconditions", None)
            if not callable(check):
                continue
            adapter_id = str(getattr(adapter, "adapter_id", type(adapter).__name__))
            try:
                raw = check(
                    execution_context,
                    conditions,
                    operation_id=operation_id,
                )
                result = dict(raw) if isinstance(raw, Mapping) else {}
            except Exception as exc:
                adapter_errors.append({"adapter_id": adapter_id, "error": type(exc).__name__})
                continue
            if (
                result.get("adapter_id") != adapter_id
                or result.get("operation_id") != operation_id
                or "claim_id" not in result
                or result.get("claim_id") != execution_context.claim_id
                or "fencing_token" not in result
                or result.get("fencing_token") != execution_context.fencing_token
                or result.get("state") not in {"verified", "blocked", "unsupported", "unknown"}
                or (result.get("state") == "verified" and result.get("verified") is not True)
            ):
                adapter_errors.append({"adapter_id": adapter_id, "error": "precondition_receipt_identity_invalid"})
                continue
            envelope_resources = result.get("resource_ids", ())
            if (
                isinstance(envelope_resources, (str, bytes))
                or not isinstance(envelope_resources, Sequence)
                or any(not isinstance(item, str) or item not in available_resource_ids for item in envelope_resources)
            ):
                adapter_errors.append({"adapter_id": adapter_id, "error": "precondition_envelope_scope_invalid"})
                continue
            rows = result.get("preconditions", ())
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
                adapter_errors.append({"adapter_id": adapter_id, "error": "invalid_precondition_receipt"})
                continue
            for row in rows:
                if not isinstance(row, Mapping) or row.get("name") not in requested:
                    adapter_errors.append({"adapter_id": adapter_id, "error": "unknown_precondition_receipt"})
                    continue
                condition = next(item for item in conditions if item.name == row["name"])
                if row.get("schema_version") != condition.schema_version:
                    adapter_errors.append({"adapter_id": adapter_id, "error": "precondition_schema_version_invalid"})
                    continue
                resource_ids = row.get("resource_ids", ())
                if (
                    isinstance(resource_ids, (str, bytes))
                    or not isinstance(resource_ids, Sequence)
                    or any(not isinstance(item, str) or item not in available_resource_ids for item in resource_ids)
                ):
                    adapter_errors.append({"adapter_id": adapter_id, "error": "precondition_resource_scope_invalid"})
                    continue
                seen[str(row["name"])].append({"adapter_id": adapter_id, **dict(row)})

        normalized: list[dict[str, Any]] = []
        all_verified = True
        for condition in conditions:
            reports = seen.get(condition.name, [])
            if not reports:
                state = "unsupported" if not adapter_errors else "unknown"
                normalized.append({"name": condition.name, "state": state, "reports": []})
                all_verified = False
                continue
            states = [str(report.get("state", "unknown")) for report in reports]
            if all(state == "verified" for state in states):
                state = "verified"
            elif any(state == "blocked" for state in states):
                state = "blocked"
            elif any(state == "unsupported" for state in states):
                state = "unsupported"
            else:
                state = "unknown"
            normalized.append({"name": condition.name, "state": state, "reports": reports})
            all_verified = all_verified and state == "verified"
        receipt = {
            "state": "verified" if all_verified else "blocked",
            "operation_id": operation_id,
            "preconditions": normalized,
            "adapter_errors": adapter_errors,
        }
        self._record_operation(operation_id, receipt, status="complete")
        if all_verified:
            self._persist_precondition_evidence(attempt_id, receipt)
        return receipt

    def _persist_precondition_evidence(
        self, attempt_id: str, receipt: Mapping[str, Any]
    ) -> None:
        """Persist verified read-only adapter evidence before executor launch."""
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist")
            attempt_data = _without_revision(attempt)
            evidence = list(attempt_data.get("evidence", ()))
            evidence.append({"kind": "execution_preconditions", **_json_value(receipt)})
            attempt_data.update({"evidence": evidence, "precondition_receipt": _json_value(receipt)})
            tx.put("attempts", attempt_id, attempt_data, attempt[_REVISION])

    def _block_attempt_on_precondition(
        self,
        job_id: str,
        attempt_id: str,
        start_operation_id: str,
        *,
        lease: Mapping[str, Any] | None,
        reservation_id: str | None,
        phase_id: str | None,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Keep the job queued/blocked when requested evidence is unavailable."""
        cleanup_error: str | None = None
        try:
            if reservation_id and phase_id:
                self.allocator.release_phase(reservation_id, phase_id)
            elif lease and lease.get("lease_id"):
                self._release_unstarted_job_lease(lease)
        except Exception as exc:
            cleanup_error = type(exc).__name__
        now = _iso(_now(self.clock))
        final_state = "unknown" if cleanup_error else "blocked"
        blocked_reason = "execution_precondition_unverified"
        if cleanup_error:
            blocked_reason = "execution_precondition_failed_and_grant_cleanup_unverified"
        operation_receipt = {
            "state": final_state,
            "reason": blocked_reason,
            "preconditions": _json_value(receipt),
            "cleanup_error": cleanup_error,
        }
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is not None:
                attempt_data = _without_revision(attempt)
                attempt_data.update({
                    "state": final_state,
                    "precondition_receipt": _json_value(receipt),
                    "precondition_operation_id": receipt.get("operation_id"),
                    "cleanup_verified": cleanup_error is None,
                    "finished_at": now,
                    "error": {"reason": blocked_reason, "cleanup_error": cleanup_error},
                })
                tx.put("attempts", attempt_id, attempt_data, attempt[_REVISION])
            job = tx.get("jobs", job_id)
            if job is not None:
                job_data = _without_revision(job)
                job_data.update({
                    "state": final_state,
                    "attempt_count": max(0, int(job_data.get("attempt_count", 1)) - 1),
                    "blocked_reason": blocked_reason,
                    "blocked_reason_code": str(receipt.get("state", "unknown")),
                    "cleanup_verified": cleanup_error is None,
                    "updated_at": now,
                })
                if cleanup_error is None:
                    job_data.update({"lease_id": None, "lease_fencing_token": None, "reservation_phase_id": None})
                tx.put("jobs", job_id, job_data, job[_REVISION])
            operation = tx.operation(start_operation_id)
            if operation is not None and operation.get("status") in {"intent", "unknown"}:
                tx.record_receipt(start_operation_id, operation_receipt, status="unknown" if cleanup_error else "failed")
            tx.event("job.execution_precondition_blocked", job_id, {"attempt_id": attempt_id, "reason": blocked_reason})
        return self.get_job(job_id)

    def _job_execution_context(
        self,
        tx: Any,
        *,
        lease_id: str | None,
        reservation_id: str | None,
        phase_id: str | None,
    ) -> ExecutionContext:
        """Bind a job launch to its current, transactionally verified grant."""
        if phase_id is not None:
            if reservation_id is None:
                raise ConflictError("reservation phase is missing its protected reservation")
            reservation = self._require_live_reservation(reservation_id, transaction=tx)
            phase = tx.get("reservation_phase", phase_id)
            if (
                phase is None
                or phase.get("reservation_id") != reservation_id
                or _state(phase.get("state")) != "active"
            ):
                raise ConflictError("reservation phase is no longer active")
            return self._execution_context_from_claim(
                tx,
                phase.get("allocations", ()),
                claim_id=phase_id,
                fencing_token=int(reservation["fencing_token"]),
                purpose=ExecutionPurpose.JOB,
            )
        if lease_id is not None:
            lease = tx.get("lease", lease_id)
            if lease is None or lease.get("state") != LeaseState.ACTIVE.value:
                raise ConflictError("job resource lease is not active")
            expires_at = _parse_time(lease.get("expires_at"), "lease expires_at")
            if expires_at <= _now(self.clock):
                raise ConflictError("job resource lease has expired")
            return self._execution_context_from_claim(
                tx,
                lease.get("allocations", ()),
                claim_id=lease_id,
                fencing_token=int(lease["fencing_token"]),
                purpose=ExecutionPurpose.JOB,
            )
        if reservation_id is not None:
            reservation = self._require_live_reservation(reservation_id, transaction=tx)
            return ExecutionContext(
                claim_id=reservation_id,
                fencing_token=int(reservation["fencing_token"]),
                purpose=ExecutionPurpose.JOB,
            )
        return ExecutionContext(purpose=ExecutionPurpose.JOB)

    def _execution_context_from_claim(
        self,
        tx: Any,
        raw_allocations: Sequence[Mapping[str, Any]],
        *,
        claim_id: str,
        fencing_token: int,
        purpose: ExecutionPurpose,
    ) -> ExecutionContext:
        allocations = tuple(ResourceAllocation.from_dict(item) for item in raw_allocations)
        resources: list[Resource] = []
        for allocation in allocations:
            resource_record = tx.get("resource", allocation.resource_id)
            if resource_record is None:
                raise ConflictError(
                    "granted resource descriptor is missing",
                    details={"resource_id": allocation.resource_id},
                )
            health = tx.get("resource_health", allocation.resource_id)
            if health is not None and health.get("available") is False:
                raise ConflictError(
                    "granted resource inventory is unavailable before execution",
                    details={
                        "resource_id": allocation.resource_id,
                        "availability_reason": health.get("availability_reason"),
                    },
                )
            resources.append(Resource.from_dict(_without_revision(resource_record)))
        return ExecutionContext(
            allocations=allocations,
            resources=tuple(resources),
            claim_id=claim_id,
            fencing_token=fencing_token,
            purpose=purpose,
        )

    def _reconcile_jobs(self) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        failures: list[str] = []
        if self.executor is None:
            return updated
        for job in self.ledger.list("jobs"):
            if _state(job.get("state")) not in {"running", "unknown", "cancelling"}:
                continue
            try:
                result = self._reconcile_one(job)
                if result is not None:
                    updated.append(result)
            except Exception as exc:
                job_id = str(job.get("id", job.get("job_id")))
                failures.append(f"{job_id}:{type(exc).__name__}")
                attempt_id = str(job.get("attempt_id") or "")
                if attempt_id:
                    self._mark_attempt_unknown(job_id, attempt_id, f"execution reconciliation failed: {type(exc).__name__}")
                else:
                    self._update_job(job_id, {"state": "unknown", "blocked_reason": "missing_attempt_identity"})
        if failures:
            raise ReconciliationRequiredError(
                "one or more executions could not be reconciled",
                details={"failures": failures},
            )
        return updated

    def _reconcile_one(self, job: Mapping[str, Any]) -> dict[str, Any] | None:
        job_id = str(job.get("id", job.get("job_id")))
        attempt_id = job.get("attempt_id")
        if not attempt_id:
            self._update_job(job_id, {"state": "unknown", "blocked_reason": "missing_attempt_identity"})
            return self.get_job(job_id)
        attempt = self._get("attempts", str(attempt_id))
        handle = _handle(attempt.get("execution_handle"))
        if handle is None:
            resolve = getattr(self.executor, "resolve", None)
            if not callable(resolve):
                self._mark_attempt_unknown(job_id, str(attempt_id), "execution handle missing and adapter cannot resolve attempt identity")
                return self.get_job(job_id)
            try:
                handle = resolve(attempt_id=str(attempt_id))
            except Exception as exc:
                self._mark_attempt_unknown(job_id, str(attempt_id), f"execution identity lookup failed: {type(exc).__name__}")
                return self.get_job(job_id)
            if handle is None:
                fence = getattr(self.executor, "fence_missing_attempt", None)
                if callable(fence):
                    fence_operation_id = f"job.missing-start-fence:{attempt_id}"
                    fence_intent = self._ensure_operation(
                        fence_operation_id,
                        "job.missing_start_fence",
                        {"job_id": job_id, "attempt_id": str(attempt_id)},
                        str(job.get("owner", "system")),
                    )
                    try:
                        if fence_intent.get("status") == "complete":
                            receipt = fence_intent.get("receipt") or {}
                        else:
                            receipt = fence(attempt_id=str(attempt_id))
                            if not isinstance(receipt, Mapping):
                                receipt = {"state": "unknown", "cleanup_verified": False, "reason": "adapter returned an invalid fence receipt"}
                            self._record_operation(
                                fence_operation_id,
                                receipt,
                                status="complete" if receipt.get("fenced") is True else "unknown",
                            )
                    except Exception as exc:
                        receipt = {"state": "unknown", "cleanup_verified": False, "error": type(exc).__name__}
                        self._record_operation(fence_operation_id, receipt, status="unknown")
                    if isinstance(receipt, Mapping) and receipt.get("cleanup_verified") is True and receipt.get("fenced") is True:
                        result = {
                            **dict(receipt),
                            "execution_outcome": "unknown_but_inactive_and_fenced",
                        }
                        self._store_attempt_result(
                            job_id,
                            str(attempt_id),
                            result,
                            state="cancelled",
                            cleanup_verified=True,
                        )
                        self._release_job_lease(self.get_job(job_id), result)
                        return self.get_job(job_id)
                    if isinstance(receipt, Mapping) and isinstance(receipt.get("execution_handle"), Mapping):
                        try:
                            active_handle = ExecutionHandle.from_dict(receipt["execution_handle"])
                        except (TypeError, ValueError):
                            active_handle = None
                        if active_handle is not None and active_handle.identity == str(attempt_id):
                            self._persist_resolved_handle(job_id, str(attempt_id), active_handle)
                            return self.get_job(job_id)
                    self._mark_attempt_unknown(
                        job_id,
                        str(attempt_id),
                        str(receipt.get("reason", "systemd execution could not be safely fenced"))
                        if isinstance(receipt, Mapping)
                        else "systemd execution could not be safely fenced",
                        receipt if isinstance(receipt, Mapping) else None,
                    )
                    return self.get_job(job_id)
                self._mark_attempt_unknown(job_id, str(attempt_id), "no execution unit found for persisted start intent")
                return self.get_job(job_id)
            if not isinstance(handle, ExecutionHandle):
                handle = ExecutionHandle.from_dict(handle)
            self._persist_resolved_handle(job_id, str(attempt_id), handle)
            attempt = self._get("attempts", str(attempt_id))
        observation = self.executor.inspect(handle)
        state = _state(observation.get("state"))
        if state == "running":
            return self._update_job(job_id, {"state": "running", "blocked_reason": None})
        if state not in {"succeeded", "failed", "cancelled"}:
            self._mark_attempt_unknown(job_id, str(attempt_id), str(observation.get("reason", "adapter observation is ambiguous")), observation)
            return self.get_job(job_id)
        operation_id = f"collect:{attempt_id}"
        intent = self._ensure_operation(operation_id, "job.collect", {"job_id": job_id, "attempt_id": attempt_id}, str(job.get("owner", "system")))
        if intent.get("status") == "complete":
            collected = intent.get("receipt") or {}
        else:
            try:
                collected = dict(self.executor.collect(handle))
            except Exception as exc:
                collected = {"state": "unknown", "cleanup_verified": False, "error": type(exc).__name__}
            self._record_operation(operation_id, collected, status="complete" if collected.get("cleanup_verified") is True else "unknown")
        combined = {**dict(observation), **dict(collected)}
        cleanup_verified = observation.get("cleanup_verified") is True and collected.get("cleanup_verified") is True
        result_state = state
        if not cleanup_verified:
            self._mark_attempt_unknown(job_id, str(attempt_id), "execution result is known but resource cleanup is not verified", combined)
            return self.get_job(job_id)
        self._store_attempt_result(job_id, str(attempt_id), combined, state=result_state, cleanup_verified=True)
        self._release_job_lease(self.get_job(job_id), combined)
        return self.get_job(job_id)

    def _store_start_receipt(
        self,
        job_id: str,
        attempt_id: str,
        operation_id: str,
        handle: ExecutionHandle | None,
        *,
        state: str,
        error: Mapping[str, Any] | None = None,
    ) -> None:
        now = _iso(_now(self.clock))
        receipt = {"state": state, "attempt_id": attempt_id, "execution_handle": handle.to_dict() if handle else None, "error": dict(error or {})}
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist")
            updated_attempt = _without_revision(attempt)
            updated_attempt.update({"state": state, "execution_handle": handle.to_dict() if handle else None, "started_at": now if handle else None, "error": dict(error or {})})
            tx.put("attempts", attempt_id, updated_attempt, attempt[_REVISION])
            job = tx.get("jobs", job_id)
            if job is not None:
                updated_job = _without_revision(job)
                updated_job.update({"state": state if state == "unknown" else "running", "updated_at": now})
                tx.put("jobs", job_id, updated_job, job[_REVISION])
            tx.record_receipt(operation_id, receipt, status="complete" if state == "running" else "unknown")

    def _try_preemption(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        now: datetime,
        service_by_owner: Mapping[str, float],
    ) -> bool:
        propose = getattr(self.policy, "propose_preemption", None)
        can_allocate = getattr(self.allocator, "can_allocate", None)
        if not callable(propose) or not callable(can_allocate) or self.executor is None:
            return False
        running = [item for item in self.ledger.list("jobs") if _state(item.get("state")) == "running"]
        if not running:
            return False
        rank = getattr(self.policy, "rank_requests", None)
        ordered = rank(candidates, now=now, service_by_owner=service_by_owner) if callable(rank) else candidates
        by_id = {str(item.get("id", item.get("job_id"))): item for item in candidates}
        ordered_jobs = [by_id[item.request_id] for item in ordered] if callable(rank) else list(ordered)
        observations = {"now": now, "service_by_owner": dict(service_by_owner)}
        for requester in ordered_jobs:
            requester_id = str(requester.get("id", requester.get("job_id")))
            requester_spec = JobSpec.from_dict(requester["spec"])
            if self._job_can_fit(requester) or not requester_spec.resources or requester.get("reservation_id"):
                continue
            try:
                proposal = propose(
                    requester,
                    running,
                    observations,
                    would_unblock=lambda target, spec=requester_spec: self._preemption_would_unblock(spec, target),
                )
            except TypeError as exc:
                raise UnsupportedCapabilityError("configured policy preemption contract is incompatible") from exc
            if proposal is None:
                continue
            if self._execute_preemption(proposal, requester_id):
                return True
        return False

    def _preemption_would_unblock(self, requester_spec: JobSpec, target: Mapping[str, Any]) -> bool:
        target_id = str(target.get("id", target.get("job_id", "")))
        target = self.ledger.get("jobs", target_id) if target_id else None
        if target is None or _state(target.get("state")) != "running":
            return False
        if target.get("reservation_id") or target.get("reservation_phase_id"):
            return False
        spec = JobSpec.from_dict(target["spec"])
        if not spec.interruption.cancellation_supported:
            return False
        if spec.priority_class != "background":
            return False
        attempt_count = target.get("attempt_count", 0)
        if spec.preemption_action.value == "abandon":
            if not spec.flexible:
                return False
        elif not spec.interruption.restart_safe or isinstance(attempt_count, bool) or not isinstance(attempt_count, int) or attempt_count >= spec.max_attempts:
            return False
        lease_id = target.get("lease_id")
        attempt_id = target.get("attempt_id")
        if not isinstance(lease_id, str) or not isinstance(attempt_id, str):
            return False
        lease = _plain_lease(self._get_lease(lease_id))
        if lease.get("state") != LeaseState.ACTIVE.value or lease.get("reservation_id") or lease.get("transferred_to"):
            return False
        if lease.get("owner_id") != target.get("owner", target.get("owner_id")):
            return False
        attempt = self.ledger.get("attempts", attempt_id)
        if attempt is None or attempt.get("state") != "running" or _handle(attempt.get("execution_handle")) is None:
            return False
        try:
            return bool(self.allocator.can_allocate(requester_spec.resources, ignore_lease_ids=(lease_id,)))
        except TypeError as exc:
            raise UnsupportedCapabilityError("allocator does not support safe preemption feasibility checks") from exc
        except Exception:
            return False

    def _execute_preemption(self, proposal: Any, requester_id: str) -> bool:
        proposal_data = _json_value(proposal)
        if not isinstance(proposal_data, Mapping):
            return False
        target_id = proposal_data.get("target_job_id")
        attempt_id = proposal_data.get("target_attempt_id")
        lease_id = proposal_data.get("target_lease_id")
        if not all(isinstance(item, str) and item for item in (target_id, attempt_id, lease_id)):
            return False
        requester = self.ledger.get("jobs", requester_id)
        target = self.ledger.get("jobs", target_id)
        if requester is None or target is None or _state(requester.get("state")) not in {"queued", "blocked"}:
            return False
        if target.get("lease_binding") == "supervised":
            return False  # The caller's reserved lease is not policy-owned capacity.
        if not self._preemption_would_unblock(JobSpec.from_dict(requester["spec"]), target):
            return False
        if target.get("attempt_id") != attempt_id or target.get("lease_id") != lease_id:
            return False
        operation_id = f"preempt:{requester_id}:{attempt_id}"
        payload = {
            "requester_job_id": requester_id,
            "target_job_id": target_id,
            "target_attempt_id": attempt_id,
            "target_lease_id": lease_id,
            "proposal": dict(proposal_data),
        }
        operation = self._ensure_operation(operation_id, "job.preempt", payload, str(requester.get("owner", "scheduler")))
        if operation.get("status") == "complete":
            receipt = operation.get("receipt") or {}
            return receipt.get("target_requeued") is True or receipt.get("target_abandoned") is True
        return self._resume_preemption(operation_id, payload)

    def _resume_preemption(self, operation_id: str, payload: Mapping[str, Any]) -> bool:
        requester_id = str(payload.get("requester_job_id", ""))
        target_id = str(payload.get("target_job_id", ""))
        attempt_id = str(payload.get("target_attempt_id", ""))
        lease_id = str(payload.get("target_lease_id", ""))
        requester = self.ledger.get("jobs", requester_id) if requester_id else None
        target = self.ledger.get("jobs", target_id) if target_id else None
        if target is None or target.get("attempt_id") != attempt_id or target.get("lease_id") != lease_id:
            self._record_operation(operation_id, {"state": "unknown", "reason": "preemption_identity_mismatch"}, status="unknown")
            return False
        state = _state(target.get("state"))
        if state in _TERMINAL and state != "cancelled":
            self._record_operation(operation_id, {"state": "failed", "reason": "target_completed_before_preemption"}, status="failed")
            return False
        if state == "running":
            if requester is None or _state(requester.get("state")) not in {"queued", "blocked"}:
                self._record_operation(operation_id, {"state": "failed", "reason": "requester_no_longer_queued"}, status="failed")
                return False
            requester_spec = JobSpec.from_dict(requester["spec"])
            if self._job_can_fit(requester):
                self._record_operation(operation_id, {"state": "failed", "reason": "requester_no_longer_needs_preemption"}, status="failed")
                return False
            spec = JobSpec.from_dict(target["spec"])
            if (
                target.get("reservation_id")
                or target.get("lease_binding") == "supervised"
                or spec.priority_class != "background"
                or not spec.interruption.cancellation_supported
                or not self._preemption_would_unblock(requester_spec, target)
            ):
                self._record_operation(operation_id, {"state": "failed", "reason": "preemption_contract_changed"}, status="failed")
                return False
            try:
                cancelled = self.cancel_job(target_id, reason=f"preempted_for:{requester_id}")
            except Exception as exc:
                self._record_operation(operation_id, {"state": "unknown", "reason": type(exc).__name__}, status="unknown")
                return False
            target = self.ledger.get("jobs", target_id)
            if target is None or _state(cancelled.get("state")) != "cancelled":
                self._record_operation(operation_id, {"state": "unknown", "reason": "target_cancellation_unverified"}, status="unknown")
                return False
        if _state(target.get("state")) == "queued":
            return self._finish_preemption(operation_id, requester_id, target_id, attempt_id, lease_id)
        if _state(target.get("state")) != "cancelled" or target.get("cleanup_verified") is not True:
            self._record_operation(operation_id, {"state": "unknown", "reason": "target_not_safely_cancelled"}, status="unknown")
            return False
        attempt = self.ledger.get("attempts", attempt_id)
        attempt_result = attempt.get("result") if isinstance(attempt, Mapping) else None
        if not isinstance(attempt_result, Mapping) or attempt_result.get("cancellation_reason") != f"preempted_for:{requester_id}":
            self._record_operation(operation_id, {"state": "failed", "reason": "target_cancelled_by_another_operation"}, status="failed")
            return False
        try:
            lease = _plain_lease(self._get_lease(lease_id))
        except Exception:
            self._record_operation(operation_id, {"state": "unknown", "reason": "target_lease_missing"}, status="unknown")
            return False
        if lease.get("state") != LeaseState.RELEASED.value:
            self._record_operation(operation_id, {"state": "unknown", "reason": "target_lease_release_unverified"}, status="unknown")
            return False
        # Refresh physical observations after cleanup before releasing the
        # admission barrier to the requesting job.
        self._discover_resources()
        requester = self.ledger.get("jobs", requester_id) if requester_id else None
        feasible = False
        if requester is not None and _state(requester.get("state")) in {"queued", "blocked"}:
            requester_spec = JobSpec.from_dict(requester["spec"])
            feasible = self._job_can_fit(requester)
        return self._finish_preemption(
            operation_id,
            requester_id,
            target_id,
            attempt_id,
            lease_id,
            requester_feasible=feasible,
        )

    def _finish_preemption(
        self,
        operation_id: str,
        requester_id: str,
        target_id: str,
        attempt_id: str,
        lease_id: str,
        *,
        requester_feasible: bool = False,
    ) -> bool:
        now = _iso(_now(self.clock))
        target_record = self.ledger.get("jobs", target_id)
        target_spec = JobSpec.from_dict(target_record.get("spec", {})) if target_record else None
        abandon = target_spec is not None and target_spec.preemption_action.value == "abandon"
        receipt = {
            "state": "preempted",
            "requester_job_id": requester_id,
            "target_job_id": target_id,
            "target_attempt_id": attempt_id,
            "target_lease_id": lease_id,
            "target_requeued": not abandon,
            "target_abandoned": abandon,
            "requester_feasible_after_cleanup": requester_feasible,
        }
        with self.ledger.transaction() as tx:
            target = tx.get("jobs", target_id)
            attempt = tx.get("attempts", attempt_id)
            operation = tx.operation(operation_id)
            if target is None or attempt is None or operation is None:
                raise ReconciliationRequiredError("preemption journal is incomplete")
            if target.get("lease_id") != lease_id or target.get("attempt_id") != attempt_id:
                raise ConflictError("target identity changed before preemption was recorded")
            if _state(target.get("state")) == "queued":
                if operation.get("status") in {"intent", "unknown"}:
                    tx.record_receipt(operation_id, receipt, status="complete")
                return True
            if _state(target.get("state")) != "cancelled" or target.get("cleanup_verified") is not True:
                raise ReconciliationRequiredError("target lacks verified cancellation evidence")
            attempt_data = _without_revision(attempt)
            attempt_data["preemption"] = {
                "requester_job_id": requester_id,
                "reason": "cancel_and_abandon" if abandon else "cancel_and_restart",
                "recorded_at": now,
            }
            tx.put("attempts", attempt_id, attempt_data, attempt[_REVISION])
            target_data = _without_revision(target)
            target_data.update({
                    "preemption_count": int(target_data.get("preemption_count", 0)) + 1,
                    "updated_at": now,
                })
            if not abandon:
                target_data.update({
                    "state": "queued",
                    "lease_id": None,
                    "lease_fencing_token": None,
                    "reservation_phase_id": None,
                    "result": None,
                    "cleanup_verified": False,
                    "blocked_reason": f"preempted_for:{requester_id}",
                })
            tx.put("jobs", target_id, target_data, target[_REVISION])
            tx.event("job.preempted", target_id, {"requester_job_id": requester_id, "attempt_id": attempt_id})
            if operation.get("status") in {"intent", "unknown"}:
                tx.record_receipt(operation_id, receipt, status="complete")
        return True

    def _reconcile_preemptions(self) -> list[dict[str, Any]]:
        operations = self.ledger.operations() if callable(getattr(self.ledger, "operations", None)) else []
        recovered: list[dict[str, Any]] = []
        for operation in operations:
            if operation.get("operation") != "job.preempt" or operation.get("status") not in {"intent", "unknown"}:
                continue
            payload = operation.get("request")
            if isinstance(payload, Mapping) and self._resume_preemption(str(operation.get("operation_id")), payload):
                recovered.append({"operation_id": operation.get("operation_id"), "state": "complete"})
        return recovered

    def _reconcile_lease_demands(self) -> list[dict[str, Any]]:
        recovered = []
        for demand in self.ledger.list("lease_demands"):
            if demand.get("state") != "pending":
                continue
            key = f"{demand.get('owner')}:{demand.get('demand_id')}"
            result = self._process_lease_demand(key)
            recovered.append({"demand_id": demand.get("demand_id"), "state": result.get("state")})
        return recovered

    def _persist_resolved_handle(self, job_id: str, attempt_id: str, handle: ExecutionHandle) -> None:
        now = _iso(_now(self.clock))
        start_operation_id = f"start:{attempt_id}"
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist")
            updated = _without_revision(attempt)
            updated.update({"state": "running", "execution_handle": handle.to_dict(), "started_at": handle.started_at.isoformat() if handle.started_at else now})
            tx.put("attempts", attempt_id, updated, attempt[_REVISION])
            operation = tx.operation(start_operation_id)
            if operation and operation.get("status") in {"intent", "unknown"}:
                tx.record_receipt(start_operation_id, {"state": "running", "execution_handle": handle.to_dict(), "reconciled": True})
            job = tx.get("jobs", job_id)
            if job is not None:
                job_data = _without_revision(job)
                job_data.update({"state": "running", "updated_at": now})
                tx.put("jobs", job_id, job_data, job[_REVISION])

    def _store_attempt_result(
        self,
        job_id: str,
        attempt_id: str,
        result: Mapping[str, Any],
        *,
        state: str,
        cleanup_verified: bool,
    ) -> None:
        now = _iso(_now(self.clock))
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is None:
                raise NotFoundError(f"attempt {attempt_id!r} does not exist")
            attempt_data = _without_revision(attempt)
            attempt_data.update({"state": state, "result": dict(result), "cleanup_verified": cleanup_verified, "finished_at": now})
            tx.put("attempts", attempt_id, attempt_data, attempt[_REVISION])
            start_operation_id = attempt_data.get("operation_id")
            start_operation = tx.operation(str(start_operation_id)) if start_operation_id else None
            if start_operation is not None and start_operation.get("status") in {"intent", "unknown"}:
                tx.record_receipt(
                    str(start_operation_id),
                    {
                        "state": state,
                        "attempt_id": attempt_id,
                        "execution_handle": attempt_data.get("execution_handle"),
                        "result": dict(result),
                        "cleanup_verified": cleanup_verified,
                    },
                    status="complete" if cleanup_verified else "unknown",
                )
            job = tx.get("jobs", job_id)
            if job is not None:
                job_data = _without_revision(job)
                job_data.update({"state": state, "result": dict(result), "cleanup_verified": cleanup_verified, "updated_at": now, "blocked_reason": None})
                tx.put("jobs", job_id, job_data, job[_REVISION])
                tx.event(f"job.{state}", job_id, {"attempt_id": attempt_id, "cleanup_verified": cleanup_verified})

    def _mark_attempt_unknown(self, job_id: str, attempt_id: str, reason: str, observation: Mapping[str, Any] | None = None) -> None:
        now = _iso(_now(self.clock))
        with self.ledger.transaction() as tx:
            attempt = tx.get("attempts", attempt_id)
            if attempt is not None:
                attempt_data = _without_revision(attempt)
                error = dict(attempt_data.get("error") or {})
                error["reconciliation_reason"] = reason
                attempt_data.update({"state": "unknown", "error": error, "observation": _json_value(observation or {}), "updated_at": now})
                tx.put("attempts", attempt_id, attempt_data, attempt[_REVISION])
            job = tx.get("jobs", job_id)
            if job is not None and _state(job.get("state")) not in _TERMINAL:
                job_data = _without_revision(job)
                job_data.update({"state": "unknown", "blocked_reason": reason[:500], "cleanup_verified": False, "updated_at": now})
                tx.put("jobs", job_id, job_data, job[_REVISION])

    def _release_job_lease(self, job: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        if job.get("lease_binding") == "supervised":
            # The caller owns this lease beyond the attached job. The job's
            # systemd cleanup receipt is retained on its attempt and is later
            # consumed by explicit lease release or expiry reconciliation.
            return
        phase_id = job.get("reservation_phase_id")
        reservation_id = job.get("reservation_id")
        if phase_id and reservation_id:
            self.allocator.release_phase(str(reservation_id), str(phase_id))
            return
        lease_id = job.get("lease_id")
        if not lease_id:
            return
        lease = _plain_lease(self._get_lease(str(lease_id)))
        token = int(lease["fencing_token"])
        if lease.get("state") != LeaseState.RELEASED.value:
            requested = self.allocator.request_release(str(lease_id), token)
            token = int(requested["fencing_token"])
            reclaiming = self.allocator.begin_reclamation(str(lease_id), token)
            token = int(reclaiming["fencing_token"])
            release_receipt = {
                "verified": True,
                "lease_id": lease_id,
                "fencing_token": token,
                "resource_ids": [item["resource_id"] for item in lease.get("allocations", [])],
                "evidence": _json_value(receipt),
            }
            self.allocator.verify_release(str(lease_id), token, release_receipt)

    def _release_unstarted_job_lease(self, lease: Mapping[str, Any]) -> None:
        lease_id = str(lease.get("lease_id", ""))
        if not lease_id:
            return
        current = _plain_lease(self._get_lease(lease_id))
        if current.get("state") == LeaseState.RELEASED.value:
            return
        requested = self.allocator.request_release(lease_id, int(current["fencing_token"]))
        token = int(requested["fencing_token"])
        reclaiming = self.allocator.begin_reclamation(lease_id, token)
        token = int(reclaiming["fencing_token"])
        receipt = {
            "verified": True,
            "lease_id": lease_id,
            "fencing_token": token,
            "resource_ids": [item["resource_id"] for item in current.get("allocations", ())],
            "evidence": {"attempt_started": False, "operation_intent_committed": False},
        }
        self.allocator.verify_release(lease_id, token, receipt)

    def _ensure_operation(self, operation_id: str, operation: str, payload: Mapping[str, Any], owner: str) -> dict[str, Any]:
        current = self.ledger.operation(operation_id) if callable(getattr(self.ledger, "operation", None)) else None
        if current is not None:
            result = dict(current)
            result["receipt"] = self._operation_receipt(result)
            result["created"] = False
            return result
        with self.ledger.transaction() as tx:
            current = tx.operation(operation_id)
            if current is not None:
                result = dict(current)
                receipts = result.get("receipts", ())
                result["receipt"] = receipts[-1].get("receipt") if receipts else None
                result["created"] = False
                return result
            tx.operation_intent(operation, payload, operation_id=operation_id, owner=owner)
        return {"operation_id": operation_id, "status": "intent", "receipt": None, "created": True}

    @staticmethod
    def _operation_receipt(operation: Mapping[str, Any]) -> Mapping[str, Any] | None:
        receipts = operation.get("receipts", ())
        if receipts and isinstance(receipts[-1], Mapping):
            receipt = receipts[-1].get("receipt")
            return receipt if isinstance(receipt, Mapping) else None
        return None

    def _record_operation(self, operation_id: str, receipt: Mapping[str, Any], *, status: str) -> None:
        with self.ledger.transaction() as tx:
            current = tx.operation(operation_id)
            if current is None:
                raise ConflictError(f"operation intent {operation_id!r} is missing")
            if current.get("status") in {"complete", "failed"}:
                return
            tx.record_receipt(operation_id, _json_value(receipt), status=status)

    def _schedules(self) -> ScheduleCoordinator:
        if self._schedule_coordinator is None:
            self._schedule_coordinator = ScheduleCoordinator(
                self.ledger,
                self._scheduled_submit,
                self.clock,
                cancel_job=self._cancel_scheduled_job,
            )
        return self._schedule_coordinator

    def _scheduled_submit(self, spec: Mapping[str, Any], owner: str, key: str) -> Mapping[str, Any]:
        return self.submit_job(spec, owner, key)

    def _cancel_scheduled_job(self, job_id: str, owner: str, idempotency_key: str) -> Mapping[str, Any]:
        job = self.get_job(job_id)
        if job.get("owner", job.get("owner_id")) != owner:
            raise PermissionDeniedError("scheduled cancellation owner does not match the canonical job")
        return self.cancel_job(job_id, reason=f"schedule_cancel:{idempotency_key}")

    def _handoffs(self) -> Any:
        if self._handoff_coordinator is None:
            try:
                from .handoff import HandoffCoordinator
            except ImportError as exc:
                raise UnsupportedCapabilityError("handoff coordinator is not available") from exc
            self._handoff_coordinator = HandoffCoordinator(
                self.ledger,
                self.allocator,
                self.backend_adapter,
                self.continuation_adapter,
                self.submit_job,
                self.get_job,
                self.clock,
                cancel_job=self.cancel_job,
            )
        return self._handoff_coordinator

    def _has_pending_handoffs(self) -> bool:
        return any(_state(item.get("state")) not in _TERMINAL | {"complete", "cancelled"} for item in self.ledger.list("handoff"))

    def _discover_resources(self) -> list[dict[str, str]]:
        with self._inventory_lock:
            return self._discover_resources_locked()

    def _discover_resources_locked(self) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []
        reconcile = getattr(self.allocator, "reconcile_resource_inventory", None)
        for adapter in self.resource_adapters:
            adapter_id = str(getattr(adapter, "adapter_id", type(adapter).__name__))
            observed: Sequence[Any] | None = None
            error: str | None = None
            snapshot_fn = getattr(adapter, "snapshot", None)
            snapshot: Mapping[str, Any] | None = None
            if callable(snapshot_fn):
                try:
                    raw_snapshot = snapshot_fn()
                    if not isinstance(raw_snapshot, Mapping):
                        raise ValidationError("resource snapshot must be an object")
                    snapshot = raw_snapshot
                    inventory = raw_snapshot.get("gpu_inventory")
                    if isinstance(inventory, Mapping) and inventory.get("state") != "observed":
                        error = str(inventory.get("error") or "resource_inventory_unknown")
                    elif raw_snapshot.get("state") == "unknown" and not isinstance(inventory, Mapping):
                        error = str(raw_snapshot.get("error") or "resource_inventory_unknown")
                    else:
                        raw_resources = raw_snapshot.get("resources", ())
                        if isinstance(raw_resources, (str, bytes)) or not isinstance(raw_resources, Sequence):
                            raise ValidationError("resource snapshot resources must be an array")
                        observed = raw_resources
                except Exception as exc:
                    error = type(exc).__name__
            else:
                discover = getattr(adapter, "discover", None)
                if not callable(discover):
                    continue
                try:
                    result = discover()
                    if isinstance(result, (str, bytes)) or not isinstance(result, Sequence):
                        raise ValidationError("resource discovery must return an array")
                    observed = result
                except Exception as exc:
                    error = type(exc).__name__

            if error is not None:
                if callable(reconcile):
                    try:
                        reconcile(adapter_id, (), error=error)
                    except Exception as exc:
                        warnings.append({"adapter_id": adapter_id, "error": type(exc).__name__})
                warnings.append({"adapter_id": adapter_id, "error": error})
                continue

            assert observed is not None
            resources = list(observed)
            if snapshot is not None:
                resources = self._enrich_inventory_observations(resources, snapshot)
            try:
                if callable(reconcile):
                    reconcile(adapter_id, resources)
                else:
                    register = self.allocator.register_resource
                    try:
                        parameters = inspect.signature(register).parameters
                    except (TypeError, ValueError):
                        parameters = {}
                    for resource in resources:
                        if "source_adapter_id" in parameters:
                            register(resource, source_adapter_id=adapter_id)
                        else:
                            register(resource)
            except Exception as exc:
                if callable(reconcile):
                    try:
                        reconcile(adapter_id, (), error=f"snapshot_rejected:{type(exc).__name__}")
                    except Exception as reconcile_exc:
                        warnings.append({"adapter_id": adapter_id, "error": type(reconcile_exc).__name__})
                warnings.append({"adapter_id": adapter_id, "error": type(exc).__name__})
        self._inventory_warnings = warnings
        return warnings

    def _matching_inventory_is_fresh(self, requirements: Sequence[ResourceRequirement]) -> bool:
        """Fail closed when a matching known resource adapter did not refresh."""
        failed_adapters = {item.get("adapter_id") for item in self._inventory_warnings}
        if not failed_adapters:
            return True
        status_fn = getattr(self.allocator, "list_resource_status", None)
        if callable(status_fn):
            records = status_fn()
        else:
            health = {item.get("resource_id"): item for item in self.ledger.list("resource_health")}
            records = [
                {"resource": item, **health.get(item.get("resource_id"), {})}
                for item in self.ledger.list("resource")
            ]
        for status in records:
            if status.get("source_adapter_id") not in failed_adapters:
                continue
            raw_resource = status.get("resource", status)
            if not isinstance(raw_resource, Mapping):
                continue
            fields = {key: value for key, value in raw_resource.items() if key in Resource.__dataclass_fields__}
            try:
                resource = Resource.from_dict(fields)
            except ValidationError:
                continue
            if any(_requirement_matches_resource(requirement, resource) for requirement in requirements):
                return False
        return True

    def _require_fresh_matching_inventory(self, requirements: Sequence[ResourceRequirement]) -> None:
        if not self._matching_inventory_is_fresh(requirements):
            raise ConflictError("matching resource inventory is unavailable or could not be reconciled")

    def _enrich_inventory_observations(
        self, resources: Sequence[Any], snapshot: Mapping[str, Any]
    ) -> list[Any]:
        """Bind GPU telemetry and managed-process proof to this inventory sample."""
        inventory_at = snapshot.get("observed_at")
        telemetry = snapshot.get("process_telemetry")
        telemetry = telemetry if isinstance(telemetry, Mapping) else {}
        telemetry_state = telemetry.get("state", "unknown")
        rows = telemetry.get("processes")
        rows = rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else None
        handles, handles_complete = self._active_execution_handles()
        managed_ids: set[int] = set()
        mapping_state = "observed" if not handles and handles_complete else "unknown"
        resolver = getattr(self.executor, "managed_process_ids", None)
        if handles and callable(resolver) and handles_complete:
            try:
                mapping = resolver(handles)
                process_ids = mapping.get("process_ids") if isinstance(mapping, Mapping) else None
                if (
                    isinstance(mapping, Mapping)
                    and mapping.get("state") == "observed"
                    and isinstance(process_ids, Sequence)
                    and not isinstance(process_ids, (str, bytes))
                ):
                    managed_ids = {int(item) for item in process_ids if isinstance(item, int) and not isinstance(item, bool)}
                    mapping_state = "observed"
                else:
                    mapping_state = "unknown"
            except Exception:
                mapping_state = "unknown"

        enriched: list[Any] = []
        for raw in resources:
            try:
                resource = raw if isinstance(raw, Resource) else Resource.from_dict(raw)
            except Exception:
                enriched.append(raw)
                continue
            if resource.kind.casefold() not in _GPU_KINDS:
                enriched.append(resource)
                continue
            attrs = dict(resource.attributes)
            attrs["inventory_observed_at"] = inventory_at
            attrs["process_observed_at"] = inventory_at
            attrs["process_telemetry_state"] = telemetry_state
            attrs["process_mapping_state"] = mapping_state
            gpu_uuid = attrs.get("uuid")
            matching_rows = [
                row for row in (rows or ())
                if isinstance(row, Mapping) and (row.get("gpu_uuid") == gpu_uuid or row.get("gpu_uuid") is None)
            ]
            unknown_count = sum(
                1 for row in matching_rows
                if row.get("identity_unknown") or row.get("gpu_uuid") is None or not isinstance(row.get("pid"), int)
            )
            visible_ids = {
                int(row["pid"])
                for row in matching_rows
                if not row.get("identity_unknown")
                and row.get("gpu_uuid") == gpu_uuid
                and isinstance(row.get("pid"), int)
                and not isinstance(row.get("pid"), bool)
            }
            verified_managed = visible_ids.intersection(managed_ids) if mapping_state == "observed" else set()
            attrs["managed_process_ids"] = sorted(verified_managed)
            attrs["unmanaged_process_count"] = len(visible_ids - verified_managed)
            attrs["process_identity_unknown_count"] = unknown_count
            attrs["process_mapping_state"] = mapping_state
            enriched.append(replace(resource, attributes=attrs))
        return enriched

    def _active_execution_handles(self) -> tuple[list[ExecutionHandle], bool]:
        handles: list[ExecutionHandle] = []
        complete = True

        def include_handle(value: Any) -> None:
            nonlocal complete
            handle = _handle(value)
            if handle is None:
                complete = False
            elif handle not in handles:
                handles.append(handle)

        for job in self.ledger.list("jobs"):
            if _state(job.get("state")) not in {"running", "unknown", "cancelling"}:
                continue
            attempt_id = job.get("attempt_id")
            attempt = self.ledger.get("attempts", str(attempt_id)) if attempt_id else None
            include_handle(attempt.get("execution_handle") if attempt else None)

        for backend in self.ledger.list("backend"):
            # A stopped or evicted backend is known not to own a live process.
            # Every other state, including unfamiliar or ambiguous states, is
            # treated as potentially physical so missing handles fail closed.
            if _state(backend.get("state")) in {"stopped", "evicted", "released"}:
                continue
            include_handle(backend.get("execution_handle"))
        return handles, complete

    def _capabilities(self) -> dict[str, Any]:
        adapters = []
        for adapter in (*self.resource_adapters, self.executor, self.backend_adapter, self.continuation_adapter):
            if adapter is None:
                continue
            caps = getattr(adapter, "capabilities", ())
            try:
                value = caps() if callable(caps) else caps
            except Exception as exc:
                value = {"error": type(exc).__name__}
            adapters.append({"adapter_id": getattr(adapter, "adapter_id", type(adapter).__name__), "capabilities": _json_value(value)})
        allocator_caps = getattr(self.allocator, "capabilities", None)
        if allocator_caps is not None:
            try:
                value = allocator_caps() if callable(allocator_caps) else allocator_caps
            except Exception as exc:
                value = {"error": type(exc).__name__}
            adapters.append({"adapter_id": "resource-allocator", "capabilities": _json_value(value)})
        supported_paths: set[str] = set()
        for status in getattr(self.allocator, "list_resource_status", lambda: [])():
            resource = status.get("resource") if isinstance(status, Mapping) else None
            if not isinstance(resource, Mapping) or status.get("available") is False:
                continue
            capability = self._external_resource_lease_capability(resource)
            if capability.get("supported") is True:
                supported_paths.add(str(capability.get("mode")))
        return {
            "adapters": adapters,
            "extensions": {
                "priority_lease_demand": {
                    "supported": True,
                    "operation": "leases.demand",
                    "minimum_principal_priority": 70,
                    "preemption": "flexible background jobs only; verified cleanup before lease grant",
                },
                "resource_forecast": {"supported": True, "operation": "resources.forecast", "max_horizon_seconds": 600, "max_candidates": 64, "reserves_capacity": False},
                "packet_runtime": {"supported": True, "expected_duration_is_advisory": True, "kill_timeout_includes_termination_grace": True},
                "autobench_history_retention": {"supported": callable(getattr(self.ledger, "compact_autobench_history", None)), "operation": "history.compact", "owner_scoped": True},
            },
            "external_leases": {
                "supported": bool(supported_paths),
                "modes": sorted(supported_paths),
                "gpu_observation_constraints": {
                    "allocation_mode": "exclusive",
                    "bundle": "GPU-only and single-adapter",
                    "release": "repeatable read-only quiescence and baseline verification",
                },
                "supervised_cpu_memory_constraints": {
                    "consumption": "jobs.submit with lease_id and current fencing_token",
                    "binding": "systemd unit launched with the exact lease ExecutionContext",
                    "concurrency": "one queued or running job per lease",
                    "unmanaged_processes": "unsupported; same-user cgroups are not hostile-code isolation",
                    "cpu_memory_semantics": "ledger admission plus per-job cgroup ceilings; not physical reservation",
                },
                "unsupported_without_a_trusted_reclamation_adapter": [
                    "unmanaged cpu leases",
                    "unmanaged memory leases",
                    "mixed-adapter bundles",
                ],
            },
        }

    def _external_resource_lease_capability(self, resource_data: Mapping[str, Any]) -> dict[str, Any]:
        """Describe resource-local release support without granting it."""
        resource_id = resource_data.get("resource_id")
        if not isinstance(resource_id, str):
            return {"supported": False, "reason": "resource_identity_unknown"}
        statuses = getattr(self.allocator, "list_resource_status", lambda: [])()
        status = next(
            (
                item for item in statuses
                if isinstance(item, Mapping)
                and isinstance(item.get("resource"), Mapping)
                and item["resource"].get("resource_id") == resource_id
            ),
            None,
        )
        if status is None or status.get("available") is False:
            return {"supported": False, "reason": "resource_inventory_unavailable"}
        adapter_id = status.get("source_adapter_id")
        adapter = next(
            (
                item for item in self.resource_adapters
                if str(getattr(item, "adapter_id", type(item).__name__)) == adapter_id
            ),
            None,
        )
        if adapter is not None and callable(getattr(adapter, "reclaim_lease", None)):
            return {"supported": True, "mode": "trusted_adapter_reclamation", "adapter_id": adapter_id}
        kind = str(resource_data.get("kind", "")).casefold()
        if kind in {"cpu", "memory", "host_memory", "ram"}:
            try:
                resource = Resource.from_dict(resource_data)
            except ValidationError:
                return {"supported": False, "reason": "resource_descriptor_invalid"}
            if self._supports_supervised_lease((resource,), str(adapter_id)):
                return {
                    "supported": True,
                    "mode": "scheduler_supervised_jobs_only",
                    "usage": LeaseUsage.SUPERVISED.value,
                    "adapter_id": adapter_id,
                    "unmanaged_consumers": "unsupported",
                    "enforcement": "per-job systemd cgroup ceilings; host admission is ledger accounting",
                }
        if (
            kind in _GPU_KINDS
            and adapter is not None
            and callable(getattr(adapter, "capture_lease_baseline", None))
            and callable(getattr(adapter, "inspect_lease_release", None))
        ):
            capacities = resource_data.get("capacities", {})
            attributes = resource_data.get("attributes", {})
            capacity = capacities.get("memory_bytes") if isinstance(capacities, Mapping) else None
            free = attributes.get("memory_free_bytes_observation") if isinstance(attributes, Mapping) else None
            budget = {}
            if all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                   for value in (capacity, free)):
                budget["admission_memory_bytes"] = math.floor(min(capacity, free) * (1 - self.allocator.gpu_headroom_fraction))
            return {
                **budget,
                "supported": True,
                "mode": "exclusive_gpu_quiescence",
                "required_allocation_mode": "exclusive",
                "release_method": "repeatable_read_only_observation",
                "adapter_id": adapter_id,
            }
        return {"supported": False, "reason": "trusted_reclamation_adapter_unavailable"}

    @staticmethod
    def _resource_view(item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            return _json_value(item)
        if not isinstance(item, Mapping):
            raise ValidationError("resource adapters must return Resource objects or mappings")
        return _without_revision(item)

    def _service_by_owner(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        now = _now(self.clock)
        window_start = now - timedelta(seconds=float(self.config.fairness_window_seconds))
        for job in self.ledger.list("jobs"):
            state = _state(job.get("state"))
            if state not in _TERMINAL and state not in {"running", "unknown", "cancelling"}:
                continue
            attempt_id = job.get("attempt_id")
            attempt = self.ledger.get("attempts", str(attempt_id)) if attempt_id else None
            if attempt is None:
                continue
            started = _optional_time(attempt.get("started_at")) or _optional_time(attempt.get("created_at"))
            if started is None:
                continue
            if state in _TERMINAL:
                ended = _optional_time(attempt.get("finished_at")) or _optional_time(job.get("updated_at"))
            else:
                ended = now
            if ended is None or ended <= window_start or started >= now:
                continue
            seconds = max(0.0, (min(ended, now) - max(started, window_start)).total_seconds())
            if seconds <= 0:
                continue
            owner = str(job.get("owner", job.get("owner_id", "unknown")))
            totals[owner] = totals.get(owner, 0.0) + seconds
        return totals

    def _job_detail(self, job: Mapping[str, Any]) -> dict[str, Any]:
        view = _without_revision(job)
        attempt_id = view.get("attempt_id")
        if attempt_id:
            attempt = self.ledger.get("attempts", str(attempt_id))
            view["attempt"] = _without_revision(attempt) if attempt else None
        if view.get("lease_id"):
            lease = self.ledger.get("lease", str(view["lease_id"]))
            view["lease_state"] = lease.get("state") if lease else "missing"
        return _json_value(view)

    def _handoff_detail(self, handoff: Mapping[str, Any]) -> dict[str, Any]:
        view = _without_revision(handoff)
        return _json_value(view)

    def _list(self, kind: str, *, owner: str | None = None) -> list[dict[str, Any]]:
        records = self.ledger.list(kind)
        return [record for record in records if owner is None or record.get("owner", record.get("owner_id")) == owner]

    def _get(self, kind: str, entity_id: str) -> dict[str, Any]:
        record = self.ledger.get(kind, entity_id)
        if record is None:
            raise NotFoundError(f"{kind} {entity_id!r} does not exist")
        return record

    def _get_lease(self, lease_id: str) -> Any:
        return self.allocator.get_lease(lease_id)

    def _update_job(self, job_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        return self._update("jobs", job_id, {**dict(changes), "updated_at": _iso(_now(self.clock))})

    def _update(self, kind: str, entity_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            current = tx.get(kind, entity_id)
            if current is None:
                raise NotFoundError(f"{kind} {entity_id!r} does not exist")
            updated = _without_revision(current)
            updated.update(_json_value(changes))
            saved = _put(tx, kind, entity_id, updated, current)
        return saved

    def _owner_from_args(self, args: Mapping[str, Any], actor: Principal, grant: str) -> str:
        owner = args.get("owner_id", actor.principal_id)
        if not isinstance(owner, str) or not owner:
            raise ValidationError("owner_id must be a non-empty string")
        if owner != actor.principal_id and not self._has(actor, grant):
            raise PermissionDeniedError("requested owner is outside the principal's grant")
        return owner

    def _require_owner(self, record: Mapping[str, Any], actor: Principal, *, owner_key: str, any_owner_action: str) -> None:
        owner = record.get(owner_key, record.get("owner_id"))
        if owner != actor.principal_id and not self._has(actor, any_owner_action):
            raise PermissionDeniedError("entity belongs to another principal")

    def _require_backend_scope(self, actor: Principal, backend_id: str) -> None:
        if not any(fnmatch.fnmatchcase(backend_id, grant) for grant in actor.backend_ids):
            raise PermissionDeniedError("backend ID is outside the principal's explicit backend grant")

    def _source_grants(self, actor: Principal) -> frozenset[str]:
        configured = self.config.event_sources.get(actor.principal_id, ())
        scoped = (
            grant.split(":", 1)[1]
            for grant in actor.actions
            if grant.startswith("events.publish:") and ":" in grant
        )
        if "events.publish" in actor.actions:
            scoped = (*scoped, "*")
        return frozenset((*configured, *scoped))

    def _authorize(self, actor: Principal, operation: str) -> None:
        # Every authenticated caller may inspect its own configured grants.
        # The request has no caller-supplied identity and reveals no other
        # principal's policy or credential material.
        if operation == "access.inspect":
            return
        event_scoped = operation == "events.publish" and any(
            action.startswith("events.publish:") for action in actor.actions
        )
        retention_scoped = operation == "history.compact" and self._has(actor, "history.compact:autobench")
        autobench_history_scoped = operation == "history.status" and self._has(actor, "history.compact:autobench")
        if not self._has(actor, operation) and not event_scoped and not retention_scoped and not autobench_history_scoped:
            raise PermissionDeniedError(f"principal is not authorized for {operation}")

    @staticmethod
    def _has(actor: Principal, action: str) -> bool:
        return any(fnmatch.fnmatchcase(action, grant) for grant in actor.actions)

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()


class _NullContext:
    """Tiny context manager used for read-only waits outside the app lock."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None


def _config(value: SchedulerConfig | Mapping[str, Any] | Any | None) -> SchedulerConfig:
    if value is None:
        return SchedulerConfig()
    if isinstance(value, SchedulerConfig):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
    else:
        data = {}
        for name in SchedulerConfig.__dataclass_fields__:
            if hasattr(value, name):
                data[name] = getattr(value, name)
        if hasattr(value, "state_directory") and "state_root" not in data:
            data["state_root"] = getattr(value, "state_directory")
        if hasattr(value, "resolved_artifact_directory") and "artifact_root" not in data:
            data["artifact_root"] = getattr(value, "resolved_artifact_directory")
        if hasattr(value, "artifact_directory") and "artifact_root" not in data:
            data["artifact_root"] = getattr(value, "artifact_directory")
    if "artifact_root" not in data and "artifact_directory" in data:
        data["artifact_root"] = data.pop("artifact_directory")
    if "state_root" not in data and "state_directory" in data:
        data["state_root"] = data.pop("state_directory")
    allowed = set(SchedulerConfig.__dataclass_fields__)
    accepted = {key: child for key, child in data.items() if key in allowed}
    result = SchedulerConfig(**accepted)
    return result


def _required(args: Mapping[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} is required")
    return value


def _idempotency_key(args: Mapping[str, Any], fallback: Any = None) -> str:
    value = args.get("idempotency_key", fallback)
    if not isinstance(value, str) or not value:
        raise ValidationError("idempotency_key is required")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _requirement_matches_resource(requirement: ResourceRequirement, resource: Resource) -> bool:
    if requirement.kind.casefold() != resource.kind.casefold():
        return False
    for key, wanted in requirement.selectors.items():
        if key == "resource_id":
            actual = resource.resource_id
        elif key == "host_id":
            actual = resource.host_id
        elif key == "kind":
            actual = resource.kind
        elif key == "capabilities":
            actual = sorted(resource.capabilities)
        elif key.startswith("topology."):
            actual = _get_path(resource.topology, key.removeprefix("topology."))
        elif key.startswith("capacity."):
            actual = _get_path(resource.capacities, key.removeprefix("capacity."))
        else:
            actual = _get_path(resource.attributes, key)
        if not _selector_matches(actual, wanted):
            return False
    constraints = requirement.topology_constraints
    required = _string_values(constraints.get("required_conflict_domains", ()))
    if not required.issubset(resource.conflict_domains):
        return False
    if _string_values(constraints.get("avoid_conflict_domains", ())).intersection(resource.conflict_domains):
        return False
    topology = constraints.get("topology", {})
    if isinstance(topology, Mapping):
        for key, wanted in topology.items():
            if not _selector_matches(_get_path(resource.topology, str(key)), wanted):
                return False
    return True


def _exclusive_domains(requirement: ResourceRequirement, resource: Resource) -> set[str]:
    value = requirement.topology_constraints.get("exclusive_conflict_domains", ())
    if value is True:
        return set(resource.conflict_domains)
    domains = _string_values(value)
    if requirement.mode == AllocationMode.EXCLUSIVE:
        domains.update(resource.conflict_domains)
    return domains


def _string_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(item) for item in value}
    return set()


def _get_path(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _selector_matches(actual: Any, wanted: Any) -> bool:
    if isinstance(wanted, Mapping) and set(wanted).issubset({"eq", "in", "not_in", "exists"}):
        if "exists" in wanted and bool(wanted["exists"]) != (actual is not None):
            return False
        if "eq" in wanted and actual != wanted["eq"]:
            return False
        if "in" in wanted and actual not in wanted["in"]:
            return False
        if "not_in" in wanted and actual in wanted["not_in"]:
            return False
        return True
    return actual == wanted


def _fencing_arg(args: Mapping[str, Any], lease: Any) -> int:
    current = int(getattr(lease, "fencing_token", lease.get("fencing_token") if isinstance(lease, Mapping) else 0))
    if "fencing_token" not in args:
        raise ValidationError("fencing_token is required for lease mutation")
    supplied = args["fencing_token"]
    if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied != current:
        raise ConflictError("fencing_token does not match the current lease generation")
    return current


def _plain_lease(value: Any) -> dict[str, Any]:
    return _json_value(value)


def _handle(value: Any) -> ExecutionHandle | None:
    if value is None:
        return None
    if isinstance(value, ExecutionHandle):
        return value
    if isinstance(value, Mapping):
        try:
            return ExecutionHandle.from_dict(value)
        except (TypeError, ValueError, SchedulerError):
            return None
    return None


def _put(tx: Any, kind: str, entity_id: str, record: Mapping[str, Any], current: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _without_revision(record)
    expected_revision = current[_REVISION] if current is not None else None
    result = tx.put(kind, entity_id, data, expected_revision)
    return dict(result) if isinstance(result, Mapping) else {**data, _REVISION: result}


def _receipt_covers(
    receipt: Mapping[str, Any],
    lease: Mapping[str, Any],
    *,
    fencing_token: int,
    operation_id: str | None = None,
) -> bool:
    expected = {
        item.get("resource_id")
        for item in lease.get("allocations", ())
        if isinstance(item, Mapping) and isinstance(item.get("resource_id"), str)
    }
    if receipt.get("lease_id") != lease.get("lease_id") or receipt.get("fencing_token") != fencing_token:
        return False
    if operation_id is not None and receipt.get("operation_id") != operation_id:
        return False
    witnessed = receipt.get("resource_ids")
    return (
        receipt.get("verified") is True
        and isinstance(witnessed, Sequence)
        and not isinstance(witnessed, (str, bytes))
        and all(isinstance(item, str) for item in witnessed)
        and expected == set(witnessed)
    )


def _call_cancel(executor: Any, handle: ExecutionHandle, operation_id: str, timeout: float) -> Mapping[str, Any]:
    cancel = executor.cancel
    try:
        parameters = inspect.signature(cancel).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {"timeout_seconds": timeout}
    if "operation_id" in parameters or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        kwargs["operation_id"] = operation_id
    result = cancel(handle, **kwargs)
    if not isinstance(result, Mapping):
        raise ValidationError("executor cancellation receipt must be an object")
    return result


def _timeout(spec: Any, default: float) -> float:
    if isinstance(spec, Mapping):
        value = spec.get("timeout_seconds", default)
    else:
        value = getattr(spec, "timeout_seconds", default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _profile_timeout(config: SchedulerConfig, profile_id: str) -> float:
    profile = config.backend_profiles.get(profile_id, {})
    return profile.get("ready_timeout_seconds", 60.0) if isinstance(profile, Mapping) else 60.0


__all__ = ["OPERATION_NAMES", "Scheduler", "SchedulerConfig"]
