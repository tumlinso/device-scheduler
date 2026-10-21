"""Durable request--evict--run--restore orchestration.

The coordinator owns the protocol, not backend implementation. Every external
operation receives a stable operation identity and its intent is committed
before the adapter is called. The protected reservation remains held until a
restored backend is ready and its residency leases have been atomically
settled back from the reservation.
"""

from __future__ import annotations

import hashlib
import base64
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .errors import ConflictError, NotFoundError, ValidationError
from .models import (
    AllocationMode,
    BackendSpec,
    ExecutionContext,
    ExecutionPurpose,
    HandoffSpec,
    HandoffState,
    Resource,
    ResourceAllocation,
)


_REVISION = "_revision"
_TERMINAL_JOBS = {"succeeded", "failed", "cancelled"}
_TERMINAL_HANDOFFS = {HandoffState.COMPLETE.value, HandoffState.CANCELLED.value}
_BASELINE_MAX_AGE = timedelta(seconds=30)


def _plain(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != _REVISION}


def _state(value: Any) -> str:
    return str(getattr(value, "value", value))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} adapter response must be an object")
    return value


class HandoffCoordinator:
    """Advance durable finite batches through safe backend eviction and restore.

    Adapter methods that perform external effects must be idempotent for the
    supplied stable operation ID. Job submission must honor the stable
    idempotency key. An uncertain job state is never resubmitted or retried.
    """

    api_version = "1"

    def __init__(
        self,
        ledger: Any,
        allocator: Any,
        backend_adapter: Any,
        continuation_adapter: Any,
        submit_job: Callable[..., Mapping[str, Any]],
        get_job: Callable[[str], Mapping[str, Any]],
        clock: Any,
        cancel_job: Callable[[str, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.allocator = allocator
        self.backend_adapter = backend_adapter
        self.continuation_adapter = continuation_adapter
        self.submit_job = submit_job
        self.get_job = get_job
        self.clock = clock
        self.cancel_job = cancel_job
        self._lock = threading.RLock()

    def _now(self) -> datetime:
        value = self.clock.now() if hasattr(self.clock, "now") else self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("handoff clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _reservation_id(handoff_id: str) -> str:
        # Keep the identity readable while remaining independent of ID syntax.
        digest = hashlib.sha256(handoff_id.encode("utf-8")).hexdigest()[:24]
        return f"handoff-{digest}"

    @staticmethod
    def _operation_id(handoff_id: str, action: str, target: str = "", attempt: int = 0) -> str:
        digest = hashlib.sha256(f"{handoff_id}\0{action}\0{target}\0{attempt}".encode()).hexdigest()[:24]
        return f"handoff-{digest}-{action}"

    @staticmethod
    def _delivery_id(handoff_id: str, continuation_ref: str) -> str:
        encoded = base64.urlsafe_b64encode(handoff_id.encode("utf-8")).decode("ascii").rstrip("=")
        digest = hashlib.sha256(continuation_ref.encode("utf-8")).hexdigest()[:16]
        return f"ds.{encoded}.{digest}"

    @staticmethod
    def _timestamp(value: Any, label: str) -> datetime:
        if not isinstance(value, str) or not value:
            raise ValidationError(f"reclamation baseline {label} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"reclamation baseline {label} must be an ISO timestamp") from exc
        if parsed.tzinfo is None:
            raise ValidationError(f"reclamation baseline {label} must include a timezone")
        return parsed.astimezone(timezone.utc)

    @classmethod
    def _validate_reclamation_baseline(
        cls, value: Mapping[str, Any], *, backend_id: str, operation_id: str
    ) -> dict[str, Any]:
        baseline = dict(value)
        if baseline.get("state") != "observed":
            raise ValidationError("reclamation baseline state must be observed")
        if baseline.get("backend_id") != backend_id or baseline.get("operation_id") != operation_id:
            raise ValidationError("reclamation baseline identity does not match the requested backend operation")
        for field in ("profile_id", "profile_version"):
            if not isinstance(baseline.get(field), str) or not baseline[field]:
                raise ValidationError(f"reclamation baseline {field} is required")
        cls._timestamp(baseline.get("observed_at"), "observed_at")
        identity = baseline.get("execution_identity")
        if not isinstance(identity, Mapping) or any(
            not isinstance(identity.get(field), str) or not identity[field]
            for field in ("unit_name", "invocation_id", "control_group")
        ):
            raise ValidationError("reclamation baseline must identify the exact managed execution")
        if baseline.get("process_mapping_state") != "observed":
            raise ValidationError("reclamation baseline process mapping must be observed")
        gpus = baseline.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            raise ValidationError("reclamation baseline must include at least one GPU observation")
        seen_uuids: set[str] = set()
        for gpu in gpus:
            if not isinstance(gpu, Mapping):
                raise ValidationError("reclamation baseline GPU entries must be objects")
            for field in ("resource_id", "gpu_uuid"):
                if not isinstance(gpu.get(field), str) or not gpu[field]:
                    raise ValidationError(f"reclamation baseline GPU {field} is required")
            if gpu["gpu_uuid"] in seen_uuids:
                raise ValidationError("reclamation baseline contains duplicate GPU UUIDs")
            seen_uuids.add(gpu["gpu_uuid"])
            for field in ("memory_free_bytes", "used_gpu_memory_bytes"):
                number = gpu.get(field)
                if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                    raise ValidationError(f"reclamation baseline GPU {field} must be a known non-negative integer")
            if gpu["used_gpu_memory_bytes"] <= 0:
                raise ValidationError("reclamation baseline must observe positive managed GPU memory")
            cls._timestamp(gpu.get("inventory_observed_at"), "inventory_observed_at")
            cls._timestamp(gpu.get("process_observed_at"), "process_observed_at")
            backend_pids = gpu.get("backend_process_ids")
            process_pids = gpu.get("process_ids")
            for label, pids in (("backend_process_ids", backend_pids), ("process_ids", process_pids)):
                if not isinstance(pids, list) or not pids or any(
                    isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
                ):
                    raise ValidationError(f"reclamation baseline GPU {label} must contain known positive PIDs")
            if not set(backend_pids) <= set(process_pids):
                raise ValidationError("reclamation baseline managed PIDs are absent from the sampled process set")
            if set(process_pids) != set(backend_pids):
                raise ValidationError("reclamation baseline contains non-backend GPU processes")
        evidence = baseline.get("evidence")
        if (
            not isinstance(evidence, (Mapping, Sequence))
            or isinstance(evidence, (str, bytes))
            or not evidence
        ):
            raise ValidationError("reclamation baseline must include evidence references")
        return baseline

    def _baseline_is_fresh(self, baseline: Mapping[str, Any]) -> bool:
        now = self._now()
        try:
            timestamps = [
                self._timestamp(baseline.get("observed_at"), "observed_at"),
                *[
                    self._timestamp(gpu.get(field), field)
                    for gpu in baseline.get("gpus", [])
                    for field in ("inventory_observed_at", "process_observed_at")
                ],
            ]
        except (ValidationError, TypeError, AttributeError):
            return False
        return all(t <= now + timedelta(seconds=5) and now - t <= _BASELINE_MAX_AGE for t in timestamps)

    def _read(self, handoff_id: str) -> dict[str, Any]:
        value = self.ledger.get("handoff", handoff_id)
        if value is None:
            raise NotFoundError(f"Unknown handoff {handoff_id}")
        return value

    def _write(self, record: Mapping[str, Any], *, event: str | None = None) -> dict[str, Any]:
        value = _plain(record)
        entity_id = value["handoff_id"]
        with self.ledger.transaction() as tx:
            current = tx.get("handoff", entity_id)
            if current is None:
                raise NotFoundError(f"Unknown handoff {entity_id}")
            value["updated_at"] = self._now().isoformat()
            saved = tx.put("handoff", entity_id, value, current[_REVISION])
            if event:
                tx.event("handoff", entity_id, {"event": event, "state": value.get("state"), "phase": value.get("phase")})
            return saved

    def _initial(self, spec: HandoffSpec, owner: str, idempotency_key: str) -> dict[str, Any]:
        self._validate_request(spec, owner)
        reservation_id = self._reservation_id(spec.handoff_id)
        refs = set(spec.continuation_refs)
        backends: dict[str, dict[str, Any]] = {}
        for backend_id in spec.backend_ids:
            entity = self.ledger.get("backend", backend_id)
            if entity is None:
                raise NotFoundError(f"Unknown managed backend {backend_id}")
            raw_spec = entity.get("spec", entity)
            backend = BackendSpec.from_dict(raw_spec)
            if backend.backend_id != backend_id:
                raise ValidationError(f"backend entity {backend_id} contains a mismatched spec")
            if _state(entity.get("state")) != "ready":
                raise ConflictError(f"backend {backend_id} must be ready before a handoff")
            lease_id = entity.get("lease_id") or entity.get("residency_lease_id") or entity.get("active_lease_id")
            if not lease_id:
                raise ConflictError(f"backend {backend_id} has no accounted residency lease")
            lease = self.ledger.get("lease", str(lease_id))
            if lease is None or lease.get("state") != "active":
                raise ConflictError(f"backend {backend_id} residency lease is not active")
            source_allocations = tuple(
                ResourceAllocation.from_dict(item) for item in lease.get("allocations", ())
            )
            if not source_allocations:
                raise ConflictError(f"backend {backend_id} residency lease has no physical allocations")
            source_resources: list[Resource] = []
            for allocation in source_allocations:
                resource_record = self.ledger.get("resource", allocation.resource_id)
                if resource_record is None:
                    raise ConflictError(
                        f"backend {backend_id} residency resource {allocation.resource_id} is absent"
                    )
                resource_fields = {
                    "resource_id", "host_id", "kind", "capacities", "attributes", "topology",
                    "conflict_domains", "capabilities", "schema_version",
                }
                source_resources.append(Resource.from_dict({
                    key: value for key, value in resource_record.items() if key in resource_fields
                }))
            refs.update(backend.attached_consumers)
            backends[backend_id] = {
                "spec": backend.to_dict(),
                "lease_id": str(lease_id),
                "allocations": [item.to_dict() for item in source_allocations],
                "resources": [item.to_dict() for item in source_resources],
                "execution_handle": entity.get("execution_handle"),
            }
        validate_ref = getattr(self.continuation_adapter, "validate_ref", None)
        if callable(validate_ref):
            for ref in sorted(refs):
                try:
                    validate_ref(ref)
                except Exception as exc:
                    raise ValidationError(f"continuation reference {ref!r} is not configured") from exc
        record = {
            "handoff_id": spec.handoff_id,
            "owner_id": owner,
            "spec": spec.to_dict(),
            "state": HandoffState.REQUESTED.value,
            "phase": "reserve",
            "resume_state": None,
            "reservation_id": reservation_id,
            "reservation": None,
            "backend_snapshots": backends,
            "reclamation_baselines": {},
            "reclamation_baseline_attempts": {},
            "continuation_refs": sorted(refs),
            "job_ids": [],
            "job_results": {},
            "suspended_refs": [],
            "effects": {},
            "cancel_requested": False,
            "cancel_reason": None,
            "created_at": self._now().isoformat(),
            "updated_at": self._now().isoformat(),
            "idempotency_key": idempotency_key,
            "error": None,
        }
        return record

    def _validate_request(self, spec: HandoffSpec, owner: str) -> None:
        if not isinstance(spec, HandoffSpec):
            raise ValidationError("handoff spec must be a HandoffSpec")
        if spec.owner_id != owner:
            raise ValidationError("handoff owner must match the authenticated caller")
        if spec.deadline is not None and spec.deadline <= self._now():
            raise ValidationError("handoff deadline has already passed")
        if any(job.owner_id != owner for job in spec.jobs):
            raise ValidationError("handoff jobs must have the authenticated owner")
        grants = {grant.backend_id: grant for grant in spec.eviction_grants}
        if len(grants) != len(spec.eviction_grants):
            raise ValidationError("handoff must contain only one grant per backend")
        for backend_id in spec.backend_ids:
            grant = grants.get(backend_id)
            if grant is None:
                raise ValidationError(f"missing explicit eviction grant for backend {backend_id}")
            if grant.scope not in {f"backend:{backend_id}", "*"}:
                raise ValidationError(f"eviction grant scope does not cover backend {backend_id}")
            if grant.expires_at <= self._now():
                raise ValidationError(f"eviction grant for backend {backend_id} has expired")

    def submit(
        self,
        spec: HandoffSpec | Mapping[str, Any],
        owner: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Persist a request only; effects begin on a later call to ``tick``."""
        model = spec if isinstance(spec, HandoffSpec) else HandoffSpec.from_dict(spec)
        if not idempotency_key or not owner:
            raise ValidationError("owner and idempotency_key are required")
        with self._lock:
            existing = self.ledger.get("handoff", model.handoff_id)
            if existing is not None:
                if existing.get("owner_id") != owner or existing.get("spec") != model.to_dict():
                    raise ConflictError("handoff ID is already bound to a different owner or specification")
                if existing.get("idempotency_key") != idempotency_key:
                    raise ConflictError("handoff ID was submitted with a different idempotency key")
                return existing
            record = self._initial(model, owner, idempotency_key)
            payload = {"spec": model.to_dict(), "owner_id": owner}

            def create(tx: Any) -> dict[str, Any]:
                saved = tx.put("handoff", model.handoff_id, record)
                tx.event("handoff", model.handoff_id, {"event": "requested", "state": "requested"})
                return saved

            saved = self.ledger.idempotent(owner, idempotency_key, "submit_handoff", payload, create)
            if (saved.get("handoff_id") != model.handoff_id or saved.get("owner_id") != owner
                    or saved.get("spec") != model.to_dict()):
                raise ConflictError("idempotency key is already bound to a different handoff")
            return saved

    def get(self, handoff_id: str) -> dict[str, Any]:
        return self._read(handoff_id)

    def cancel(self, handoff_id: str, reason: str = "caller") -> dict[str, Any]:
        if not reason:
            raise ValidationError("cancellation reason must be non-empty")
        with self._lock:
            record = self._read(handoff_id)
            if record.get("state") in _TERMINAL_HANDOFFS:
                return record
            if record.get("state") == HandoffState.UNKNOWN.value:
                raise ConflictError("an unknown handoff requires reconciliation before cancellation")
            if record.get("cancel_requested"):
                return record
            record["cancel_requested"] = True
            record["cancel_reason"] = reason
            return self._write(record, event="cancellation_requested")

    def recover(self, handoff_id: str) -> dict[str, Any]:
        """Resume a known blocked phase; never clear an ambiguous unknown state."""
        with self._lock:
            record = self._read(handoff_id)
            state = record.get("state")
            if state == HandoffState.UNKNOWN.value:
                return record
            if state != HandoffState.BLOCKED.value:
                return record
            resume_state = record.get("resume_state")
            if not resume_state:
                raise ConflictError("blocked handoff has no safe recovery phase")
            if record.get("phase") == "restore":
                attempts = dict(record.get("effect_attempts", {}))
                effects = dict(record.get("effects", {}))
                retryable = [name for name, value in effects.items()
                             if name.startswith("restore:") and
                             isinstance(value, Mapping) and
                             isinstance(value.get("receipt"), Mapping) and
                             value["receipt"].get("retry_safe") is True]
                for name in retryable:
                    effects.pop(name, None)
                    attempts[name] = int(attempts.get(name, 0)) + 1
                if retryable:
                    record["effects"] = effects
                    record["effect_attempts"] = attempts
            record["state"] = resume_state
            record["resume_state"] = None
            record["error"] = None
            if record.get("phase") == "restore":
                record.pop("restore_started_at", None)
                record.pop("readiness_observations", None)
            return self._write(record, event="recovery_requested")

    def _block(self, record: dict[str, Any], message: str, *, unknown: bool = False) -> dict[str, Any]:
        if record.get("state") not in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
            record["resume_state"] = record.get("state")
        record["state"] = HandoffState.UNKNOWN.value if unknown else HandoffState.BLOCKED.value
        record["error"] = message
        return self._write(record, event="unknown" if unknown else "blocked")

    def _set_phase(self, record: dict[str, Any], phase: str, *, state: str | None = None) -> dict[str, Any]:
        record["phase"] = phase
        if state:
            record["state"] = state
        record["error"] = None
        record["resume_state"] = None
        return self._write(record, event="phase_advanced")

    def _effect(
        self,
        record: dict[str, Any],
        name: str,
        operation: str,
        target: str,
        payload: Mapping[str, Any],
        callback: Callable[[], Mapping[str, Any]],
        *,
        reconcile: Callable[[], tuple[str, Mapping[str, Any] | None]] | None = None,
        retry_if_not_applied: bool = False,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        prior = record.get("effects", {}).get(name)
        if isinstance(prior, Mapping) and isinstance(prior.get("receipt"), Mapping):
            return record, prior["receipt"]
        attempt = int(record.get("effect_attempts", {}).get(name, 0))
        operation_id = self._operation_id(record["handoff_id"], operation, target, attempt)
        op = self.ledger.operation(operation_id)
        created_intent = op is None
        if created_intent:
            self.ledger.operation_intent(
                f"handoff.{operation}", payload, operation_id=operation_id,
                owner=record["owner_id"],
            )
        else:
            status = op.get("status")
            if status == "complete":
                receipts = op.get("receipts", [])
                completed = next((item.get("receipt") for item in reversed(receipts)
                                  if item.get("status") == "complete"), None)
                if not isinstance(completed, Mapping):
                    return self._block(record, f"operation {operation_id} is complete but has no recoverable receipt", unknown=True), {}
                return self._persist_effect_receipt(record, name, operation_id, completed)
            if status != "intent":
                return self._block(record, f"operation {operation_id} is {status!r} without a handoff receipt", unknown=True), {}
            if reconcile is None:
                return self._block(record, f"operation {operation_id} has an unresolved durable intent", unknown=True), {}
            try:
                outcome, recovered = reconcile()
            except Exception as exc:
                return self._block(record, f"operation {operation_id} reconciliation is unavailable: {type(exc).__name__}: {exc}"), {}
            if outcome == "applied" and isinstance(recovered, Mapping):
                return self._persist_effect_receipt(record, name, operation_id, recovered)
            if outcome != "not_applied" or not retry_if_not_applied:
                return self._block(record, f"operation {operation_id} remains unresolved after inspection"), {}
        try:
            receipt = _mapping(callback(), f"{operation} operation")
        except Exception as exc:
            return self._block(record, f"{operation} outcome is uncertain: {type(exc).__name__}: {exc}", unknown=True), {}
        return self._persist_effect_receipt(record, name, operation_id, receipt)

    def _persist_effect_receipt(
        self,
        record: dict[str, Any],
        name: str,
        operation_id: str,
        receipt: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        receipt_data = dict(receipt)
        effects = dict(record.get("effects", {}))
        effects[name] = {"operation_id": operation_id, "receipt": receipt_data}
        record["effects"] = effects
        with self.ledger.transaction() as tx:
            tx.record_receipt(operation_id, receipt_data, status="complete")
            current = tx.get("handoff", record["handoff_id"])
            if current is None:
                raise NotFoundError(f"Unknown handoff {record['handoff_id']}")
            clean = _plain(record)
            clean["updated_at"] = self._now().isoformat()
            saved = tx.put("handoff", record["handoff_id"], clean, current[_REVISION])
            tx.event("handoff", record["handoff_id"], {"event": "effect_receipt", "effect": name})
        return saved, receipt_data

    def _renew_if_needed(self, record: dict[str, Any]) -> dict[str, Any]:
        reservation = record.get("reservation")
        if not reservation or reservation.get("state") == "released":
            return record
        now = self._now()
        expires = datetime.fromisoformat(str(reservation["expires_at"]).replace("Z", "+00:00"))
        ttl = self._reservation_ttl(record["spec"])
        if expires > now + timedelta(seconds=ttl / 2):
            return record
        try:
            renewed = self.allocator.renew_envelope(
                record["reservation_id"], int(reservation["fencing_token"]), ttl
            )
        except Exception as exc:
            return self._block(record, f"protected reservation renewal failed: {type(exc).__name__}: {exc}", unknown=True)
        if renewed.get("state") != "active":
            return self._block(
                record,
                f"protected reservation is {renewed.get('state')!r}; handoff cannot continue without an active verified envelope",
                unknown=True,
            )
        record["reservation"] = renewed
        return self._write(record, event="reservation_renewed")

    @staticmethod
    def _reservation_ttl(spec_data: Mapping[str, Any]) -> float:
        spec = HandoffSpec.from_dict(spec_data)
        return max(120.0, spec.drain_timeout_seconds + spec.run_timeout_seconds + spec.restore_timeout_seconds + 60.0)

    def _do_reserve(self, record: dict[str, Any]) -> dict[str, Any]:
        spec = HandoffSpec.from_dict(record["spec"])
        backends = record["backend_snapshots"]
        lease_ids = [str(item["lease_id"]) for item in backends.values()]
        backend_lease_ids = {backend_id: str(item["lease_id"]) for backend_id, item in backends.items()}
        try:
            reservation = self.allocator.transfer_leases_to_envelope(
                lease_ids,
                record["reservation_id"],
                record["owner_id"],
                self._reservation_ttl(record["spec"]),
                idempotency_key=f"handoff:{record['handoff_id']}:reservation",
                requirements=spec.resource_envelope,
                backend_lease_ids=backend_lease_ids,
            )
        except Exception as exc:
            return self._block(record, f"cannot establish protected round-trip reservation: {type(exc).__name__}: {exc}")
        if reservation.get("state") != "active":
            return self._block(
                record,
                f"protected round-trip reservation is {reservation.get('state')!r}, expected active",
                unknown=True,
            )
        record["reservation"] = reservation
        return self._set_phase(record, "close_admission", state=HandoffState.RESERVED.value)

    def _backend(self, record: Mapping[str, Any], backend_id: str) -> BackendSpec:
        return BackendSpec.from_dict(record["backend_snapshots"][backend_id]["spec"])

    @staticmethod
    def _allocation_is_covered(
        requested: ResourceAllocation, granted: Mapping[str, Any]
    ) -> bool:
        if requested.resource_id != granted.get("resource_id"):
            return False
        try:
            requested_mode = AllocationMode(requested.mode)
            granted_mode = AllocationMode(granted.get("mode", AllocationMode.SHARED.value))
        except (TypeError, ValueError):
            return False
        if requested_mode == AllocationMode.EXCLUSIVE and granted_mode != AllocationMode.EXCLUSIVE:
            return False
        quantities = granted.get("quantities")
        if not isinstance(quantities, Mapping):
            return False
        return all(
            isinstance(quantities.get(key), (int, float))
            and not isinstance(quantities.get(key), bool)
            and math.isfinite(float(quantities[key]))
            and float(quantities[key]) + 1e-9 >= float(value)
            for key, value in requested.quantities.items()
        )

    def _restore_execution_context(self, record: Mapping[str, Any], backend_id: str) -> ExecutionContext:
        """Bind restoration to the backend's original grant within the live envelope."""
        reservation = self.ledger.get("reservation", record["reservation_id"])
        if not isinstance(reservation, Mapping) or reservation.get("state") != "active":
            state = reservation.get("state") if isinstance(reservation, Mapping) else "missing"
            raise ConflictError(f"protected reservation is {state!r}; backend restore requires an active envelope")
        source = record["backend_snapshots"].get(backend_id)
        if not isinstance(source, Mapping):
            raise ValidationError(f"backend {backend_id} has no immutable source allocation snapshot")
        raw_allocations = source.get("allocations")
        raw_resources = source.get("resources")
        if not isinstance(raw_allocations, (list, tuple)) or not raw_allocations:
            raise ValidationError(f"backend {backend_id} has no original residency allocations")
        if not isinstance(raw_resources, (list, tuple)) or not raw_resources:
            raise ValidationError(f"backend {backend_id} has no original resource descriptors")
        allocations = tuple(ResourceAllocation.from_dict(item) for item in raw_allocations)
        resources = tuple(Resource.from_dict(item) for item in raw_resources)
        envelope = reservation.get("allocations")
        if not isinstance(envelope, (list, tuple)):
            raise ValidationError("protected reservation has no allocation envelope")
        envelope_by_id = {
            str(item.get("resource_id")): item for item in envelope if isinstance(item, Mapping)
        }
        for allocation in allocations:
            grant = envelope_by_id.get(allocation.resource_id)
            if grant is None or not self._allocation_is_covered(allocation, grant):
                raise ConflictError(
                    f"protected reservation does not cover backend {backend_id} source allocation "
                    f"{allocation.resource_id}"
                )
        reservation_id = str(reservation.get("reservation_id", ""))
        fencing_token = reservation.get("fencing_token")
        if reservation_id != record["reservation_id"]:
            raise ConflictError("protected reservation identity does not match handoff")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int) or fencing_token <= 0:
            raise ConflictError("protected reservation has no valid fencing token")
        return ExecutionContext(
            allocations=allocations,
            resources=resources,
            claim_id=reservation_id,
            fencing_token=fencing_token,
            purpose=ExecutionPurpose.BACKEND,
        )

    def _reconcile_admission(
        self, record: Mapping[str, Any], backend_id: str, *, should_be_closed: bool
    ) -> tuple[str, Mapping[str, Any] | None]:
        observation = _mapping(
            self.backend_adapter.inspect(self._backend(record, backend_id)),
            "backend admission inspection",
        )
        nested = observation.get("admission_observation")
        admission = nested if isinstance(nested, Mapping) else observation
        document = admission.get("document")
        if not isinstance(document, Mapping):
            document = admission
        closed = document.get("closed", admission.get("admission_closed", observation.get("admission_closed")))
        operation_id = document.get("operation_id", admission.get("admission_operation_id", observation.get("admission_operation_id")))
        epoch = document.get("epoch", admission.get("admission_epoch", observation.get("admission_epoch")))
        admission_state = admission.get("admission_state", document.get("state", observation.get("admission_state")))
        expected_operation_id = record["handoff_id"]
        known = isinstance(closed, bool) and admission_state not in {"unknown", "unsupported"}
        if not known:
            return "unknown", None
        if operation_id == expected_operation_id:
            if closed == should_be_closed:
                return "applied", {
                    "state": "admission_closed" if should_be_closed else "open",
                    "admission_closed": should_be_closed,
                    "admission_open": not should_be_closed,
                    "operation_id": expected_operation_id,
                    "epoch": epoch,
                    "reconciled": True,
                }
            if closed != should_be_closed:
                return "not_applied", None
        if should_be_closed and closed is False:
            return "not_applied", None
        # A default/unconfigured FrameShift gate is closed to all provider
        # requests, but it has not yet recorded this handoff's fencing epoch.
        if should_be_closed and closed is True and operation_id is None and admission.get("configured") is False:
            return "not_applied", None
        return "unknown", None

    def _reconcile_continuation(
        self,
        record: Mapping[str, Any],
        continuation_ref: str,
        action: str,
        *,
        delivery_id: str | None = None,
    ) -> tuple[str, Mapping[str, Any] | None]:
        barrier = _mapping(
            self.continuation_adapter.inspect(
                continuation_ref, handoff_id=record["handoff_id"]
            ),
            "continuation barrier inspection",
        )
        state = barrier.get("state")
        if action == "suspend":
            if state in {"acknowledged", "delivered", "resumed"} and barrier.get("safe_boundary") is True and barrier.get("durable") is True:
                return "applied", {
                    "state": state, "safe_boundary": True, "durable": True,
                    "handoff_id": record["handoff_id"], "reconciled": True,
                }
            if state in {"absent", "requested"}:
                return "not_applied", None
            return "unknown", None
        observed_delivery_id = barrier.get("delivery_id")
        if action == "deliver":
            if state in {"delivered", "resumed"} and observed_delivery_id == delivery_id and barrier.get("durable") is True:
                return "applied", {
                    "state": "delivered", "delivered": True, "durable": True,
                    "delivery_id": delivery_id, "reconciled": True,
                }
            if state == "acknowledged" and barrier.get("safe_boundary") is True and observed_delivery_id is None:
                return "not_applied", None
            return "unknown", None
        if action == "resume":
            if state == "resumed" and observed_delivery_id == delivery_id:
                return "applied", {
                    "state": "resumed", "resumed": True,
                    "delivery_id": delivery_id, "reconciled": True,
                }
            if state == "delivered" and observed_delivery_id == delivery_id and barrier.get("durable") is True:
                return "not_applied", None
        return "unknown", None

    def _close_admission(self, record: dict[str, Any]) -> dict[str, Any]:
        spec = HandoffSpec.from_dict(record["spec"])
        if not record.get("drain_started_at"):
            started = self._now()
            deadline = started + timedelta(seconds=spec.drain_timeout_seconds)
            if spec.deadline is not None:
                deadline = min(deadline, spec.deadline)
            record["drain_started_at"] = started.isoformat()
            record["drain_deadline"] = deadline.isoformat()
            record = self._write(record, event="drain_started")
        for backend_id in spec.backend_ids:
            name = f"drain:{backend_id}"
            record, receipt = self._effect(
                record, name, "drain", backend_id,
                {"backend_id": backend_id, "handoff_id": record["handoff_id"]},
                lambda backend=self._backend(record, backend_id): self.backend_adapter.drain(
                    backend, operation_id=record["handoff_id"], timeout_seconds=spec.drain_timeout_seconds
                ),
                reconcile=lambda backend_id=backend_id: self._reconcile_admission(
                    record, backend_id, should_be_closed=True
                ),
                retry_if_not_applied=True,
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("admission_closed") is not True:
                return self._block(record, f"backend {backend_id} did not confirm admission closure")
        return self._set_phase(record, "suspend_consumers", state=HandoffState.DRAINING.value)

    def _suspend_consumers(self, record: dict[str, Any]) -> dict[str, Any]:
        deadline = datetime.fromisoformat(record["drain_deadline"].replace("Z", "+00:00"))
        for ref in record["continuation_refs"]:
            name = f"suspend:{ref}"
            record, receipt = self._effect(
                record, name, "suspend", ref,
                {"continuation_ref": ref, "handoff_id": record["handoff_id"]},
                lambda ref=ref: self.continuation_adapter.suspend(
                    ref, handoff_id=record["handoff_id"], deadline=deadline
                ),
                reconcile=lambda ref=ref: self._reconcile_continuation(
                    record, ref, "suspend"
                ),
                retry_if_not_applied=True,
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("safe_boundary") is not True or receipt.get("durable") is not True:
                return self._block(record, f"continuation {ref} did not acknowledge a durable safe boundary")
            suspended = set(record.get("suspended_refs", []))
            if ref not in suspended:
                suspended.add(ref)
                record["suspended_refs"] = sorted(suspended)
                record = self._write(record, event="continuation_suspended")
        return self._set_phase(record, "verify_drain", state=HandoffState.DRAINING.value)

    def _verify_drain(self, record: dict[str, Any]) -> dict[str, Any]:
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            try:
                observation = _mapping(self.backend_adapter.inspect(self._backend(record, backend_id)), "backend inspection")
            except Exception as exc:
                return self._block(record, f"backend {backend_id} drain state is unknown: {type(exc).__name__}: {exc}", unknown=True)
            active = observation.get("active_generations")
            if observation.get("state") not in {"observed", "ready", "idle"}:
                return self._block(record, f"backend {backend_id} drain state is not known", unknown=True)
            if isinstance(active, bool) or not isinstance(active, int) or active < 0:
                return self._block(record, f"backend {backend_id} active generation count is unknown", unknown=True)
            if active:
                if self._now() >= datetime.fromisoformat(str(record["drain_deadline"]).replace("Z", "+00:00")):
                    return self._block(record, f"backend {backend_id} did not reach a turn boundary before the drain timeout")
                return record
        return self._set_phase(record, "capture_baseline", state=HandoffState.DRAINING.value)

    def _capture_reclamation_baselines(self, record: dict[str, Any]) -> dict[str, Any]:
        capture = getattr(self.backend_adapter, "capture_reclamation_baseline", None)
        if not callable(capture):
            return self._block(
                record,
                "backend adapter does not support the required pre-eviction resource reclamation baseline",
            )
        baselines = dict(record.get("reclamation_baselines", {}))
        attempts = dict(record.get("reclamation_baseline_attempts", {}))
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            existing = baselines.get(backend_id)
            if isinstance(existing, Mapping) and self._baseline_is_fresh(existing):
                continue
            attempt = int(attempts.get(backend_id, 0))
            operation_id = f"{record['handoff_id']}:baseline:{attempt}"
            # This read-only observation may be repeated after a crash. Persist
            # the next identity first so a partial sample cannot be confused with
            # the durable baseline consumed by eviction verification.
            attempts[backend_id] = attempt + 1
            record["reclamation_baseline_attempts"] = attempts
            record = self._write(record, event="reclamation_baseline_capture_started")
            try:
                response = _mapping(
                    capture(self._backend(record, backend_id), operation_id=operation_id),
                    "reclamation baseline",
                )
                baseline = self._validate_reclamation_baseline(
                    response, backend_id=backend_id, operation_id=operation_id
                )
                backend_spec = self._backend(record, backend_id)
                if (
                    baseline.get("profile_id") != backend_spec.profile_id
                    or baseline.get("profile_version")
                    != backend_spec.pinned_configuration.get("profile_version")
                ):
                    raise ValidationError("reclamation baseline profile identity does not match the pinned backend")
            except Exception as exc:
                return self._block(
                    record,
                    f"backend {backend_id} cannot establish a trusted pre-eviction baseline: {type(exc).__name__}: {exc}",
                )
            if not self._baseline_is_fresh(baseline):
                return self._block(record, f"backend {backend_id} reclamation baseline is stale")
            baselines[backend_id] = baseline
            record["reclamation_baselines"] = baselines
            record = self._write(record, event="reclamation_baseline_persisted")
        return self._set_phase(record, "evict", state=HandoffState.EVICTING.value)

    def _eviction_started(self, record: Mapping[str, Any]) -> bool:
        effects = record.get("effects", {})
        if any(str(name).startswith("evict:") for name in effects):
            return True
        attempts = record.get("effect_attempts", {})
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            operation_id = self._operation_id(
                record["handoff_id"], "evict", backend_id,
                int(attempts.get(f"evict:{backend_id}", 0)),
            )
            if self.ledger.operation(operation_id) is not None:
                return True
        return False

    def _evict(self, record: dict[str, Any]) -> dict[str, Any]:
        baselines = dict(record.get("reclamation_baselines", {}))
        if any(
            not isinstance(baselines.get(backend_id), Mapping)
            or not self._baseline_is_fresh(baselines[backend_id])
            for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids
        ):
            if self._eviction_started(record):
                return self._block(
                    record,
                    "a pre-eviction reclamation baseline expired after physical eviction began; refusing to recapture against changed resources",
                )
            return self._set_phase(record, "capture_baseline", state=HandoffState.DRAINING.value)
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            name = f"evict:{backend_id}"
            prior = record.get("effects", {}).get(name)
            if isinstance(prior, Mapping) and isinstance(prior.get("receipt"), Mapping):
                receipt = prior["receipt"]
            else:
                try:
                    observation = _mapping(self.backend_adapter.inspect(self._backend(record, backend_id)), "backend inspection")
                except Exception as exc:
                    return self._block(record, f"backend {backend_id} final pre-eviction inspection is unknown: {type(exc).__name__}: {exc}", unknown=True)
                active = observation.get("active_generations")
                if observation.get("state") not in {"observed", "ready", "idle"}:
                    return self._block(record, f"backend {backend_id} final drain state is not known", unknown=True)
                if isinstance(active, bool) or not isinstance(active, int) or active < 0:
                    return self._block(record, f"backend {backend_id} final active-generation count is unknown", unknown=True)
                if active != 0:
                    return self._block(record, f"backend {backend_id} acquired an active generation after drain verification")
                record, receipt = self._effect(
                    record, name, "evict", backend_id,
                    {"backend_id": backend_id, "handoff_id": record["handoff_id"]},
                    lambda backend=self._backend(record, backend_id): self.backend_adapter.evict(
                        backend,
                        operation_id=record["handoff_id"],
                        reclamation_baseline=baselines[backend_id],
                    ),
                )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            reclamation = receipt.get("resource_reclamation")
            if receipt.get("state") != "evicted" or receipt.get("cleanup_verified") is not True:
                return self._block(record, f"backend {backend_id} eviction is not verified", unknown=receipt.get("state") == "unknown")
            if not isinstance(reclamation, Mapping) or reclamation.get("state") != "reclaimed" or not reclamation.get("evidence"):
                return self._block(record, f"backend {backend_id} resource reclamation is not verified", unknown=True)
        return self._set_phase(record, "submit_jobs", state=HandoffState.EVICTING.value)

    def _submit_jobs(self, record: dict[str, Any]) -> dict[str, Any]:
        spec = HandoffSpec.from_dict(record["spec"])
        if not record.get("run_started_at"):
            record["run_started_at"] = self._now().isoformat()
            record = self._write(record, event="execution_started")
        submitted = list(record.get("job_ids", []))
        for index, job in enumerate(spec.jobs):
            key = f"handoff:{spec.handoff_id}:job:{index}"
            try:
                response = _mapping(
                    self.submit_job(job.to_dict(), record["owner_id"], key, reservation_id=record["reservation_id"]),
                    "job submission",
                )
            except Exception as exc:
                return self._block(record, f"job submission outcome is uncertain for batch item {index}: {type(exc).__name__}: {exc}", unknown=True)
            job_id = response.get("id") or response.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                return self._block(record, f"scheduler returned no stable job ID for batch item {index}", unknown=True)
            if index < len(submitted) and submitted[index] != job_id:
                return self._block(record, f"idempotent job key returned a different job ID for batch item {index}", unknown=True)
            if index == len(submitted):
                submitted.append(job_id)
                record["job_ids"] = submitted
                record = self._write(record, event="job_submitted")
        return self._set_phase(record, "wait_jobs", state=HandoffState.RUNNING.value)

    def _persist_restored_handle(
        self, record: dict[str, Any], backend_id: str, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        handle = receipt.get("execution_handle")
        if not isinstance(handle, Mapping) or not handle:
            return self._block(
                record,
                f"backend {backend_id} restoration did not return a durable execution handle",
                unknown=True,
            )
        error: str | None = None
        with self.ledger.transaction() as tx:
            backend = tx.get("backend", backend_id)
            if backend is None:
                error = f"restored backend {backend_id} disappeared from the ledger"
            elif (backend.get("handoff_id") != record["reservation_id"]
                    or backend.get("transferred_lease_id") != record["backend_snapshots"][backend_id]["lease_id"]):
                error = f"restored backend {backend_id} no longer belongs to this handoff"
            elif not (backend.get("execution_handle") == dict(handle)
                      and backend.get("restoration_operation_id") == receipt.get("operation_id")):
                updated = _plain(backend)
                updated["execution_handle"] = dict(handle)
                updated["state"] = "restoring"
                updated["restoration_operation_id"] = receipt.get("operation_id")
                updated["restoration_observation"] = dict(receipt)
                updated["updated_at"] = self._now().isoformat()
                tx.put("backend", backend_id, updated, backend[_REVISION])
                tx.event("backend_restoration_handle_recorded", backend_id, {
                    "handoff_id": record["handoff_id"],
                    "operation_id": receipt.get("operation_id"),
                })
        if error:
            return self._block(record, error, unknown=True)
        return record

    def _wait_jobs(self, record: dict[str, Any]) -> dict[str, Any]:
        spec = HandoffSpec.from_dict(record["spec"])
        deadline = datetime.fromisoformat(str(record.get("run_started_at", record["updated_at"])).replace("Z", "+00:00")) + timedelta(seconds=spec.run_timeout_seconds)
        all_done = True
        results = dict(record.get("job_results", {}))
        if record.get("cancel_requested") and self.cancel_job is None:
            return self._block(record, "handoff cancellation requires a configured safe job cancellation callback")
        for job_id in record.get("job_ids", []):
            try:
                job = _mapping(self.get_job(job_id), "job inspection")
            except Exception as exc:
                return self._block(record, f"job {job_id} inspection is uncertain: {type(exc).__name__}: {exc}", unknown=True)
            state = _state(job.get("state"))
            if state == "unknown":
                return self._block(record, f"job {job_id} has an unknown execution outcome", unknown=True)
            if state not in _TERMINAL_JOBS:
                all_done = False
                if self._now() >= deadline and not record.get("cancel_requested"):
                    record["cancel_requested"] = True
                    record["cancel_reason"] = "handoff run timeout"
                    record = self._write(record, event="run_timeout_cancellation_requested")
                if record.get("cancel_requested"):
                    if self.cancel_job is None:
                        return self._block(record, "handoff cancellation requires a configured safe job cancellation callback")
                    cancel_name = f"cancel-job:{job_id}"
                    record, receipt = self._effect(
                        record, cancel_name, "cancel_job", job_id,
                        {"job_id": job_id, "reason": record.get("cancel_reason") or "caller"},
                        lambda job_id=job_id: self.cancel_job(job_id, record.get("cancel_reason") or "caller"),
                    )
                    if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                        return record
                    if receipt.get("id") not in {None, job_id} and receipt.get("job_id") not in {None, job_id}:
                        return self._block(record, f"cancellation receipt named a different job than {job_id}", unknown=True)
                continue
            if job.get("cleanup_verified") is not True:
                return self._block(record, f"terminal job {job_id} has no verified execution cleanup", unknown=True)
            results[job_id] = {"state": state, "result": job.get("result"), "cleanup_verified": True}
        if results != record.get("job_results", {}):
            record["job_results"] = results
            record = self._write(record, event="job_results_persisted")
        if not all_done:
            return record
        return self._set_phase(record, "restore", state=HandoffState.CLEANING.value)

    def _restore(self, record: dict[str, Any]) -> dict[str, Any]:
        if not record.get("restore_started_at"):
            record["restore_started_at"] = self._now().isoformat()
            record = self._write(record, event="restore_started")
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            name = f"restore:{backend_id}"
            if f"evict:{backend_id}" not in record.get("effects", {}):
                continue
            try:
                execution_context = self._restore_execution_context(record, backend_id)
            except Exception as exc:
                return self._block(
                    record,
                    f"backend {backend_id} cannot be safely bound to the protected restoration allocation: "
                    f"{type(exc).__name__}: {exc}",
                )
            operation_attempt = int(record.get("effect_attempts", {}).get(name, 0))
            restore_operation_id = (
                record["handoff_id"] if operation_attempt == 0
                else f"{record['handoff_id']}:restore-retry:{operation_attempt}"
            )
            record, receipt = self._effect(
                record, name, "restore", backend_id,
                {
                    "backend_id": backend_id,
                    "handoff_id": record["handoff_id"],
                    "execution_context": execution_context.to_dict(),
                },
                lambda backend=self._backend(record, backend_id): self.backend_adapter.restore(
                    backend, operation_id=restore_operation_id, execution_context=execution_context
                ),
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("state") not in {"restored", "ready", "starting"}:
                return self._block(record, f"backend {backend_id} restoration failed; reservation remains protected", unknown=receipt.get("state") == "unknown" or receipt.get("retry_safe") is not True)
            record = self._persist_restored_handle(record, backend_id, receipt)
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
        return self._set_phase(record, "verify_ready", state=HandoffState.RESTORING.value)

    def _verify_ready(self, record: dict[str, Any]) -> dict[str, Any]:
        spec = HandoffSpec.from_dict(record["spec"])
        observations: dict[str, Any] = {}
        for backend_id in spec.backend_ids:
            try:
                receipt = _mapping(self.backend_adapter.ready(
                    self._backend(record, backend_id), timeout_seconds=spec.restore_timeout_seconds,
                    operation_id=record["handoff_id"],
                ), "backend readiness")
            except Exception as exc:
                return self._block(record, f"backend {backend_id} readiness is unknown: {type(exc).__name__}: {exc}", unknown=True)
            observations[backend_id] = dict(receipt)
            if receipt.get("ready") is not True:
                started = datetime.fromisoformat(record.get("restore_started_at", record["updated_at"]).replace("Z", "+00:00"))
                if self._now() < started + timedelta(seconds=spec.restore_timeout_seconds):
                    record["readiness_observations"] = observations
                    return self._write(record, event="readiness_pending")
                return self._block(record, f"backend {backend_id} is not ready before the restoration deadline; reservation remains protected")
        record["readiness_observations"] = observations
        return self._set_phase(record, "settle_reservation", state=HandoffState.DELIVERING.value)

    def _settle(self, record: dict[str, Any]) -> dict[str, Any]:
        try:
            restored = self.allocator.settle_envelope_to_leases(
                record["reservation_id"],
                ttl_seconds=self._reservation_ttl(record["spec"]),
                receipt={"verified": True, "backend_ready": True, "reservation_id": record["reservation_id"], "handoff_id": record["handoff_id"]},
            )
        except Exception as exc:
            return self._block(record, f"backend residency lease settlement failed: {type(exc).__name__}: {exc}")
        record["restored_leases"] = restored
        return self._set_phase(record, "reopen_admission", state=HandoffState.DELIVERING.value)

    def _reopen_admission(self, record: dict[str, Any]) -> dict[str, Any]:
        for backend_id in HandoffSpec.from_dict(record["spec"]).backend_ids:
            name = f"reopen:{backend_id}"
            if f"drain:{backend_id}" not in record.get("effects", {}):
                continue
            record, receipt = self._effect(
                record, name, "reopen", backend_id,
                {"backend_id": backend_id, "handoff_id": record["handoff_id"]},
                lambda backend=self._backend(record, backend_id): self.backend_adapter.reopen(
                    backend, operation_id=record["handoff_id"]
                ),
                reconcile=lambda backend_id=backend_id: self._reconcile_admission(
                    record, backend_id, should_be_closed=False
                ),
                retry_if_not_applied=True,
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("admission_open") is not True and receipt.get("state") not in {"ready", "open"}:
                return self._block(record, f"backend {backend_id} admission did not reopen")
        return self._set_phase(record, "deliver", state=HandoffState.DELIVERING.value)

    def _deliver(self, record: dict[str, Any]) -> dict[str, Any]:
        results = record.get("job_results", {})
        payload = {
            "handoff_id": record["handoff_id"],
            "cancelled": bool(record.get("cancel_requested")),
            "jobs": [
                {"job_id": job_id, **result}
                for job_id, result in sorted(results.items())
            ],
            "error": record.get("cancel_reason") if record.get("cancel_requested") else None,
        }
        refs = record.get("suspended_refs", []) if record.get("cancel_requested") else record["continuation_refs"]
        for ref in refs:
            delivery_id = self._delivery_id(record["handoff_id"], ref)
            name = f"deliver:{ref}"
            record, receipt = self._effect(
                record, name, "deliver", ref,
                {"continuation_ref": ref, "delivery_id": delivery_id, "result": payload},
                lambda ref=ref, delivery_id=delivery_id: self.continuation_adapter.deliver(
                    ref, delivery_id=delivery_id, result=payload
                ),
                reconcile=lambda ref=ref, delivery_id=delivery_id: self._reconcile_continuation(
                    record, ref, "deliver", delivery_id=delivery_id
                ),
                retry_if_not_applied=True,
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("durable") is not True or receipt.get("delivered") is not True:
                return self._block(record, f"result delivery to continuation {ref} is not durable")
        return self._set_phase(record, "resume", state=HandoffState.RESUMING.value)

    def _resume(self, record: dict[str, Any]) -> dict[str, Any]:
        refs = record.get("suspended_refs", []) if record.get("cancel_requested") else record["continuation_refs"]
        for ref in refs:
            delivery_id = self._delivery_id(record["handoff_id"], ref)
            name = f"resume:{ref}"
            record, receipt = self._effect(
                record, name, "resume", ref,
                {"continuation_ref": ref, "delivery_id": delivery_id},
                lambda ref=ref, delivery_id=delivery_id: self.continuation_adapter.resume(
                    ref, delivery_id=delivery_id
                ),
                reconcile=lambda ref=ref, delivery_id=delivery_id: self._reconcile_continuation(
                    record, ref, "resume", delivery_id=delivery_id
                ),
                retry_if_not_applied=True,
            )
            if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                return record
            if receipt.get("resumed") is not True:
                return self._block(record, f"continuation {ref} did not confirm resumption")
        cancelled = bool(record.get("cancel_requested"))
        record["state"] = HandoffState.CANCELLED.value if cancelled else HandoffState.COMPLETE.value
        record["phase"] = "cancelled" if cancelled else "complete"
        record["completed_at"] = self._now().isoformat()
        record["error"] = None
        return self._write(record, event="cancelled" if cancelled else "complete")

    def _cancel_before_eviction(self, record: dict[str, Any]) -> dict[str, Any]:
        if not record.get("reservation"):
            record["state"] = HandoffState.CANCELLED.value
            record["phase"] = "cancelled"
            record["completed_at"] = self._now().isoformat()
            return self._write(record, event="cancelled_before_reservation")
        if any(name.startswith("evict:") for name in record.get("effects", {})):
            return self._set_phase(record, "restore", state=HandoffState.CLEANING.value)
        if not record.get("restore_started_at"):
            record["restore_started_at"] = self._now().isoformat()
            record = self._write(record, event="cancel_cleanup_started")
        return self._set_phase(record, "verify_ready", state=HandoffState.RESTORING.value)

    def _advance(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
            return record
        if record.get("state") in _TERMINAL_HANDOFFS:
            return record
        phase = record.get("phase")
        if record.get("cancel_requested"):
            if phase == "reserve" and not record.get("reservation"):
                return self._cancel_before_eviction(record)
            if phase in {"close_admission", "suspend_consumers", "verify_drain", "capture_baseline", "evict", "submit_jobs"} and not record.get("job_ids"):
                return self._cancel_before_eviction(record)
            if phase == "wait_jobs":
                return self._wait_jobs(record)
        record = self._renew_if_needed(record)
        if record.get("state") in {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
            return record
        phase = record.get("phase")
        if phase == "reserve": return self._do_reserve(record)
        if phase == "close_admission": return self._close_admission(record)
        if phase == "suspend_consumers": return self._suspend_consumers(record)
        if phase == "verify_drain": return self._verify_drain(record)
        if phase == "capture_baseline": return self._capture_reclamation_baselines(record)
        if phase == "evict": return self._evict(record)
        if phase == "submit_jobs": return self._submit_jobs(record)
        if phase == "wait_jobs": return self._wait_jobs(record)
        if phase == "restore": return self._restore(record)
        if phase == "verify_ready": return self._verify_ready(record)
        if phase == "settle_reservation": return self._settle(record)
        if phase == "reopen_admission": return self._reopen_admission(record)
        if phase == "deliver": return self._deliver(record)
        if phase == "resume": return self._resume(record)
        return self._block(record, f"unknown handoff phase {phase!r}", unknown=True)

    def tick(self) -> dict[str, Any]:
        """Advance at most one phase per active handoff and return the snapshot."""
        with self._lock:
            changed: list[str] = []
            for item in self.ledger.list("handoff"):
                if item.get("state") in _TERMINAL_HANDOFFS | {HandoffState.BLOCKED.value, HandoffState.UNKNOWN.value}:
                    continue
                current = self._advance(item)
                changed.append(current["handoff_id"])
            return {"advanced": changed, "handoffs": self.ledger.list("handoff")}


__all__ = ["HandoffCoordinator"]
