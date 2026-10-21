"""Project Control compatibility bridge to Device Scheduler MCP operations.

This adapter keeps the narrow HostResourceFacade reservation/owner DTO shape
for migrated callers while forwarding authority to Device Scheduler. It does
not maintain a second ledger and does not synthesize process-quiescence or
reclamation evidence. Legacy preemption and residency-protection APIs fail
closed; managed model services must migrate to backend/handoff contracts.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import itertools
import json
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


class ProjectControlAdapterError(RuntimeError):
    """The scheduler could not safely honor a Project Control resource call."""


class UnsupportedLegacyOperation(ProjectControlAdapterError):
    """A Project Control-only lifecycle operation has no lease equivalent."""


class ReclamationPending(ProjectControlAdapterError):
    """Device Scheduler has not verified that an owner's resources are free."""


_ACTIVE_LEASE_STATES = frozenset({"active", "release_requested", "reclaiming", "unknown", "expired"})
_GPU_KIND_ALIASES = frozenset({"accelerator", "gpu", "nvidia_gpu", "cuda_gpu"})


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _process_start(pid: int | None) -> str | None:
    """Record a Linux process generation token as metadata, never as a receipt."""
    if pid is None or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        return None


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectControlAdapterError(f"scheduler returned an invalid {label}")
    return dict(value)


def _await_sync(awaitable: Any) -> Any:
    """Run the async public SDK from a synchronous Project Control call site."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # Project Control's current host facade is synchronous and can be reached
    # from async service code. Run the independent MCP request on a helper
    # thread instead of trying to nest an event loop in the caller's thread.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ds-pc-bridge") as pool:
        return pool.submit(asyncio.run, awaitable).result()


class DeviceSchedulerProjectControlAdapter:
    """Synchronous HostResourceFacade bridge over the public async MCP client.

    The injected client must be an authenticated ``DeviceSchedulerClient`` (or
    a contract-compatible fake) with public ``call(operation, arguments)``.
    Its principal needs operation grants for `resources.list`, `leases.request`,
    `leases.list`, `leases.renew`, and `leases.release`, plus
    `leases.request:any_owner`, `leases.read:any_owner`, and
    `leases.manage:any_owner`. Device Scheduler remains the only allocator.
    """

    def __init__(self, client: Any, *, lease_ttl_seconds: float = 60.0,
                 poll_seconds: float = 0.1, clock: Any = time.monotonic,
                 sleeper: Any = time.sleep) -> None:
        if client is None or not callable(getattr(client, "call", None)):
            raise TypeError("client must expose the public call(operation, arguments) method")
        if isinstance(lease_ttl_seconds, bool) or not isinstance(lease_ttl_seconds, (int, float)) or lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self._client = client
        self._lease_ttl_seconds = float(lease_ttl_seconds)
        self._poll_seconds = float(poll_seconds)
        self._clock = clock
        self._sleep = sleeper

    def _call(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            result = self._client.call(operation, dict(arguments or {}))
            if inspect.isawaitable(result):
                result = _await_sync(result)
        except Exception as exc:
            # The public SDK surfaces structured MCP domain errors as
            # exceptions with a stable ``code`` attribute. Translate only
            # allocation conflicts to the facade's no-grant result; transport,
            # authorization and validation failures must remain errors.
            code = getattr(exc, "code", None)
            if code in {"conflict", "resource_conflict", "admission_denied", "idempotency_conflict"}:
                raise _ResourceConflict(str(exc)) from exc
            if isinstance(code, str):
                raise ProjectControlAdapterError(f"{operation} failed ({code}): {exc}") from exc
            raise ProjectControlAdapterError(f"{operation} failed: {type(exc).__name__}") from exc
        result = _mapping(result, f"{operation} response")
        error = result.get("error")
        if isinstance(error, Mapping):
            code = str(error.get("code", "operation_failed"))
            message = str(error.get("message", "Device Scheduler operation failed"))
            if code in {"conflict", "resource_conflict", "admission_denied", "idempotency_conflict"}:
                raise _ResourceConflict(message)
            raise ProjectControlAdapterError(f"{operation} failed ({code}): {message}")
        return result

    def _resources(self) -> list[dict[str, Any]]:
        result = self._call("resources.list")
        values = result.get("resources")
        if not isinstance(values, list):
            raise ProjectControlAdapterError("resources.list omitted its resources array")
        resources = []
        for value in values:
            item = _mapping(value, "resource")
            nested = item.get("resource")
            if isinstance(nested, Mapping):
                item = {**dict(nested), **{key: child for key, child in item.items() if key != "resource"}}
            resources.append(item)
        return resources

    @staticmethod
    def _kind(resource: Mapping[str, Any]) -> str:
        return str(resource.get("kind", ""))

    @staticmethod
    def _legacy_resource_id(resource: Mapping[str, Any]) -> str:
        kind = str(resource.get("kind", ""))
        attrs = resource.get("attributes") if isinstance(resource.get("attributes"), Mapping) else {}
        if kind.casefold() in _GPU_KIND_ALIASES and isinstance(attrs.get("uuid"), str):
            return f"accelerator:{attrs['uuid']}"
        return str(resource.get("resource_id", ""))

    @staticmethod
    def _legacy_id_from_resource_id(resource_id: str) -> str | None:
        # NvidiaAdapter's stable identity is gpu:<host>:<UUID>. The UUID
        # suffix preserves the old DTO if inventory disappears after a grant.
        if resource_id.startswith("gpu:"):
            parts = resource_id.split(":", 2)
            if len(parts) == 3 and parts[2]:
                return f"accelerator:{parts[2]}"
        return None

    @staticmethod
    def _isolation_members(resource: Mapping[str, Any]) -> set[str]:
        if str(resource.get("kind", "")).casefold() != "isolation":
            return set()
        attrs = resource.get("attributes") if isinstance(resource.get("attributes"), Mapping) else {}
        members = attrs.get("member_resource_ids", ())
        if not isinstance(members, (list, tuple, set, frozenset)):
            return set()
        return {str(item) for item in members}

    @staticmethod
    def _legacy_tags(resource: Mapping[str, Any], isolation: Sequence[Mapping[str, Any]] = ()) -> dict[str, str]:
        attrs = resource.get("attributes") if isinstance(resource.get("attributes"), Mapping) else {}
        capacities = resource.get("capacities") if isinstance(resource.get("capacities"), Mapping) else {}
        tags: dict[str, str] = {}
        for name in ("uuid", "pci_bus_id", "name"):
            if attrs.get(name) is not None:
                tags[name] = str(attrs[name])
        if attrs.get("index_observation") is not None:
            tags["index"] = str(attrs["index_observation"])
        if attrs.get("memory_free_bytes_observation") is not None:
            try:
                tags["memory_free_mib"] = str(int(float(attrs["memory_free_bytes_observation"]) / (1024 * 1024)))
            except (TypeError, ValueError, OverflowError):
                pass
        if attrs.get("utilization_percent_observation") is not None:
            tags["utilization_percent"] = str(attrs["utilization_percent_observation"])
        if capacities.get("memory_bytes") is not None:
            try:
                tags["memory_total_mib"] = str(int(float(capacities["memory_bytes"]) / (1024 * 1024)))
            except (TypeError, ValueError, OverflowError):
                pass
        resource_id = str(resource.get("resource_id", ""))
        domains = [
            item for item in isolation
            if resource_id in DeviceSchedulerProjectControlAdapter._isolation_members(item)
        ]
        if len(domains) == 1:
            domain_attrs = domains[0].get("attributes") if isinstance(domains[0].get("attributes"), Mapping) else {}
            tags["nvlink_domain"] = str(domain_attrs.get("domain_id", domains[0].get("resource_id", "")))
        return tags

    @staticmethod
    def _legacy_enabled(resource: Mapping[str, Any]) -> bool:
        if resource.get("available") is False:
            return False
        capacities = resource.get("capacities") if isinstance(resource.get("capacities"), Mapping) else {}
        return bool(capacities) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            for value in capacities.values()
        )

    def list(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        resources = self._resources()
        isolation = [item for item in resources if self._kind(item).casefold() == "isolation"]
        wanted_kind = "accelerator" if kind is not None and kind.casefold() in _GPU_KIND_ALIASES else kind
        result = []
        for resource in resources:
            actual_kind = self._kind(resource)
            legacy_kind = "accelerator" if actual_kind.casefold() in _GPU_KIND_ALIASES else actual_kind
            if wanted_kind is not None and legacy_kind != wanted_kind:
                continue
            result.append({
                "id": self._legacy_resource_id(resource),
                "kind": legacy_kind,
                "tags": self._legacy_tags(resource, isolation),
                "enabled": self._legacy_enabled(resource),
            })
        return sorted(result, key=lambda item: str(item["id"]))

    def discover_gpus(self) -> list[dict[str, Any]]:
        """Return DS's current GPU inventory; discovery itself is server-owned."""
        return self.list(kind="accelerator")

    @staticmethod
    def _validate_legacy_request(value: object | None) -> dict[str, Any]:
        raw = {} if value is None else _mapping(value, "legacy resource request")
        allowed = {
            "schema_version", "kind", "ids", "count", "tags", "exclusive_resources",
            "isolate_pcie_root", "isolate_nvlink_domain", "cpu_threads", "cpu_heavy", "ram_bytes",
        }
        extra = set(raw) - allowed
        if extra:
            raise ProjectControlAdapterError(f"unsupported legacy resource request fields: {', '.join(sorted(extra))}")
        if raw.get("schema_version", 1) != 1:
            raise ProjectControlAdapterError("only Project Control resource request schema version 1 is supported")
        kind = raw.get("kind", "accelerator")
        if not isinstance(kind, str) or not kind:
            raise ProjectControlAdapterError("resource request kind must be non-empty text")
        ids = raw.get("ids", ())
        if not isinstance(ids, (list, tuple)) or any(not isinstance(item, str) or not item for item in ids):
            raise ProjectControlAdapterError("resource request ids must be an array of non-empty strings")
        if len(set(ids)) != len(ids):
            raise ProjectControlAdapterError("resource request ids must not contain duplicates")
        if "ids" in raw and "count" in raw:
            raise ProjectControlAdapterError("resource request may declare ids or count, not both")
        count = raw.get("count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ProjectControlAdapterError("resource request count must be a non-negative integer")
        tags = raw.get("tags", {})
        if not isinstance(tags, Mapping) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in tags.items()):
            raise ProjectControlAdapterError("resource request tags must map strings to strings")
        exclusive = raw.get("exclusive_resources", ())
        if not isinstance(exclusive, (list, tuple)) or any(not isinstance(item, str) or not item for item in exclusive):
            raise ProjectControlAdapterError("exclusive_resources must be an array of non-empty strings")
        if len(set(exclusive)) != len(exclusive):
            raise ProjectControlAdapterError("exclusive_resources must not contain duplicates")
        bool_fields = ("isolate_pcie_root", "isolate_nvlink_domain", "cpu_heavy")
        for field in bool_fields:
            if field in raw and not isinstance(raw[field], bool):
                raise ProjectControlAdapterError(f"{field} must be a boolean")
        for field in ("cpu_threads", "ram_bytes"):
            number = raw.get(field, 0)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ProjectControlAdapterError(f"{field} must be a non-negative integer")
        return {
            "kind": kind,
            "ids": tuple(ids),
            "count": count,
            "tags": dict(tags),
            "exclusive_resources": tuple(exclusive),
            "isolate_pcie_root": raw.get("isolate_pcie_root", False),
            "isolate_nvlink_domain": raw.get("isolate_nvlink_domain", False),
            "cpu_threads": raw.get("cpu_threads", 0),
            "cpu_heavy": raw.get("cpu_heavy", False),
            "ram_bytes": raw.get("ram_bytes", 0),
        }

    @staticmethod
    def _gpu_capacity(resource: Mapping[str, Any]) -> float | None:
        capacities = resource.get("capacities") if isinstance(resource.get("capacities"), Mapping) else {}
        amount = capacities.get("memory_bytes")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            return None
        return float(amount)

    def _resource_requirements(self, raw: object | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        request = self._validate_legacy_request(raw)
        resources = self._resources()
        by_legacy_id = {self._legacy_resource_id(item): item for item in resources}
        by_id = {str(item.get("resource_id")): item for item in resources}
        requirements: list[dict[str, Any]] = []
        selected_gpu_ids: set[str] = set()
        kind = request["kind"]

        if request["isolate_pcie_root"]:
            raise UnsupportedLegacyOperation("Device Scheduler has no qualified PCIe-root isolation mapping for this legacy request")

        if request["ids"] or request["count"]:
            device_kind = "gpu" if kind.casefold() in _GPU_KIND_ALIASES else kind
            candidates = [
                item for item in resources
                if self._kind(item).casefold() == device_kind.casefold() and item.get("available", True) is not False
            ]
            isolation = [item for item in resources if self._kind(item).casefold() == "isolation"]
            candidate_tags = {str(item.get("resource_id")): self._legacy_tags(item, isolation) for item in candidates}
            candidates = [
                item for item in candidates
                if all(candidate_tags[str(item.get("resource_id"))].get(key) == value for key, value in request["tags"].items())
            ]
            if request["ids"]:
                selected: list[dict[str, Any]] = []
                for legacy_id in request["ids"]:
                    item = by_legacy_id.get(str(legacy_id))
                    if item is None or self._kind(item).casefold() != device_kind.casefold() or item.get("available", True) is False:
                        raise ProjectControlAdapterError(f"requested resource is absent from current Device Scheduler inventory: {legacy_id}")
                    if item not in candidates:
                        raise ProjectControlAdapterError(f"requested resource does not match its kind/tags: {legacy_id}")
                    selected.append(item)
                target_sets = [[item] for item in selected]
            elif request["count"]:
                if len(candidates) < request["count"]:
                    raise _ResourceConflict("Device Scheduler inventory has fewer matching resources than requested")
                target_sets = [candidates] * request["count"]
            else:
                target_sets = []

            if request["ids"]:
                selected_gpu_ids.update(str(item.get("resource_id")) for item in selected
                                        if self._kind(item).casefold() in _GPU_KIND_ALIASES)

            for choices in target_sets:
                if device_kind.casefold() in _GPU_KIND_ALIASES:
                    usable = [(item, self._gpu_capacity(item)) for item in choices]
                    usable = [(item, capacity) for item, capacity in usable if capacity is not None]
                    if not usable:
                        raise UnsupportedLegacyOperation(f"resource kind {device_kind!r} has no supported positive capacity")
                    # Project Control's coordinator reserved whole GPU IDs.
                    # Exclusive mode fences the resource's conflict domains;
                    # a one-byte quantity satisfies the core's positive
                    # dimensional request and its idle-process telemetry check.
                    # It is not a memory reservation or enforcement claim.
                    quantity_key = "memory_bytes"
                    quantity = 1.0
                    selector = {"resource_id": {"in": [str(item["resource_id"]) for item, _ in usable]}}
                    req = {
                        "kind": device_kind,
                        "quantities": {quantity_key: quantity},
                        "selectors": selector,
                        "mode": "exclusive",
                        "same_host": True,
                    }
                    requirements.append(req)
                else:
                    # The old host coordinator supported exact whole-resource
                    # reservations for non-GPU kinds as well.
                    usable = [
                        item for item in choices
                        if isinstance(item.get("capacities"), Mapping)
                        and any(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
                                for value in item["capacities"].values())
                    ]
                    if not usable:
                        raise UnsupportedLegacyOperation(f"resource kind {device_kind!r} has no supported positive capacity")
                    common_keys = set.intersection(*[
                        {str(key) for key, value in item["capacities"].items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0}
                        for item in usable
                    ])
                    quantities = {
                        key: min(float(item["capacities"][key]) for item in usable)
                        for key in common_keys
                    }
                    if not quantities:
                        raise UnsupportedLegacyOperation(f"resource kind {device_kind!r} has no common positive capacity")
                    requirements.append({
                        "kind": device_kind,
                        "quantities": quantities,
                        "selectors": {"resource_id": {"in": [str(item["resource_id"]) for item in usable]}},
                        "mode": "exclusive",
                        "same_host": True,
                    })

        if request["cpu_heavy"] or request["cpu_threads"]:
            cpu_resources = [item for item in resources if self._kind(item).casefold() == "cpu"]
            if not cpu_resources:
                raise UnsupportedLegacyOperation("CPU reservation requested but Device Scheduler has no observed CPU resource")
            capacities = [
                item.get("capacities", {}).get("cores")
                for item in cpu_resources
                if isinstance(item.get("capacities"), Mapping)
            ]
            capacities = [value for value in capacities if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0]
            if not capacities:
                raise UnsupportedLegacyOperation("CPU reservation requested but no positive CPU capacity is observed")
            capacity = min(float(value) for value in capacities)
            cores = float(request["cpu_threads"])
            if request["cpu_heavy"] and cores <= 0:
                cores = float(max(1, int(capacity * 0.75)))
            requirements.append({
                "kind": "cpu", "quantities": {"cores": cores},
                "selectors": {"capacity.cores": {"eq": capacity}}, "mode": "shared", "same_host": True,
            })
        if request["ram_bytes"]:
            memory_resources = [item for item in resources if self._kind(item).casefold() == "memory"]
            if not memory_resources:
                raise UnsupportedLegacyOperation("RAM reservation requested but Device Scheduler has no observed host-memory resource")
            capacities = [
                item.get("capacities", {}).get("bytes")
                for item in memory_resources
                if isinstance(item.get("capacities"), Mapping)
            ]
            capacities = [value for value in capacities if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0]
            if not capacities:
                raise UnsupportedLegacyOperation("RAM reservation requested but no positive host-memory capacity is observed")
            capacity = min(float(value) for value in capacities)
            requirements.append({
                "kind": "memory", "quantities": {"bytes": float(request["ram_bytes"])},
                "selectors": {"capacity.bytes": {"eq": capacity}}, "mode": "shared", "same_host": True,
            })

        requested_domain_ids: set[str] = set()
        isolation = [item for item in resources if self._kind(item).casefold() == "isolation"]
        for legacy_domain in request["exclusive_resources"]:
            if legacy_domain in by_id and self._kind(by_id[legacy_domain]).casefold() == "isolation":
                requested_domain_ids.add(legacy_domain)
                continue
            if legacy_domain.startswith("interference:nvlink:") and selected_gpu_ids:
                matching_domains = {
                    str(item["resource_id"])
                    for item in isolation
                    if self._isolation_members(item).intersection(selected_gpu_ids)
                    and isinstance(item.get("attributes"), Mapping)
                    and item["attributes"].get("closure") == "complete-observed-gpu-inventory"
                }
                if not matching_domains:
                    raise UnsupportedLegacyOperation("legacy NVLink exclusion has no complete Device Scheduler topology equivalent")
                requested_domain_ids.update(matching_domains)
                continue
            raise UnsupportedLegacyOperation(f"legacy isolation domain has no exact Device Scheduler mapping: {legacy_domain}")

        if request["isolate_nvlink_domain"]:
            if not selected_gpu_ids:
                raise UnsupportedLegacyOperation("NVLink isolation needs explicitly selected GPUs in the legacy request")
            matching = [
                item for item in isolation
                if self._isolation_members(item).intersection(selected_gpu_ids)
                and isinstance(item.get("attributes"), Mapping)
                and item["attributes"].get("closure") == "complete-observed-gpu-inventory"
            ]
            if not matching:
                raise UnsupportedLegacyOperation("complete topology-derived isolation domains are unavailable")
            requested_domain_ids.update(str(item["resource_id"]) for item in matching)

        for resource_id in sorted(requested_domain_ids):
            domain = by_id.get(resource_id)
            if domain is None:
                raise ProjectControlAdapterError(f"isolation resource disappeared before lease request: {resource_id}")
            capacities = domain.get("capacities") if isinstance(domain.get("capacities"), Mapping) else {}
            quantity = float(capacities.get("slots", 0))
            if quantity <= 0:
                raise UnsupportedLegacyOperation("topology isolation resource has no positive slot capacity")
            requirements.append({
                "kind": "isolation", "quantities": {"slots": quantity},
                "selectors": {"resource_id": resource_id}, "mode": "exclusive", "same_host": True,
            })

        if not requirements:
            raise ProjectControlAdapterError("empty legacy owner requests cannot create a Device Scheduler lease")
        return requirements, resources

    @staticmethod
    def _owner_context(owner_id: str, *, owner_kind: str, project_root: str | Path,
                       job_id: str | None = None, attempt_id: str | None = None,
                       service_id: str | None = None, pid: int | None = None) -> dict[str, str]:
        root = str(Path(project_root).expanduser().resolve())
        process_pid = os.getpid() if pid is None else pid
        if isinstance(process_pid, bool) or not isinstance(process_pid, int) or process_pid <= 0:
            raise ProjectControlAdapterError("owner process pid must be a positive integer")
        context = {
            "issuer": "project-control",
            "owner_id": owner_id,
            "owner_kind": owner_kind,
            "project_root": root,
            "pid": str(process_pid),
        }
        for name, value in (("job_id", job_id), ("attempt_id", attempt_id), ("service_id", service_id)):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ProjectControlAdapterError(f"{name} must be non-empty text when supplied")
                context[name] = value
        token = _process_start(process_pid)
        if token is not None:
            context["process_start"] = token
        return context

    @staticmethod
    def _new_owner_id(owner_kind: str, stable_key: str | None = None) -> str:
        suffix = _digest(stable_key)[:32] if stable_key is not None else uuid.uuid4().hex
        return f"pc:{owner_kind}:{suffix}"

    @staticmethod
    def _legacy_owner_id_for_background(project_root: str | Path, job_id: str, attempt_id: str) -> str:
        root = str(Path(project_root).expanduser().resolve())
        return DeviceSchedulerProjectControlAdapter._new_owner_id(
            "background", _canonical_json({"project_root": root, "job_id": job_id, "attempt_id": attempt_id})
        )

    def _request(self, owner_id: str, context: Mapping[str, str], raw: object | None,
                 *, idempotency_key: str | None = None) -> dict[str, Any] | None:
        requirements, _ = self._resource_requirements(raw)
        payload = {
            "request_id": owner_id,
            "owner_id": owner_id,
            "owner_context": dict(context),
            "requirements": requirements,
            "lease_ttl_seconds": self._lease_ttl_seconds,
        }
        # Stable background owners must also have one stable idempotency key.
        # A retry with a changed PID or request body then conflicts at the
        # scheduler instead of creating a second lease for the same attempt.
        payload["idempotency_key"] = idempotency_key or f"pc-lease-{_digest(payload)[:40]}"
        try:
            result = self._call("leases.request", {"request": payload})
        except _ResourceConflict:
            return None
        lease = result.get("lease", result)
        if not isinstance(lease, Mapping):
            raise ProjectControlAdapterError("leases.request omitted its lease")
        if lease.get("owner_id") != owner_id:
            raise ProjectControlAdapterError("leases.request returned a different owner identity")
        if str(lease.get("state")) != "active":
            return None
        return dict(lease)

    def reserve_background(self, *, project_root: str | Path, job_id: str, attempt_id: str,
                           resource_request: object, pid: int | None = None) -> dict[str, Any] | None:
        if not isinstance(job_id, str) or not job_id or not isinstance(attempt_id, str) or not attempt_id:
            raise ProjectControlAdapterError("job_id and attempt_id are required for a background owner")
        owner_id = self._legacy_owner_id_for_background(project_root, job_id, attempt_id)
        context = self._owner_context(owner_id, owner_kind="background", project_root=project_root,
                                      job_id=job_id, attempt_id=attempt_id, pid=pid)
        owner_key = f"pc-owner-{_digest(owner_id)[:40]}"
        lease = self._request(owner_id, context, resource_request, idempotency_key=owner_key)
        return None if lease is None else self._reservation_view(lease)

    def reserve_service(self, *, project_root: str | Path, service_id: str,
                        resource_request: object, priority_class: str = "idle_model_residency",
                        pid: int | None = None) -> dict[str, Any] | None:
        if not isinstance(service_id, str) or not service_id:
            raise ProjectControlAdapterError("service_id is required for a service owner")
        if not isinstance(priority_class, str) or not priority_class:
            raise ProjectControlAdapterError("priority_class must be non-empty text")
        raise UnsupportedLegacyOperation(
            "managed service resource ownership must use a registered Device Scheduler backend; "
            "a lease cannot preserve service priority, process supervision, or residency cleanup"
        )

    def begin_foreground(self, *, project_root: str | Path, resource_request: object,
                         priority_class: str = "foreground_gpu", pid: int | None = None) -> dict[str, Any]:
        if not isinstance(priority_class, str) or not priority_class:
            raise ProjectControlAdapterError("priority_class must be non-empty text")
        # Admission is immediate and atomic; this value does not request
        # priority or preemption from Device Scheduler.
        owner_id = self._new_owner_id("foreground")
        context = self._owner_context(owner_id, owner_kind="foreground", project_root=project_root,
                                      pid=pid)
        lease = self._request(owner_id, context, resource_request)
        if lease is None:
            raise _ResourceConflict("foreground resources are unavailable; Device Scheduler does not preempt legacy lease owners")
        return self._reservation_view(lease)

    def _reservation_view(self, lease: Mapping[str, Any]) -> dict[str, Any]:
        return self._reservation_view_for_lease(lease)

    def _lease_records(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        arguments = {"owner_id": owner_id} if owner_id is not None else {}
        result = self._call("leases.list", arguments)
        values = result.get("leases")
        if not isinstance(values, list):
            raise ProjectControlAdapterError("leases.list omitted its leases array")
        leases = [_mapping(item, "lease") for item in values]
        if owner_id is not None:
            leases = [item for item in leases if item.get("owner_id") == owner_id]
        return leases

    def _legacy_resources_for_lease(self, lease: Mapping[str, Any]) -> list[str]:
        allocations = lease.get("allocations", ())
        if not isinstance(allocations, (list, tuple)):
            raise ProjectControlAdapterError("lease allocations must be an array")
        resources_by_id = {str(item.get("resource_id")): item for item in self._resources()}
        result = []
        for allocation in allocations:
            if not isinstance(allocation, Mapping):
                raise ProjectControlAdapterError("lease allocation must be an object")
            resource_id = str(allocation.get("resource_id", ""))
            resource = resources_by_id.get(resource_id)
            if resource is None:
                fallback = self._legacy_id_from_resource_id(resource_id)
                if fallback is not None:
                    result.append(fallback)
                continue
            if self._kind(resource).casefold() in _GPU_KIND_ALIASES:
                result.append(self._legacy_resource_id(resource))
            elif self._kind(resource).casefold() == "isolation":
                # The old coordinator exposed held NVLink exclusion rows in
                # owner.resources. Their native labels were not always
                # reconstructable after inventory churn, so preserve the
                # durable Device Scheduler identity verbatim.
                result.append(resource_id)
            elif self._kind(resource).casefold() not in {"cpu", "memory"}:
                result.append(resource_id)
        return sorted(set(result))

    def _reservation_view_for_lease(self, lease: Mapping[str, Any]) -> dict[str, Any]:
        allocations = lease.get("allocations", ())
        if not isinstance(allocations, (list, tuple)):
            raise ProjectControlAdapterError("lease allocations must be an array")
        context = lease.get("owner_context")
        owner_kind = context.get("owner_kind") if isinstance(context, Mapping) else None
        resources_by_id = {str(item.get("resource_id")): item for item in self._resources()}
        resource_ids: list[str] = []
        for allocation in allocations:
            if not isinstance(allocation, Mapping):
                raise ProjectControlAdapterError("lease allocation must be an object")
            resource_id = str(allocation.get("resource_id", ""))
            resource = resources_by_id.get(resource_id)
            if resource is not None:
                kind = self._kind(resource).casefold()
                if kind in _GPU_KIND_ALIASES:
                    resource_ids.append(self._legacy_resource_id(resource))
                elif owner_kind == "foreground" and kind not in {"cpu", "memory"}:
                    resource_ids.append(resource_id)
            elif resource is None:
                fallback = self._legacy_id_from_resource_id(resource_id)
                if fallback is not None:
                    resource_ids.append(fallback)
                elif owner_kind == "foreground":
                    resource_ids.append(resource_id)
        return {"owner_id": str(lease["owner_id"]), "resource_ids": sorted(set(resource_ids))}

    def _matching_owner_leases(self, owner_id: str) -> list[dict[str, Any]]:
        if not owner_id.startswith("pc:"):
            return []
        records = self._lease_records()
        matched = []
        for lease in records:
            context = lease.get("owner_context")
            if (
                lease.get("owner_id") == owner_id
                and isinstance(context, Mapping)
                and context.get("issuer") == "project-control"
                and context.get("owner_id") == owner_id
            ):
                matched.append(lease)
        return matched

    def _current_owner_lease(self, owner_id: str) -> dict[str, Any] | None:
        matches = self._matching_owner_leases(owner_id)
        if len(matches) > 1:
            active = [item for item in matches if str(item.get("state")) in _ACTIVE_LEASE_STATES]
            if len(active) > 1:
                raise ProjectControlAdapterError(f"multiple unreleased Device Scheduler leases match owner {owner_id}")
            if active:
                return active[0]
            return max(matches, key=lambda item: str(item.get("expires_at", "")))
        return matches[0] if matches else None

    def owner(self, owner_id: str) -> dict[str, Any] | None:
        lease = self._current_owner_lease(owner_id)
        if lease is None:
            return None
        context = lease.get("owner_context")
        if not isinstance(context, Mapping) or context.get("issuer") != "project-control":
            return None
        state = str(lease.get("state", "unknown"))
        resources = [] if state == "released" else self._legacy_resources_for_lease(lease)
        try:
            pid = int(context["pid"]) if context.get("pid") else None
        except (TypeError, ValueError):
            pid = None
        cpu_threads = 0.0
        ram_bytes = 0.0
        allocations = lease.get("allocations", ())
        resources_by_id = {str(item.get("resource_id")): item for item in self._resources()}
        if isinstance(allocations, (list, tuple)):
            for allocation in allocations:
                if not isinstance(allocation, Mapping):
                    continue
                resource = resources_by_id.get(str(allocation.get("resource_id", "")), {})
                kind = str(resource.get("kind", "")).casefold()
                quantities = allocation.get("quantities")
                if not isinstance(quantities, Mapping):
                    continue
                if kind == "cpu":
                    cpu_threads += float(quantities.get("cores", 0))
                elif kind == "memory":
                    ram_bytes += float(quantities.get("bytes", 0))
        return {
            "id": owner_id,
            "owner_id": owner_id,
            "owner_kind": context.get("owner_kind", "external"),
            "project_root": context.get("project_root"),
            "job_id": context.get("job_id"),
            "attempt_id": context.get("attempt_id"),
            "service_id": context.get("service_id"),
            "pid": pid,
            "process_start": context.get("process_start"),
            "state": "active" if state == "active" else state,
            "preempt_requested": None,
            "priority_class": None,
            "preemptible": False,
            "residency_protected": False,
            "residency_metadata_json": "{}",
            "residency_release_verified": False,
            "cpu_threads": int(cpu_threads),
            "ram_bytes": int(ram_bytes),
            # Lease DTOs deliberately do not claim the HostCoordinator's
            # acquisition/heartbeat timestamps. Expiry is the authoritative
            # renewal deadline and is exposed separately below.
            "acquired_at": None,
            "heartbeat_at": None,
            "resources": resources,
            "lease_id": lease.get("lease_id"),
            "fencing_token": lease.get("fencing_token"),
            "expires_at": lease.get("expires_at"),
        }

    def owners_for_attempt(self, project_root: str | Path, attempt_id: str) -> list[dict[str, Any]]:
        root = str(Path(project_root).expanduser().resolve())
        result = []
        for lease in self._lease_records():
            context = lease.get("owner_context")
            if (
                not str(lease.get("owner_id", "")).startswith("pc:")
                or not isinstance(context, Mapping)
                or context.get("issuer") != "project-control"
                or context.get("owner_id") != lease.get("owner_id")
            ):
                continue
            if context.get("project_root") == root and context.get("attempt_id") == attempt_id:
                owner = self.owner(str(lease.get("owner_id", "")))
                if owner is not None:
                    result.append(owner)
        return sorted(result, key=lambda item: (str(item.get("expires_at", "")), str(item["owner_id"])))

    def conflicts(self, resource_ids: Iterable[str]) -> list[str]:
        wanted = {str(item) for item in resource_ids}
        if not wanted:
            return []
        current_resources = self._resources()
        ds_to_legacy = {str(item.get("resource_id")): self._legacy_resource_id(item) for item in current_resources}
        legacy_to_ds = {legacy: ds_id for ds_id, legacy in ds_to_legacy.items()}
        translated = {legacy_to_ds[item] for item in wanted if item in legacy_to_ds}
        translated.update(item for item in wanted if item in ds_to_legacy)
        if len(translated) != len(wanted):
            missing = sorted(wanted - set(legacy_to_ds) - set(ds_to_legacy))
            raise ProjectControlAdapterError(f"cannot map legacy resource IDs to Device Scheduler inventory: {missing}")
        conflicts: set[str] = set()
        for lease in self._lease_records():
            if str(lease.get("state")) not in _ACTIVE_LEASE_STATES:
                continue
            allocations = lease.get("allocations", ())
            if not isinstance(allocations, (list, tuple)):
                raise ProjectControlAdapterError("lease allocations must be an array")
            if any(isinstance(item, Mapping) and str(item.get("resource_id")) in translated for item in allocations):
                conflicts.add(str(lease.get("owner_id", "")))
        return sorted(item for item in conflicts if item)

    def wait_for_quiescence(self, resource_ids: Iterable[str], *, timeout_seconds: float = 30.0,
                            poll_seconds: float | None = None) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        interval = self._poll_seconds if poll_seconds is None else max(0.01, float(poll_seconds))
        deadline = self._clock() + timeout
        while True:
            if not self.conflicts(resource_ids):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(min(interval, max(0.0, deadline - self._clock())))

    def heartbeat(self, owner_id: str, *, pid: int | None = None) -> None:
        lease = self._current_owner_lease(owner_id)
        if lease is None or str(lease.get("state")) != "active":
            raise ProjectControlAdapterError(f"no active Device Scheduler lease for owner {owner_id}")
        context = lease.get("owner_context")
        if not isinstance(context, Mapping) or context.get("issuer") != "project-control":
            raise ProjectControlAdapterError("lease is not in the Project Control owner namespace")
        if pid is not None and str(pid) != str(context.get("pid")):
            raise UnsupportedLegacyOperation("owner process identity is immutable; migrate this lifecycle to a managed backend")
        self._call("leases.renew", {
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "ttl_seconds": self._lease_ttl_seconds,
        })

    def release(self, owner_id: str, *, residency_quiescence: dict[str, Any] | None = None,
                residency_capability: str | None = None, generation: str | None = None) -> None:
        if residency_quiescence is not None or residency_capability is not None or generation is not None:
            raise UnsupportedLegacyOperation("legacy residency-release proofs are not accepted; use a managed backend/handoff")
        lease = self._current_owner_lease(owner_id)
        if lease is None:
            return
        state = str(lease.get("state", "unknown"))
        if state == "released":
            return
        result = self._call("leases.release", {
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
        })
        released = result.get("lease", result)
        if not isinstance(released, Mapping) or str(released.get("state")) != "released":
            pending_state = released.get("state", "unknown") if isinstance(released, Mapping) else "unknown"
            raise ReclamationPending(f"Device Scheduler has not verified release for {owner_id} (state={pending_state})")

    def activate_foreground(self, reservation: Mapping[str, Any]) -> bool:
        owner_id = reservation.get("owner_id") if isinstance(reservation, Mapping) else None
        if not isinstance(owner_id, str) or not owner_id:
            raise ProjectControlAdapterError("foreground reservation omitted owner_id")
        owner = self.owner(owner_id)
        if owner is None or owner.get("state") != "active":
            return False
        requested = reservation.get("resource_ids")
        return (
            isinstance(requested, (list, tuple))
            and sorted(str(item) for item in requested) == sorted(owner.get("resources", ()))
        )

    def compound_gpu_bundles(self, count: int) -> list[dict[str, Any]]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ProjectControlAdapterError("GPU bundle count must be positive")
        resources = self._resources()
        isolation = [item for item in resources if self._kind(item).casefold() == "isolation"]
        gpus = [item for item in resources if self._kind(item).casefold() in _GPU_KIND_ALIASES]
        groups: dict[str, list[dict[str, Any]]] = {}
        for gpu in gpus:
            resource_id = str(gpu.get("resource_id", ""))
            domains = [item for item in isolation if resource_id in self._isolation_members(item)]
            group = str(domains[0].get("resource_id")) if len(domains) == 1 else resource_id
            groups.setdefault(group, []).append(gpu)
        candidate_sets = [combo for group in groups.values() for combo in itertools.combinations(group, count)]
        if not candidate_sets:
            candidate_sets = list(itertools.combinations(gpus, count))
        bundles = []
        for combo in candidate_sets:
            legacy_ids = sorted(self._legacy_resource_id(item) for item in combo)
            ds_ids = {str(item.get("resource_id")) for item in combo}
            domains = sorted(
                str(item.get("resource_id")) for item in isolation
                if self._isolation_members(item).intersection(ds_ids)
            )
            if self.conflicts([*legacy_ids, *domains]):
                continue
            bundles.append({
                "resource_ids": legacy_ids,
                # Returned values are stable DS isolation resource IDs, so a
                # caller that passes the bundle back can request those exact
                # conflict domains without relying on legacy topology labels.
                "exclusive_resources": domains,
                "selection": {"topology_domains": domains},
            })
        return sorted(bundles, key=lambda item: (len(item["exclusive_resources"]), item["resource_ids"]))

    def owner_activity(self, owner_ids: Sequence[str]) -> tuple[int, int]:
        wanted = {str(item) for item in owner_ids}
        active = [
            lease for owner_id in wanted for lease in self._matching_owner_leases(owner_id)
            if str(lease.get("state")) in _ACTIVE_LEASE_STATES
        ]
        return len({str(item.get("owner_id")) for item in active}), sum(
            len(item.get("allocations", ())) for item in active
        )

    def release_observation_counts(self) -> tuple[int, int, int]:
        active = [item for item in self._lease_records() if str(item.get("state")) in _ACTIVE_LEASE_STATES]
        return len({str(item.get("owner_id")) for item in active}), sum(
            len(item.get("allocations", ())) for item in active
        ), 0

    def upsert(self, resources: list[dict[str, Any]]) -> None:
        raise UnsupportedLegacyOperation("resource discovery is Device Scheduler-owned; local inventory writes are disabled")

    def replace(self, kind: str, resources: list[dict[str, Any]]) -> None:
        raise UnsupportedLegacyOperation("resource discovery is Device Scheduler-owned; local inventory writes are disabled")

    def preempt_requested(self, owner_id: str) -> bool:
        raise UnsupportedLegacyOperation("lease owners have no legacy preemption signal; migrate managed execution to backend/job control")

    def request_preemption(self, owner_id: str) -> bool:
        raise UnsupportedLegacyOperation("lease preemption is not an implicit reclamation grant")

    def set_priority(self, owner_id: str, priority_class: str) -> bool:
        raise UnsupportedLegacyOperation("legacy lease priority mutation is unsupported; assign job priority through a server-side grant")

    def protect_residency(self, owner_id: str, *, memory_baseline: dict[str, float]) -> dict[str, Any]:
        raise UnsupportedLegacyOperation("residency protection belongs to a registered managed backend")

    def record_residency_process(self, owner_id: str, *, pid: int,
                                 residency_capability: str, generation: str) -> None:
        raise UnsupportedLegacyOperation("residency process tracking belongs to a registered managed backend")

    def reconcile_current_service_owners(self, *, project_root: str | Path,
                                         pid: int, live_owner_ids: Iterable[str]) -> list[str]:
        raise UnsupportedLegacyOperation("stale owner reconciliation is Device Scheduler-owned; no local owner rows are swept")

    def register_drain_callback(self, owner_id: str, callback: Any) -> None:
        raise UnsupportedLegacyOperation("process drain callbacks are local execution semantics; migrate to a managed backend")

    def drain_if_requested(self, owner_id: str) -> bool:
        raise UnsupportedLegacyOperation("lease owners have no legacy preemption signal; migrate to a managed backend")


class _ResourceConflict(ProjectControlAdapterError):
    """A safe no-grant outcome that maps to HostResourceFacade's None result."""

