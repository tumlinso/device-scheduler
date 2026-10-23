"""Atomic logical resource allocation over a durable ledger.

The allocator owns only the ledger decision. It never stops processes, unloads
models, or assumes that expiry proves reclamation. Every lease state except
``released`` continues to consume capacity until a caller records verified
release evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .errors import ConflictError, NotFoundError, ValidationError
from .models import (
    AllocationMode,
    Lease,
    LeaseState,
    Resource,
    ResourceAllocation,
    ResourceRequest,
    ResourceRequirement,
)
from .validation import owner_context_mapping


_REVISION = "_revision"
_GPU_KINDS = {"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"}
_GPU_MEMORY_NAMES = {"memory", "memory_bytes", "memory_mib", "memory_gib", "gpu_memory", "gpu_memory_bytes", "gpu_memory_mib", "vram", "vram_bytes", "vram_mib"}
_RESOURCE_FIELDS = frozenset({
    "resource_id", "host_id", "kind", "capacities", "attributes", "topology",
    "conflict_domains", "capabilities", "schema_version",
})


def _plain(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != _REVISION}


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _json_data(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("expected a model with to_dict() or a mapping")
    return dict(value)


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


def _quantity_key_is_gpu_memory(name: str) -> bool:
    key = name.casefold()
    return key in _GPU_MEMORY_NAMES or ("memory" in key and ("gpu" in key or "vram" in key))


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _positive_ttl(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("ttl_seconds must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValidationError("ttl_seconds must be a finite positive number")
    return result


def _observation_time(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _quantity_to_bytes(resource: Resource, key: str, amount: float) -> float | None:
    folded = key.casefold()
    if "byte" in folded:
        return float(amount)
    if "mib" in folded:
        return float(amount) * (1024.0**2)
    if "gib" in folded:
        return float(amount) * (1024.0**3)
    if folded in {"memory", "gpu_memory", "vram"}:
        # Unitless memory claims are ambiguous and cannot safely be compared to
        # the NVIDIA adapter's byte-valued free-memory observation.
        return None
    return None


class ResourceAllocator:
    """Deterministic bundle allocator whose writes serialize across SQLite clients."""

    def __init__(
        self,
        ledger: Any,
        *,
        gpu_headroom_fraction: float = 0.10,
        gpu_observation_max_age_seconds: float = 30.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(gpu_headroom_fraction, bool) or not 0 <= gpu_headroom_fraction < 1:
            raise ValueError("gpu_headroom_fraction must be in [0, 1)")
        if (
            isinstance(gpu_observation_max_age_seconds, bool)
            or not math.isfinite(float(gpu_observation_max_age_seconds))
            or gpu_observation_max_age_seconds <= 0
        ):
            raise ValueError("gpu_observation_max_age_seconds must be finite and positive")
        self.ledger = ledger
        self.gpu_headroom_fraction = float(gpu_headroom_fraction)
        self.gpu_observation_max_age_seconds = float(gpu_observation_max_age_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def register_resource(
        self,
        resource: Resource | Mapping[str, Any],
        *,
        source_adapter_id: str | None = None,
        observed_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        model = resource if isinstance(resource, Resource) else Resource.from_dict(resource)
        if source_adapter_id is not None and (not isinstance(source_adapter_id, str) or not source_adapter_id.strip()):
            raise ValidationError("source_adapter_id must be a non-empty string")
        stamp = _observation_time(observed_at) if observed_at is not None else self._now()
        if stamp is None:
            raise ValidationError("observed_at must be an ISO timestamp or aware datetime")
        with self.ledger.transaction() as tx:
            current = tx.get("resource", model.resource_id)
            health = self._resource_health(tx, model.resource_id, current)
            if source_adapter_id is None:
                source_adapter_id = health.get("source_adapter_id")
            elif health.get("source_adapter_id") not in (None, source_adapter_id):
                raise ConflictError(
                    "resource identity is already owned by another adapter",
                    details={"resource_id": model.resource_id, "source_adapter_id": health.get("source_adapter_id")},
                )
            stored = tx.put("resource", model.resource_id, model.to_dict(), current[_REVISION] if current else None)
            health_record = {
                "resource_id": model.resource_id,
                "source_adapter_id": source_adapter_id,
                "available": True,
                "availability_reason": None,
                "last_seen_at": stamp.isoformat(),
            }
            prior_health = tx.get("resource_health", model.resource_id)
            tx.put(
                "resource_health", model.resource_id, health_record,
                prior_health[_REVISION] if prior_health else None,
            )
            return stored

    def reconcile_resource_inventory(
        self,
        adapter_id: str,
        resources: Sequence[Resource | Mapping[str, Any]],
        *,
        error: str | None = None,
        observed_at: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically publish an adapter snapshot without dropping old claims.

        A failed or partial discovery marks the adapter's prior resources
        unavailable. A successful empty inventory marks all prior resources
        absent. Both retain their ledger entities and claims for reconciliation.
        """

        if not isinstance(adapter_id, str) or not adapter_id.strip():
            raise ValidationError("adapter_id must be a non-empty string")
        if error is not None and (not isinstance(error, str) or not error.strip()):
            raise ValidationError("error must be a non-empty string when supplied")
        models = tuple(item if isinstance(item, Resource) else Resource.from_dict(item) for item in resources)
        ids = [item.resource_id for item in models]
        if len(set(ids)) != len(ids):
            raise ValidationError("inventory snapshot contains duplicate resource IDs")
        stamp = _observation_time(observed_at) if observed_at is not None else self._now()
        if stamp is None:
            raise ValidationError("observed_at must be an ISO timestamp or aware datetime")
        seen = set(ids)
        result: list[dict[str, Any]] = []
        with self.ledger.transaction() as tx:
            for model in models:
                prior = tx.get("resource", model.resource_id)
                prior_health = self._resource_health(tx, model.resource_id, prior)
                # Do not let two adapters silently take ownership of one stable
                # resource identity.
                if prior is not None and prior_health.get("source_adapter_id") not in (None, adapter_id):
                    raise ConflictError(
                        "resource identity is already owned by another adapter",
                        details={"resource_id": model.resource_id, "source_adapter_id": prior_health.get("source_adapter_id")},
                    )
                expected = prior[_REVISION] if prior is not None else None
                result.append(tx.put("resource", model.resource_id, model.to_dict(), expected))
                health_record = {
                    "resource_id": model.resource_id,
                    "source_adapter_id": adapter_id,
                    "available": error is None,
                    "availability_reason": None if error is None else f"inventory_error: {error}",
                    "last_seen_at": stamp.isoformat(),
                }
                prior_health = tx.get("resource_health", model.resource_id)
                tx.put(
                    "resource_health", model.resource_id, health_record,
                    prior_health[_REVISION] if prior_health else None,
                )
            health_records = {item.get("resource_id"): item for item in tx.list("resource_health")}
            for prior in tx.list("resource"):
                health = health_records.get(prior.get("resource_id")) or self._resource_health(tx, str(prior.get("resource_id")), prior)
                if health.get("source_adapter_id") != adapter_id or prior.get("resource_id") in seen:
                    continue
                if error is None:
                    reason = "resource_absent_from_successful_inventory"
                else:
                    reason = f"inventory_error: {error}"
                if health.get("available") is False and health.get("availability_reason") == reason:
                    continue
                health_record = _plain(health)
                health_record["available"] = False
                health_record["availability_reason"] = reason
                health_record["last_inventory_attempt_at"] = stamp.isoformat()
                tx.put("resource_health", health["resource_id"], health_record, health[_REVISION])
            tx.event(
                "resource_inventory_reconciled",
                adapter_id,
                {"resource_ids": sorted(seen), "available": error is None, "error": error, "observed_at": stamp.isoformat()},
            )
        return result

    def list_resources(self, *, available_only: bool = True) -> list[Resource]:
        return [Resource.from_dict(item["resource"]) for item in self.list_resource_status() if not available_only or item["available"]]

    def list_resource_status(self) -> list[dict[str, Any]]:
        """Return resource DTOs alongside ledger-owned inventory health metadata."""
        with self.ledger.transaction() as tx:
            health = {item.get("resource_id"): item for item in tx.list("resource_health")}
            result: list[dict[str, Any]] = []
            for record in tx.list("resource"):
                resource_id = str(record.get("resource_id"))
                status = self._resource_health(tx, resource_id, record, health)
                result.append({
                    "resource": self._resource_model(record).to_dict(),
                    "available": status.get("available", True),
                    "source_adapter_id": status.get("source_adapter_id"),
                    "availability_reason": status.get("availability_reason"),
                    "last_seen_at": status.get("last_seen_at"),
                })
            return result

    def get_resource(self, resource_id: str) -> Resource:
        with self.ledger.transaction() as tx:
            record = tx.get("resource", resource_id)
            if record is None:
                raise NotFoundError(f"Unknown resource {resource_id}")
            health = self._resource_health(tx, resource_id, record)
            if health.get("available", True) is False:
                raise ConflictError(
                    "resource is currently unavailable",
                    details={"resource_id": resource_id, "availability_reason": health.get("availability_reason")},
                )
            return self._resource_model(record)

    def request_lease(self, request: ResourceRequest | Mapping[str, Any]) -> Lease:
        spec = request if isinstance(request, ResourceRequest) else ResourceRequest.from_dict(request)
        key = spec.idempotency_key or spec.request_id
        request_data = spec.to_dict()
        # Adding the default usage must not change existing external-request digests.
        if request_data.get("usage") == "external":
            request_data.pop("usage", None)

        def create(tx: Any) -> dict[str, Any]:
            lease = self._allocate_lease_tx(tx, spec, request_data=request_data)
            tx.event("lease_granted", lease.lease_id, {"owner_id": lease.owner_id, "fencing_token": lease.fencing_token})
            return lease.to_dict()

        stored = self.ledger.idempotent(spec.owner_id, key, "request_lease", request_data, create)
        return Lease.from_dict(stored)

    def get_lease(self, lease_id: str) -> Lease:
        record = self.ledger.get("lease", lease_id)
        if record is None:
            raise NotFoundError(f"Unknown lease {lease_id}")
        return self._lease_model(record)

    def list_leases(
        self,
        *,
        owner_id: str | None = None,
        state: LeaseState | str | None = None,
    ) -> list[Lease]:
        state_value = _enum_value(state) if state is not None else None
        return [
            self._lease_model(record)
            for record in self.ledger.list("lease")
            if (owner_id is None or record.get("owner_id") == owner_id)
            and (state_value is None or record.get("state") == state_value)
        ]

    def preview_allocation(
        self,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]],
        *,
        ignore_lease_ids: Iterable[str] = (),
    ) -> tuple[ResourceAllocation, ...]:
        """Return a feasible placement without changing ledger state.

        The result is an observation, not a reservation: another request may
        win capacity before the caller submits its lease request.
        """

        reqs = tuple(
            item if isinstance(item, ResourceRequirement) else ResourceRequirement.from_dict(item)
            for item in requirements
        )
        if not reqs:
            raise ValidationError("allocation preview requires at least one resource requirement")
        ignored = frozenset(ignore_lease_ids)
        with self.ledger.transaction() as tx:
            resources = self._resources(tx)
            claims = self._claims(tx, resources, ignored_lease_ids=ignored)
            selected = self._select_bundle(
                reqs, resources, claims, available_resource_ids=self._available_resource_ids(tx)
            )
            allocations, _ = self._aggregate(selected)
            return tuple(allocations)

    def can_allocate(
        self,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]],
        *,
        ignore_lease_ids: Iterable[str] = (),
    ) -> bool:
        try:
            self.preview_allocation(requirements, ignore_lease_ids=ignore_lease_ids)
        except ConflictError:
            return False
        return True

    def _allocate_lease_tx(
        self,
        tx: Any,
        request: ResourceRequest,
        *,
        request_data: Mapping[str, Any] | None = None,
        ignored_lease_ids: frozenset[str] = frozenset(),
    ) -> Lease:
        now = self._now()
        self._expire_due_tx(tx, now)
        resources = self._resources(tx)
        claims = self._claims(tx, resources, ignored_lease_ids=ignored_lease_ids)
        selected = self._select_bundle(
            request.requirements, resources, claims, available_resource_ids=self._available_resource_ids(tx)
        )
        allocations, exclusive_domains = self._aggregate(selected)
        lease_id = str(uuid.uuid4())
        lease = Lease(
            lease_id=lease_id,
            owner_id=request.owner_id,
            allocations=tuple(allocations),
            fencing_token=tx.next_fencing_token(),
            expires_at=now + timedelta(seconds=request.lease_ttl_seconds),
            state=LeaseState.ACTIVE,
            request_digest=self._digest(request_data or request.to_dict()),
            owner_context=request.owner_context,
            usage=request.usage,
        )
        record = lease.to_dict()
        record["exclusive_conflict_domains"] = sorted(exclusive_domains)
        record["request_id"] = request.request_id
        record["created_at"] = now.isoformat()
        record["usage_events"] = [{
            "created_at": now.isoformat(),
            "allocations": [item.to_dict() for item in allocations],
        }]
        tx.put("lease", lease_id, record)
        return lease

    def renew_lease(self, lease_id: str, fencing_token: int, ttl_seconds: float) -> Lease:
        ttl = _positive_ttl(ttl_seconds)
        expired = False
        with self.ledger.transaction() as tx:
            lease = self._lease_record(tx, lease_id)
            self._assert_fencing_token(lease, fencing_token)
            if lease["state"] != LeaseState.ACTIVE.value:
                raise ConflictError("only an active lease can be renewed", details={"state": lease["state"]})
            now = self._now()
            expires_at = datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00"))
            if expires_at <= now:
                lease["state"] = LeaseState.RECLAIMING.value
                lease["fencing_token"] = tx.next_fencing_token()
                lease["reconciliation_reason"] = "lease expired before renewal; resource reclamation is unverified"
                self._put_updated(tx, "lease", lease_id, lease)
                tx.event("lease_expired", lease_id, {
                    "state": LeaseState.RECLAIMING.value,
                    "fencing_token": lease["fencing_token"],
                })
                expired = True
            else:
                lease["fencing_token"] = tx.next_fencing_token()
                lease["expires_at"] = (now + timedelta(seconds=ttl)).isoformat()
                lease = self._put_updated(tx, "lease", lease_id, lease)
                tx.event("lease_renewed", lease_id, {"fencing_token": lease["fencing_token"], "expires_at": lease["expires_at"]})
        if expired:
            raise ConflictError("expired lease cannot be renewed before reconciliation")
        return self.get_lease(lease_id)

    def request_release(self, lease_id: str, fencing_token: int) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            lease = self._lease_record(tx, lease_id)
            self._assert_fencing_token(lease, fencing_token)
            state = LeaseState(lease["state"])
            if state == LeaseState.RELEASED:
                return lease
            if state == LeaseState.ACTIVE:
                lease["state"] = LeaseState.RELEASE_REQUESTED.value
                # Fence the original holder before reclamation starts.
                lease["fencing_token"] = tx.next_fencing_token()
                lease = self._put_updated(tx, "lease", lease_id, lease)
                tx.event("lease_release_requested", lease_id, {"fencing_token": lease["fencing_token"]})
            return lease

    def begin_reclamation(self, lease_id: str, fencing_token: int) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            lease = self._lease_record(tx, lease_id)
            self._assert_fencing_token(lease, fencing_token)
            state = LeaseState(lease["state"])
            if state == LeaseState.RELEASED:
                return lease
            if state not in (LeaseState.RELEASE_REQUESTED, LeaseState.RECLAIMING, LeaseState.UNKNOWN, LeaseState.EXPIRED):
                raise ConflictError("lease must be release-requested before reclamation", details={"state": state.value})
            if state != LeaseState.RECLAIMING:
                lease["state"] = LeaseState.RECLAIMING.value
                lease = self._put_updated(tx, "lease", lease_id, lease)
                tx.event("lease_reclaiming", lease_id, {})
            return lease

    def mark_unknown(self, lease_id: str, fencing_token: int, reason: str) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            lease = self._lease_record(tx, lease_id)
            self._assert_fencing_token(lease, fencing_token)
            if lease["state"] == LeaseState.RELEASED.value:
                raise ConflictError("a released lease cannot become unknown")
            lease["state"] = LeaseState.UNKNOWN.value
            lease["reconciliation_reason"] = reason
            lease = self._put_updated(tx, "lease", lease_id, lease)
            tx.event("lease_unknown", lease_id, {"reason": reason})
            return lease

    def verify_release(
        self, lease_id: str, fencing_token: int, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            lease = self._lease_record(tx, lease_id)
            self._assert_fencing_token(lease, fencing_token)
            if lease["state"] == LeaseState.RELEASED.value:
                if lease.get("release_receipt") != dict(receipt):
                    raise ConflictError("lease was released with different evidence")
                return lease
            if lease["state"] not in {
                LeaseState.RELEASE_REQUESTED.value,
                LeaseState.RECLAIMING.value,
                LeaseState.UNKNOWN.value,
            }:
                raise ConflictError("lease must enter reclamation before release can be verified", details={"state": lease["state"]})
            if receipt.get("verified") is not True:
                raise ValidationError("release receipt must contain verified=true")
            if receipt.get("lease_id") != lease_id:
                raise ValidationError("release receipt names a different lease")
            if receipt.get("fencing_token") != fencing_token:
                raise ValidationError("release receipt fencing_token does not match the current lease fence")
            expected_resources = {item["resource_id"] for item in lease["allocations"]}
            witnessed = receipt.get("resource_ids")
            if not isinstance(witnessed, (list, tuple, set, frozenset)) or not all(
                isinstance(item, str) for item in witnessed
            ):
                raise ValidationError("release receipt must enumerate verified resource_ids")
            if set(witnessed) != expected_resources:
                raise ValidationError("release receipt resource_ids must exactly match the allocated resources")
            lease["state"] = LeaseState.RELEASED.value
            lease["release_receipt"] = dict(receipt)
            lease = self._put_updated(tx, "lease", lease_id, lease)
            tx.event("lease_released", lease_id, {"receipt": dict(receipt)})
            return lease

    def expire_due(self) -> list[str]:
        with self.ledger.transaction() as tx:
            return self._expire_due_tx(tx, self._now())

    def _expire_due_tx(self, tx: Any, now: datetime) -> list[str]:
        expired: list[str] = []
        for record in tx.list("lease"):
            lease_id = record.get("lease_id") or record.get("id")
            if record.get("state") != LeaseState.ACTIVE.value:
                continue
            expires_at = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
            if expires_at > now:
                continue
            record["state"] = LeaseState.RECLAIMING.value
            record["fencing_token"] = tx.next_fencing_token()
            record["reconciliation_reason"] = "lease expired; resource reclamation is unverified"
            self._put_updated(tx, "lease", lease_id, record)
            tx.event("lease_expired", lease_id, {
                "state": LeaseState.RECLAIMING.value,
                "fencing_token": record["fencing_token"],
            })
            expired.append(lease_id)
        for record in tx.list("reservation"):
            if record.get("state") != "active":
                continue
            expires_at = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
            if expires_at > now:
                continue
            reservation_id = record.get("reservation_id") or record.get("id")
            record["state"] = "reclaiming"
            record["fencing_token"] = tx.next_fencing_token()
            record["reconciliation_reason"] = "reservation expired; resource reclamation is unverified"
            self._put_updated(tx, "reservation", reservation_id, record)
            tx.event("reservation_expired", reservation_id, {
                "state": "reclaiming",
                "fencing_token": record["fencing_token"],
            })
            expired.append(reservation_id)
        return expired

    def reserve_envelope(
        self,
        reservation_id: str,
        owner_id: str,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]],
        ttl_seconds: float,
        *,
        idempotency_key: str,
        owner_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if not reservation_id or not owner_id or not idempotency_key:
            raise ValidationError("reservation_id, owner_id, and idempotency_key are required")
        reqs = tuple(item if isinstance(item, ResourceRequirement) else ResourceRequirement.from_dict(item) for item in requirements)
        if not reqs:
            raise ValidationError("reservation must contain at least one resource requirement")
        if owner_context is not None:
            owner_context = owner_context_mapping(owner_context)
        ttl_seconds = _positive_ttl(ttl_seconds)
        payload = {
            "reservation_id": reservation_id,
            "owner_id": owner_id,
            "requirements": [item.to_dict() for item in reqs],
            "ttl_seconds": ttl_seconds,
            "owner_context": dict(owner_context or {}),
        }

        def create(tx: Any) -> dict[str, Any]:
            self._expire_due_tx(tx, self._now())
            resources = self._resources(tx)
            claims = self._claims(tx, resources)
            selected = self._select_bundle(
                reqs, resources, claims, available_resource_ids=self._available_resource_ids(tx)
            )
            allocations, exclusive_domains = self._aggregate(selected)
            record = self._reservation_record(
                tx,
                reservation_id,
                owner_id,
                allocations,
                exclusive_domains,
                ttl_seconds,
                payload,
                owner_context=owner_context,
            )
            record = tx.put("reservation", reservation_id, record)
            tx.event("reservation_created", reservation_id, {"owner_id": owner_id})
            return record

        return self.ledger.idempotent(owner_id, idempotency_key, "reserve_envelope", payload, create)

    def transfer_lease_to_envelope(
        self,
        lease_id: str,
        reservation_id: str,
        owner_id: str,
        ttl_seconds: float,
        *,
        idempotency_key: str,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]] = (),
        backend_lease_ids: Mapping[str, str] | None = None,
        owner_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self.transfer_leases_to_envelope(
            [lease_id], reservation_id, owner_id, ttl_seconds,
            idempotency_key=idempotency_key, requirements=requirements,
            backend_lease_ids=backend_lease_ids, owner_context=owner_context,
        )

    def transfer_leases_to_envelope(
        self,
        lease_ids: Sequence[str],
        reservation_id: str,
        owner_id: str,
        ttl_seconds: float,
        *,
        idempotency_key: str,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]] = (),
        backend_lease_ids: Mapping[str, str] | None = None,
        owner_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        source_ids = tuple(lease_ids)
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValidationError("lease_ids must be non-empty and unique")
        if not reservation_id or not owner_id or not idempotency_key:
            raise ValidationError("reservation_id, owner_id, and idempotency_key are required")
        ttl_seconds = _positive_ttl(ttl_seconds)
        reqs = tuple(item if isinstance(item, ResourceRequirement) else ResourceRequirement.from_dict(item) for item in requirements)
        if owner_context is not None:
            owner_context = owner_context_mapping(owner_context)
        payload = {
            "lease_ids": list(source_ids),
            "reservation_id": reservation_id,
            "owner_id": owner_id,
            "ttl_seconds": ttl_seconds,
            "requirements": [item.to_dict() for item in reqs],
            "backend_lease_ids": dict(sorted((backend_lease_ids or {}).items())),
            "owner_context": dict(owner_context or {}),
        }

        def transfer(tx: Any) -> dict[str, Any]:
            sources: list[dict[str, Any]] = []
            for lease_id in source_ids:
                source = self._lease_record(tx, lease_id)
                if source["state"] != LeaseState.ACTIVE.value:
                    raise ConflictError("only active residency leases can be transferred", details={"lease_id": lease_id, "state": source["state"]})
                sources.append(source)
            resources = self._resources(tx)
            claims = self._claims(tx, resources, ignored_lease_ids=frozenset(source_ids))
            extra: list[tuple[ResourceRequirement, Resource]] = []
            if reqs:
                extra = self._select_bundle(
                    reqs, resources, claims, available_resource_ids=self._available_resource_ids(tx)
                )
            baseline: dict[str, dict[str, Any]] = {}
            exclusive_domains: set[str] = set()
            for source in sources:
                exclusive_domains.update(source.get("exclusive_conflict_domains", []))
                for allocation in source["allocations"]:
                    item = baseline.setdefault(
                        allocation["resource_id"],
                        {"resource_id": allocation["resource_id"], "quantities": {}, "mode": "shared"},
                    )
                    for key, quantity in allocation["quantities"].items():
                        item["quantities"][key] = item["quantities"].get(key, 0.0) + quantity
                    if allocation["mode"] == AllocationMode.EXCLUSIVE.value:
                        item["mode"] = AllocationMode.EXCLUSIVE.value
            for requirement, resource in extra:
                item = baseline.setdefault(
                    resource.resource_id,
                    {"resource_id": resource.resource_id, "quantities": {}, "mode": "shared"},
                )
                for key, quantity in requirement.quantities.items():
                    item["quantities"][key] = max(item["quantities"].get(key, 0.0), quantity)
                if requirement.mode == AllocationMode.EXCLUSIVE:
                    item["mode"] = AllocationMode.EXCLUSIVE.value
                exclusive_domains.update(self._exclusive_domains(requirement, resource))

            allocations = [
                ResourceAllocation(
                    resource_id=item["resource_id"],
                    quantities=item["quantities"],
                    mode=AllocationMode(item["mode"]),
                )
                for _, item in sorted(baseline.items())
            ]
            # Validate only the incremental envelope against current claims. The
            # source residency allocation has already been admitted and remains
            # held continuously by the parent reservation.
            self._validate_replacement_envelope(
                allocations,
                sources,
                claims,
                resources,
                exclusive_domains,
                self._available_resource_ids(tx),
            )
            record = self._reservation_record(
                tx,
                reservation_id,
                owner_id,
                allocations,
                exclusive_domains,
                ttl_seconds,
                payload,
                transferred_lease_ids=list(source_ids),
                owner_context=self._transfer_owner_context(sources, owner_context),
                usage_events=self._transferred_usage_events(sources, allocations),
            )
            record["transferred_from_owners"] = sorted({source["owner_id"] for source in sources})
            record["source_owner_contexts"] = {
                source["lease_id"]: dict(source.get("owner_context", {})) for source in sources
            }
            backend_updates: list[tuple[str, dict[str, Any], str]] = []
            if backend_lease_ids is not None:
                for backend_id, lease_id in backend_lease_ids.items():
                    if lease_id not in source_ids:
                        raise ValidationError(f"backend {backend_id} references a lease outside this transfer")
                    backend = tx.get("backend", backend_id)
                    if backend is None:
                        raise NotFoundError(f"Unknown backend {backend_id}")
                    active_lease = backend.get("lease_id") or backend.get("residency_lease_id") or backend.get("active_lease_id")
                    if active_lease != lease_id:
                        raise ConflictError("backend lease identity changed before transfer", details={"backend_id": backend_id})
                    backend_updates.append((backend_id, _plain(backend), lease_id))
            else:
                for backend in tx.list("backend"):
                    lease_id = backend.get("lease_id") or backend.get("residency_lease_id") or backend.get("active_lease_id")
                    if lease_id in source_ids:
                        backend_id = backend.get("backend_id") or backend.get("id")
                        backend_updates.append((backend_id, _plain(backend), lease_id))
            mapped_backends = {backend_id for backend_id, _, _ in backend_updates}
            explicit_backends = set((backend_lease_ids or {}).keys())
            for backend_id, backend, lease_id in backend_updates:
                backend["pre_handoff_state"] = backend.get("state")
                for field in ("lease_id", "residency_lease_id", "active_lease_id"):
                    if backend.get(field) == lease_id:
                        backend[field] = None
                backend["transferred_lease_id"] = lease_id
                backend["handoff_id"] = reservation_id
                backend["state"] = "draining"
                self._put_updated(tx, "backend", backend_id, backend)
                tx.event("backend_reservation_transferred", backend_id, {"reservation_id": reservation_id, "lease_id": lease_id})
            if backend_lease_ids is not None and mapped_backends != explicit_backends:
                raise ConflictError("not all explicitly mapped backend leases were transferred")
            for source in sources:
                lease_id = source["lease_id"]
                source["state"] = LeaseState.RELEASED.value
                source["transferred_to"] = reservation_id
                source["transferred_at"] = self._now().isoformat()
                self._put_updated(tx, "lease", lease_id, source)
                tx.event("lease_transferred", lease_id, {"reservation_id": reservation_id})
            record = tx.put("reservation", reservation_id, record)
            tx.event("reservation_created", reservation_id, {"owner_id": owner_id, "transferred_lease_ids": list(source_ids)})
            return record

        return self.ledger.idempotent(owner_id, idempotency_key, "transfer_leases_to_envelope", payload, transfer)

    def phase_lease(
        self,
        reservation_id: str,
        phase_id: str,
        requirements: Sequence[ResourceRequirement | Mapping[str, Any]],
    ) -> dict[str, Any]:
        reqs = tuple(item if isinstance(item, ResourceRequirement) else ResourceRequirement.from_dict(item) for item in requirements)
        if not reqs:
            raise ValidationError("phase lease must contain resource requirements")
        with self.ledger.transaction() as tx:
            reservation = self._reservation_record_get(tx, reservation_id)
            if reservation["state"] != "active":
                raise ConflictError("reservation is not active", details={"state": reservation["state"]})
            existing_phase = tx.get("reservation_phase", phase_id)
            if existing_phase is not None:
                if existing_phase["reservation_id"] != reservation_id or existing_phase["requirements"] != [r.to_dict() for r in reqs]:
                    raise ConflictError("phase_id was already used for a different reservation or requirement")
                return _plain(existing_phase)
            active_phases = [
                phase for phase in tx.list("reservation_phase")
                if phase["reservation_id"] == reservation_id and phase["state"] == "active"
            ]
            if active_phases:
                raise ConflictError("reservation already has an active phase lease")
            resources = self._resources(tx)
            available_resource_ids = self._available_resource_ids(tx)
            parent_allocations = {item["resource_id"]: item for item in reservation["allocations"]}
            options: dict[int, list[Resource]] = {}
            for req in reqs:
                if req.kind.casefold() in _GPU_KINDS and req.mode == AllocationMode.SHARED and not any(
                    _quantity_key_is_gpu_memory(key) for key in req.quantities
                ):
                    raise ConflictError("shared GPU phase requires an explicit memory budget", details={"reason": "gpu_memory_budget_required"})
            for index, req in enumerate(reqs):
                candidates = [
                    resources[rid]
                    for rid in parent_allocations
                    if rid in resources
                    and rid in available_resource_ids
                    and self._resource_matches(req, resources[rid])
                ]
                feasible: list[Resource] = []
                for resource in candidates:
                    parent = parent_allocations[resource.resource_id]
                    if req.mode == AllocationMode.EXCLUSIVE and parent["mode"] != AllocationMode.EXCLUSIVE.value:
                        continue
                    if not self._quantities_within(req.quantities, parent["quantities"]):
                        continue
                    domains = _string_set(req.topology_constraints.get("exclusive_conflict_domains", []))
                    if not domains.issubset(set(reservation.get("exclusive_conflict_domains", []))):
                        continue
                    feasible.append(resource)
                options[index] = feasible
                if not feasible:
                    raise ConflictError("phase requirements exceed the protected reservation envelope", details={"reservation_id": reservation_id, "kind": req.kind})
            order = sorted(range(len(reqs)), key=lambda index: (len(options[index]), index))
            selected: list[tuple[ResourceRequirement, Resource]] = []
            allocations: list[ResourceAllocation] | None = None
            exclusive_domains: set[str] = set()

            def search(position: int) -> bool:
                nonlocal allocations, exclusive_domains
                if position == len(order):
                    candidate_allocations, candidate_domains = self._aggregate(selected)
                    for allocation in candidate_allocations:
                        parent = parent_allocations[allocation.resource_id]
                        if not self._quantities_within(allocation.quantities, parent["quantities"]):
                            return False
                        if allocation.mode == AllocationMode.EXCLUSIVE and parent["mode"] != AllocationMode.EXCLUSIVE.value:
                            return False
                    claims = self._claims(tx, resources)
                    for selected_index, (requirement, resource) in enumerate(selected):
                        if resource.kind.casefold() in _GPU_KINDS and not self._gpu_telemetry_allows(
                            requirement, resource, claims, selected[:selected_index], for_transfer=False
                        ):
                            return False
                    allocations = candidate_allocations
                    exclusive_domains = candidate_domains
                    return True
                index = order[position]
                requirement = reqs[index]
                for resource in options[index]:
                    if any(item.same_host for item in reqs):
                        host = next((chosen.host_id for _, chosen in selected), None)
                        if host is not None and resource.host_id != host:
                            continue
                    if not self._topology_matches(requirement, resource, selected):
                        continue
                    selected.append((requirement, resource))
                    if search(position + 1):
                        return True
                    selected.pop()
                return False

            if not search(0) or allocations is None:
                raise ConflictError(
                    "phase requirements exceed the protected reservation envelope or bundle constraints",
                    details={"reservation_id": reservation_id},
                )
            phase = {
                "phase_id": phase_id,
                "reservation_id": reservation_id,
                "requirements": [r.to_dict() for r in reqs],
                "allocations": [a.to_dict() for a in allocations],
                "exclusive_conflict_domains": sorted(exclusive_domains),
                "state": "active",
                "created_at": self._now().isoformat(),
            }
            phase = tx.put("reservation_phase", phase_id, phase)
            tx.event("reservation_phase_started", phase_id, {"reservation_id": reservation_id})
            return phase

    def release_phase(self, reservation_id: str, phase_id: str) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            phase = tx.get("reservation_phase", phase_id)
            if phase is None or phase["reservation_id"] != reservation_id:
                raise NotFoundError(f"Unknown reservation phase {phase_id}")
            if phase["state"] == "released":
                return phase
            phase["state"] = "released"
            phase["released_at"] = self._now().isoformat()
            phase = self._put_updated(tx, "reservation_phase", phase_id, phase)
            tx.event("reservation_phase_released", phase_id, {"reservation_id": reservation_id})
            return phase

    def release_envelope(self, reservation_id: str) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            record = self._reservation_record_get(tx, reservation_id)
            if record["state"] == "released":
                return record
            if record["state"] not in {"release_requested", "reclaiming", "unknown"}:
                record["state"] = "release_requested"
                record = self._put_updated(tx, "reservation", reservation_id, record)
                tx.event("reservation_release_requested", reservation_id, {})
            return record

    def renew_envelope(self, reservation_id: str, fencing_token: int, ttl_seconds: float) -> dict[str, Any]:
        ttl = _positive_ttl(ttl_seconds)
        with self.ledger.transaction() as tx:
            record = self._reservation_record_get(tx, reservation_id)
            if record["state"] not in {"active", "release_requested", "reclaiming", "unknown"}:
                raise ConflictError("reservation in this state cannot be renewed", details={"state": record["state"]})
            if int(record["fencing_token"]) != fencing_token:
                raise ConflictError("stale reservation fencing token", details={"actual_fencing_token": record["fencing_token"]})
            now = self._now()
            record["fencing_token"] = tx.next_fencing_token()
            record["expires_at"] = (now + timedelta(seconds=ttl)).isoformat()
            record = self._put_updated(tx, "reservation", reservation_id, record)
            tx.event("reservation_renewed", reservation_id, {"fencing_token": record["fencing_token"], "expires_at": record["expires_at"]})
            return record

    def verify_envelope_release(self, reservation_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            record = self._reservation_record_get(tx, reservation_id)
            if record["state"] == "released":
                if record.get("release_receipt") != dict(receipt):
                    raise ConflictError("reservation was released with different evidence")
                return record
            if record["state"] not in {"release_requested", "reclaiming", "unknown"}:
                raise ConflictError(
                    "reservation must enter reclamation before release can be verified",
                    details={"state": record["state"]},
                )
            if receipt.get("verified") is not True:
                raise ValidationError("reservation release receipt must contain verified=true")
            if receipt.get("reservation_id") != reservation_id:
                raise ValidationError("release receipt names a different reservation")
            expected_resources = {item["resource_id"] for item in record["allocations"]}
            witnessed = receipt.get("resource_ids")
            if not isinstance(witnessed, (list, tuple, set, frozenset)) or not all(
                isinstance(item, str) for item in witnessed
            ):
                raise ValidationError("release receipt must enumerate verified resource_ids")
            if not expected_resources.issubset(set(witnessed)):
                raise ValidationError("release receipt does not cover every reserved resource")
            if record.get("transferred_lease_ids"):
                raise ConflictError("transferred backend residency must be settled back into leases")
            phase_ids = [
                phase["phase_id"] for phase in tx.list("reservation_phase")
                if phase["reservation_id"] == reservation_id and phase["state"] == "active"
            ]
            if phase_ids:
                raise ConflictError("cannot release an envelope with active phase leases", details={"phase_ids": phase_ids})
            record["state"] = "released"
            record["release_receipt"] = dict(receipt)
            record = self._put_updated(tx, "reservation", reservation_id, record)
            tx.event("reservation_released", reservation_id, {"receipt": dict(receipt)})
            return record

    def settle_envelope_to_leases(
        self,
        reservation_id: str,
        *,
        ttl_seconds: float,
        receipt: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Atomically restore backend lease ownership and release excess envelope.

        Returned items preserve source lease order and include both
        ``source_lease_id`` and the fresh, fenced lease dictionary. Repeating an
        identical settlement returns the same lease identities.
        """

        ttl_seconds = _positive_ttl(ttl_seconds)
        if receipt.get("verified") is not True or receipt.get("backend_ready") is not True:
            raise ValidationError("settlement receipt must verify backend readiness")
        with self.ledger.transaction() as tx:
            reservation = self._reservation_record_get(tx, reservation_id)
            receipt_digest = self._digest({"ttl_seconds": ttl_seconds, "receipt": receipt})
            if reservation["state"] == "released" and reservation.get("restored_leases"):
                if reservation.get("settlement_digest") != receipt_digest:
                    raise ConflictError("reservation was settled with a different receipt or TTL")
                return [self._restored_lease_result(tx, item) for item in reservation["restored_leases"]]
            if reservation["state"] not in {"active", "release_requested", "reclaiming", "unknown"}:
                raise ConflictError("reservation cannot be settled in its current state", details={"state": reservation["state"]})
            if receipt.get("reservation_id") != reservation_id:
                raise ValidationError("settlement receipt names a different reservation")
            active_phases = [
                phase["phase_id"] for phase in tx.list("reservation_phase")
                if phase["reservation_id"] == reservation_id and phase["state"] == "active"
            ]
            if active_phases:
                raise ConflictError("cannot settle while a phase lease is active", details={"phase_ids": active_phases})
            source_ids = list(reservation.get("transferred_lease_ids", []))
            if not source_ids:
                raise ConflictError("reservation has no transferred backend leases to restore")
            now = self._now()
            restored: list[dict[str, Any]] = []
            for source_id in source_ids:
                source = self._lease_record(tx, source_id)
                if source.get("transferred_to") != reservation_id:
                    raise ConflictError("source lease transfer identity does not match reservation")
                new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"device-scheduler:{reservation_id}:{source_id}"))
                prior = tx.get("lease", new_id)
                if prior is None:
                    model = Lease(
                        lease_id=new_id,
                        owner_id=source["owner_id"],
                        allocations=tuple(ResourceAllocation.from_dict(item) for item in source["allocations"]),
                        fencing_token=tx.next_fencing_token(),
                expires_at=now + timedelta(seconds=ttl_seconds),
                        state=LeaseState.ACTIVE,
                        request_digest=source.get("request_digest"),
                        owner_context=source.get("owner_context", {}),
                    )
                    lease_record = model.to_dict()
                    lease_record["exclusive_conflict_domains"] = list(source.get("exclusive_conflict_domains", []))
                    lease_record["source_lease_id"] = source_id
                    lease_record["restored_from_reservation"] = reservation_id
                    lease_record["created_at"] = now.isoformat()
                    lease_record["usage_events"] = [{
                        "created_at": now.isoformat(),
                        "allocations": [item.to_dict() for item in model.allocations],
                    }]
                    lease_record = tx.put("lease", new_id, lease_record)
                else:
                    lease_record = _plain(prior)
                    if lease_record.get("restored_from_reservation") != reservation_id:
                        raise ConflictError("deterministic replacement lease ID is already occupied")
                restored.append({"source_lease_id": source_id, "lease_id": new_id})

                for backend in tx.list("backend"):
                    backend_id = backend.get("backend_id") or backend.get("id")
                    if backend.get("handoff_id") != reservation_id or backend.get("transferred_lease_id") != source_id:
                        continue
                    backend = _plain(backend)
                    backend["lease_id"] = new_id
                    if "residency_lease_id" in backend:
                        backend["residency_lease_id"] = new_id
                    if "active_lease_id" in backend:
                        backend["active_lease_id"] = new_id
                    backend["state"] = "ready"
                    backend["restored_at"] = now.isoformat()
                    backend.pop("handoff_id", None)
                    backend.pop("transferred_lease_id", None)
                    backend.pop("pre_handoff_state", None)
                    self._put_updated(tx, "backend", backend_id, backend)
                    tx.event("backend_residency_restored", backend_id, {"reservation_id": reservation_id, "lease_id": new_id})

            reservation["state"] = "released"
            reservation["settlement_digest"] = receipt_digest
            reservation["settlement_receipt"] = dict(receipt)
            reservation["restored_leases"] = restored
            reservation["settled_at"] = now.isoformat()
            self._put_updated(tx, "reservation", reservation_id, reservation)
            tx.event("reservation_settled", reservation_id, {"restored_leases": restored})
            return [self._restored_lease_result(tx, item) for item in restored]

    def _restored_lease_result(self, tx: Any, item: Mapping[str, str]) -> dict[str, Any]:
        record = self._lease_record(tx, item["lease_id"])
        lease = Lease.from_dict({
            key: record[key]
            for key in ("lease_id", "owner_id", "allocations", "fencing_token", "expires_at", "state", "request_digest", "schema_version", "owner_context")
            if key in record
        })
        return {"source_lease_id": item["source_lease_id"], "lease": lease.to_dict()}

    def mark_envelope_unknown(self, reservation_id: str, reason: str) -> dict[str, Any]:
        with self.ledger.transaction() as tx:
            record = self._reservation_record_get(tx, reservation_id)
            if record["state"] == "released":
                raise ConflictError("released reservation cannot become unknown")
            record["state"] = "unknown"
            record["reconciliation_reason"] = reason
            record = self._put_updated(tx, "reservation", reservation_id, record)
            tx.event("reservation_unknown", reservation_id, {"reason": reason})
            return record

    def _reservation_record(
        self,
        tx: Any,
        reservation_id: str,
        owner_id: str,
        allocations: Sequence[ResourceAllocation],
        exclusive_domains: Iterable[str],
        ttl_seconds: float,
        request_payload: Mapping[str, Any],
        *,
        transferred_lease_ids: list[str] | None = None,
        owner_context: Mapping[str, str] | None = None,
        usage_events: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ttl = _positive_ttl(ttl_seconds)
        now = self._now()
        if tx.get("reservation", reservation_id) is not None:
            raise ConflictError(f"Reservation {reservation_id} already exists")
        context = dict(owner_context_mapping(owner_context))
        return {
            "reservation_id": reservation_id,
            "owner_id": owner_id,
            "allocations": [item.to_dict() for item in allocations],
            "exclusive_conflict_domains": sorted(set(exclusive_domains)),
            "fencing_token": tx.next_fencing_token(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
            "state": "active",
            "request_digest": self._digest(request_payload),
            "transferred_lease_ids": transferred_lease_ids or [],
            "created_at": now.isoformat(),
            "owner_context": context,
            "usage_events": list(usage_events) if usage_events is not None else [{
                "created_at": now.isoformat(),
                "allocations": [item.to_dict() for item in allocations],
            }],
        }

    @staticmethod
    def _usage_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
        events = record.get("usage_events")
        if isinstance(events, list) and all(isinstance(item, Mapping) for item in events):
            return [dict(item) for item in events]
        allocations = record.get("allocations", [])
        return [{"created_at": record.get("created_at"), "allocations": allocations}]

    @staticmethod
    def _transfer_owner_context(
        sources: Sequence[Mapping[str, Any]], owner_context: Mapping[str, str] | None
    ) -> dict[str, str]:
        if owner_context is not None:
            return dict(owner_context_mapping(owner_context))
        contexts = [dict(source.get("owner_context", {})) for source in sources]
        if contexts and all(context == contexts[0] for context in contexts[1:]):
            return contexts[0]
        return dict(owner_context_mapping({}))

    def _transferred_usage_events(
        self,
        sources: Sequence[Mapping[str, Any]],
        envelope_allocations: Sequence[ResourceAllocation],
    ) -> list[dict[str, Any]]:
        events = [event for source in sources for event in self._usage_events(source)]
        source_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        envelope_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for source in sources:
            for allocation in source.get("allocations", []):
                for key, value in allocation.get("quantities", {}).items():
                    source_totals[allocation["resource_id"]][key] += float(value)
        for allocation in envelope_allocations:
            for key, value in allocation.quantities.items():
                envelope_totals[allocation.resource_id][key] += float(value)
        delta: list[dict[str, Any]] = []
        for resource_id, quantities in envelope_totals.items():
            additional = {
                key: max(0.0, value - source_totals[resource_id].get(key, 0.0))
                for key, value in quantities.items()
                if value - source_totals[resource_id].get(key, 0.0) > 1e-9
            }
            if additional:
                delta.append({"resource_id": resource_id, "quantities": additional, "mode": "shared"})
        if delta:
            events.append({"created_at": self._now().isoformat(), "allocations": delta})
        return events

    def _validate_replacement_envelope(
        self,
        allocations: Sequence[ResourceAllocation],
        source_leases: Sequence[Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
        resources: Mapping[str, Resource],
        exclusive_domains: set[str],
        available_resource_ids: set[str],
    ) -> None:
        old: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for lease in source_leases:
            for allocation in lease["allocations"]:
                for key, quantity in allocation["quantities"].items():
                    old[allocation["resource_id"]][key] += quantity
        for allocation in allocations:
            resource = resources[allocation.resource_id]
            if allocation.resource_id not in available_resource_ids:
                raise ConflictError(
                    "transferred envelope references an unavailable resource",
                    details={"resource_id": allocation.resource_id},
                )
            if resource.kind.casefold() in _GPU_KINDS:
                requirement = ResourceRequirement(
                    kind=resource.kind,
                    quantities=allocation.quantities,
                    mode=allocation.mode,
                )
                if not self._gpu_telemetry_allows(requirement, resource, claims, (), for_transfer=True):
                    raise ConflictError(
                        "GPU inventory or process ownership is not sufficiently observed for backend transfer",
                        details={"resource_id": resource.resource_id, "reason": "gpu_observation_or_process_ownership_unknown"},
                    )
            old_quantities = old[resource.resource_id]
            candidate_domains = set(exclusive_domains)
            if allocation.mode == AllocationMode.EXCLUSIVE:
                candidate_domains.update(resource.conflict_domains)
            for key, quantity in allocation.quantities.items():
                limit = resource.capacities.get(key, 0.0)
                if resource.kind.casefold() in _GPU_KINDS and _quantity_key_is_gpu_memory(key):
                    limit *= 1.0 - self.gpu_headroom_fraction
                old_quantity = old_quantities.get(key, 0.0)
                if old_quantity <= limit + 1e-9 and quantity > limit + 1e-9:
                    raise ConflictError("transferred envelope exceeds resource capacity", details={"resource_id": resource.resource_id, "quantity": key})
            for other in claims:
                if set(other["exclusive_domains"]).intersection(candidate_domains):
                    raise ConflictError("transferred envelope conflicts in a protected conflict domain")
                if set(other["resource_domains"]).intersection(candidate_domains):
                    raise ConflictError("transferred envelope requests an occupied conflict domain")
                if set(other["exclusive_domains"]).intersection(resource.conflict_domains):
                    raise ConflictError("transferred envelope conflicts with an exclusive conflict domain")
                if other["resource_id"] == resource.resource_id:
                    if allocation.mode == AllocationMode.EXCLUSIVE or other["mode"] == "exclusive":
                        raise ConflictError("transferred envelope conflicts with another lease", details={"resource_id": resource.resource_id})
                    for key, quantity in allocation.quantities.items():
                        used = other["quantities"].get(key, 0.0)
                        old_quantity = old_quantities.get(key, 0.0)
                        limit = resource.capacities.get(key, 0.0)
                        if resource.kind.casefold() in _GPU_KINDS and _quantity_key_is_gpu_memory(key):
                            limit *= 1.0 - self.gpu_headroom_fraction
                        if used + max(0.0, quantity - old_quantity) > limit + 1e-9:
                            raise ConflictError("transferred envelope exceeds available resource capacity", details={"resource_id": resource.resource_id, "quantity": key})

    def _resources(self, tx: Any) -> dict[str, Resource]:
        values: dict[str, Resource] = {}
        for item in tx.list("resource"):
            resource = self._resource_model(item)
            values[resource.resource_id] = resource
        return values

    @staticmethod
    def _resource_model(record: Mapping[str, Any]) -> Resource:
        data = {key: value for key, value in _plain(record).items() if key in _RESOURCE_FIELDS}
        return Resource.from_dict(data)

    @staticmethod
    def _resource_health(
        tx: Any,
        resource_id: str,
        resource_record: Mapping[str, Any] | None = None,
        health_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        health = (health_by_id or {}).get(resource_id)
        if health is None:
            health = tx.get("resource_health", resource_id)
        if health is not None:
            return _plain(health)
        # Read the earlier ledger projection during migration, but all new
        # writes keep operational health out of the strict Resource DTO.
        legacy = resource_record or {}
        return {
            "resource_id": resource_id,
            "available": legacy.get("available", True),
            "source_adapter_id": legacy.get("source_adapter_id"),
            "availability_reason": legacy.get("availability_reason"),
            "last_seen_at": legacy.get("last_seen_at"),
        }

    def _available_resource_ids(self, tx: Any) -> set[str]:
        health_by_id = {item.get("resource_id"): item for item in tx.list("resource_health")}
        available: set[str] = set()
        for item in tx.list("resource"):
            resource_id = str(item.get("resource_id") or item.get("id"))
            health = self._resource_health(tx, resource_id, item, health_by_id)
            if health.get("available", True) is True:
                available.add(resource_id)
        return available

    def _claims(
        self,
        tx: Any,
        resources: Mapping[str, Resource],
        *,
        ignored_lease_ids: frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for lease in tx.list("lease"):
            lease_id = lease.get("lease_id") or lease.get("id")
            if lease_id in ignored_lease_ids or lease["state"] == LeaseState.RELEASED.value:
                continue
            usage_events = self._usage_events(lease)
            for allocation in lease["allocations"]:
                resource = resources.get(allocation["resource_id"])
                claims.append(
                    {
                        "resource_id": allocation["resource_id"],
                        "quantities": dict(allocation["quantities"]),
                        "mode": allocation["mode"],
                        "exclusive_domains": set(lease.get("exclusive_conflict_domains", []))
                        | (set(resource.conflict_domains) if resource and allocation["mode"] == "exclusive" else set()),
                        "resource_domains": set(resource.conflict_domains) if resource else set(),
                        "claim_id": lease_id,
                        "created_at": lease.get("created_at"),
                        "usage_events": usage_events,
                    }
                )
        for reservation in tx.list("reservation"):
            if reservation.get("state") == "released":
                continue
            usage_events = self._usage_events(reservation)
            usage_events.extend(
                {
                    "created_at": phase.get("created_at"),
                    "allocations": phase.get("allocations", []),
                }
                for phase in tx.list("reservation_phase")
                if phase.get("reservation_id") == reservation.get("reservation_id")
                and phase.get("state") == "active"
            )
            for allocation in reservation["allocations"]:
                resource = resources.get(allocation["resource_id"])
                claims.append(
                    {
                        "resource_id": allocation["resource_id"],
                        "quantities": dict(allocation["quantities"]),
                        "mode": allocation["mode"],
                        "exclusive_domains": set(reservation.get("exclusive_conflict_domains", []))
                        | (set(resource.conflict_domains) if resource and allocation["mode"] == "exclusive" else set()),
                        "resource_domains": set(resource.conflict_domains) if resource else set(),
                        "claim_id": reservation["reservation_id"],
                        "created_at": reservation.get("created_at"),
                        "usage_events": usage_events,
                    }
                )
        return claims

    def _select_bundle(
        self,
        requirements: Sequence[ResourceRequirement],
        resources: Mapping[str, Resource],
        claims: Sequence[Mapping[str, Any]],
        *,
        available_resource_ids: set[str] | None = None,
    ) -> list[tuple[ResourceRequirement, Resource]]:
        candidates = {
            index: [
                resource
                for resource in sorted(resources.values(), key=lambda item: item.resource_id)
                if (available_resource_ids is None or resource.resource_id in available_resource_ids)
                and self._resource_matches(requirement, resource)
            ]
            for index, requirement in enumerate(requirements)
        }
        order = sorted(range(len(requirements)), key=lambda index: (len(candidates[index]), index))
        if any(not candidates[index] for index in order):
            missing = next(index for index in order if not candidates[index])
            req = requirements[missing]
            raise ConflictError("no resource matches requirement", details={"kind": req.kind, "selectors": dict(req.selectors)})
        selected: list[tuple[ResourceRequirement, Resource]] = []
        bundle_same_host = any(requirement.same_host for requirement in requirements)
        nodes = 0
        max_nodes = 100_000

        def search(position: int) -> bool:
            nonlocal nodes
            if position == len(order):
                return True
            index = order[position]
            requirement = requirements[index]
            for resource in candidates[index]:
                nodes += 1
                if nodes > max_nodes:
                    raise ConflictError("resource bundle search exceeded its bounded search budget")
                if bundle_same_host:
                    host_for_bundle = next((chosen.host_id for _, chosen in selected), None)
                    if host_for_bundle is not None and resource.host_id != host_for_bundle:
                        continue
                if not self._topology_matches(requirement, resource, selected):
                    continue
                if not self._fits(requirement, resource, claims, selected):
                    continue
                selected.append((requirement, resource))
                if search(position + 1):
                    return True
                selected.pop()
            return False

        if not search(0):
            raise ConflictError("no feasible atomic resource bundle is currently available")
        return selected

    def _resource_matches(self, requirement: ResourceRequirement, resource: Resource) -> bool:
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
        required_domains = _string_set(constraints.get("required_conflict_domains", []))
        if not required_domains.issubset(set(resource.conflict_domains)):
            return False
        avoid_domains = _string_set(constraints.get("avoid_conflict_domains", []))
        if avoid_domains.intersection(resource.conflict_domains):
            return False
        topology = constraints.get("topology", {})
        if isinstance(topology, Mapping):
            for key, wanted in topology.items():
                if not _selector_matches(_get_path(resource.topology, key), wanted):
                    return False
        return True

    def _topology_matches(
        self,
        requirement: ResourceRequirement,
        resource: Resource,
        selected: Sequence[tuple[ResourceRequirement, Resource]],
    ) -> bool:
        same_topology: set[str] = set()
        for item, _ in selected:
            keys = item.topology_constraints.get("same_topology", [])
            if isinstance(keys, str):
                keys = [keys]
            same_topology.update(str(key) for key in keys)
        keys = requirement.topology_constraints.get("same_topology", [])
        if isinstance(keys, str):
            keys = [keys]
        same_topology.update(str(key) for key in keys)
        for key in same_topology:
            prior_values = [_get_path(chosen.topology, key) for _, chosen in selected]
            current_value = _get_path(resource.topology, key)
            if prior_values and (
                current_value is None
                or any(value is None or current_value != value for value in prior_values)
            ):
                return False
        return True

    def _fits(
        self,
        requirement: ResourceRequirement,
        resource: Resource,
        claims: Sequence[Mapping[str, Any]],
        selected: Sequence[tuple[ResourceRequirement, Resource]],
    ) -> bool:
        if resource.kind.casefold() in _GPU_KINDS and not self._gpu_telemetry_allows(
            requirement, resource, claims, selected, for_transfer=False
        ):
            return False
        if resource.kind.casefold() in _GPU_KINDS and requirement.mode == AllocationMode.SHARED:
            if not any(_quantity_key_is_gpu_memory(key) for key in requirement.quantities):
                raise ValidationError("shared GPU admission requires an explicit GPU memory budget")
        used: dict[str, float] = defaultdict(float)
        existing_exclusive = False
        held_domains: set[str] = set()
        occupied_domains: set[str] = set()
        selected_same_resource = False
        for claim in claims:
            held_domains.update(claim["exclusive_domains"])
            occupied_domains.update(claim["resource_domains"])
            if claim["resource_id"] == resource.resource_id:
                existing_exclusive |= claim["mode"] == AllocationMode.EXCLUSIVE.value
                for key, quantity in claim["quantities"].items():
                    used[key] += float(quantity)
        for prior_req, prior_resource in selected:
            occupied_domains.update(prior_resource.conflict_domains)
            held_domains.update(self._exclusive_domains(prior_req, prior_resource))
            if prior_req.mode == AllocationMode.EXCLUSIVE:
                held_domains.update(prior_resource.conflict_domains)
            if prior_resource.resource_id == resource.resource_id:
                selected_same_resource = True
                existing_exclusive |= prior_req.mode == AllocationMode.EXCLUSIVE
                for key, quantity in prior_req.quantities.items():
                    used[key] += float(quantity)
        candidate_domains = self._exclusive_domains(requirement, resource)
        if requirement.mode == AllocationMode.EXCLUSIVE:
            candidate_domains.update(resource.conflict_domains)
        if requirement.mode == AllocationMode.EXCLUSIVE and (
            selected_same_resource or any(claim["resource_id"] == resource.resource_id for claim in claims)
        ):
            return False
        if existing_exclusive and requirement.mode == AllocationMode.SHARED:
            return False
        if (
            candidate_domains.intersection(occupied_domains)
            or candidate_domains.intersection(held_domains)
            or resource.conflict_domains.intersection(held_domains)
        ):
            return False
        if requirement.mode == AllocationMode.EXCLUSIVE:
            for claim in claims:
                if claim["resource_id"] == resource.resource_id:
                    return False
        for key, quantity in requirement.quantities.items():
            capacity = resource.capacities.get(key)
            if capacity is None:
                return False
            effective_capacity = float(capacity)
            if resource.kind.casefold() in _GPU_KINDS and requirement.mode == AllocationMode.SHARED and _quantity_key_is_gpu_memory(key):
                effective_capacity *= 1.0 - self.gpu_headroom_fraction
            if used[key] + float(quantity) > effective_capacity + 1e-9:
                return False
        return True

    def _gpu_telemetry_allows(
        self,
        requirement: ResourceRequirement,
        resource: Resource,
        claims: Sequence[Mapping[str, Any]],
        selected: Sequence[tuple[ResourceRequirement, Resource]],
        *,
        for_transfer: bool,
    ) -> bool:
        """Fail closed unless NVIDIA memory and process observations are fresh.

        Shared allocation compares new claims against observed free memory and
        subtracts scheduler claims created after that observation. Whole-device
        exclusivity additionally requires a clean process table. A transfer may
        carry only source-backed backend residency whose visible PIDs the
        adapter and application have proven belong to scheduler cgroups.
        """

        attrs = resource.attributes
        inventory_at = _observation_time(attrs.get("inventory_observed_at"))
        process_at = _observation_time(attrs.get("process_observed_at"))
        if inventory_at is None or process_at is None:
            return False
        now = self._now()
        for stamp in (inventory_at, process_at):
            age = (now - stamp).total_seconds()
            if age < 0 or age > self.gpu_observation_max_age_seconds:
                return False
        process_state = attrs.get("process_telemetry_state")
        if process_state is None:
            process_state = "observed" if attrs.get("process_telemetry") == "compute-apps-visible" else "unknown"
        if process_state != "observed" or attrs.get("process_telemetry") != "compute-apps-visible":
            return False
        process_count = attrs.get("process_count_observation")
        if isinstance(process_count, bool) or not isinstance(process_count, int) or process_count < 0:
            return False
        free = attrs.get("memory_free_bytes_observation")
        if isinstance(free, bool) or not isinstance(free, (int, float)) or not math.isfinite(float(free)) or free < 0:
            return False
        unknown_count = attrs.get("process_identity_unknown_count")
        unmanaged_count = attrs.get("unmanaged_process_count")
        if unknown_count is None:
            unknown_count = process_count if process_count else 0
        if unmanaged_count is None:
            unmanaged_count = process_count if process_count else 0
        if (
            isinstance(unknown_count, bool)
            or not isinstance(unknown_count, int)
            or unknown_count < 0
            or isinstance(unmanaged_count, bool)
            or not isinstance(unmanaged_count, int)
            or unmanaged_count < 0
        ):
            return False
        if unknown_count > process_count or unmanaged_count > process_count:
            return False
        managed_ids = attrs.get("managed_process_ids", ())
        if isinstance(managed_ids, (str, bytes)) or not isinstance(managed_ids, (list, tuple, set, frozenset)):
            return False
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in managed_ids):
            return False
        if for_transfer:
            if unknown_count or unmanaged_count:
                return False
            if process_count > len(set(managed_ids)):
                return False
        elif requirement.mode == AllocationMode.EXCLUSIVE:
            # Even known managed processes must first release their lease and
            # disappear from the device before another exclusive grant.
            if process_count != 0 or unknown_count or unmanaged_count:
                return False

        memory_quantities: dict[str, float] = defaultdict(float)
        for key, amount in requirement.quantities.items():
            if _quantity_key_is_gpu_memory(key):
                converted = _quantity_to_bytes(resource, key, float(amount))
                if converted is None:
                    return False
                memory_quantities[key] += converted
        if not memory_quantities or for_transfer:
            return True

        # The free-memory observation already reflects work that existed when
        # sampled. Subtract only claims created later, including earlier members
        # of this atomic bundle, to avoid both double counting and stale-snapshot
        # over-admission.
        post_observation = 0.0
        for claim in claims:
            if claim.get("resource_id") != resource.resource_id:
                continue
            for event in claim.get("usage_events", ()):
                if not isinstance(event, Mapping):
                    continue
                event_time = _observation_time(event.get("created_at"))
                if event_time is not None and event_time <= min(inventory_at, process_at):
                    continue
                for allocation in event.get("allocations", ()):
                    if not isinstance(allocation, Mapping) or allocation.get("resource_id") != resource.resource_id:
                        continue
                    quantities = allocation.get("quantities", {})
                    if not isinstance(quantities, Mapping):
                        continue
                    for key, amount in quantities.items():
                        if not _quantity_key_is_gpu_memory(str(key)):
                            continue
                        converted = _quantity_to_bytes(resource, str(key), float(amount))
                        if converted is None:
                            return False
                        post_observation += converted
        for prior_requirement, prior_resource in selected:
            if prior_resource.resource_id != resource.resource_id:
                continue
            for key, amount in prior_requirement.quantities.items():
                if _quantity_key_is_gpu_memory(key):
                    converted = _quantity_to_bytes(resource, key, float(amount))
                    if converted is None:
                        return False
                    post_observation += converted
        new_memory = sum(memory_quantities.values())
        observed_budget = float(free) * (1.0 - self.gpu_headroom_fraction)
        return post_observation + new_memory <= observed_budget + 1e-6

    @staticmethod
    def _exclusive_domains(requirement: ResourceRequirement, resource: Resource) -> set[str]:
        domains = requirement.topology_constraints.get("exclusive_conflict_domains", [])
        if domains is True:
            return set(resource.conflict_domains)
        return _string_set(domains)

    def _aggregate(
        self, selected: Sequence[tuple[ResourceRequirement, Resource]]
    ) -> tuple[list[ResourceAllocation], set[str]]:
        totals: dict[str, dict[str, Any]] = {}
        exclusive_domains: set[str] = set()
        for requirement, resource in selected:
            current = totals.setdefault(
                resource.resource_id,
                {"quantities": defaultdict(float), "mode": AllocationMode.SHARED},
            )
            for key, quantity in requirement.quantities.items():
                current["quantities"][key] += float(quantity)
            if requirement.mode == AllocationMode.EXCLUSIVE:
                current["mode"] = AllocationMode.EXCLUSIVE
            exclusive_domains.update(self._exclusive_domains(requirement, resource))
        allocations = [
            ResourceAllocation(resource_id=resource_id, quantities=dict(data["quantities"]), mode=data["mode"])
            for resource_id, data in sorted(totals.items())
        ]
        return allocations, exclusive_domains

    def _lease_record(self, tx: Any, lease_id: str) -> dict[str, Any]:
        record = tx.get("lease", lease_id)
        if record is None:
            raise NotFoundError(f"Unknown lease {lease_id}")
        return _plain(record)

    @staticmethod
    def _lease_model(record: Mapping[str, Any]) -> Lease:
        fields = (
            "lease_id", "owner_id", "allocations", "fencing_token", "expires_at",
            "state", "request_digest", "schema_version", "owner_context", "usage",
        )
        return Lease.from_dict({key: record[key] for key in fields if key in record})

    @staticmethod
    def _assert_fencing_token(record: Mapping[str, Any], fencing_token: int) -> None:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise ValidationError("fencing_token must be an integer")
        if int(record["fencing_token"]) != fencing_token:
            raise ConflictError(
                "stale lease fencing token",
                details={"expected_fencing_token": record["fencing_token"], "received_fencing_token": fencing_token},
            )

    def _reservation_record_get(self, tx: Any, reservation_id: str) -> dict[str, Any]:
        record = tx.get("reservation", reservation_id)
        if record is None:
            raise NotFoundError(f"Unknown reservation {reservation_id}")
        return _plain(record)

    @staticmethod
    def _put_updated(tx: Any, kind: str, entity_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        expected = record.get(_REVISION)
        if expected is None:
            current = tx.get(kind, entity_id)
            expected = current[_REVISION] if current else None
        return tx.put(kind, entity_id, _plain(record), expected)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("allocator clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _digest(value: Any) -> str:
        def default(obj: Any) -> Any:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if hasattr(obj, "value"):
                return obj.value
            raise TypeError(f"Cannot digest {type(obj).__name__}")

        encoded = json.dumps(value, default=default, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _quantities_within(required: Mapping[str, float], available: Mapping[str, float]) -> bool:
        return all(key in available and float(amount) <= float(available[key]) + 1e-9 for key, amount in required.items())


__all__ = ["ResourceAllocator"]
