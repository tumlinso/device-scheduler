"""Versioned ports between environment-neutral services and adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    AdapterCapability,
    BackendSpec,
    CapabilityPrecondition,
    ExecutionContext,
    ExecutionHandle,
    HandoffSpec,
    JobSpec,
    Lease,
    Resource,
    ResourceRequest,
)

PORT_API_VERSION = "1"


@runtime_checkable
class Clock(Protocol):
    api_version: str

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


@runtime_checkable
class Ledger(Protocol):
    """Durable canonical state store; implementations define transaction scope."""

    api_version: str

    def capabilities(self) -> Sequence[AdapterCapability]: ...

    def transaction(self) -> AbstractContextManager[LedgerTransaction]: ...

    def get(self, kind: str, entity_id: str) -> dict[str, Any] | None: ...

    def list(self, kind: str) -> list[dict[str, Any]]: ...

    def put(
        self, kind: str, entity_id: str, data: Mapping[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]: ...

    def event(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> str: ...

    def idempotent(
        self,
        owner: str,
        key: str,
        operation: str,
        payload: Mapping[str, Any],
        callback: Callable[[Any], Any],
    ) -> Any: ...

    def events(
        self,
        *,
        kind: str | None = None,
        entity_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def operation(self, operation_id: str) -> dict[str, Any] | None: ...

    def operations(self, *, entity_id: str | None = None) -> list[dict[str, Any]]: ...

    def operation_intent(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        owner: str = "system",
        idempotency_key: str | None = None,
    ) -> str: ...

    def record_receipt(
        self, operation_id: str, receipt: Mapping[str, Any], *, status: str = "complete"
    ) -> str: ...

    def enqueue_outbox(
        self, topic: str, payload: Mapping[str, Any], *, outbox_id: str | None = None
    ) -> str: ...

    def outbox(self, *, consumer: str | None = None) -> list[dict[str, Any]]: ...

    def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]: ...

    def mark_outbox_delivered(self, outbox_id: str) -> None: ...

    def mark_outbox_failed(self, outbox_id: str, error: str) -> None: ...

    def next_fencing_token(self) -> int: ...


@runtime_checkable
class LedgerTransaction(Protocol):
    """Transaction-scoped view of the same ledger operations."""

    def get(self, kind: str, entity_id: str) -> dict[str, Any] | None: ...

    def list(self, kind: str) -> list[dict[str, Any]]: ...

    def put(
        self, kind: str, entity_id: str, data: Mapping[str, Any], expected_revision: int | None = None
    ) -> dict[str, Any]: ...

    def event(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> str: ...

    def events(
        self,
        *,
        kind: str | None = None,
        entity_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...

    def operation(self, operation_id: str) -> dict[str, Any] | None: ...

    def operation_intent(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        owner: str = "system",
        idempotency_key: str | None = None,
    ) -> str: ...

    def record_receipt(
        self, operation_id: str, receipt: Mapping[str, Any], *, status: str = "complete"
    ) -> str: ...

    def enqueue_outbox(
        self, topic: str, payload: Mapping[str, Any], *, outbox_id: str | None = None
    ) -> str: ...

    def idempotent(
        self,
        owner: str,
        key: str,
        operation: str,
        payload: Mapping[str, Any],
        callback: Callable[[LedgerTransaction], Any],
    ) -> Any: ...

    def next_fencing_token(self) -> int: ...


@runtime_checkable
class ResourceAdapter(Protocol):
    api_version: str
    adapter_id: str

    def capabilities(self) -> Sequence[AdapterCapability]: ...

    def discover(self) -> Sequence[Resource]: ...

    def inspect(self, resource_id: str) -> Resource | None: ...

    def check_execution_preconditions(
        self,
        execution_context: ExecutionContext,
        preconditions: Sequence[CapabilityPrecondition],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class LeaseReclamationAdapter(Protocol):
    """Optional port for leases whose consumers are outside supervised jobs.

    Implementations must return evidence scoped to the exact lease, current
    fencing token, operation identity, and full allocation. A TTL or owner
    acknowledgement alone is never reclamation evidence.
    """

    api_version: str
    adapter_id: str

    def reclaim_lease(self, lease: Lease, *, operation_id: str) -> Mapping[str, Any]: ...


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Supervise a job using only resources represented by its actual grant.

    ``JobSpec.resources`` expresses requirements; ``execution_context`` carries
    the admitted allocations and descriptors. Missing GPU grant data never
    authorizes inheriting the adapter's ambient device visibility. A
    supervised lease is consumable only through this port; adapters must bind
    the launched unit to the supplied grant and prove its cleanup before the
    application can release that lease.
    """

    api_version: str
    adapter_id: str

    def capabilities(self) -> Sequence[AdapterCapability]: ...

    def start(
        self,
        spec: JobSpec,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | None = None,
    ) -> ExecutionHandle: ...

    def inspect(self, handle: ExecutionHandle) -> Mapping[str, Any]: ...

    def cancel(
        self, handle: ExecutionHandle, *, operation_id: str, timeout_seconds: float
    ) -> Mapping[str, Any]: ...

    def collect(self, handle: ExecutionHandle) -> Mapping[str, Any]: ...


@runtime_checkable
class BackendAdapter(Protocol):
    """Control a managed backend under the scheduler's recorded allocations.

    A GPU-backed restoration must receive its protected reservation's actual
    allocation context; a backend profile alone is not an allocation grant.
    Initial-start preparation is a separate admission gate; it does not reopen
    an existing drain epoch or stand in for the handoff lifecycle.
    """

    api_version: str
    adapter_id: str

    def capabilities(self) -> Sequence[AdapterCapability]: ...

    def inspect(self, backend: BackendSpec) -> Mapping[str, Any]: ...

    def prepare_initial_start(
        self, backend: BackendSpec, *, operation_id: str
    ) -> Mapping[str, Any]: ...

    def drain(
        self, backend: BackendSpec, *, operation_id: str, timeout_seconds: float
    ) -> Mapping[str, Any]: ...

    def capture_reclamation_baseline(
        self, backend: BackendSpec, *, operation_id: str
    ) -> Mapping[str, Any]: ...

    def evict(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        reclamation_baseline: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def restore(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        execution_context: ExecutionContext | None = None,
    ) -> Mapping[str, Any]: ...

    def ready(
        self,
        backend: BackendSpec,
        *,
        timeout_seconds: float,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    def reopen(self, backend: BackendSpec, *, operation_id: str) -> Mapping[str, Any]: ...


@runtime_checkable
class ContinuationAdapter(Protocol):
    api_version: str
    adapter_id: str

    def capabilities(self) -> Sequence[AdapterCapability]: ...

    def register_consumer(
        self,
        continuation_ref: str,
        *,
        backend_id: str,
        contract_id: str,
    ) -> Mapping[str, Any]: ...

    def revoke_consumer(
        self, continuation_ref: str, *, backend_id: str, contract_id: str
    ) -> Mapping[str, Any]: ...

    def suspend(self, continuation_ref: str, *, handoff_id: str, deadline: datetime) -> Mapping[str, Any]: ...

    def inspect(self, continuation_ref: str, *, handoff_id: str) -> Mapping[str, Any]: ...

    def deliver(
        self, continuation_ref: str, *, delivery_id: str, result: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def resume(self, continuation_ref: str, *, delivery_id: str) -> Mapping[str, Any]: ...


@runtime_checkable
class PolicySelection(Protocol):
    """A policy's proposed queue choice; authority remains with core."""

    request_id: str
    owner_id: str
    priority_class: str
    score: float
    reason_code: str
    explanation: str


@runtime_checkable
class SchedulerPolicy(Protocol):
    """Current policy extension point for choosing among queued requests.

    Policies receive candidate records and read-only feasibility predicates.
    They propose one candidate; the application verifies the proposal and the
    allocator remains authoritative for admission. No adapter side effect or
    lifecycle transition is performed by this port.
    """

    api_version: str
    policy_id: str
    version: str

    def choose_next(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        can_fit: Callable[[Mapping[str, Any]], bool],
        safe_backfill: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
        now: datetime | None = None,
        service_by_owner: Mapping[str, float] | None = None,
    ) -> PolicySelection | None: ...


@runtime_checkable
class ExplainableSchedulerPolicy(SchedulerPolicy, Protocol):
    """Optional decision-explanation capability implemented by built-in policy."""

    def explain(
        self,
        request: Mapping[str, Any],
        *,
        now: datetime | None = None,
        service_by_owner: Mapping[str, float] | None = None,
    ) -> PolicySelection: ...


@runtime_checkable
class RankableSchedulerPolicy(SchedulerPolicy, Protocol):
    """Optional queue-ranking capability for inspection tooling."""

    def rank_requests(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        now: datetime | None = None,
        service_by_owner: Mapping[str, float] | None = None,
    ) -> Sequence[PolicySelection]: ...


@runtime_checkable
class PreemptionSelection(Protocol):
    """Advisory target identity; core must re-read and validate every target."""

    requester_job_id: str
    target_job_id: str
    target_attempt_id: str
    target_lease_id: str
    reason_code: str
    requester_score: float
    target_score: float
    policy_id: str
    policy_version: str
    action: str


@runtime_checkable
class PreemptionPolicy(SchedulerPolicy, Protocol):
    """Optional proposal capability; it cannot authorize interruption itself."""

    def propose_preemption(
        self,
        request: Mapping[str, Any],
        running: Sequence[Mapping[str, Any]],
        observations: Mapping[str, Any] | None = None,
        *,
        would_unblock: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> PreemptionSelection | None: ...

