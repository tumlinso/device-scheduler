"""Adversarial application/ledger qualification over a real SQLite database.

These tests exercise the public application boundary with deterministic fake
physical adapters. They do not claim systemd, GPU, or FrameShift qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core.allocator import ResourceAllocator
from device_scheduler_core.application import Scheduler, SchedulerConfig
from device_scheduler_core.errors import IdempotencyConflictError, PermissionDeniedError
from device_scheduler_core.models import (
    BackendSpec,
    ExecutionContext,
    EvictionGrant,
    ExecutionHandle,
    HandoffSpec,
    JobSpec,
    Principal,
    Resource,
    ResourceRequest,
    ResourceRequirement,
)
from device_scheduler_core.policy import PolicyDecision


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2030, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.value

    def monotonic(self) -> float:
        return 0.0

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class RecordingExecutor:
    api_version = "1"
    adapter_id = "test.executor"

    def __init__(self, state: str = "running") -> None:
        self.state = state
        self.started: list[tuple[str, ...]] = []
        self.handles: dict[str, ExecutionHandle] = {}
        self.contexts: dict[str, ExecutionContext] = {}

    def capabilities(self) -> list[dict[str, Any]]:
        return []

    def start(
        self,
        spec: JobSpec,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | None = None,
    ) -> ExecutionHandle:
        self.started.append(tuple(spec.argv or ()))
        assert execution_context is not None
        self.contexts[attempt_id] = execution_context
        handle = ExecutionHandle(
            adapter_id=self.adapter_id,
            external_id=f"unit-{attempt_id}",
            identity=attempt_id,
            started_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            metadata={"execution_root": str(execution_root)},
        )
        self.handles[attempt_id] = handle
        return handle

    def resolve(self, *, attempt_id: str) -> ExecutionHandle | None:
        return self.handles.get(attempt_id)

    def inspect(self, handle: ExecutionHandle) -> Mapping[str, Any]:
        return {"state": self.state, "cleanup_verified": self.state != "unknown"}

    def collect(self, handle: ExecutionHandle) -> Mapping[str, Any]:
        return {"state": self.state, "cleanup_verified": self.state != "unknown", "exit_code": 0}

    def cancel(self, handle: ExecutionHandle, *, operation_id: str, timeout_seconds: float) -> Mapping[str, Any]:
        self.state = "cancelled"
        return {"state": "cancelled", "cleanup_verified": True, "operation_id": operation_id}


class SimulatedProcessCrash(BaseException):
    pass


class CrashAfterStartExecutor(RecordingExecutor):
    def start(
        self,
        spec: JobSpec,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | None = None,
    ) -> ExecutionHandle:
        super().start(
            spec,
            attempt_id=attempt_id,
            execution_root=execution_root,
            execution_context=execution_context,
        )
        raise SimulatedProcessCrash("simulated daemon exit after external start")


class FakeBackend:
    def __init__(self, clock: MutableClock) -> None:
        self.clock = clock
        self.closed = False
        self.evicted = False
        self.calls: list[str] = []
        self.restore_contexts: list[ExecutionContext] = []

    def inspect(self, backend: BackendSpec) -> Mapping[str, Any]:
        return {"active_generations": 0, "state": "stopped" if self.evicted else "ready"}

    def drain(self, backend: BackendSpec, *, operation_id: str, timeout_seconds: float) -> Mapping[str, Any]:
        self.calls.append("drain")
        self.closed = True
        return {"admission_closed": True, "operation_id": operation_id}

    def capture_reclamation_baseline(
        self, backend: BackendSpec, *, operation_id: str
    ) -> Mapping[str, Any]:
        sampled_at = self.clock.now().isoformat()
        return {
            "state": "observed",
            "backend_id": backend.backend_id,
            "operation_id": operation_id,
            "profile_id": backend.profile_id,
            "profile_version": backend.pinned_configuration["profile_version"],
            "observed_at": sampled_at,
            "source": "deterministic-test-fixture",
            "execution_identity": {
                "unit_name": "test-backend.service",
                "invocation_id": "test-invocation",
                "control_group": "/user.slice/test-backend.service",
            },
            "process_mapping_state": "observed",
            "gpus": [{
                "resource_id": "gpu:host0:GPU-TEST-0",
                "gpu_uuid": "GPU-TEST-0",
                "memory_free_bytes": 8 * 1024 * 1024 * 1024,
                "used_gpu_memory_bytes": 2 * 1024 * 1024 * 1024,
                "backend_process_ids": [4321],
                "process_ids": [4321],
                "inventory_observed_at": sampled_at,
                "process_observed_at": sampled_at,
            }],
            "evidence": [{"kind": "test", "id": "gpu-baseline"}],
            "limitations": ["deterministic test fixture; no physical GPU assertion"],
        }

    def evict(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        reclamation_baseline: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls.append("evict")
        assert reclamation_baseline["backend_id"] == backend.backend_id
        self.evicted = True
        return {
            "state": "evicted",
            "cleanup_verified": True,
            "resource_reclamation": {"state": "reclaimed", "evidence": "test-fake"},
            "operation_id": operation_id,
        }

    def restore(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        execution_context: ExecutionContext | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append("restore")
        assert execution_context is not None
        self.restore_contexts.append(execution_context)
        self.evicted = False
        return {"state": "restored", "operation_id": operation_id}

    def ready(self, backend: BackendSpec, *, timeout_seconds: float, operation_id: str | None = None) -> Mapping[str, Any]:
        self.calls.append("ready")
        return {"ready": not self.evicted, "state": "ready", "operation_id": operation_id}

    def reopen(self, backend: BackendSpec, *, operation_id: str) -> Mapping[str, Any]:
        self.calls.append("reopen")
        self.closed = False
        return {"admission_open": True, "operation_id": operation_id}


class FakeContinuations:
    def suspend(self, ref: str, *, handoff_id: str, deadline: datetime) -> Mapping[str, Any]:
        return {"safe_boundary": True, "durable": True}

    def deliver(self, ref: str, *, delivery_id: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"delivered": True, "durable": True}

    def resume(self, ref: str, *, delivery_id: str) -> Mapping[str, Any]:
        return {"resumed": True}


@dataclass
class ChoosingPolicy:
    """A visible custom policy to prove admission calls configured policy code."""

    policy_id: str = "test.reverse"
    version: str = "7"
    calls: int = 0

    def choose_next(self, requests: list[Mapping[str, Any]], **_: Any) -> PolicyDecision | None:
        self.calls += 1
        if not requests:
            return None
        selected = next(
            (item for item in requests if item.get("spec", {}).get("argv") == ["/bin/second"]),
            requests[0],
        )
        job_id = str(selected.get("id", selected.get("job_id")))
        return PolicyDecision(job_id, str(selected["owner"]), "normal", 1.0, "test_choice", "selected by test policy")

    def explain(self, request: Mapping[str, Any], **_: Any) -> PolicyDecision:
        job_id = str(request.get("request_id", request.get("job_id", "explain")))
        return PolicyDecision(job_id, str(request.get("owner_id", "test")), "normal", 1.0, "test_choice", "test")


def _build(
    tmp_path: Path,
    *,
    clock: MutableClock | None = None,
    executor: RecordingExecutor | None = None,
    backend: TestBackend | None = None,
    continuations: FakeContinuations | None = None,
    policy: Any | None = None,
) -> tuple[Scheduler, Ledger, ResourceAllocator, MutableClock, RecordingExecutor]:
    clock = clock or MutableClock()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    allocator = ResourceAllocator(ledger, clock=clock.now)
    executor = executor or RecordingExecutor()
    scheduler = Scheduler(
        ledger,
        allocator,
        executor=executor,
        backend_adapter=backend,
        continuation_adapter=continuations,
        clock=clock,
        config=SchedulerConfig(
            state_root=tmp_path / "state",
            execution_root=tmp_path / "exec",
            artifact_root=tmp_path / "artifacts",
            policy=policy,
        ),
    )
    return scheduler, ledger, allocator, clock, executor


def _actor(work: Path, *actions: str, backend_ids: tuple[str, ...] = ()) -> Principal:
    return Principal(
        principal_id="agent:one",
        actions=frozenset(actions),
        execution_roots=(str(work),),
        backend_ids=frozenset(backend_ids),
    )


def _job_args(work: Path, key: str, *, argv: tuple[str, ...] = ("/bin/true",)) -> dict[str, Any]:
    return {"idempotency_key": key, "spec": {"argv": list(argv), "working_directory": str(work)}}


def _register_backend(allocator: ResourceAllocator, ledger: Ledger, clock: MutableClock, *, backend: str = "model-backend") -> None:
    # Model one coherent, current hardware sample. Both allocator freshness
    # checks and lease expiry use the same injected clock as this observation.
    # The UUID is deliberately distinct from the ledger resource ID so tests
    # cannot pass by accidentally treating source aliases as resource identity.
    observed_at = clock.now().isoformat()
    gpu_uuid = "GPU-TEST-0"
    total_bytes = 16 * 1024 * 1024 * 1024
    allocator.reconcile_resource_inventory("nvidia-smi", [Resource(
        f"gpu:host0:{gpu_uuid}", "host0", "gpu", {"memory_bytes": total_bytes},
        attributes={
            "uuid": gpu_uuid,
            "inventory_observed_at": observed_at,
            "process_observed_at": observed_at,
            "memory_free_bytes_observation": total_bytes,
            "process_telemetry": "compute-apps-visible",
            "process_telemetry_state": "observed",
            "process_count_observation": 0,
            "process_identity_unknown_count": 0,
            "unmanaged_process_count": 0,
            "managed_process_ids": [],
            "process_mapping_state": "observed",
        },
    )], observed_at=clock.now())
    residency = ResourceRequirement("gpu", {"memory_bytes": 2 * 1024 * 1024 * 1024})
    lease = allocator.request_lease({
        "request_id": "backend-residency",
        "owner_id": "backend-owner",
        "requirements": [residency.to_dict()],
        "lease_ttl_seconds": 3600,
        "idempotency_key": "backend-residency",
    })
    spec = BackendSpec(
        backend_id=backend,
        host_id="host0",
        adapter_id="test.backend",
        profile_id="llama",
        configuration_ref="trusted-config",
        residency_requirements=(residency,),
        attached_consumers=("frame:one",),
        pinned_configuration={"profile_version": "test-v1"},
    )
    ledger.put("backend", backend, {
        "id": backend,
        "backend_id": backend,
        "owner": "operator",
        "spec": spec.to_dict(),
        "state": "ready",
        "lease_id": lease.lease_id,
        "execution_handle": {"external_id": "backend-unit"},
    })


def _handoff_args(work: Path, clock: MutableClock, *, job_resources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "idempotency_key": "handoff:one",
        "spec": {
            "handoff_id": "handoff-one",
            "owner_id": "agent:one",
            "jobs": [{
                "argv": ["/bin/true"],
                "working_directory": str(work),
                "resources": job_resources or [],
            }],
            "backend_ids": ["model-backend"],
            "continuation_refs": ["frame:one"],
            "eviction_grants": [{
                "grant_id": "grant-one",
                "backend_id": "model-backend",
                "granted_by": "agent:one",
                "scope": "backend:model-backend",
                "expires_at": (clock.now() + timedelta(hours=1)).isoformat(),
            }],
            "resource_envelope": [{"kind": "gpu", "quantities": {"memory_bytes": 8 * 1024 * 1024 * 1024}}],
            "drain_timeout_seconds": 30,
            "run_timeout_seconds": 120,
            "restore_timeout_seconds": 30,
        },
    }


def _handoff_scheduler(tmp_path: Path, work: Path) -> tuple[Scheduler, Ledger, ResourceAllocator, MutableClock, RecordingExecutor, FakeBackend]:
    clock = MutableClock()
    backend = FakeBackend(clock)
    continuations = FakeContinuations()
    scheduler, ledger, allocator, clock, executor = _build(
        tmp_path, clock=clock, backend=backend, continuations=continuations,
    )
    _register_backend(allocator, ledger, clock)
    return scheduler, ledger, allocator, clock, executor, backend


def test_duplicate_submission_is_stable_and_different_payload_is_rejected(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler, *_ = _build(tmp_path)
    actor = _actor(work, "jobs.submit")

    first = scheduler.dispatch("jobs.submit", _job_args(work, "same-key"), actor)
    duplicate = scheduler.dispatch("jobs.submit", _job_args(work, "same-key"), actor)
    assert duplicate["id"] == first["id"]
    with pytest.raises(IdempotencyConflictError):
        scheduler.dispatch("jobs.submit", _job_args(work, "same-key", argv=("/bin/false",)), actor)


def test_submission_and_reads_remain_bound_to_authenticated_owner(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler, *_ = _build(tmp_path)
    owner = _actor(work, "jobs.submit", "jobs.get")
    with pytest.raises(PermissionDeniedError, match="owner"):
        scheduler.dispatch("jobs.submit", {**_job_args(work, "forged"), "owner_id": "agent:other"}, owner)

    submitted = scheduler.dispatch("jobs.submit", _job_args(work, "owned"), owner)
    other = Principal(principal_id="agent:other", actions=frozenset({"jobs.get"}), execution_roots=(str(work),))
    with pytest.raises(PermissionDeniedError, match="another principal"):
        scheduler.dispatch("jobs.get", {"job_id": submitted["id"]}, other)


def test_restart_marks_missing_execution_unknown_without_relaunch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = CrashAfterStartExecutor()
    scheduler, *_ = _build(tmp_path, executor=executor)
    actor = _actor(work, "jobs.submit")
    submitted = scheduler.dispatch("jobs.submit", _job_args(work, "crash-key"), actor)

    with pytest.raises(SimulatedProcessCrash):
        scheduler.tick()
    assert len(executor.started) == 1
    # Model a unit that disappeared before restart reconciliation. The stable
    # attempt cannot be resolved, so startup must preserve uncertainty.
    executor.handles.clear()
    restarted, ledger, _allocator, _clock, _executor = _build(tmp_path, executor=executor)

    restarted.tick()
    assert restarted.get_job(submitted["id"])["state"] == "unknown"
    assert len(executor.started) == 1
    assert ledger.list("attempts")[0]["state"] == "unknown"


def test_unverified_cleanup_keeps_external_capacity_unavailable(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler, _ledger, allocator, _clock, _executor = _build(tmp_path)
    allocator.register_resource(Resource("cpu0", "host0", "cpu", {"cores": 2}))
    actor = _actor(work, "leases.request", "leases.release")
    # Model a restored legacy claim whose physical adapter is now unavailable.
    # New external CPU requests are refused before grant; existing claims still
    # need honest reconciliation and must never disappear on a release request.
    lease = allocator.request_lease(ResourceRequest(
        request_id="cpu-request", owner_id=actor.principal_id,
        requirements=(ResourceRequirement("cpu", {"cores": 2}),),
        lease_ttl_seconds=600, idempotency_key="cpu-lease",
    )).to_dict()

    released = scheduler.dispatch("leases.release", {
        "lease_id": lease["lease_id"],
        "fencing_token": lease["fencing_token"],
    }, actor)
    assert released["state"] == "unknown"
    assert allocator.can_allocate([{"kind": "cpu", "quantities": {"cores": 1}}]) is False


def test_public_handoff_submission_enforces_eviction_grant_and_execution_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    scheduler, ledger, allocator, clock, _executor, _backend = _handoff_scheduler(tmp_path, work)
    actor = _actor(work, "handoffs.submit", "backends.evict", backend_ids=("model-backend",))

    bad_cwd = _handoff_args(outside, clock)
    with pytest.raises(PermissionDeniedError, match="execution roots"):
        scheduler.dispatch("handoffs.submit", bad_cwd, actor)
    no_eviction = _actor(work, "handoffs.submit", backend_ids=("model-backend",))
    with pytest.raises(PermissionDeniedError, match="backends.evict"):
        scheduler.dispatch("handoffs.submit", _handoff_args(work, clock), no_eviction)
    assert ledger.list("handoff") == []
    assert allocator.list_leases()[0].state.value == "active"


def test_configured_policy_selects_admission_order(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    policy = ChoosingPolicy()
    executor = RecordingExecutor(state="running")
    scheduler, *_ = _build(tmp_path, executor=executor, policy=policy)
    actor = _actor(work, "jobs.submit")
    first = scheduler.dispatch("jobs.submit", _job_args(work, "first", argv=("/bin/first",)), actor)
    second = scheduler.dispatch("jobs.submit", _job_args(work, "second", argv=("/bin/second",)), actor)

    started = scheduler.tick()["started_jobs"]
    assert executor.started == [("/bin/second",), ("/bin/first",)]
    assert {item["id"] for item in started} == {first["id"], second["id"]}
    assert policy.calls >= 2


def test_handoff_internal_job_does_not_start_after_envelope_release_request(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler, _ledger, allocator, clock, executor, _backend = _handoff_scheduler(tmp_path, work)
    actor = _actor(work, "handoffs.submit", "backends.evict", backend_ids=("model-backend",))
    scheduler.dispatch("handoffs.submit", _handoff_args(work, clock), actor)

    # Advance the durable handoff until its finite work batch is persisted, then
    # interleave an allocator release request before the daemon admits that job.
    for _ in range(10):
        handoff = scheduler._handoffs().get("handoff-one")
        if handoff["phase"] == "submit_jobs":
            break
        scheduler.tick()
    assert scheduler._handoffs().get("handoff-one")["phase"] == "submit_jobs"
    scheduler._handoffs().tick()
    queued_job = scheduler.get_job(scheduler._handoffs().get("handoff-one")["job_ids"][0])
    assert queued_job["state"] == "queued"

    reservation_id = queued_job["reservation_id"]
    allocator.release_envelope(reservation_id)
    scheduler.tick()

    assert allocator.ledger.get("reservation", reservation_id)["state"] != "active"
    assert executor.started == [], "handoff work must not launch without an active protected reservation"


def test_handoff_job_resource_request_must_fit_protected_envelope(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler, _ledger, allocator, clock, executor, _backend = _handoff_scheduler(tmp_path, work)
    actor = _actor(work, "handoffs.submit", "backends.evict", backend_ids=("model-backend",))
    # The protected envelope is 8 GiB; this job asks for 12 GiB.
    scheduler.dispatch(
        "handoffs.submit",
        _handoff_args(work, clock, job_resources=[{"kind": "gpu", "quantities": {"memory_bytes": 12 * 1024 * 1024 * 1024}}]),
        actor,
    )
    for _ in range(14):
        scheduler.tick()
    handoff = scheduler._handoffs().get("handoff-one")
    reservation = allocator.ledger.get("reservation", handoff["reservation_id"])
    assert reservation["state"] == "active"
    assert executor.started == [], "a handoff batch item cannot exceed its protected resource envelope"
