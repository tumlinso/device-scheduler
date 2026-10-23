"""Application-boundary integration tests over the real SQLite ledger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Mapping

import pytest

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core.allocator import ResourceAllocator
from device_scheduler_core.application import Scheduler, SchedulerConfig
from device_scheduler_core.errors import (
    ConflictError,
    PermissionDeniedError,
    ReconciliationRequiredError,
    UnsupportedCapabilityError,
    ValidationError,
)
from device_scheduler_core.models import (
    AdapterCapability,
    EnforcementLevel,
    ExecutionContext,
    ExecutionHandle,
    Principal,
    Resource,
)


NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


class FakeClock:
    api_version = "1"

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0


class MutableClock(FakeClock):
    def __init__(self) -> None:
        self.current = NOW

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class FakeExecutor:
    api_version = "1"
    adapter_id = "test.executor"

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.handles: dict[str, ExecutionHandle] = {}
        self.state = "succeeded"
        self.cleanup_verified = True
        self.contexts: dict[str, ExecutionContext] = {}

    def capabilities(self) -> list[dict[str, Any]]:
        return []

    def start(
        self,
        spec: Any,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | None = None,
    ) -> ExecutionHandle:
        self.starts.append(attempt_id)
        assert execution_context is not None
        self.contexts[attempt_id] = execution_context
        handle = ExecutionHandle(
            adapter_id=self.adapter_id,
            external_id=f"unit-{attempt_id}",
            identity=attempt_id,
            started_at=NOW,
            metadata={"execution_root": str(execution_root)},
        )
        self.handles[attempt_id] = handle
        return handle

    def resolve(self, *, attempt_id: str) -> ExecutionHandle | None:
        return self.handles.get(attempt_id)

    def inspect(self, handle: ExecutionHandle) -> Mapping[str, Any]:
        return {"state": self.state, "cleanup_verified": self.cleanup_verified}

    def collect(self, handle: ExecutionHandle) -> Mapping[str, Any]:
        return {
            "state": self.state,
            "cleanup_verified": self.cleanup_verified,
            "exit_code": 0,
            "stdout": "finished\n",
            "stderr": "",
        }

    def cancel(self, handle: ExecutionHandle, *, operation_id: str, timeout_seconds: float) -> Mapping[str, Any]:
        self.state = "cancelled"
        return {"state": "cancelled", "cleanup_verified": True, "operation_id": operation_id}


class LeaseBoundFakeExecutor(FakeExecutor):
    adapter_id = "systemd-user-unit"

    def capabilities(self) -> list[AdapterCapability]:
        return [
            AdapterCapability(name, True, EnforcementLevel.ADVISORY, {})
            for name in (
                "process-tree-cgroup-supervision",
                "granted-resource-cgroup-limits",
                "supervised-lease-job-binding",
            )
        ]


class UnverifiedCancelFakeExecutor(LeaseBoundFakeExecutor):
    def cancel(self, handle: ExecutionHandle, *, operation_id: str, timeout_seconds: float) -> Mapping[str, Any]:
        return {"state": "unknown", "cleanup_verified": False, "operation_id": operation_id}


class HostLeaseResourceAdapter:
    api_version = "1"
    adapter_id = "linux-host"

    def __init__(self) -> None:
        capability = frozenset({"supervised-lease-job-only", "systemd-cgroup-request-dependent"})
        self.resources = [
            Resource("host:local:cpu", "host:local", "cpu", {"cores": 4}, capabilities=capability),
            Resource("host:local:memory", "host:local", "memory", {"bytes": 8 * 1024**3}, capabilities=capability),
        ]

    def snapshot(self) -> Mapping[str, Any]:
        return {"state": "observed", "resources": self.resources}

    def discover(self) -> list[Resource]:
        return list(self.resources)

    def inspect(self, resource_id: str) -> Resource | None:
        return next((item for item in self.resources if item.resource_id == resource_id), None)


class SimulatedProcessCrash(BaseException):
    """Escape normal error recovery after the external unit was created."""


class CrashAfterExternalStart(FakeExecutor):
    def start(
        self,
        spec: Any,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | None = None,
    ) -> ExecutionHandle:
        handle = super().start(
            spec,
            attempt_id=attempt_id,
            execution_root=execution_root,
            execution_context=execution_context,
        )
        raise SimulatedProcessCrash("daemon stopped after the systemd unit started")


class UnverifiedReclaimer:
    api_version = "1"
    adapter_id = "test.reclaimer"

    def reclaim_lease(self, lease: Mapping[str, Any], *, operation_id: str) -> Mapping[str, Any]:
        return {
            "verified": False,
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "resource_ids": [],
            "operation_id": operation_id,
        }


class VerifiedReclaimer:
    api_version = "1"
    adapter_id = "test.expiring-resource"

    def __init__(self, *, verified: bool = True) -> None:
        self.verified = verified
        self.calls: list[str] = []

    def reclaim_lease(self, lease: Mapping[str, Any], *, operation_id: str) -> Mapping[str, Any]:
        self.calls.append(operation_id)
        return {
            "verified": self.verified,
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "resource_ids": [item["resource_id"] for item in lease["allocations"]] if self.verified else [],
            "operation_id": operation_id,
            "evidence": {"operation_id": operation_id},
        }


class ObservedGpuAdapter:
    api_version = "1"
    adapter_id = "test.gpu-observer"

    def __init__(self, *, baseline_valid: bool = True, release_states: tuple[str, ...] = ("reclaimed",)) -> None:
        self.baseline_valid = baseline_valid
        self.release_states = list(release_states)
        self.baseline_calls: list[str] = []
        self.release_calls: list[str] = []

    def snapshot(self) -> Mapping[str, Any]:
        stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        resource = Resource(
            "gpu:host:GPU-test",
            "host:local",
            "gpu",
            {"gpu_memory_bytes": 8_000_000_000},
            attributes={
                "uuid": "GPU-test",
                "memory_free_bytes_observation": 7_000_000_000,
                "process_telemetry": "compute-apps-visible",
                "process_count_observation": 0,
            },
        )
        return {
            "adapter": self.adapter_id,
            "state": "observed",
            "observed_at": stamp,
            "gpu_inventory": {"state": "observed"},
            "process_telemetry": {"state": "observed", "processes": []},
            "resources": [resource],
        }

    def capture_lease_baseline(
        self,
        lease: Mapping[str, Any],
        resources: tuple[Resource, ...],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        self.baseline_calls.append(operation_id)
        return {
            "state": "observed",
            "verified": True,
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "resource_ids": [item.resource_id for item in resources],
            "operation_id": operation_id if self.baseline_valid else "wrong-operation",
            "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "baseline": {"adapter_id": self.adapter_id, "resource_ids": [item.resource_id for item in resources]},
        }

    def inspect_lease_release(
        self,
        lease: Mapping[str, Any],
        baseline: Mapping[str, Any],
        resources: tuple[Resource, ...],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        self.release_calls.append(operation_id)
        state = self.release_states.pop(0) if len(self.release_states) > 1 else self.release_states[0]
        verified = state == "reclaimed"
        return {
            "verified": verified,
            "state": state,
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "resource_ids": [item.resource_id for item in resources],
            "operation_id": operation_id,
            "evidence": {"baseline_resource_ids": baseline["resource_ids"]},
        }


class BlockingBaselineGpuAdapter(ObservedGpuAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.capture_entered = Event()
        self.continue_capture = Event()
        self.lease_seen: Mapping[str, Any] | None = None

    def capture_lease_baseline(
        self,
        lease: Mapping[str, Any],
        resources: tuple[Resource, ...],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        self.lease_seen = dict(lease)
        self.capture_entered.set()
        if not self.continue_capture.wait(5):
            raise TimeoutError("test did not release external GPU baseline capture")
        return super().capture_lease_baseline(lease, resources, operation_id=operation_id)


class PreconditionAdapter:
    api_version = "1"
    adapter_id = "test.preconditions"

    def __init__(self, *, state: str = "verified", identity_matches: bool = True) -> None:
        self.state = state
        self.identity_matches = identity_matches
        self.calls: list[str] = []

    def check_execution_preconditions(
        self,
        execution_context: ExecutionContext,
        preconditions: tuple[Any, ...],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        self.calls.append(operation_id)
        return {
            "state": self.state,
            "verified": self.state == "verified",
            "adapter_id": self.adapter_id,
            "operation_id": operation_id,
            "claim_id": execution_context.claim_id if self.identity_matches else "wrong-claim",
            "fencing_token": execution_context.fencing_token,
            "resource_ids": [item.resource_id for item in execution_context.allocations],
            "preconditions": [
                {
                    "name": condition.name,
                    "schema_version": condition.schema_version,
                    "state": self.state,
                    "resource_ids": [item.resource_id for item in execution_context.allocations],
                    "evidence": {"parameters": dict(condition.parameters)},
                }
                for condition in preconditions
            ],
        }


class FakeBackendAdapter:
    api_version = "1"
    adapter_id = "test.backend"

    def __init__(self, *, initial_state: str = "closed", initial_operation: str | None = "backend-start:backend:one") -> None:
        self.initial_state = initial_state
        self.initial_operation = initial_operation
        self.preparations: list[str] = []
        self.restores: list[str] = []
        self.reopens: list[str] = []

    def prepare_initial_start(self, backend: Any, *, operation_id: str) -> Mapping[str, Any]:
        self.preparations.append(operation_id)
        return {
            "state": self.initial_state,
            "backend_id": backend.backend_id,
            "operation_id": self.initial_operation or operation_id,
            "admission_closed": True,
            "admission_epoch": 7,
            "active_generations": 0,
            "active_generations_source": "frameshift-provider-leases-upper-bound",
        }

    def restore(self, backend: Any, *, operation_id: str, execution_context: ExecutionContext | None = None) -> Mapping[str, Any]:
        self.restores.append(operation_id)
        assert execution_context is not None
        return {"state": "ready", "operation_id": operation_id, "execution_context": execution_context.to_dict()}

    def ready(self, backend: Any, *, timeout_seconds: float, operation_id: str | None = None) -> Mapping[str, Any]:
        return {"ready": True, "state": "ready", "operation_id": operation_id}

    def reopen(self, backend: Any, *, operation_id: str) -> Mapping[str, Any]:
        self.reopens.append(operation_id)
        return {"opened": True, "admission_epoch": 7, "operation_id": operation_id}

    def inspect(self, backend: Any) -> Mapping[str, Any]:
        return {"state": "stopped"}


class BlockingInventory:
    api_version = "1"
    adapter_id = "test.blocking-inventory"

    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()

    def snapshot(self) -> Mapping[str, Any]:
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test did not release inventory snapshot")
        return {
            "state": "unknown",
            "gpu_inventory": {"state": "unknown", "error": "test inventory unavailable"},
            "resources": [],
        }


def _scheduler(
    tmp_path: Path,
    *,
    executor: Any | None = None,
    adapters: tuple[Any, ...] = (),
    clock: Any | None = None,
) -> Scheduler:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    return Scheduler(
        ledger,
        ResourceAllocator(ledger, clock=clock.now if clock is not None else None),
        executor=executor,
        resource_adapters=adapters,
        clock=clock or FakeClock(),
        config=SchedulerConfig(
            state_root=tmp_path / "state",
            execution_root=tmp_path / "executions",
            artifact_root=tmp_path / "artifacts",
        ),
    )


def _principal(*actions: str, roots: tuple[Path, ...] = (), max_priority: int = 50) -> Principal:
    return Principal(
        principal_id="agent:one",
        actions=frozenset(actions),
        execution_roots=tuple(str(item) for item in roots),
        max_priority=max_priority,
    )


def _job_args(cwd: Path, *, key: str = "job:key", **changes: Any) -> dict[str, Any]:
    spec = {
        "argv": ["/usr/bin/true"],
        "working_directory": str(cwd),
        **changes,
    }
    return {"spec": spec, "idempotency_key": key}


def _scheduler_with_gpu_process_snapshot(
    tmp_path: Path,
    *,
    backend_state: str,
    backend_handle: Any,
    process_rows: list[dict[str, Any]],
    managed_process_ids: list[int],
) -> tuple[Scheduler, FakeExecutor, Resource]:
    resource = Resource(
        "gpu:host:GPU-managed",
        "host:local",
        "gpu",
        {"gpu_memory_bytes": 16 * 1024**2},
        attributes={"uuid": "GPU-managed"},
    )

    class ProcessInventory:
        api_version = "1"
        adapter_id = "test.gpu-inventory"

        def snapshot(self) -> Mapping[str, Any]:
            return {
                "state": "observed",
                "observed_at": NOW.isoformat(),
                "gpu_inventory": {"state": "observed"},
                "process_telemetry": {"state": "observed", "processes": process_rows},
                "resources": [resource],
            }

    executor = FakeExecutor()
    observed_handles: list[ExecutionHandle] = []

    def resolve_managed_processes(handles: list[ExecutionHandle]) -> Mapping[str, Any]:
        observed_handles.extend(handles)
        return {"state": "observed", "process_ids": managed_process_ids}

    executor.managed_process_ids = resolve_managed_processes  # type: ignore[attr-defined]
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(ProcessInventory(),))
    scheduler.ledger.put("backend", "backend:llama", {
        "id": "backend:llama",
        "backend_id": "backend:llama",
        "owner": "agent:one",
        "state": backend_state,
        "execution_handle": backend_handle,
    })
    executor.observed_handles = observed_handles  # type: ignore[attr-defined]
    return scheduler, executor, resource


def _resource_from_listing(response: Mapping[str, Any], resource_id: str) -> Mapping[str, Any]:
    for item in response["resources"]:
        resource = item.get("resource", item)
        if resource.get("resource_id") == resource_id:
            return resource
    raise AssertionError(f"resource {resource_id!r} missing from resource listing")


def test_inventory_attributes_managed_backend_pids_and_keeps_unmanaged_pids(tmp_path: Path) -> None:
    handle = ExecutionHandle(
        adapter_id="systemd-user-unit",
        external_id="device-scheduler-backend-llama.service",
        identity="backend-invocation-1",
        metadata={"invocation_id": "invocation-1", "control_group": "/user.slice/backend-llama.service"},
    )
    scheduler, executor, resource = _scheduler_with_gpu_process_snapshot(
        tmp_path,
        backend_state="ready",
        backend_handle=handle.to_dict(),
        process_rows=[
            {"gpu_uuid": "GPU-managed", "pid": 4321, "identity_unknown": False},
            {"gpu_uuid": "GPU-managed", "pid": 9876, "identity_unknown": False},
        ],
        managed_process_ids=[4321],
    )

    response = scheduler.dispatch("resources.list", {}, _principal("resources.list"))
    observed = _resource_from_listing(response, resource.resource_id)
    attributes = observed["attributes"]

    assert executor.observed_handles == [handle]
    assert attributes["process_mapping_state"] == "observed"
    assert attributes["managed_process_ids"] == [4321]
    assert attributes["unmanaged_process_count"] == 1


@pytest.mark.parametrize(
    ("backend_state", "backend_handle"),
    [
        ("starting", None),
        ("unknown", {"adapter_id": "systemd-user-unit", "external_id": "backend.service", "identity": ""}),
    ],
    ids=("starting-without-handle", "unknown-with-invalid-handle"),
)
def test_active_backend_without_valid_handle_keeps_gpu_processes_unmanaged(
    tmp_path: Path, backend_state: str, backend_handle: Any
) -> None:
    scheduler, executor, resource = _scheduler_with_gpu_process_snapshot(
        tmp_path,
        backend_state=backend_state,
        backend_handle=backend_handle,
        process_rows=[{"gpu_uuid": "GPU-managed", "pid": 4321, "identity_unknown": False}],
        managed_process_ids=[4321],
    )

    response = scheduler.dispatch("resources.list", {}, _principal("resources.list"))
    attributes = _resource_from_listing(response, resource.resource_id)["attributes"]

    assert executor.observed_handles == []
    assert attributes["process_mapping_state"] == "unknown"
    assert attributes["managed_process_ids"] == []
    assert attributes["unmanaged_process_count"] == 1


def test_stopped_backend_without_handle_does_not_make_handle_inventory_incomplete(tmp_path: Path) -> None:
    scheduler, _executor, _resource = _scheduler_with_gpu_process_snapshot(
        tmp_path,
        backend_state="stopped",
        backend_handle=None,
        process_rows=[],
        managed_process_ids=[],
    )

    assert scheduler._active_execution_handles() == ([], True)


def test_dispatch_submit_tick_and_collect_normal_job(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor)
    principal = _principal("jobs.submit", "jobs.get", roots=(work,))

    submitted = scheduler.dispatch("jobs.submit", _job_args(work), principal)
    assert submitted["state"] == "queued"
    assert submitted["owner"] == principal.principal_id
    assert "_revision" not in submitted
    replayed = scheduler.dispatch("jobs.submit", _job_args(work), principal)
    assert replayed["id"] == submitted["id"]
    assert "_revision" not in replayed

    first = scheduler.tick()
    assert len(first["started_jobs"]) == 1
    assert first["started_jobs"][0]["state"] == "running"
    assert len(executor.starts) == 1
    empty_context = executor.contexts[first["started_jobs"][0]["attempt_id"]]
    assert empty_context.allocations == ()
    assert empty_context.resources == ()
    decision = scheduler.ledger.get("decisions", first["started_jobs"][0]["decision_id"])
    assert decision["policy_configuration"]["policy_id"] == "builtin.weighted-fair"
    assert decision["policy_configuration"]["source"] == {
        "kind": "builtin",
        "name": "builtin.weighted-fair",
    }
    assert decision["observations"]["requests"][0]["can_fit"] is True
    assert scheduler.status()["scheduler"]["policy_configuration"] == decision["policy_configuration"]

    second = scheduler.tick()
    assert second["reconciled_jobs"][0]["state"] == "succeeded"
    completed = scheduler.dispatch("jobs.get", {"job_id": submitted["id"]}, principal)
    assert completed["result"]["exit_code"] == 0
    assert completed["cleanup_verified"] is True


def test_executor_context_uses_only_the_actual_granted_resource_descriptors(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor)
    granted = Resource("cpu:host-a", "host:local", "cpu", {"cores": 2})
    decoy = Resource("cpu:host-b", "host:local", "cpu", {"cores": 8})
    scheduler.allocator.register_resource(granted)
    scheduler.allocator.register_resource(decoy)

    submitted = scheduler.dispatch(
        "jobs.submit",
        _job_args(
            work,
            key="job:context-bound",
            resources=[{
                "kind": "cpu",
                "quantities": {"cores": 1},
                "selectors": {"resource_id": "cpu:host-a"},
            }],
        ),
        _principal("jobs.submit", roots=(work,)),
    )
    started = scheduler.tick()["started_jobs"][0]
    context = executor.contexts[started["attempt_id"]]

    assert [item.resource_id for item in context.allocations] == ["cpu:host-a"]
    assert [item.resource_id for item in context.resources] == ["cpu:host-a"]
    assert context.resources[0] == granted
    assert context.allocations[0].quantities == {"cores": 1.0}
    assert context.claim_id == started["lease_id"]
    assert context.fencing_token == started["lease_fencing_token"]
    stored_context = scheduler.ledger.get("attempts", started["attempt_id"])["execution_context"]
    assert stored_context == context.to_dict()


def test_restart_resolves_started_attempt_without_replaying_launch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = CrashAfterExternalStart()
    scheduler = _scheduler(tmp_path, executor=executor)
    principal = _principal("jobs.submit", roots=(work,))
    submitted = scheduler.dispatch("jobs.submit", _job_args(work), principal)

    with pytest.raises(SimulatedProcessCrash):
        scheduler.tick()
    assert len(executor.starts) == 1
    stored = scheduler.get_job(submitted["id"])
    assert stored["state"] == "running"
    assert stored["attempt_id"] is not None
    assert scheduler.ledger.get("attempts", stored["attempt_id"])["execution_handle"] is None

    restarted = _scheduler(tmp_path, executor=executor)
    result = restarted.tick()
    assert len(executor.starts) == 1
    assert result["reconciled_jobs"][0]["state"] == "succeeded"
    assert restarted.get_job(submitted["id"])["cleanup_verified"] is True


def test_handoff_job_requires_live_reservation_even_without_resource_requirements(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor)
    job = scheduler.submit_job(
        {
            "owner_id": "agent:one",
            "argv": ["/usr/bin/true"],
            "working_directory": str(work),
        },
        "agent:one",
        "handoff:job:0",
        reservation_id="reservation:missing",
    )

    result = scheduler.tick()
    assert result["started_jobs"] == []
    assert executor.starts == []
    assert scheduler.get_job(job["id"])["state"] == "blocked"


def test_control_cancels_owned_execution_and_rejects_other_owner_reads(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    executor.state = "running"
    scheduler = _scheduler(tmp_path, executor=executor)
    principal = _principal("jobs.submit", "jobs.get", "jobs.control", roots=(work,))
    submitted = scheduler.dispatch("jobs.submit", _job_args(work), principal)
    assert scheduler.tick()["started_jobs"][0]["state"] == "running"

    stranger = Principal("agent:two", frozenset({"jobs.get"}), (str(work),))
    with pytest.raises(PermissionDeniedError, match="another principal"):
        scheduler.dispatch("jobs.get", {"job_id": submitted["id"]}, stranger)

    cancelled = scheduler.dispatch(
        "jobs.control", {"job_id": submitted["id"], "action": "cancel"}, principal
    )
    assert cancelled["state"] == "cancelled"
    assert cancelled["cleanup_verified"] is True


def test_dispatch_enforces_owner_priority_cwd_and_exclusive_tag(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    scheduler = _scheduler(tmp_path)
    principal = _principal("jobs.submit", roots=(work,), max_priority=20)

    with pytest.raises(PermissionDeniedError, match="owner"):
        scheduler.dispatch(
            "jobs.submit",
            _job_args(work, owner_id="agent:other"),
            principal,
        )
    with pytest.raises(PermissionDeniedError, match="execution roots"):
        scheduler.dispatch("jobs.submit", _job_args(outside, key="outside:key"), principal)

    result = scheduler.dispatch(
        "jobs.submit",
        _job_args(
            work,
            tags=["gpu.exclusive"],
            resources=[{"kind": "gpu", "quantities": {"gpu_memory_bytes": 1024}}],
            priority=100,
            priority_class="interactive",
        ),
        principal,
    )
    assert result["owner"] == principal.principal_id
    assert result["spec"]["priority"] == 20
    assert result["spec"]["priority_class"] == "background"
    assert result["spec"]["resources"][0]["mode"] == "exclusive"


def test_unverified_external_lease_release_remains_held(tmp_path: Path) -> None:
    adapter = UnverifiedReclaimer()
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    scheduler.allocator.reconcile_resource_inventory(
        adapter.adapter_id, [Resource("cpu:host", "host:local", "cpu", {"cores": 1})]
    )
    principal = _principal("leases.request", "leases.release", "leases.get")
    lease = scheduler.dispatch(
        "leases.request",
        {
            "idempotency_key": "lease:create",
            "request": {
                "request_id": "request:cpu",
                "owner_id": principal.principal_id,
                "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                "lease_ttl_seconds": 3600,
            },
        },
        principal,
    )

    with pytest.raises(ValidationError, match="fencing_token is required"):
        scheduler.dispatch("leases.release", {"lease_id": lease["lease_id"]}, principal)
    with pytest.raises(ConflictError, match="fencing_token"):
        scheduler.dispatch(
            "leases.release",
            {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"] + 1},
            principal,
        )
    assert scheduler.allocator.get_lease(lease["lease_id"]).state.value == "active"

    released = scheduler.dispatch(
        "leases.release",
        {
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "verified": True,
        },
        principal,
    )
    assert released["state"] == "unknown"
    assert released["fencing_token"] > lease["fencing_token"]


@pytest.mark.parametrize(
    ("verified", "expected_state"),
    [(True, "released"), (False, "unknown")],
)
def test_expired_external_lease_uses_journaled_adapter_reclamation(
    tmp_path: Path, verified: bool, expected_state: str
) -> None:
    clock = MutableClock()
    adapter = VerifiedReclaimer(verified=verified)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    allocator = ResourceAllocator(ledger, clock=clock.now)
    allocator.reconcile_resource_inventory(
        adapter.adapter_id,
        [Resource("cpu:external", "host:local", "cpu", {"cores": 1})],
        observed_at=clock.now(),
    )
    scheduler = Scheduler(
        ledger,
        allocator,
        resource_adapters=(adapter,),
        clock=clock,
        config=SchedulerConfig(state_root=tmp_path / "state"),
    )
    lease = scheduler.dispatch(
        "leases.request",
        {
            "idempotency_key": "lease:expiring",
            "request": {
                "request_id": "request:expiring",
                "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                "lease_ttl_seconds": 10,
            },
        },
        _principal("leases.request"),
    )
    clock.advance(11)

    summary = scheduler.tick()

    stored = allocator.get_lease(lease["lease_id"])
    assert stored.state.value == expected_state
    assert len(adapter.calls) == 1
    operations = [item for item in ledger.operations() if item["operation"] == "lease.release"]
    assert len(operations) == 1
    assert operations[0]["status"] == ("complete" if verified else "unknown")
    assert summary["reconciled_expired_leases"][0]["state"] == expected_state
    if not verified:
        scheduler.tick()
        assert len(adapter.calls) == 1  # unresolved external effects are never blindly replayed
        assert allocator.can_allocate([{"kind": "cpu", "quantities": {"cores": 1}}]) is False


def test_expired_managed_backend_residency_is_not_reclaimed_as_external_lease(tmp_path: Path) -> None:
    clock = MutableClock()
    adapter = VerifiedReclaimer()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    allocator = ResourceAllocator(ledger, clock=clock.now)
    allocator.reconcile_resource_inventory(
        adapter.adapter_id,
        [Resource("cpu:backend", "host:local", "cpu", {"cores": 1})],
        observed_at=clock.now(),
    )
    scheduler = Scheduler(
        ledger,
        allocator,
        resource_adapters=(adapter,),
        clock=clock,
        config=SchedulerConfig(state_root=tmp_path / "state"),
    )
    lease = scheduler.dispatch(
        "leases.request",
        {
            "idempotency_key": "lease:backend-residency",
            "request": {
                "request_id": "request:backend-residency",
                "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                "lease_ttl_seconds": 10,
            },
        },
        _principal("leases.request"),
    )
    with ledger.transaction() as tx:
        tx.put(
            "backend",
            "backend:managed",
            {"id": "backend:managed", "backend_id": "backend:managed", "owner": "agent:backend", "state": "ready", "lease_id": lease["lease_id"]},
            None,
        )
    clock.advance(11)

    summary = scheduler.tick()

    assert allocator.get_lease(lease["lease_id"]).state.value == "reclaiming"
    assert adapter.calls == []
    assert summary["reconciled_expired_leases"][0]["reason"] == "managed_backend_residency_protected"
    assert allocator.can_allocate([{"kind": "cpu", "quantities": {"cores": 1}}]) is False


def test_external_cpu_lease_without_trusted_reclamation_is_refused_before_grant(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler.allocator.register_resource(Resource("cpu:host", "host:local", "cpu", {"cores": 1}))
    principal = _principal("leases.request")

    with pytest.raises(UnsupportedCapabilityError, match="no trusted reclamation adapter"):
        scheduler.dispatch(
            "leases.request",
            {
                "idempotency_key": "lease:unsupported-cpu",
                "request": {
                    "request_id": "request:unsupported-cpu",
                    "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                    "lease_ttl_seconds": 600,
                },
            },
            principal,
        )

    assert scheduler.allocator.list_leases() == []
    assert scheduler.allocator.can_allocate([{"kind": "cpu", "quantities": {"cores": 1}}]) is True


def _request_supervised_host_lease(
    scheduler: Scheduler,
    principal: Principal,
    *,
    key: str = "lease:supervised",
    ttl: int = 600,
    mode: str = "shared",
) -> dict[str, Any]:
    return scheduler.dispatch(
        "leases.request",
        {
            "idempotency_key": key,
            "request": {
                "request_id": key,
                "requirements": [
                    {"kind": "cpu", "quantities": {"cores": 1}, "mode": mode},
                    {"kind": "memory", "quantities": {"bytes": 1024**2}, "mode": mode},
                ],
                "lease_ttl_seconds": ttl,
                "usage": "supervised",
            },
        },
        principal,
    )


@pytest.mark.parametrize("mode", ["shared", "exclusive"])
def test_supervised_cpu_memory_lease_binds_one_job_and_remains_owned_after_job_cleanup(
    tmp_path: Path, mode: str
) -> None:
    executor = LeaseBoundFakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(HostLeaseResourceAdapter(),))
    principal = _principal(
        "leases.request", "leases.renew", "leases.release", "jobs.submit",
        roots=(tmp_path,),
    )
    lease = _request_supervised_host_lease(scheduler, principal, mode=mode)
    assert lease["usage"] == "supervised"
    assert scheduler._get_lease(lease["lease_id"]).to_dict()["usage"] == "supervised"

    submitted = scheduler.dispatch(
        "jobs.submit",
        {
            **_job_args(tmp_path, key="job:lease-bound"),
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
        },
        principal,
    )
    assert submitted["lease_id"] == lease["lease_id"]
    assert submitted["lease_binding"] == "supervised"

    with pytest.raises(ConflictError, match="only one queued or running job"):
        scheduler.dispatch(
            "jobs.submit",
            {
                **_job_args(tmp_path, key="job:lease-bound-2"),
                "lease_id": lease["lease_id"],
                "fencing_token": lease["fencing_token"],
            },
            principal,
        )

    tick_result = scheduler.tick()
    assert len(executor.contexts) == 1, {
        "tick": tick_result,
        "job": scheduler.get_job(submitted["job_id"]),
        "lease": scheduler.allocator.get_lease(lease["lease_id"]).to_dict(),
        "initial_lease": lease,
    }
    context = next(iter(executor.contexts.values()))
    assert context.claim_id == lease["lease_id"]
    assert context.fencing_token == lease["fencing_token"]
    assert {item.resource_id for item in context.resources} == {
        "host:local:cpu", "host:local:memory",
    }
    scheduler.tick()
    assert scheduler.allocator.get_lease(lease["lease_id"]).state.value == "active"
    assert scheduler.get_job(submitted["job_id"])["cleanup_verified"] is True

    renewed = scheduler.dispatch(
        "leases.renew",
        {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"], "ttl_seconds": 900},
        principal,
    )
    assert renewed["state"] == "active"
    released = scheduler.dispatch(
        "leases.release",
        {"lease_id": lease["lease_id"], "fencing_token": renewed["fencing_token"]},
        principal,
    )
    assert released["state"] == "released"
    assert released["release_receipt"]["evidence"]["mode"] == "scheduler-supervised-jobs-only"


def test_supervised_lease_release_cancels_attached_job_and_holds_unknown_cleanup(tmp_path: Path) -> None:
    executor = UnverifiedCancelFakeExecutor()
    executor.state = "running"
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(HostLeaseResourceAdapter(),))
    principal = _principal(
        "leases.request", "leases.release", "jobs.submit",
        roots=(tmp_path,),
    )
    lease = _request_supervised_host_lease(scheduler, principal, key="lease:unclean")
    submitted = scheduler.dispatch(
        "jobs.submit",
        {
            **_job_args(tmp_path, key="job:unclean"),
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
        },
        principal,
    )
    scheduler.tick()
    released = scheduler.dispatch(
        "leases.release",
        {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]},
        principal,
    )
    assert released["state"] == "unknown"
    assert scheduler.get_job(submitted["job_id"])["cleanup_verified"] is False
    assert scheduler.allocator.can_allocate([{"kind": "cpu", "quantities": {"cores": 4}}]) is False


def test_supervised_lease_rejects_stale_fence_and_gpu_or_unmanaged_use(tmp_path: Path) -> None:
    executor = LeaseBoundFakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(HostLeaseResourceAdapter(),))
    principal = _principal(
        "leases.request", "jobs.submit", roots=(tmp_path,),
    )
    lease = _request_supervised_host_lease(scheduler, principal, key="lease:stale")
    with pytest.raises(ConflictError, match="fencing token is stale"):
        scheduler.dispatch(
            "jobs.submit",
            {
                **_job_args(tmp_path, key="job:stale"),
                "lease_id": lease["lease_id"],
                "fencing_token": lease["fencing_token"] + 1,
            },
            principal,
        )
    with pytest.raises(ValidationError, match="inherits its allocation"):
        scheduler.dispatch(
            "jobs.submit",
            {
                **_job_args(
                    tmp_path,
                    key="job:resource-override",
                    resources=[{"kind": "cpu", "quantities": {"cores": 1}}],
                ),
                "lease_id": lease["lease_id"],
                "fencing_token": lease["fencing_token"],
            },
            principal,
        )


def test_expired_supervised_lease_cancels_bound_job_before_releasing_capacity(tmp_path: Path) -> None:
    clock = MutableClock()
    executor = LeaseBoundFakeExecutor()
    executor.state = "running"
    scheduler = _scheduler(
        tmp_path, executor=executor, adapters=(HostLeaseResourceAdapter(),), clock=clock
    )
    principal = _principal(
        "leases.request", "jobs.submit", roots=(tmp_path,),
    )
    lease = _request_supervised_host_lease(
        scheduler, principal, key="lease:expires", ttl=10
    )
    submitted = scheduler.dispatch(
        "jobs.submit",
        {
            **_job_args(tmp_path, key="job:expires"),
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
        },
        principal,
    )
    scheduler.tick()
    assert scheduler.get_job(submitted["job_id"])["state"] == "running"

    clock.advance(11)
    result = scheduler.tick()

    assert scheduler.get_job(submitted["job_id"])["state"] == "cancelled"
    assert scheduler.allocator.get_lease(lease["lease_id"]).state.value == "released"
    assert result["reconciled_expired_leases"][0]["state"] == "released"


def test_external_gpu_lease_uses_persisted_baseline_and_repeatable_release_observation(tmp_path: Path) -> None:
    adapter = ObservedGpuAdapter(release_states=("blocked", "reclaimed"))
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    principal = _principal("leases.request", "leases.release", "leases.get")
    request = {
        "request_id": "request:gpu-exclusive",
        "requirements": [{
            "kind": "gpu",
            "selectors": {"resource_id": "gpu:host:GPU-test"},
            "quantities": {"gpu_memory_bytes": 1},
            "mode": "exclusive",
        }],
        "lease_ttl_seconds": 600,
    }
    lease = scheduler.dispatch(
        "leases.request",
        {"idempotency_key": "lease:gpu-exclusive", "request": request},
        principal,
    )
    evidence = scheduler.ledger.get("lease_evidence", lease["lease_id"])
    assert evidence["initial_fencing_token"] == lease["fencing_token"]
    assert evidence["resource_ids"] == ["gpu:host:GPU-test"]
    assert len(adapter.baseline_calls) == 1

    blocked = scheduler.dispatch(
        "leases.release",
        {"lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]},
        principal,
    )
    assert blocked["state"] == "unknown"
    assert len(adapter.release_calls) == 1
    gpu_request = request["requirements"]
    assert scheduler.allocator.can_allocate(gpu_request) is False

    released = scheduler.dispatch(
        "leases.release",
        {"lease_id": lease["lease_id"], "fencing_token": blocked["fencing_token"]},
        principal,
    )
    assert released["state"] == "released"
    assert len(adapter.release_calls) == 2
    assert len(set(adapter.release_calls)) == 2
    assert scheduler.ledger.get("lease", lease["lease_id"])["release_receipt"]["resource_ids"] == ["gpu:host:GPU-test"]


def test_external_gpu_lease_baseline_identity_failure_keeps_grant_unknown(tmp_path: Path) -> None:
    adapter = ObservedGpuAdapter(baseline_valid=False)
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    principal = _principal("leases.request")
    request = {
        "request_id": "request:bad-baseline",
        "requirements": [{
            "kind": "gpu",
            "selectors": {"resource_id": "gpu:host:GPU-test"},
            "quantities": {"gpu_memory_bytes": 1},
            "mode": "exclusive",
        }],
        "lease_ttl_seconds": 600,
    }

    with pytest.raises(ReconciliationRequiredError, match="baseline was not verified"):
        scheduler.dispatch(
            "leases.request",
            {"idempotency_key": "lease:bad-baseline", "request": request},
            principal,
        )

    lease = scheduler.allocator.list_leases()[0]
    assert lease.state.value == "unknown"
    assert scheduler.allocator.can_allocate(request["requirements"]) is False


def test_release_racing_external_gpu_baseline_capture_cannot_release_unobserved_grant(tmp_path: Path) -> None:
    adapter = BlockingBaselineGpuAdapter()
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    principal = _principal("leases.request", "leases.release")
    request = {
        "request_id": "request:baseline-race",
        "requirements": [{
            "kind": "gpu",
            "selectors": {"resource_id": "gpu:host:GPU-test"},
            "quantities": {"gpu_memory_bytes": 1},
            "mode": "exclusive",
        }],
        "lease_ttl_seconds": 600,
    }
    errors: list[BaseException] = []

    def submit() -> None:
        try:
            scheduler.dispatch(
                "leases.request",
                {"idempotency_key": "lease:baseline-race", "request": request},
                principal,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=submit, daemon=True)
    thread.start()
    assert adapter.capture_entered.wait(2)
    assert adapter.lease_seen is not None
    release_result = scheduler.dispatch(
        "leases.release",
        {
            "lease_id": adapter.lease_seen["lease_id"],
            "fencing_token": adapter.lease_seen["fencing_token"],
        },
        principal,
    )
    adapter.continue_capture.set()
    thread.join(2)

    assert not thread.is_alive()
    assert isinstance(errors[0], ReconciliationRequiredError)
    assert release_result["state"] == "unknown"
    assert scheduler.ledger.get("lease_evidence", adapter.lease_seen["lease_id"]) is None
    assert scheduler.allocator.can_allocate(request["requirements"]) is False


@pytest.mark.parametrize("adapter_state,identity_matches", [("blocked", True), ("verified", False)])
def test_execution_preconditions_fail_closed_before_executor_start(
    tmp_path: Path, adapter_state: str, identity_matches: bool
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    adapter = PreconditionAdapter(state=adapter_state, identity_matches=identity_matches)
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(adapter,))
    principal = _principal("jobs.submit", roots=(work,))
    submitted = scheduler.dispatch(
        "jobs.submit",
        _job_args(
            work,
            key=f"job:precondition:{adapter_state}:{identity_matches}",
            preconditions=[{"name": "device.quiescence", "parameters": {"samples": 3}}],
        ),
        principal,
    )

    summary = scheduler.tick()

    stored = scheduler.get_job(submitted["id"])
    assert stored["state"] == "blocked"
    assert executor.starts == []
    assert len(adapter.calls) == 1
    attempt = scheduler.ledger.get("attempts", stored["attempt_id"])
    assert attempt["state"] == "blocked"
    assert attempt["cleanup_verified"] is True
    assert attempt["precondition_receipt"]["state"] == "blocked"
    assert summary["started_jobs"] == []


def test_verified_execution_precondition_is_durable_attempt_evidence(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    adapter = PreconditionAdapter()
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(adapter,))
    principal = _principal("jobs.submit", roots=(work,))
    submitted = scheduler.dispatch(
        "jobs.submit",
        _job_args(work, key="job:precondition:verified", preconditions=[{"name": "device.quiescence"}]),
        principal,
    )

    started = scheduler.tick()["started_jobs"][0]

    assert started["state"] == "running"
    attempt = scheduler.ledger.get("attempts", started["attempt_id"])
    assert attempt["precondition_receipt"]["state"] == "verified"
    assert attempt["evidence"][0]["kind"] == "execution_preconditions"
    assert attempt["evidence"][0]["operation_id"] == f"preconditions:{started['attempt_id']}"
    assert len(executor.starts) == 1
    assert scheduler.get_job(submitted["id"])["state"] == "running"


def test_backend_initial_start_requires_exact_closed_admission_receipt(tmp_path: Path) -> None:
    for label, initial_state, initial_operation, should_start in (
        ("closed", "closed", "backend-start:backend:one", True),
        ("open", "open", "backend-start:backend:one", False),
        ("wrong-operation", "closed", "handoff:old:reopen", False),
    ):
        adapter = FakeBackendAdapter(initial_state=initial_state, initial_operation=initial_operation)
        ledger = Ledger(tmp_path / f"backend-{label}.sqlite3")
        allocator = ResourceAllocator(ledger)
        allocator.register_resource(Resource("cpu:backend", "host:local", "cpu", {"cores": 2}))
        scheduler = Scheduler(
            ledger,
            allocator,
            backend_adapter=adapter,
            clock=FakeClock(),
            config=SchedulerConfig(
                state_root=tmp_path / f"state-{label}",
                backend_profiles={"profile:llama": {"version": "1", "configuration_ref": "config:llama"}},
            ),
        )
        principal = Principal(
            "agent:backend",
            frozenset({"backends.register", "backends.control"}),
            backend_ids=("backend:*",),
        )
        registration = scheduler.dispatch(
            "backends.register",
            {
                "idempotency_key": f"register:{label}",
                "spec": {
                    "backend_id": "backend:one",
                    "host_id": "host:local",
                    "adapter_id": adapter.adapter_id,
                    "profile_id": "profile:llama",
                    "configuration_ref": "config:llama",
                    "residency_requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                },
            },
            principal,
        )

        result = scheduler.dispatch(
            "backends.control",
            {"backend_id": "backend:one", "action": "start"},
            principal,
        )

        assert registration["state"] == "stopped"
        assert result["state"] == ("ready" if should_start else "unknown")
        assert adapter.preparations == ["backend-start:backend:one"]
        assert len(adapter.restores) == (1 if should_start else 0)
        assert adapter.reopens == (["backend-start:backend:one"] if should_start else [])


def test_slow_tick_inventory_does_not_block_client_job_submission(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    inventory = BlockingInventory()
    scheduler = _scheduler(tmp_path, adapters=(inventory,))
    principal = _principal("jobs.submit", roots=(work,))
    tick_results: list[dict[str, Any]] = []
    submit_results: list[dict[str, Any]] = []
    submitted = Event()

    tick_thread = Thread(target=lambda: tick_results.append(scheduler.tick()), daemon=True)
    tick_thread.start()
    assert inventory.entered.wait(1)

    def submit() -> None:
        try:
            submit_results.append(scheduler.dispatch("jobs.submit", _job_args(work), principal))
        finally:
            submitted.set()

    submit_thread = Thread(target=submit, daemon=True)
    submit_thread.start()
    try:
        assert submitted.wait(1), "client submit blocked behind slow tick adapter I/O"
    finally:
        inventory.release.set()
    submit_thread.join(2)
    tick_thread.join(2)
    assert not submit_thread.is_alive()
    assert not tick_thread.is_alive()
    assert submit_results[0]["state"] == "queued"
    assert tick_results[0]["status"] == "degraded"


def test_unknown_inventory_cannot_reuse_last_known_resource_capacity(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    inventory = BlockingInventory()
    inventory.release.set()
    scheduler = _scheduler(tmp_path, adapters=(inventory,))
    resource = Resource("accel:one", "host:local", "accelerator", {"slots": 1})
    scheduler.allocator.reconcile_resource_inventory(inventory.adapter_id, [resource])
    principal = _principal("leases.request")

    with pytest.raises(ConflictError, match="matching resource inventory"):
        scheduler.dispatch(
            "leases.request",
            {
                "idempotency_key": "lease:unknown-inventory",
                "request": {
                    "request_id": "request:unknown-inventory",
                    "requirements": [{"kind": "accelerator", "quantities": {"slots": 1}}],
                    "lease_ttl_seconds": 600,
                },
            },
            principal,
        )
    stored = scheduler.ledger.get("resource_health", "accel:one")
    assert stored["available"] is False
    assert scheduler.allocator.can_allocate(
        [{"kind": "accelerator", "quantities": {"slots": 1}}]
    ) is False


def test_tick_reconciliation_error_skips_new_execution_admission(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    scheduler = _scheduler(tmp_path, executor=executor)
    principal = _principal("jobs.submit", roots=(work,))
    scheduler.dispatch("jobs.submit", _job_args(work), principal)

    def fail_reconciliation() -> list[dict[str, Any]]:
        raise OSError("injected reconciliation failure")

    scheduler._reconcile_jobs = fail_reconciliation  # type: ignore[method-assign]
    summary = scheduler.tick()
    assert summary["admission_skipped"] == "pre_admission_error"
    assert summary["errors"][0]["component"] == "job_reconciliation"
    assert executor.starts == []


def test_safe_backfill_preserves_disjoint_aged_request_placement(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    scheduler = _scheduler(tmp_path)
    scheduler.allocator.register_resource(Resource("cpu:a", "host:local", "cpu", {"cores": 1}))
    scheduler.allocator.register_resource(Resource("cpu:b", "host:local", "cpu", {"cores": 1}))
    protected = {
        "id": "job:protected",
        "owner": "agent:large",
        "owner_id": "agent:large",
        "created_at": "2026-10-21T11:50:00Z",
        "state": "blocked",
        "spec": {
            "argv": ["/usr/bin/true"],
            "owner_id": "agent:large",
            "working_directory": str(work),
            "resources": [{"kind": "cpu", "selectors": {"resource_id": "cpu:a"}, "quantities": {"cores": 1}}],
        },
    }
    candidate = {
        "id": "job:backfill",
        "owner": "agent:small",
        "owner_id": "agent:small",
        "created_at": "2026-10-21T11:59:00Z",
        "state": "queued",
        "spec": {
            "argv": ["/usr/bin/true"],
            "owner_id": "agent:small",
            "working_directory": str(work),
            "resources": [{"kind": "cpu", "selectors": {"resource_id": "cpu:b"}, "quantities": {"cores": 1}}],
        },
    }
    assert scheduler._safe_backfill(candidate, protected) is True
    decision = scheduler.policy.choose_next(
        [protected, candidate],
        can_fit=lambda item: item["id"] == candidate["id"],
        safe_backfill=scheduler._safe_backfill,
        now=NOW,
    )
    assert decision is not None
    assert decision.request_id == candidate["id"]
    assert decision.reason_code == "safe_backfill"


def test_external_owner_grant_persists_descriptive_owner_context(tmp_path: Path) -> None:
    adapter = VerifiedReclaimer()
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    scheduler.allocator.reconcile_resource_inventory(
        adapter.adapter_id, [Resource("cpu:host", "host:local", "cpu", {"cores": 1})]
    )
    principal = Principal(
        "allocator-service:trusted-bridge",
        frozenset({"leases.request", "leases.request:any_owner"}),
    )
    lease = scheduler.dispatch(
        "leases.request",
        {
            "idempotency_key": "resource-lease:one",
            "request": {
                "request_id": "resource-request:one",
                "owner_id": "external-tenant:project-a:attempt:one",
                "owner_context": {"issuer": "trusted-bridge", "attempt_id": "attempt:one"},
                "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                "lease_ttl_seconds": 600,
            },
        },
        principal,
    )
    assert lease["owner_id"] == "external-tenant:project-a:attempt:one"
    assert lease["owner_context"] == {"issuer": "trusted-bridge", "attempt_id": "attempt:one"}

    unprivileged = Principal("agent:ordinary", frozenset({"leases.request"}))
    with pytest.raises(PermissionDeniedError, match="leases.request:any_owner"):
        scheduler.dispatch(
            "leases.request",
            {
                "idempotency_key": "external:lease:forged",
                "request": {
                    "request_id": "external:request:forged",
                    "owner_id": "external-tenant:project-a:attempt:forged",
                    "owner_context": {"issuer": "external", "attempt_id": "attempt:forged"},
                    "requirements": [{"kind": "cpu", "quantities": {"cores": 1}}],
                    "lease_ttl_seconds": 600,
                },
            },
            unprivileged,
        )


def test_restart_safe_background_job_can_be_preempted_and_requeued(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    executor = FakeExecutor()
    executor.state = "running"
    scheduler = _scheduler(tmp_path, executor=executor)
    scheduler.allocator.register_resource(Resource("cpu:host", "host:local", "cpu", {"cores": 1}))
    requirement = [{"kind": "cpu", "quantities": {"cores": 1}}]
    background = scheduler.submit_job(
        {
            "owner_id": "agent:background",
            "argv": ["/usr/bin/true"],
            "working_directory": str(work),
            "resources": requirement,
            "priority_class": "background",
            "priority": 5,
            "max_attempts": 2,
            "interruption": {"restart_safe": True, "cancellation_supported": True},
        },
        "agent:background",
        "background:job",
    )
    first = scheduler.tick()
    assert first["started_jobs"][0]["id"] == background["id"]
    old_attempt_id = scheduler.get_job(background["id"])["attempt_id"]

    interactive = scheduler.submit_job(
        {
            "owner_id": "agent:interactive",
            "argv": ["/usr/bin/true"],
            "working_directory": str(work),
            "resources": requirement,
            "priority_class": "interactive",
            "priority": 90,
        },
        "agent:interactive",
        "interactive:job",
    )
    second = scheduler.tick()

    assert [item["id"] for item in second["started_jobs"]] == [interactive["id"]]
    requeued = scheduler.get_job(background["id"])
    assert requeued["state"] == "queued"
    assert requeued["attempt_count"] == 1  # preemption does not silently reset max_attempts
    assert requeued["preemption_count"] == 1
    assert scheduler.ledger.get("attempts", old_attempt_id)["preemption"]["requester_job_id"] == interactive["id"]
    preemption_ops = [item for item in scheduler.ledger.operations() if item["operation"] == "job.preempt"]
    assert len(preemption_ops) == 1
    assert preemption_ops[0]["status"] == "complete"


def test_external_gpu_capability_advertises_budget_after_observed_headroom(tmp_path: Path) -> None:
    adapter = ObservedGpuAdapter()
    scheduler = _scheduler(tmp_path, adapters=(adapter,))
    resource = Resource("gpu:test", "host:local", "gpu", {"memory_bytes": 8_000_000_000},
                        attributes={"memory_free_bytes_observation": 7_000_000_000})
    scheduler.allocator.reconcile_resource_inventory(adapter.adapter_id, [resource])
    capability = scheduler._external_resource_lease_capability(resource.to_dict())
    assert capability["admission_memory_bytes"] == 6_300_000_000
    assert capability["required_allocation_mode"] == "exclusive"


def test_pre_upgrade_job_idempotency_record_replays_unchanged(tmp_path: Path) -> None:
    from device_scheduler_core.models import JobSpec
    scheduler = _scheduler(tmp_path)
    spec = JobSpec(owner_id="agent:one", argv=("/bin/true",), working_directory=str(tmp_path))
    payload = {"spec": spec.to_dict(), "reservation_id": None}
    prior = {"id": "legacy-job", "owner": "agent:one", "state": "queued", "spec": spec.to_dict()}
    scheduler.ledger.idempotent("agent:one", "legacy-job-key", "jobs.submit", payload, lambda tx: prior)
    assert scheduler.submit_job(spec.to_dict(), "agent:one", "legacy-job-key")["id"] == "legacy-job"


def test_pre_upgrade_external_lease_idempotency_record_replays_unchanged(tmp_path: Path) -> None:
    from device_scheduler_core.models import ResourceRequest, ResourceRequirement
    scheduler = _scheduler(tmp_path)
    scheduler.allocator.reconcile_resource_inventory("legacy", [Resource("cpu:test", "host:test", "cpu", {"cores": 1})])
    request = ResourceRequest("legacy-request", "agent:one", (ResourceRequirement("cpu", {"cores": .25}),), 120)
    legacy_payload = request.to_dict()
    legacy_payload.pop("usage")
    def old_create(tx):
        result = scheduler.allocator._allocate_lease_tx(tx, request, request_data=legacy_payload).to_dict()
        result.pop("usage")
        return result
    old = scheduler.ledger.idempotent("agent:one", "legacy-request", "request_lease", legacy_payload, old_create)
    assert scheduler.allocator.request_lease(request).lease_id == old["lease_id"]


def test_policy_cannot_preempt_a_caller_owned_supervised_lease(tmp_path: Path) -> None:
    executor = LeaseBoundFakeExecutor()
    executor.state = "running"
    scheduler = _scheduler(tmp_path, executor=executor, adapters=(HostLeaseResourceAdapter(),))
    principal = _principal("leases.request", "jobs.submit", roots=(tmp_path,))
    lease = _request_supervised_host_lease(scheduler, principal)
    background = scheduler.dispatch("jobs.submit", {
        **_job_args(tmp_path, key="protected-background", priority_class="background", priority=0,
                    max_attempts=2, interruption={"restart_safe": True, "cancellation_supported": True}),
        "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"],
    }, principal)
    scheduler.tick()
    target = scheduler.get_job(background["id"])
    interactive = scheduler.submit_job({"owner_id": "interactive", "argv": ["/bin/true"],
        "working_directory": str(tmp_path), "resources": [{"kind": "cpu", "quantities": {"cores": 4}}],
        "priority_class": "interactive", "priority": 100}, "interactive", "interactive-request")
    scheduler.tick()
    assert scheduler.get_job(background["id"])["state"] == "running"
    assert scheduler.get_job(background["id"])["lease_id"] == lease["lease_id"]
    assert scheduler._execute_preemption({"target_job_id": background["id"],
        "target_attempt_id": target["attempt_id"], "target_lease_id": lease["lease_id"]}, interactive["id"]) is False
    assert scheduler._get_lease(lease["lease_id"]).state.value == "active"
