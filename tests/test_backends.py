from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from device_scheduler_adapters.backends import (
    BackendAdapterError,
    BackendProfile,
    FrameShiftBackendAdmissionClient,
    ManagedLlamaBackendAdapter,
)
from device_scheduler_core.models import (
    AllocationMode,
    AdapterCapability,
    BackendSpec,
    EnforcementLevel,
    ExecutionContext,
    ExecutionHandle,
    ExecutionPurpose,
    Resource,
    ResourceAllocation,
    ResourceRequirement,
)


def make_backend(*, pinned_configuration: dict[str, Any] | None = None) -> BackendSpec:
    return BackendSpec(
        backend_id="llama-main",
        host_id="lab-host",
        adapter_id=ManagedLlamaBackendAdapter.adapter_id,
        profile_id="llama-profile",
        configuration_ref="deployment:llama-main:v3",
        residency_requirements=(ResourceRequirement("gpu", {"memory_bytes": 1_000_000}),),
        attached_consumers=("frame:agent-1",),
        pinned_configuration=pinned_configuration
        or {"profile_version": "v3", "model_identity": "sha256:model-a"},
    )


def make_profile(**changes: Any) -> BackendProfile:
    data: dict[str, Any] = {
        "profile_id": "llama-profile",
        "version": "v3",
        "execution_profile_id": "execution:llama-v3",
        "health_url": "http://127.0.0.1:8080/health",
        "slots_url": "http://127.0.0.1:8080/slots",
        "access_mode": "frameshift-only",
        "frameshift_control_ref": "frameshift:local",
        "expected_model_identity": "sha256:model-a",
        "identity_json_path": "model",
        "health_json_path": "status",
        "expected_health_value": "ok",
        "readiness_timeout_seconds": 0,
        "stop_timeout_seconds": 4,
    }
    data.update(changes)
    return BackendProfile(**data)


class FakeExecution:
    def __init__(self, *, state: str = "running", cleanup_verified: bool = True) -> None:
        self.state = state
        self.cleanup_verified = cleanup_verified
        self.starts: list[dict[str, Any]] = []
        self.cancels: list[dict[str, Any]] = []
        self.handle = ExecutionHandle(
            adapter_id="systemd-user-unit",
            external_id="device-scheduler-attempt-backend.service",
            identity="attempt:backend",
            started_at=datetime.now(timezone.utc),
            metadata={
                "invocation_id": "backend-invocation",
                "control_group": "/user.slice/user-1000.slice/device-scheduler-attempt-backend.service",
                "description": "backend unit",
            },
        )

    def start(self, spec, *, attempt_id: str, execution_root, execution_context=None):
        self.starts.append(
            {
                "spec": spec,
                "attempt_id": attempt_id,
                "execution_root": execution_root,
                "execution_context": execution_context,
            }
        )
        return self.handle

    def inspect(self, handle):
        return {"state": self.state, "cleanup_verified": self.cleanup_verified}

    def cancel(self, handle, *, operation_id: str, timeout_seconds: float):
        self.cancels.append(
            {"handle": handle, "operation_id": operation_id, "timeout_seconds": timeout_seconds}
        )
        return {
            "state": "succeeded",
            "cleanup_verified": self.cleanup_verified,
            "operation_id": operation_id,
        }

    def collect(self, handle):
        return {"state": "succeeded"}


class FakeAdmission:
    def __init__(self, *, active_generations: int = 0, closed: bool = False, operation_id: str | None = None):
        self.active_generations = active_generations
        self.closed = closed
        self.operation_id = operation_id
        self.epoch = 1
        self.close_calls: list[str] = []
        self.open_calls: list[tuple[str, str | int]] = []

    def backend_admission_state(self, backend_id: str) -> dict[str, Any]:
        return {
            "backend_id": backend_id,
            "closed": self.closed,
            "operation_id": self.operation_id,
            "epoch": self.epoch,
            "active_generations": self.active_generations,
            "active_generations_source": "frameshift-provider-leases-upper-bound",
        }

    def close_backend_admission(self, backend_id: str, operation_id: str) -> dict[str, Any]:
        self.close_calls.append(operation_id)
        if self.closed and self.operation_id not in {None, operation_id}:
            return {"closed": False, "backend_id": backend_id, "operation_id": operation_id}
        if self.closed and self.operation_id == operation_id:
            return {"closed": True, "backend_id": backend_id, "operation_id": operation_id, "epoch": self.epoch,
                    "active_generations": self.active_generations,
                    "active_generations_source": "frameshift-provider-leases-upper-bound"}
        if not self.closed and self.operation_id == operation_id:
            return {"closed": False, "backend_id": backend_id, "operation_id": operation_id}
        self.epoch += 1
        self.operation_id = operation_id
        self.closed = True
        return {"closed": True, "backend_id": backend_id, "operation_id": operation_id, "epoch": self.epoch,
                "active_generations": self.active_generations,
                "active_generations_source": "frameshift-provider-leases-upper-bound"}

    def open_backend_admission(self, backend_id: str, operation_id: str, epoch: str | int) -> dict[str, Any]:
        self.open_calls.append((operation_id, epoch))
        if self.operation_id != operation_id or self.epoch != epoch:
            return {"opened": False}
        self.closed = False
        return {"opened": True, "backend_id": backend_id, "operation_id": operation_id, "epoch": epoch}


def make_adapter(
    execution: FakeExecution,
    *,
    profile: BackendProfile | None = None,
    health_document: dict[str, Any] | None = None,
    slots_document: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ManagedLlamaBackendAdapter:
    profile = profile or make_profile()
    health_document = health_document or {"status": "ok", "model": "sha256:model-a"}
    slots_document = slots_document or [{"id": 0, "is_processing": False}]
    control = kwargs.pop("admission", None) or FakeAdmission()
    kwargs.setdefault("frameshift_controls", {"frameshift:local": control})
    kwargs.setdefault(
        "resource_reclamation_baseline_capturer",
        lambda backend, operation_id: make_reclamation_baseline(execution, backend, operation_id),
    )
    def health_reader(url, *, timeout):
        document = slots_document if url.endswith("/slots") else health_document
        return {"status": 200, "json": document}

    return ManagedLlamaBackendAdapter(
        execution,
        profile_resolver=lambda _backend: profile,
        handle_resolver=lambda _backend: execution.handle,
        health_reader=health_reader,
        monotonic=lambda: 0.0,
        sleeper=lambda _seconds: None,
        **kwargs,
    )


def make_backend_execution_context(*, purpose: ExecutionPurpose = ExecutionPurpose.BACKEND) -> ExecutionContext:
    gpu = Resource(
        resource_id="gpu:lab-host:GPU-TEST-0",
        host_id="lab-host",
        kind="gpu",
        capacities={"memory_bytes": 16_000_000_000},
        attributes={"uuid": "GPU-TEST-0"},
    )
    allocation = ResourceAllocation(
        gpu.resource_id,
        {"memory_bytes": 1_000_000},
        AllocationMode.SHARED,
    )
    return ExecutionContext(
        allocations=(allocation,),
        resources=(gpu,),
        claim_id="lease:backend-reservation",
        fencing_token=7,
        purpose=purpose,
    )


def make_reclamation_baseline(
    execution: FakeExecution,
    backend: BackendSpec,
    operation_id: str,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    handle = execution.handle
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "state": "observed",
        "backend_id": backend.backend_id,
        "operation_id": operation_id,
        "profile_id": "llama-profile",
        "profile_version": "v3",
        "observed_at": (observed_at or datetime.now(timezone.utc)).isoformat(),
        "source": "nvidia-smi+invocation-verified-systemd-unit-cgroups",
        "execution_identity": {
            "unit_name": handle.external_id,
            "invocation_id": handle.metadata["invocation_id"],
            "control_group": handle.metadata["control_group"],
        },
        "process_mapping_state": "observed",
        "gpus": [
            {
                "resource_id": "gpu:lab-host:GPU-TEST-0",
                "gpu_uuid": "GPU-TEST-0",
                "memory_free_bytes": 8_000_000_000,
                "used_gpu_memory_bytes": 1_000_000_000,
                "inventory_observed_at": timestamp,
                "process_observed_at": timestamp,
                "process_ids": [12345],
                "backend_process_ids": [12345],
            }
        ],
        "evidence": [{"source": "test-fixture", "identity_verified": True}],
    }


def test_profile_requires_loopback_and_pinned_identity_path() -> None:
    with pytest.raises(ValueError, match="loopback"):
        make_profile(health_url="http://example.com/health")
    with pytest.raises(ValueError, match="identity_json_path"):
        make_profile(identity_json_path=None)
    with pytest.raises(ValueError, match="credential-free"):
        make_profile(inference_base_url="http://user:pass@127.0.0.1:8080/v1", inference_model_id="model-a")
    with pytest.raises(ValueError, match="query or fragment"):
        make_profile(inference_base_url="http://127.0.0.1:8080/v1?token=x", inference_model_id="model-a")
    with pytest.raises(ValueError, match="loopback"):
        make_profile(inference_base_url="https://example.com/v1", inference_model_id="model-a")


def test_restore_uses_pinned_registered_profile_and_verifies_model_identity(tmp_path) -> None:
    execution = FakeExecution()
    adapter = make_adapter(execution, execution_root=str(tmp_path / "backends"))

    receipt = adapter.restore(
        make_backend(),
        operation_id="handoff:one",
        execution_context=make_backend_execution_context(),
    )

    assert receipt["state"] == "ready"
    assert receipt["profile_version"] == "v3"
    assert receipt["readiness"]["identity_verified"] is True
    assert receipt["readiness"]["model_identity"] == "sha256:model-a"
    assert execution.starts[0]["spec"].profile_id == "execution:llama-v3"
    assert execution.starts[0]["attempt_id"].startswith("backend-restore-")
    assert execution.starts[0]["execution_root"].name.startswith("backend-")
    assert execution.starts[0]["execution_context"].purpose == ExecutionPurpose.BACKEND

    repeated = adapter.restore(
        make_backend(),
        operation_id="handoff:one",
        execution_context=make_backend_execution_context(),
    )
    assert repeated["attempt_id"] == receipt["attempt_id"]


def test_restore_requires_actual_backend_allocation_context_before_start() -> None:
    execution = FakeExecution()
    adapter = make_adapter(execution)

    missing = adapter.restore(make_backend(), operation_id="handoff:no-context")
    cpu_purpose = adapter.restore(
        make_backend(),
        operation_id="handoff:wrong-purpose",
        execution_context=make_backend_execution_context(purpose=ExecutionPurpose.JOB),
    )

    assert missing["state"] == "blocked"
    assert "protected resource allocation" in missing["reason"]
    assert cpu_purpose["state"] == "blocked"
    assert "purpose 'backend'" in cpu_purpose["reason"]
    assert execution.starts == []


def test_registered_cpu_only_backend_restores_under_protected_cpu_context() -> None:
    execution = FakeExecution()
    adapter = make_adapter(execution)
    cpu = Resource(
        resource_id="cpu:lab-host:0",
        host_id="lab-host",
        kind="cpu",
        capacities={"cores": 8},
    )
    cpu_backend = BackendSpec(
        backend_id="llama-cpu",
        host_id="lab-host",
        adapter_id=ManagedLlamaBackendAdapter.adapter_id,
        profile_id="llama-profile",
        configuration_ref="deployment:llama-cpu:v3",
        residency_requirements=(ResourceRequirement("cpu", {"cores": 1}),),
        pinned_configuration={"profile_version": "v3", "model_identity": "sha256:model-a"},
    )
    context = ExecutionContext(
        allocations=(ResourceAllocation(cpu.resource_id, {"cores": 2}, AllocationMode.SHARED),),
        resources=(cpu,),
        claim_id="lease:cpu-backend",
        fencing_token=9,
        purpose=ExecutionPurpose.BACKEND,
    )

    receipt = adapter.restore(
        cpu_backend,
        operation_id="startup:llama-cpu",
        execution_context=context,
    )

    assert receipt["state"] == "ready"
    assert execution.starts[0]["execution_context"].allocations[0].resource_id == cpu.resource_id


def test_restore_exposes_only_verified_secret_free_inference_endpoint() -> None:
    execution = FakeExecution()
    profile = make_profile(
        inference_base_url="http://127.0.0.1:8080/v1",
        inference_model_id="llama-local",
    )
    adapter = make_adapter(execution, profile=profile)

    restored = adapter.restore(
        make_backend(),
        operation_id="handoff:endpoint",
        execution_context=make_backend_execution_context(),
    )
    ready = adapter.ready(make_backend(), timeout_seconds=0, operation_id="handoff:ready")
    inspected = adapter.inspect(make_backend())

    expected = {
        "server_kind": "llama.cpp",
        "protocol": "openai-compatible",
        "base_url": "http://127.0.0.1:8080/v1",
        "model_id": "llama-local",
    }
    assert restored["endpoint"] == expected
    assert restored["readiness"]["endpoint"] == expected
    assert ready["endpoint"] == expected
    assert ready["readiness"]["endpoint"] == expected
    assert inspected["readiness"]["endpoint"] == expected

    unverified = make_adapter(
        FakeExecution(),
        profile=make_profile(
            expected_model_identity=None,
            identity_json_path=None,
            inference_base_url=None,
            inference_model_id=None,
        ),
    ).ready(
        make_backend(pinned_configuration={"profile_version": "v3"}),
        timeout_seconds=0,
    )
    assert unverified["ready"] is True
    assert unverified["identity_verified"] is False
    assert "endpoint" not in unverified


def test_prepare_initial_start_binds_unconfigured_gate_and_rechecks_idle_state() -> None:
    execution = FakeExecution()
    admission = FakeAdmission(closed=True, operation_id=None)
    admission.epoch = 0
    adapter = make_adapter(execution, admission=admission)

    receipt = adapter.prepare_initial_start(
        make_backend(), operation_id="backend-start:llama-main"
    )

    assert receipt["state"] == "admission_closed"
    assert receipt["admission_closed"] is True
    assert receipt["operation_id"] == "backend-start:llama-main"
    assert receipt["admission_epoch"] == 1
    assert receipt["active_generations"] == 0
    assert receipt["active_generations_source"] == "frameshift-provider-leases-upper-bound"
    assert admission.close_calls == ["backend-start:llama-main"]


def test_prepare_initial_start_closes_open_gate_and_refuses_another_closed_epoch() -> None:
    execution = FakeExecution()
    open_admission = FakeAdmission(closed=False, operation_id="previous-start")
    adapter = make_adapter(execution, admission=open_admission)
    receipt = adapter.prepare_initial_start(
        make_backend(), operation_id="backend-start:llama-main"
    )
    assert receipt["state"] == "admission_closed"
    assert open_admission.close_calls == ["backend-start:llama-main"]

    handoff_admission = FakeAdmission(closed=True, operation_id="handoff:protected")
    handoff_adapter = make_adapter(execution, admission=handoff_admission)
    blocked = handoff_adapter.prepare_initial_start(
        make_backend(), operation_id="backend-start:llama-main"
    )
    assert blocked["state"] == "blocked"
    assert blocked["current_operation_id"] == "handoff:protected"
    assert handoff_admission.close_calls == []
    reopened = handoff_adapter.reopen(
        make_backend(), operation_id="backend-start:llama-main"
    )
    assert reopened["state"] == "unknown"
    assert handoff_admission.open_calls == []


def test_non_frameshift_backend_can_start_but_is_explicitly_non_evictable() -> None:
    execution = FakeExecution()
    profile = make_profile(access_mode=None, frameshift_control_ref=None, slots_url=None)
    adapter = make_adapter(execution, profile=profile)
    backend = make_backend()

    prepared = adapter.prepare_initial_start(
        backend, operation_id="backend-start:llama-main"
    )
    restored = adapter.restore(
        backend,
        operation_id="startup:llama-main",
        execution_context=make_backend_execution_context(),
    )
    drain = adapter.drain(backend, operation_id="handoff:cannot-evict", timeout_seconds=1)

    assert prepared["state"] == "not_required"
    assert prepared["eviction_supported"] is False
    assert restored["state"] == "ready"
    assert len(execution.starts) == 1
    assert drain["state"] == "blocked"


def test_restore_fails_readiness_if_served_identity_is_wrong() -> None:
    execution = FakeExecution()
    adapter = make_adapter(execution, health_document={"status": "ok", "model": "wrong-model"})

    receipt = adapter.restore(
        make_backend(),
        operation_id="handoff:identity",
        execution_context=make_backend_execution_context(),
    )

    assert receipt["state"] == "failed"
    assert receipt["readiness"]["identity_verified"] is False
    assert receipt["readiness"]["observed_model_identity"] == "wrong-model"


def test_restore_does_not_accept_health_from_non_running_execution() -> None:
    execution = FakeExecution(state="succeeded")
    adapter = make_adapter(execution)

    receipt = adapter.restore(
        make_backend(),
        operation_id="handoff:terminal",
        execution_context=make_backend_execution_context(),
    )

    assert receipt["state"] == "failed"
    assert receipt["readiness"]["ready"] is False
    assert "execution is succeeded" in receipt["readiness"]["reason"]


def test_profile_version_mismatch_blocks_restore_before_external_effect() -> None:
    execution = FakeExecution()
    adapter = make_adapter(execution, profile=make_profile(version="v4"))

    with pytest.raises(BackendAdapterError, match="does not match pinned"):
        adapter.restore(
            make_backend(),
            operation_id="handoff:version",
            execution_context=make_backend_execution_context(),
        )

    assert execution.starts == []


def test_drain_only_closes_admission_and_requires_stable_operation_id() -> None:
    execution = FakeExecution()
    admission = FakeAdmission()
    adapter = make_adapter(
        execution,
        admission=admission,
    )

    receipt = adapter.drain(make_backend(), operation_id="handoff:drain", timeout_seconds=3)

    assert receipt["state"] == "admission_closed"
    assert receipt["admission_closed"] is True
    assert receipt["continuations_suspended"] is False
    assert admission.close_calls == ["handoff:drain"]
    assert execution.cancels == []


def test_evict_requires_grant_closed_admission_idle_generations_and_reclamation() -> None:
    execution = FakeExecution()
    admission = FakeAdmission()
    adapter = make_adapter(
        execution,
        admission=admission,
        eviction_authorizer=lambda _backend, op: {"authorized": True, "grant_id": f"grant:{op}"},
        resource_reclamation_verifier=lambda _backend, op, _baseline: {
            "state": "reclaimed", "operation_id": op, "evidence": ["gpu-memory-floor:observed"]
        },
    )

    backend = make_backend()
    baseline = make_reclamation_baseline(execution, backend, "handoff:evict:baseline:0")
    receipt = adapter.evict(backend, operation_id="handoff:evict", reclamation_baseline=baseline)

    assert receipt["state"] == "evicted"
    assert receipt["cleanup_verified"] is True
    assert receipt["resource_reclamation"]["state"] == "reclaimed"
    assert receipt["reclamation_baseline"]["operation_id"] == "handoff:evict:baseline:0"
    assert admission.close_calls == ["handoff:evict"]
    assert execution.cancels[0]["operation_id"] == "handoff:evict"
    assert execution.cancels[0]["timeout_seconds"] == 4


def test_reclamation_baseline_capture_binds_current_invocation_and_process_mapping() -> None:
    execution = FakeExecution()
    backend = make_backend()
    adapter = make_adapter(execution)

    receipt = adapter.capture_reclamation_baseline(backend, operation_id="handoff:baseline:0")

    assert receipt["state"] == "observed"
    assert receipt["operation_id"] == "handoff:baseline:0"
    assert receipt["execution_identity"] == {
        "unit_name": execution.handle.external_id,
        "invocation_id": execution.handle.metadata["invocation_id"],
        "control_group": execution.handle.metadata["control_group"],
    }
    assert receipt["process_mapping_state"] == "observed"


def test_evict_requires_fresh_baseline_bound_to_current_execution_before_any_side_effect() -> None:
    execution = FakeExecution()
    admission = FakeAdmission()
    adapter = make_adapter(
        execution,
        admission=admission,
        eviction_authorizer=lambda _backend, op: {"authorized": True, "grant_id": op},
    )
    backend = make_backend()

    missing = adapter.evict(backend, operation_id="handoff:missing-baseline")
    assert missing["state"] == "blocked"
    assert admission.close_calls == []
    assert execution.cancels == []

    stale = make_reclamation_baseline(
        execution,
        backend,
        "handoff:stale:baseline:0",
        observed_at=datetime.fromtimestamp(0, tz=timezone.utc),
    )
    stale_result = adapter.evict(
        backend, operation_id="handoff:stale", reclamation_baseline=stale
    )
    assert stale_result["state"] == "blocked"
    assert admission.close_calls == []
    assert execution.cancels == []

    mismatched = make_reclamation_baseline(
        execution, backend, "handoff:mismatch:baseline:0"
    )
    mismatched["execution_identity"]["invocation_id"] = "different-systemd-invocation"
    mismatch_result = adapter.evict(
        backend, operation_id="handoff:mismatch", reclamation_baseline=mismatched
    )
    assert mismatch_result["state"] == "blocked"
    assert admission.close_calls == []
    assert execution.cancels == []

    unattributed = make_reclamation_baseline(
        execution, backend, "handoff:unattributed:baseline:0"
    )
    unattributed["process_mapping_state"] = "unknown"
    unattributed_result = adapter.evict(
        backend,
        operation_id="handoff:unattributed",
        reclamation_baseline=unattributed,
    )
    assert unattributed_result["state"] == "blocked"
    assert admission.close_calls == []
    assert execution.cancels == []


def test_evict_blocks_without_grant_or_when_generation_is_active() -> None:
    no_grant_execution = FakeExecution()
    no_grant = make_adapter(
        no_grant_execution,
    )
    no_grant_backend = make_backend()
    receipt = no_grant.evict(
        no_grant_backend,
        operation_id="handoff:no-grant",
        reclamation_baseline=make_reclamation_baseline(
            no_grant_execution, no_grant_backend, "handoff:no-grant:baseline:0"
        ),
    )
    assert receipt["state"] == "blocked"
    assert no_grant_execution.cancels == []

    active_execution = FakeExecution()
    active_admission = FakeAdmission(active_generations=1)
    active = make_adapter(
        active_execution,
        admission=active_admission,
        eviction_authorizer=lambda _backend, _op: {"authorized": True, "grant_id": "grant:active"},
    )
    active_backend = make_backend()
    receipt = active.evict(
        active_backend,
        operation_id="handoff:active",
        reclamation_baseline=make_reclamation_baseline(
            active_execution, active_backend, "handoff:active:baseline:0"
        ),
    )
    assert receipt["state"] == "blocked"
    assert active_execution.cancels == []


def test_process_exit_without_resource_receipt_stays_unknown() -> None:
    execution = FakeExecution()
    adapter = make_adapter(
        execution,
        eviction_authorizer=lambda _backend, _op: {"authorized": True, "grant_id": "grant:1"},
    )

    backend = make_backend()
    receipt = adapter.evict(
        backend,
        operation_id="handoff:unknown-resource",
        reclamation_baseline=make_reclamation_baseline(
            execution, backend, "handoff:unknown-resource:baseline:0"
        ),
    )

    assert receipt["state"] == "unknown"
    assert receipt["cleanup_verified"] is True
    assert receipt["resource_reclamation"]["state"] == "unknown"
    assert "evidence" not in receipt["resource_reclamation"]


def test_reopen_is_explicit_and_capabilities_are_typed() -> None:
    execution = FakeExecution()
    admission = FakeAdmission(closed=True, operation_id="handoff:reopen")
    adapter = make_adapter(
        execution,
        admission=admission,
    )

    receipt = adapter.reopen(make_backend(), operation_id="handoff:reopen")
    recovered = adapter.reopen(make_backend(), operation_id="handoff:reopen")

    assert receipt["state"] == "ready"
    assert receipt["admission_open"] is True
    assert recovered["state"] == "ready"
    assert recovered["open_receipt"]["reconciled"] is True
    assert admission.open_calls == [("handoff:reopen", 1)]
    assert all(isinstance(item, AdapterCapability) for item in adapter.capabilities())
    assert next(item for item in adapter.capabilities() if item.name == "explicit-grant-model-eviction").supported is False
    assert next(item for item in adapter.capabilities() if item.name == "explicit-grant-model-eviction").enforcement is None


def test_inspect_exposes_admission_epoch_and_unknown_when_controller_unreachable() -> None:
    admission = FakeAdmission(closed=True, operation_id="handoff:one")
    adapter = make_adapter(FakeExecution(), admission=admission)

    observed = adapter.inspect(make_backend())

    assert observed["admission_state"] == "closed"
    assert observed["admission_closed"] is True
    assert observed["admission_operation_id"] == "handoff:one"
    assert observed["admission_epoch"] == 1

    broken = make_adapter(
        FakeExecution(),
        admission_state_reader=lambda _backend: (_ for _ in ()).throw(OSError("offline")),
    ).inspect(make_backend())
    assert broken["admission_state"] == "unknown"
    assert broken["admission_closed"] is None


def test_evict_rejects_unverified_admission_operation_and_physical_active_slot() -> None:
    wrong_admission = FakeAdmission()
    wrong_execution = FakeExecution()
    wrong = make_adapter(
        wrong_execution,
        admission=wrong_admission,
        admission_state_reader=lambda _backend: {
            "backend_id": "llama-main",
            "closed": True,
            "operation_id": "handoff:someone-else",
            "epoch": 1,
            "active_generations": 0,
        },
        eviction_authorizer=lambda _backend, op: {"authorized": True, "grant_id": op},
        resource_reclamation_verifier=lambda _backend, _op, _baseline: {"state": "reclaimed", "evidence": ["ok"]},
    )
    wrong_backend = make_backend()
    wrong = wrong.evict(
        wrong_backend,
        operation_id="handoff:wrong",
        reclamation_baseline=make_reclamation_baseline(
            FakeExecution(), wrong_backend, "handoff:wrong:baseline:0"
        ),
    )
    assert wrong["state"] == "unknown"
    assert wrong["cleanup_verified"] is False

    slot_admission = FakeAdmission()
    slot_execution = FakeExecution()
    active_slot_adapter = make_adapter(
        slot_execution,
        admission=slot_admission,
        slots_document=[{"id": 0, "is_processing": True}],
        eviction_authorizer=lambda _backend, op: {"authorized": True, "grant_id": op},
    )
    slot_backend = make_backend()
    active_slot = active_slot_adapter.evict(
        slot_backend,
        operation_id="handoff:active-slot",
        reclamation_baseline=make_reclamation_baseline(
            slot_execution, slot_backend, "handoff:active-slot:baseline:0"
        ),
    )
    assert active_slot["state"] == "blocked"
    assert active_slot["cleanup_verified"] is False


def test_frameshift_admission_client_uses_scoped_routes_and_validates_receipts() -> None:
    responses = [
        {
            "closed": True,
            "backend_id": "llama/main",
            "operation_id": "handoff:one",
            "epoch": 7,
            "active_generations": 1,
            "active_generations_source": "frameshift-provider-leases-upper-bound",
        },
        {
            "backend_id": "llama/main",
            "closed": True,
            "operation_id": "handoff:one",
            "epoch": 7,
            "active_generations": 0,
            "active_generations_source": "frameshift-provider-leases-upper-bound",
        },
        {
            "opened": True,
            "backend_id": "llama/main",
            "operation_id": "handoff:one",
            "epoch": 7,
        },
    ]
    calls: list[dict[str, Any]] = []

    class Response:
        status = 200

        def __init__(self, document):
            self._body = io.BytesIO(json.dumps(document).encode())

        def read(self, size):
            return self._body.read(size)

        def close(self):
            self._body.close()

    def opener(request, *, timeout):
        calls.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "body": request.data,
                "timeout": timeout,
            }
        )
        return Response(responses.pop(0))

    client = FrameShiftBackendAdmissionClient("http://127.0.0.1:8991", "scheduler-secret", opener=opener)
    closed = client.close_backend_admission("llama/main", "handoff:one")
    state = client.backend_admission_state("llama/main")
    opened = client.open_backend_admission("llama/main", "handoff:one", state["epoch"])

    assert [call["method"] for call in calls] == ["POST", "GET", "POST"]
    assert calls[0]["url"].endswith("/v1/backends/llama%2Fmain/admission/close")
    assert calls[1]["url"].endswith("/v1/backends/llama%2Fmain/admission")
    assert calls[2]["url"].endswith("/v1/backends/llama%2Fmain/admission/open")
    assert calls[0]["authorization"] == "Bearer scheduler-secret"
    assert json.loads(calls[0]["body"]) == {"operation_id": "handoff:one"}
    assert json.loads(calls[2]["body"]) == {"operation_id": "handoff:one", "epoch": 7}
    assert closed["active_generations"] == 1
    assert state["active_generations"] == 0
    assert opened["opened"] is True


def test_frameshift_admission_client_rejects_non_loopback_and_header_injection() -> None:
    with pytest.raises(ValueError, match="loopback"):
        FrameShiftBackendAdmissionClient("http://example.com", "secret")
    with pytest.raises(ValueError, match="token"):
        FrameShiftBackendAdmissionClient("http://127.0.0.1", "secret\r\nX-Leak: yes")
