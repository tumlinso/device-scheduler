from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from device_scheduler_adapters.nvidia import NvidiaAdapter
from device_scheduler_core.models import (
    AllocationMode,
    CapabilityPrecondition,
    EnforcementLevel,
    ExecutionContext,
    ExecutionPurpose,
    Resource,
    ResourceAllocation,
)


GPU_OUTPUT = """0, GPU-a, NVIDIA L40S, 0000:01:00.0, 46080, 45000, 5
1, GPU-b, NVIDIA L40S, 0000:41:00.0, 46080, 44000, 0
2, GPU-c, NVIDIA L40S, 0000:81:00.0, 46080, 43000, 0
"""
GPU_OUTPUT_TWO = """0, GPU-a, NVIDIA L40S, 0000:01:00.0, 46080, 45000, 5
1, GPU-b, NVIDIA L40S, 0000:41:00.0, 46080, 44000, 0
"""

TOPOLOGY_CONNECTED = """        GPU0 GPU1 GPU2
GPU0     X    NV2  PIX
GPU1     NV2  X    PXB
GPU2     PIX  PXB  X
"""

MIB = 1024 * 1024
GPU_A_ID = "gpu:lab-host:GPU-a"
GPU_A_IDLE = "0, GPU-a, NVIDIA L40S, 0000:01:00.0, 46080, 45000, 0\n"


class FakeNvidia:
    def __init__(
        self,
        *,
        gpu_output: str = GPU_OUTPUT,
        process_outputs: list[str] | None = None,
        topology: str = TOPOLOGY_CONNECTED,
        fail_commands: set[str] | None = None,
    ) -> None:
        self.gpu_output = gpu_output
        self.process_outputs = deque(process_outputs or ["No running processes found\n"])
        self.topology = topology
        self.fail_commands = fail_commands or set()
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], *, timeout: float):
        self.calls.append(list(argv))
        if any(token in self.fail_commands for token in argv):
            return SimpleNamespace(returncode=1, stdout="", stderr="telemetry unavailable")
        if len(argv) > 1 and argv[1].startswith("--query-gpu="):
            return SimpleNamespace(returncode=0, stdout=self.gpu_output, stderr="")
        if len(argv) > 1 and argv[1].startswith("--query-compute-apps="):
            output = self.process_outputs.popleft() if self.process_outputs else "No running processes found\n"
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        if argv[1:] == ["topo", "-m"]:
            return SimpleNamespace(returncode=0, stdout=self.topology, stderr="")
        raise AssertionError(f"unexpected nvidia-smi invocation: {argv}")


def make_adapter(fake: FakeNvidia, *, clock=None, sleeper=None) -> NvidiaAdapter:
    return NvidiaAdapter(
        runner=fake,
        hostname=lambda: "lab-host",
        clock=clock or (lambda: 0.0),
        sleeper=sleeper or (lambda _seconds: None),
    )


def make_gpu_resource(resource_id: str = GPU_A_ID, *, host_id: str = "lab-host", uuid: str = "GPU-a") -> Resource:
    return Resource(
        resource_id=resource_id,
        host_id=host_id,
        kind="gpu",
        capacities={"memory_bytes": 46080 * MIB},
        attributes={"uuid": uuid},
    )


def make_execution_context(*, resources: tuple[Resource, ...] | None = None) -> ExecutionContext:
    selected = resources or (make_gpu_resource(),)
    allocations = tuple(
        ResourceAllocation(resource.resource_id, {"memory_bytes": 8 * MIB}, AllocationMode.SHARED)
        for resource in selected
    )
    return ExecutionContext(
        allocations=allocations,
        resources=selected,
        claim_id="lease:gpu-job",
        fencing_token=7,
        purpose=ExecutionPurpose.JOB,
    )


def make_external_lease(*, fencing_token: int, mode: AllocationMode = AllocationMode.EXCLUSIVE) -> dict[str, object]:
    return {
        "lease_id": "lease:external-gpu",
        "fencing_token": fencing_token,
        "allocations": (
            ResourceAllocation(GPU_A_ID, {"memory_bytes": 8 * MIB}, mode),
        ),
    }


def test_discovery_emits_gpu_resources_and_closed_topology_isolation_resource() -> None:
    fake = FakeNvidia()
    adapter = make_adapter(fake)

    snapshot = adapter.snapshot()

    assert snapshot["gpu_inventory"]["state"] == "observed"
    assert snapshot["process_telemetry"]["state"] == "observed"
    assert snapshot["isolation_domain_support"]["state"] == "observed"
    resources = snapshot["resources"]
    gpu_resources = [resource for resource in resources if resource.kind == "gpu"]
    isolation_resources = [resource for resource in resources if resource.kind == "isolation"]
    assert len(gpu_resources) == 3
    assert len(isolation_resources) == 1
    assert all(isinstance(resource, Resource) for resource in resources)
    assert gpu_resources[0].resource_id == "gpu:lab-host:GPU-a"
    assert gpu_resources[0].capacities["memory_bytes"] == 46080 * 1024 * 1024
    assert gpu_resources[0].attributes["memory_enforcement"] == "accounting_only"
    assert gpu_resources[0].conflict_domains == frozenset({"gpu-exclusive:lab-host:GPU-a"})

    isolation = isolation_resources[0]
    assert isolation.resource_id.startswith("host:lab-host:domain:gpu-topology:")
    assert isolation.capacities == {"slots": 1.0}
    assert isolation.attributes["member_resource_ids"] == (
        "gpu:lab-host:GPU-a",
        "gpu:lab-host:GPU-b",
        "gpu:lab-host:GPU-c",
    )
    assert isolation.attributes["closure"] == "complete-observed-gpu-inventory"
    assert isolation.attributes["contention_established"] is False
    assert isolation.attributes["enforcement"] == "scheduler-admission-only"

    capability = next(item for item in adapter.capabilities() if item.name == "gpu-memory-budget")
    assert capability.enforcement is EnforcementLevel.ADVISORY
    assert any(argv[0] == "nvidia-smi" for argv in fake.calls)


def test_partial_topology_does_not_create_an_isolation_domain_or_false_gpu_conflict() -> None:
    incomplete = """        GPU0 GPU1 GPU2
GPU0     X    NV2  PIX
"""
    adapter = make_adapter(FakeNvidia(topology=incomplete))

    snapshot = adapter.snapshot()

    assert snapshot["isolation_domain_support"]["state"] == "unsupported"
    assert "matrix_incomplete" in snapshot["isolation_domain_support"]["reason"]
    assert not [resource for resource in snapshot["resources"] if resource.kind == "isolation"]
    gpu_resources = [resource for resource in snapshot["resources"] if resource.kind == "gpu"]
    assert all(len(resource.conflict_domains) == 1 for resource in gpu_resources)
    assert all(resource.topology["topology_state"] == "partial" for resource in gpu_resources)


def test_complete_matrix_with_no_nvlink_or_pcie_paths_has_no_fake_domain() -> None:
    topology = """        GPU0 GPU1
GPU0     X    SYS
GPU1     SYS  X
"""
    snapshot = make_adapter(FakeNvidia(gpu_output=GPU_OUTPUT_TWO, topology=topology)).snapshot()

    assert snapshot["isolation_domain_support"]["state"] == "observed"
    assert snapshot["isolation_domain_support"]["domains"] == []
    assert not [resource for resource in snapshot["resources"] if resource.kind == "isolation"]
    assert snapshot["topology"]["links"][0]["type"] == "sys"


def test_unavailable_memory_is_omitted_instead_of_reported_as_zero() -> None:
    output = "0, GPU-a, NVIDIA L40S, 0000:01:00.0, N/A, N/A, N/A\n"
    topology = "        GPU0\nGPU0     X\n"
    snapshot = make_adapter(FakeNvidia(gpu_output=output, topology=topology)).snapshot()
    gpu = next(resource for resource in snapshot["resources"] if resource.kind == "gpu")

    assert gpu.capacities == {}
    assert gpu.attributes["memory_free_bytes_observation"] is None
    assert gpu.attributes["utilization_percent_observation"] is None


def test_snapshot_exposes_fresh_per_gpu_process_rows_and_unknown_identity_blocks_attribution() -> None:
    known = make_adapter(
        FakeNvidia(process_outputs=["GPU-a, 4321, python, 64\n"])
    ).snapshot()
    gpu_a = next(resource for resource in known["resources"] if resource.resource_id.endswith("GPU-a"))
    assert gpu_a.attributes["process_count_observation"] == 1
    assert gpu_a.attributes["processes_observation"][0]["pid"] == 4321
    assert gpu_a.attributes["inventory_observed_at"] == known["inventory_observed_at"]
    assert gpu_a.attributes["process_observed_at"] == known["process_telemetry"]["observed_at"]

    unknown = make_adapter(
        FakeNvidia(process_outputs=["N/A, 9876, unknown-client, 128\n"])
    ).snapshot()
    assert unknown["process_telemetry"]["state"] == "partial"
    assert all(resource.attributes["process_telemetry"] == "unknown" for resource in unknown["resources"] if resource.kind == "gpu")
    assert all(resource.attributes["process_count_observation"] is None for resource in unknown["resources"] if resource.kind == "gpu")
    assert all(
        resource.attributes["processes_observation"][0]["identity_unknown"]
        for resource in unknown["resources"]
        if resource.kind == "gpu"
    )


@pytest.mark.parametrize("pid", ["0", "12.5", "N/A"])
def test_invalid_compute_process_pid_is_not_accepted_as_owned_identity(pid: str) -> None:
    snapshot = make_adapter(
        FakeNvidia(process_outputs=[f"GPU-a, {pid}, python, 64\n"])
    ).snapshot()

    assert snapshot["process_telemetry"]["state"] == "partial"
    assert snapshot["process_telemetry"]["processes"][0]["identity_unknown"] is True


def test_inventory_failure_snapshot_is_unknown_even_though_discover_is_empty() -> None:
    adapter = make_adapter(FakeNvidia(fail_commands={"--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu"}))

    snapshot = adapter.snapshot()

    assert snapshot["state"] == "unknown"
    assert snapshot["gpu_inventory"]["state"] == "unknown"
    assert snapshot["resources"] == []
    assert adapter.discover() == []


def test_process_telemetry_failure_is_unknown_not_quiescent() -> None:
    adapter = make_adapter(FakeNvidia(fail_commands={"--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"}))

    observation = adapter.observe_processes("GPU-a")

    assert observation["state"] == "unknown"
    assert observation["processes"] is None
    assert observation["reason"].startswith("nvidia_smi_failed:")


def test_quiescence_requires_consecutive_successful_empty_samples() -> None:
    samples = [
        "No running processes found\n",
        "GPU-a, 123, python, 50\n",
        "No running processes found\n",
        "No running processes found\n",
    ]
    fake = FakeNvidia(process_outputs=samples)
    now = [0.0]
    adapter = make_adapter(fake, clock=lambda: now[0], sleeper=lambda delay: now.__setitem__(0, now[0] + delay))

    result = adapter.wait_quiescent(
        "GPU-a", consecutive_samples=2, interval_seconds=0.1, timeout_seconds=1.0
    )

    assert result["state"] == "quiescent"
    assert result["consecutive_empty_samples"] == 2
    assert len(result["samples"]) == 4
    assert "does not prove hostile-process isolation" in result["limitations"][1]


def test_memory_reclamation_requires_process_absence_and_observed_free_floor() -> None:
    adapter = make_adapter(FakeNvidia())
    result = adapter.wait_memory_reclaimed("GPU-a", 44_000 * 1024 * 1024, consecutive_samples=1)

    assert result["state"] == "reclaimed"
    assert result["minimum_free_bytes"] == 44_000 * 1024 * 1024
    assert result["memory_enforcement"] == "accounting-only-observation"


@pytest.mark.parametrize("value", [-1, 1.5, True, "100"])
def test_memory_reclamation_rejects_invalid_free_memory_threshold(value: object) -> None:
    adapter = make_adapter(FakeNvidia())
    with pytest.raises(ValueError):
        adapter.wait_memory_reclaimed("GPU-a", value)  # type: ignore[arg-type]


def test_external_gpu_lease_captures_exact_exclusive_baseline_with_configured_headroom() -> None:
    fake = FakeNvidia(gpu_output=GPU_A_IDLE)
    adapter = NvidiaAdapter(
        runner=fake,
        hostname=lambda: "lab-host",
        external_lease_headroom_bytes=4 * MIB,
        external_lease_samples=2,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=1,
        sleeper=lambda _seconds: None,
    )

    receipt = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11),
        (make_gpu_resource(),),
        operation_id="lease:external:baseline:11",
    )

    assert receipt["state"] == "observed"
    assert receipt["verified"] is True
    assert receipt["lease_id"] == "lease:external-gpu"
    assert receipt["fencing_token"] == 11
    assert receipt["resource_ids"] == [GPU_A_ID]
    assert receipt["baseline"]["headroom_bytes"] == 4 * MIB
    gpu = receipt["baseline"]["gpus"][0]
    assert gpu["memory_budget_bytes"] == 8 * MIB
    assert gpu["required_free_bytes"] == 12 * MIB
    assert len(receipt["evidence"]["samples"]) == 2
    assert all("nvidia-smi" in call[0] for call in fake.calls)
    assert not any("kill" in token or "reset" in token for call in fake.calls for token in call)


def test_external_gpu_lease_refuses_shared_or_mismatched_bundles_before_observation() -> None:
    fake = FakeNvidia(gpu_output=GPU_A_IDLE)
    adapter = make_adapter(fake)

    shared = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11, mode=AllocationMode.SHARED),
        (make_gpu_resource(),),
        operation_id="lease:external:shared",
    )
    forged = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11),
        (make_gpu_resource(resource_id="gpu:lab-host:GPU-other"),),
        operation_id="lease:external:forged",
    )

    assert shared["state"] == "unsupported"
    assert forged["state"] == "blocked"
    assert fake.calls == []


def test_external_gpu_lease_requires_fresh_idle_memory_and_process_observations() -> None:
    busy = NvidiaAdapter(
        runner=FakeNvidia(),
        hostname=lambda: "lab-host",
        external_lease_samples=1,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=0,
    ).capture_lease_baseline(
        make_external_lease(fencing_token=11), (make_gpu_resource(),), operation_id="lease:busy"
    )
    unknown = NvidiaAdapter(
        runner=FakeNvidia(
            gpu_output=GPU_A_IDLE,
            process_outputs=["N/A, 99, unknown, 16\n"],
        ),
        hostname=lambda: "lab-host",
        external_lease_samples=1,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=0,
    ).capture_lease_baseline(
        make_external_lease(fencing_token=11), (make_gpu_resource(),), operation_id="lease:unknown"
    )
    insufficient = NvidiaAdapter(
        runner=FakeNvidia(
            gpu_output="0, GPU-a, NVIDIA L40S, 0000:01:00.0, 46080, 10, 0\n"
        ),
        hostname=lambda: "lab-host",
        external_lease_samples=1,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=0,
    ).capture_lease_baseline(
        make_external_lease(fencing_token=11), (make_gpu_resource(),), operation_id="lease:low-memory"
    )

    assert busy["state"] == "blocked"
    assert unknown["state"] == "unknown"
    assert insufficient["state"] == "blocked"
    assert busy["verified"] is False
    assert unknown["verified"] is False


def test_external_gpu_lease_release_requires_exact_baseline_rotated_fence_and_repeated_samples() -> None:
    adapter = NvidiaAdapter(
        runner=FakeNvidia(gpu_output=GPU_A_IDLE),
        hostname=lambda: "lab-host",
        external_lease_headroom_bytes=4 * MIB,
        external_lease_samples=2,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=1,
        sleeper=lambda _seconds: None,
    )
    baseline = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11),
        (make_gpu_resource(),),
        operation_id="lease:external:baseline:11",
    )
    released = adapter.inspect_lease_release(
        make_external_lease(fencing_token=12),
        baseline,
        (make_gpu_resource(),),
        operation_id="lease:external:release:12:sample-a",
    )
    same_fence = adapter.inspect_lease_release(
        make_external_lease(fencing_token=11),
        baseline,
        (make_gpu_resource(),),
        operation_id="lease:external:release:11",
    )
    mismatched = adapter.inspect_lease_release(
        make_external_lease(fencing_token=12),
        baseline,
        (make_gpu_resource(resource_id="gpu:lab-host:GPU-other"),),
        operation_id="lease:external:release:mismatch",
    )

    assert released["state"] == "reclaimed"
    assert released["verified"] is True
    assert released["baseline_fencing_token"] == 11
    assert released["fencing_token"] == 12
    assert released["evidence"]["sample_count"] == 2
    assert same_fence["state"] == "blocked"
    assert mismatched["state"] == "blocked"


def test_unknown_external_release_can_be_resampled_without_physical_side_effects() -> None:
    fake = FakeNvidia(gpu_output=GPU_A_IDLE)
    adapter = NvidiaAdapter(
        runner=fake,
        hostname=lambda: "lab-host",
        external_lease_headroom_bytes=4 * MIB,
        external_lease_samples=1,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=0,
    )
    baseline = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11),
        (make_gpu_resource(),),
        operation_id="lease:external:baseline:11",
    )
    first = adapter.inspect_lease_release(
        make_external_lease(fencing_token=12), baseline, (make_gpu_resource(),), operation_id="observation:one"
    )
    second = adapter.inspect_lease_release(
        make_external_lease(fencing_token=12), baseline, (make_gpu_resource(),), operation_id="observation:two"
    )

    assert first["verified"] is True and second["verified"] is True
    assert first["observation_id"] != second["observation_id"]
    assert all(call[0] == "nvidia-smi" for call in fake.calls)


def test_external_gpu_release_stays_blocked_until_persisted_memory_floor_returns() -> None:
    fake = FakeNvidia(gpu_output=GPU_A_IDLE)
    adapter = NvidiaAdapter(
        runner=fake,
        hostname=lambda: "lab-host",
        external_lease_headroom_bytes=4 * MIB,
        external_lease_samples=1,
        external_lease_interval_seconds=0,
        external_lease_timeout_seconds=0,
    )
    baseline = adapter.capture_lease_baseline(
        make_external_lease(fencing_token=11),
        (make_gpu_resource(),),
        operation_id="lease:external:baseline:11",
    )
    fake.gpu_output = "0, GPU-a, NVIDIA L40S, 0000:01:00.0, 46080, 44000, 0\n"

    receipt = adapter.inspect_lease_release(
        make_external_lease(fencing_token=12),
        baseline,
        (make_gpu_resource(),),
        operation_id="lease:external:release:12:low-free-memory",
    )

    assert receipt["state"] == "blocked"
    assert receipt["verified"] is False
    assert receipt["evidence"]["samples"][0]["gpus"][0]["required_free_bytes"] > 44000 * MIB


def test_gpu_quiescence_precondition_is_bound_to_fenced_grant_and_repeated_observations() -> None:
    fake = FakeNvidia(gpu_output=GPU_A_IDLE)
    now = [0.0]
    adapter = make_adapter(
        fake,
        clock=lambda: now[0],
        sleeper=lambda delay: now.__setitem__(0, now[0] + delay),
    )
    precondition = CapabilityPrecondition(
        "gpu.quiescence",
        {
            "consecutive_samples": 2,
            "interval_seconds": 0.1,
            "timeout_seconds": 0.2,
        },
    )

    receipt = adapter.check_execution_preconditions(
        make_execution_context(), (precondition,), operation_id="attempt:quiescence:before-start"
    )

    assert receipt["state"] == "verified"
    assert receipt["verified"] is True
    assert receipt["claim_id"] == "lease:gpu-job"
    assert receipt["fencing_token"] == 7
    check = receipt["preconditions"][0]
    assert check["name"] == "gpu.quiescence"
    assert check["state"] == "verified"
    assert check["resource_ids"] == [GPU_A_ID]
    assert check["evidence"]["observation"]["consecutive_samples"] == 2
    assert len(fake.calls) == 4


def test_gpu_quiescence_precondition_fails_closed_for_bad_context_threshold_and_unknown_telemetry() -> None:
    no_gpu = ExecutionContext(
        claim_id="lease:cpu-job",
        fencing_token=1,
        purpose=ExecutionPurpose.JOB,
    )
    unsupported = make_adapter(FakeNvidia()).check_execution_preconditions(
        no_gpu,
        (CapabilityPrecondition("gpu.quiescence"),),
        operation_id="cpu-only",
    )
    weak_threshold = make_adapter(FakeNvidia(gpu_output=GPU_A_IDLE)).check_execution_preconditions(
        make_execution_context(),
        (CapabilityPrecondition("gpu.quiescence", {"minimum_free_bytes": 1}),),
        operation_id="weak-threshold",
    )
    unknown_adapter = make_adapter(
        FakeNvidia(
            gpu_output=GPU_A_IDLE,
            process_outputs=["N/A, 123, unknown, 1\n"],
        )
    )
    unknown = unknown_adapter.check_execution_preconditions(
        make_execution_context(),
        (CapabilityPrecondition("gpu.quiescence", {"consecutive_samples": 1}),),
        operation_id="unknown-process-identity",
    )
    extras = make_adapter(FakeNvidia()).check_execution_preconditions(
        make_execution_context(),
        (CapabilityPrecondition("gpu.quiescence", {"pretend_verified": True}),),
        operation_id="unsupported-parameter",
    )

    assert unsupported["state"] == "unsupported"
    assert weak_threshold["state"] == "blocked"
    assert unknown["state"] == "unknown"
    assert extras["state"] == "unsupported"
    assert all(len(result["preconditions"]) == 1 for result in (unsupported, weak_threshold, unknown, extras))
    assert unknown["preconditions"][0]["state"] == "unknown"


def test_gpu_quiescence_precondition_blocks_busy_gpu_before_execution() -> None:
    adapter = make_adapter(FakeNvidia())  # GPU-a reports 5% utilization.
    receipt = adapter.check_execution_preconditions(
        make_execution_context(),
        (
            CapabilityPrecondition(
                "gpu.quiescence",
                {
                    "consecutive_samples": 1,
                    "interval_seconds": 0,
                    "timeout_seconds": 0,
                    "max_utilization_percent": 0,
                },
            ),
        ),
        operation_id="attempt:busy-gpu",
    )

    assert receipt["state"] == "blocked"
    assert receipt["verified"] is False
    assert receipt["preconditions"][0]["state"] == "blocked"
