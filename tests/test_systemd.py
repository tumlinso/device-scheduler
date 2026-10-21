from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from device_scheduler_adapters.systemd import (
    ExecutionAdapterError,
    StartUncertain,
    SystemdExecutionAdapter,
    attempt_unit_name,
)
from device_scheduler_core.models import (
    AllocationMode,
    ExecutionContext,
    ExecutionHandle,
    ExecutionPurpose,
    JobSpec,
    Resource,
    ResourceAllocation,
)


class FakeSystemd:
    def __init__(self, cgroup_root: Path) -> None:
        self.cgroup_root = cgroup_root
        self.calls: list[list[str]] = []
        self.units: dict[str, dict[str, str]] = {}
        self.logs = "bounded test output"
        self.log_truncated = False

    def install(self, attempt_id: str) -> str:
        unit = attempt_unit_name(attempt_id)
        cgroup = f"/user.slice/{unit}"
        path = self.cgroup_root / cgroup.lstrip("/")
        path.mkdir(parents=True, exist_ok=True)
        (path / "cgroup.events").write_text("populated 1\n", encoding="ascii")
        self.units[unit] = {
            "Id": unit,
            "Description": f"Device Scheduler attempt {attempt_id}",
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "Result": "success",
            "ExecMainCode": "0",
            "ExecMainStatus": "0",
            "ControlGroup": cgroup,
            "InvocationID": f"invocation-{attempt_id}",
            "Restart": "no",
        }
        return unit

    def __call__(self, argv: list[str], *, timeout: float, max_output_bytes: int | None = None):
        self.calls.append(list(argv))
        if argv[0] == "systemd-run":
            unit = next(value.split("=", 1)[1] for value in argv if value.startswith("--unit="))
            description = next(value.split("=", 1)[1] for value in argv if value.startswith("--description="))
            attempt_id = description.removeprefix("Device Scheduler attempt ")
            self.install(attempt_id)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] == "systemctl" and "show" in argv:
            unit = argv[3]
            properties = self.units.get(unit, {"LoadState": "not-found"})
            return SimpleNamespace(
                returncode=0,
                stdout="".join(f"{key}={value}\n" for key, value in properties.items()),
                stderr="",
            )
        if argv[0] == "systemctl" and "stop" in argv:
            unit = argv[3]
            properties = self.units[unit]
            properties.update(
                ActiveState="inactive",
                SubState="dead",
                Result="success",
                ExecMainCode="1",
                ExecMainStatus="0",
            )
            cgroup = self.cgroup_root / properties["ControlGroup"].lstrip("/")
            (cgroup / "cgroup.events").write_text("populated 0\n", encoding="ascii")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] == "journalctl":
            return SimpleNamespace(
                returncode=0,
                stdout=self.logs,
                stderr="",
                truncated=self.log_truncated,
            )
        raise AssertionError(f"unexpected command: {argv}")


def test_start_uses_deterministic_transient_unit_and_limits(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(
        runner=fake,
        cgroup_root=tmp_path / "cgroup",
        profile_resolver=lambda profile_id: {
            "argv": ["python", "-c", "print('safe')"],
            "working_directory": "/workspace",
            "environment": {},
            "limits": {
                "cpu_quota_percent": 175,
                "memory_max_bytes": 4_000_000_000,
                "tasks_max": 64,
            },
            "profile_version": "v3",
        },
    )
    spec = JobSpec(owner_id="client:test", profile_id="profile:kernel")

    handle = adapter.start(spec, attempt_id="attempt:test", execution_root=tmp_path / "runs")

    assert handle.external_id == attempt_unit_name("attempt:test")
    assert handle.identity == "attempt:test"
    assert handle.metadata["invocation_id"] == "invocation-attempt:test"
    command = fake.calls[0]
    assert command[:5] == [
        "systemd-run",
        "--user",
        f"--unit={attempt_unit_name('attempt:test')}",
        "--description=Device Scheduler attempt attempt:test",
        "--service-type=exec",
    ]
    assert "--property=RemainAfterExit=yes" in command
    assert "--property=Restart=no" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=CPUQuota=175%" in command
    assert "--property=MemoryMax=4000000000" in command
    assert "--property=TasksMax=64" in command
    assert "--setenv=CUDA_VISIBLE_DEVICES=" in command
    assert "--property=RuntimeMaxSec=3600s" in command
    assert "--collect" not in command
    assert command[-4:] == ["--", "python", "-c", "print('safe')"]


def _execution_context(*, purpose: ExecutionPurpose = ExecutionPurpose.JOB) -> ExecutionContext:
    gpu = Resource(
        resource_id="gpu:lab-host:GPU-abc",
        host_id="lab-host",
        kind="gpu",
        capacities={"memory_bytes": 16_000_000_000},
        attributes={"uuid": "GPU-abc"},
    )
    cpu = Resource(
        resource_id="cpu:lab-host:0",
        host_id="lab-host",
        kind="cpu",
        capacities={"cores": 8},
    )
    memory = Resource(
        resource_id="memory:lab-host:0",
        host_id="lab-host",
        kind="host_memory",
        capacities={"bytes": 64_000_000_000},
    )
    return ExecutionContext(
        allocations=(
            ResourceAllocation(
                gpu.resource_id,
                {"memory_bytes": 8_000_000_000},
                AllocationMode.SHARED,
            ),
            ResourceAllocation(cpu.resource_id, {"cores": 2}),
            ResourceAllocation(memory.resource_id, {"bytes": 12_000_000_000}),
        ),
        resources=(gpu, cpu, memory),
        claim_id="lease:1",
        fencing_token=1,
        purpose=purpose,
    )


def test_start_binds_gpu_visibility_and_enforces_granted_host_limits(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(
        runner=fake,
        cgroup_root=tmp_path / "cgroup",
        host_id="lab-host",
        profile_resolver=lambda _profile: {
            "argv": ["python", "-c", "pass"],
            "working_directory": "/workspace",
            "environment": {"CUDA_VISIBLE_DEVICES": "GPU-abc"},
            "limits": {"cpu_quota_percent": 150, "memory_max_bytes": 9_000_000_000},
        },
    )
    spec = JobSpec(owner_id="client:test", profile_id="profile:kernel", timeout_seconds=17)

    adapter.start(
        spec,
        attempt_id="attempt:granted-gpu",
        execution_root=tmp_path / "runs",
        execution_context=_execution_context(),
    )

    command = fake.calls[0]
    assert "--setenv=CUDA_VISIBLE_DEVICES=GPU-abc" in command
    # The profile's caps are stricter than the grant (200% CPU, 12 GB host RAM).
    assert "--property=CPUQuota=150%" in command
    assert "--property=MemoryMax=9000000000" in command
    assert "--property=RuntimeMaxSec=17s" in command
    # GPU memory is admission accounting only, never turned into a fake cgroup limit.
    assert not any("MemoryMax=8000000000" in item for item in command)


def test_grant_limits_cap_a_more_permissive_profile(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(
        runner=fake,
        cgroup_root=tmp_path / "cgroup",
        host_id="lab-host",
        profile_resolver=lambda _profile: {
            "argv": ["true"],
            "working_directory": "/tmp",
            "limits": {"cpu_quota_percent": 1000, "memory_max_bytes": 20_000_000_000},
        },
    )

    adapter.start(
        JobSpec(owner_id="client:test", profile_id="profile:loose"),
        attempt_id="attempt:grant-caps-profile",
        execution_root=tmp_path / "runs",
        execution_context=_execution_context(),
    )

    command = fake.calls[0]
    assert "--property=CPUQuota=200%" in command
    assert "--property=MemoryMax=12000000000" in command


def test_start_rejects_conflicting_or_unbound_gpu_visibility_before_launch(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(
        runner=fake,
        cgroup_root=tmp_path / "cgroup",
        host_id="lab-host",
        profile_resolver=lambda _profile: {
            "argv": ["true"],
            "working_directory": "/tmp",
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        },
    )

    with pytest.raises(ExecutionAdapterError, match="conflicts"):
        adapter.start(
            JobSpec(owner_id="client:test", profile_id="profile:kernel"),
            attempt_id="attempt:gpu-conflict",
            execution_root=tmp_path / "runs",
            execution_context=_execution_context(),
        )
    assert not any(call[0] == "systemd-run" for call in fake.calls)


def test_backend_purpose_uses_grant_binding_without_finite_job_runtime(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup", host_id="lab-host")

    adapter.start(
        JobSpec(owner_id="backend:test", argv=("true",), working_directory="/tmp"),
        attempt_id="attempt:backend-runtime",
        execution_root=tmp_path / "runs",
        execution_context=_execution_context(purpose=ExecutionPurpose.BACKEND),
    )

    command = fake.calls[0]
    assert "--setenv=CUDA_VISIBLE_DEVICES=GPU-abc" in command
    assert not any(item.startswith("--property=RuntimeMaxSec=") for item in command)


def test_start_rejects_remote_resource_descriptors_before_launch(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup", host_id="lab-host")
    remote_gpu = Resource(
        resource_id="gpu:remote:GPU-abc",
        host_id="remote",
        kind="gpu",
        capacities={"memory_bytes": 16_000_000_000},
        attributes={"uuid": "GPU-abc"},
    )
    remote_context = ExecutionContext(
        allocations=(ResourceAllocation(remote_gpu.resource_id, {"memory_bytes": 8_000_000_000}),),
        resources=(remote_gpu,),
        claim_id="lease:remote",
        fencing_token=2,
    )

    with pytest.raises(ExecutionAdapterError, match="another host"):
        adapter.start(
            JobSpec(owner_id="client:test", argv=("true",), working_directory="/tmp"),
            attempt_id="attempt:remote-gpu",
            execution_root=tmp_path / "runs",
            execution_context=remote_context,
        )
    assert not fake.calls


def test_inspection_survives_daemon_restart_and_bounds_journal_output(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    unit = fake.install("attempt-1")
    fake.logs = "0123456789abcdef"
    fake.log_truncated = True
    adapter = SystemdExecutionAdapter(
        runner=fake,
        cgroup_root=tmp_path / "cgroup",
        max_output_bytes=8,
    )
    handle = adapter.resolve(attempt_id="attempt-1")
    assert handle is not None

    receipt = adapter.inspect(handle.to_dict())

    assert receipt["state"] == "running"
    assert receipt["cleanup_verified"] is False
    assert len(receipt["logs"].encode("utf-8")) <= 8
    assert receipt["logs_truncated"] is True
    assert receipt["invocation_id"] == fake.units[unit]["InvocationID"]
    journal_command = next(call for call in fake.calls if call[0] == "journalctl")
    assert "--invocation" not in " ".join(journal_command)
    assert f"_SYSTEMD_INVOCATION_ID={fake.units[unit]['InvocationID']}" in journal_command
    assert adapter.resolve(attempt_id="missing") is None


def test_cancel_stops_only_verified_unit_and_requires_empty_cgroup(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    fake.install("attempt-stop")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-stop")
    assert handle is not None

    receipt = adapter.cancel(handle, operation_id="operation:cancel-1", timeout_seconds=4)

    assert receipt["operation_id"] == "operation:cancel-1"
    assert receipt["state"] == "succeeded"
    assert receipt["cleanup_verified"] is True
    assert [call for call in fake.calls if call[:3] == ["systemctl", "--user", "stop"]]
    assert all(call[0] not in {"kill", "pkill"} for call in fake.calls)


def test_cancel_accepts_unloaded_unit_only_after_acknowledged_stop_and_removed_cgroup(tmp_path: Path) -> None:
    class UnloadingFakeSystemd(FakeSystemd):
        def __call__(self, argv, *, timeout, max_output_bytes=None):
            result = super().__call__(argv, timeout=timeout, max_output_bytes=max_output_bytes)
            if argv[0] == "systemctl" and "stop" in argv:
                unit = argv[3]
                properties = self.units.pop(unit)
                cgroup = self.cgroup_root / properties["ControlGroup"].lstrip("/")
                (cgroup / "cgroup.events").unlink()
                cgroup.rmdir()
            return result

    fake = UnloadingFakeSystemd(tmp_path / "cgroup")
    fake.install("attempt-unloaded-after-stop")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-unloaded-after-stop")
    assert handle is not None

    receipt = adapter.cancel(
        handle,
        operation_id="operation:unloaded-after-stop",
        timeout_seconds=2,
    )

    assert receipt["cleanup_verified"] is True
    assert receipt["state"] == "unknown"
    assert receipt["reason"] == "unit was unloaded after systemd acknowledged stop"


def test_cancel_uses_persisted_cgroup_when_terminal_unit_clears_control_group(tmp_path: Path) -> None:
    class ClearingFakeSystemd(FakeSystemd):
        def __call__(self, argv, *, timeout, max_output_bytes=None):
            result = super().__call__(argv, timeout=timeout, max_output_bytes=max_output_bytes)
            if argv[0] == "systemctl" and "stop" in argv:
                self.units[argv[3]]["ControlGroup"] = ""
            return result

    fake = ClearingFakeSystemd(tmp_path / "cgroup")
    fake.install("attempt-cleared-cgroup-property")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-cleared-cgroup-property")
    assert handle is not None
    assert handle.metadata["control_group"] == f"/user.slice/{handle.external_id}"

    receipt = adapter.cancel(
        handle,
        operation_id="operation:cleared-cgroup-property",
        timeout_seconds=2,
    )

    assert receipt["cleanup_verified"] is True
    assert receipt["state"] == "succeeded"
    assert receipt["evidence"]["control_group"] == handle.metadata["control_group"]


def test_invalid_cpu_quota_is_rejected_before_systemd_run(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    adapter = SystemdExecutionAdapter(
        runner=fake,
        profile_resolver=lambda _profile: {
            "argv": ["true"],
            "working_directory": "/tmp",
            "limits": {"cpu_quota_percent": float("nan")},
        },
    )

    with pytest.raises(ValueError, match="finite positive"):
        adapter.start(
            JobSpec(owner_id="client:test", profile_id="profile:invalid"),
            attempt_id="attempt-invalid-quota",
            execution_root=tmp_path / "runs",
        )

    assert not fake.calls


def test_invocation_mismatch_never_signals_a_reused_unit(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    fake.install("attempt-reused")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-reused")
    assert handle is not None
    properties = fake.units[handle.external_id]
    properties["InvocationID"] = "new-invocation"

    receipt = adapter.cancel(handle, operation_id="operation:cancel-reused", timeout_seconds=1)

    assert receipt["state"] == "unknown"
    assert receipt["cleanup_verified"] is False
    assert not any("stop" in call for call in fake.calls)


def test_cancel_requires_a_persisted_invocation_identity(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    fake.install("attempt-unbound")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-unbound")
    assert handle is not None
    unbound = handle.to_dict()
    unbound["metadata"].pop("invocation_id")

    receipt = adapter.cancel(unbound, operation_id="operation:unbound", timeout_seconds=1)

    assert receipt["state"] == "unknown"
    assert "not bound" in receipt["reason"]
    assert not any("stop" in call for call in fake.calls)


def test_missing_cgroup_events_and_ambiguous_launch_remain_unknown(tmp_path: Path) -> None:
    fake = FakeSystemd(tmp_path / "cgroup")
    unit = fake.install("attempt-no-events")
    properties = fake.units[unit]
    cgroup = tmp_path / "cgroup" / properties["ControlGroup"].lstrip("/")
    (cgroup / "cgroup.events").unlink()
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=tmp_path / "cgroup")
    handle = adapter.resolve(attempt_id="attempt-no-events")
    assert handle is not None
    receipt = adapter.inspect(handle)
    assert receipt["cleanup_verified"] is False

    class TimedOutRunner(FakeSystemd):
        def __call__(self, argv, *, timeout, max_output_bytes=None):
            if argv[0] == "systemd-run":
                self.calls.append(list(argv))
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout=timeout, max_output_bytes=max_output_bytes)

    uncertain = SystemdExecutionAdapter(runner=TimedOutRunner(tmp_path / "other-cgroup"))
    with pytest.raises(StartUncertain):
        uncertain.start(
            JobSpec(owner_id="client:test", argv=("true",), working_directory="/tmp"),
            attempt_id="attempt-uncertain",
            execution_root=tmp_path / "runs",
        )


def test_managed_process_ids_require_verified_invocation_and_read_complete_cgroup_tree(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids\n", encoding="ascii")
    fake = FakeSystemd(root)
    unit = fake.install("attempt-pid-ownership")
    unit_group = root / fake.units[unit]["ControlGroup"].lstrip("/")
    (unit_group / "cgroup.procs").write_text("123\n", encoding="ascii")
    child = unit_group / "child.scope"
    child.mkdir()
    (child / "cgroup.procs").write_text("456\n", encoding="ascii")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=root)
    handle = adapter.resolve(attempt_id="attempt-pid-ownership")
    assert handle is not None

    observed = adapter.managed_process_ids([handle])

    assert observed["state"] == "observed"
    assert observed["process_ids"] == [123, 456]
    assert observed["scope"] == "invocation-verified-systemd-unit-cgroups"
    assert observed["attempts"][0]["invocation_id"] == "invocation-attempt-pid-ownership"


def test_managed_process_ids_fail_closed_when_invocation_changes(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory pids\n", encoding="ascii")
    fake = FakeSystemd(root)
    unit = fake.install("attempt-pid-reused")
    unit_group = root / fake.units[unit]["ControlGroup"].lstrip("/")
    (unit_group / "cgroup.procs").write_text("999\n", encoding="ascii")
    adapter = SystemdExecutionAdapter(runner=fake, cgroup_root=root)
    handle = adapter.resolve(attempt_id="attempt-pid-reused")
    assert handle is not None
    fake.units[unit]["InvocationID"] = "different-invocation"

    observed = adapter.managed_process_ids([handle])

    assert observed["state"] == "unknown"
    assert observed["process_ids"] == []
    assert observed["attempts"][0]["process_ids"] == []
