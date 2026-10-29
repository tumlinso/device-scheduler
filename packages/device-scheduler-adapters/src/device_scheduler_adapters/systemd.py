"""Linux process supervision through transient systemd user service units.

The scheduler persists its operation intent before calling this adapter.  Unit
names are deterministic from attempt IDs so a daemon restart can inspect a
launch whose receipt was not yet committed.  This adapter never signals a PID;
stop requests are scoped to a named unit after its invocation has been checked.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import selectors
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from device_scheduler_core.models import (
    AdapterCapability,
    EnforcementLevel,
    ExecutionContext,
    ExecutionHandle,
    ExecutionPurpose,
    JobSpec,
    Resource,
)


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNIT_PREFIX = "device-scheduler-attempt-"
_GPU_KINDS = {"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"}
_CPU_KINDS = {"cpu"}
_MEMORY_KINDS = {"memory", "host_memory", "ram"}
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class ExecutionAdapterError(RuntimeError):
    """An execution operation failed or could not be verified."""

    def __init__(self, message: str, *, safe_diagnostic: str | None = None) -> None:
        super().__init__(message)
        self.safe_diagnostic = safe_diagnostic


class StartUncertain(ExecutionAdapterError):
    """The start command may have had an effect; inspect this stable handle."""

    def __init__(self, message: str, handle: Mapping[str, Any]):
        super().__init__(message, safe_diagnostic=message)
        self.execution_handle = dict(handle)


class StartNotSubmitted(ExecutionAdapterError):
    """Local validation proved the execution request was never submitted."""

    execution_not_submitted = True


CommandResult = Any
CommandRunner = Callable[..., CommandResult]


def _default_runner(
    argv: Sequence[str], *, timeout: float, max_output_bytes: int | None = None
) -> CommandResult:
    """Run a small systemctl command, or stream a capped journal query.

    Journal output can be produced by an untrusted workload.  It is read with a
    byte ceiling so inspection memory use does not scale with total job output.
    """
    args = list(argv)
    if args and Path(args[0]).name == "journalctl" and max_output_bytes is not None:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdout is not None
        chunks: list[bytes] = []
        size = 0
        truncated = False
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    raise subprocess.TimeoutExpired(args, timeout)
                if not selector.select(remaining):
                    proc.kill()
                    raise subprocess.TimeoutExpired(args, timeout)
                chunk = os.read(
                    proc.stdout.fileno(), min(65536, max_output_bytes + 1 - size)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > max_output_bytes:
                    truncated = True
                    proc.terminate()
                    break
            if truncated:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        finally:
            selector.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        content = b"".join(chunks)[:max_output_bytes]
        return SimpleNamespace(
            args=args,
            returncode=proc.returncode if not truncated else 0,
            stdout=content.decode("utf-8", errors="replace"),
            stderr="",
            truncated=truncated,
        )
    return subprocess.run(
        args, text=True, capture_output=True, timeout=timeout, check=False
    )


def _value(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _result_attr(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _parse_properties(output: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def attempt_unit_name(attempt_id: str) -> str:
    """Return a stable, bounded unit name without exposing arbitrary ID text."""
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("attempt_id must be a non-empty string")
    digest = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:32]
    return f"{_UNIT_PREFIX}{digest}.service"


class SystemdExecutionAdapter:
    """Supervise argv-based executions as transient user service units.

    The injectable ``runner`` is intended for hermetic contract tests.  The
    default implementation communicates only with the user's systemd manager;
    it is never invoked implicitly by importing this module.
    """

    version = "systemd-user-unit/v1"

    def __init__(
        self,
        *,
        runner: CommandRunner = _default_runner,
        profile_resolver: Callable[[str], Mapping[str, Any]] | None = None,
        cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
        command_timeout: float = 15.0,
        stop_timeout_seconds: int = 15,
        max_output_bytes: int = 256 * 1024,
        max_output_lines: int = 2000,
        host_id: str | Callable[[], str] | None = None,
    ) -> None:
        for name, value in (
            ("command_timeout", command_timeout),
            ("stop_timeout_seconds", stop_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes < 1:
            raise ValueError("max_output_bytes must be a positive integer")
        if isinstance(max_output_lines, bool) or not isinstance(max_output_lines, int) or max_output_lines < 1:
            raise ValueError("max_output_lines must be a positive integer")
        self._runner = runner
        self._profile_resolver = profile_resolver
        self._cgroup_root = Path(cgroup_root)
        self._command_timeout = command_timeout
        self._stop_timeout_seconds = stop_timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._host_id = host_id or socket.gethostname

    api_version = "1"
    adapter_id = "systemd-user-unit"

    def capabilities(self) -> Sequence[AdapterCapability]:
        detail = {"adapter": self.version, "host_qualified": False}
        return (
            AdapterCapability(
                "systemd-transient-user-service",
                True,
                EnforcementLevel.ADVISORY,
                {**detail, "restart_survival": "host-and-user-manager-running"},
            ),
            AdapterCapability(
                "process-tree-cgroup-supervision",
                True,
                EnforcementLevel.ADVISORY,
                {**detail, "stop": "systemd-unit-only", "requires_verified_cgroup_cleanup": True},
            ),
            AdapterCapability(
                "cpu-memory-task-limits",
                True,
                EnforcementLevel.ADVISORY,
                {**detail, "properties": ["CPUQuota", "MemoryMax", "TasksMax"]},
            ),
            AdapterCapability(
                "granted-resource-cgroup-limits",
                True,
                EnforcementLevel.ADVISORY,
                {
                    **detail,
                    "cpu_allocation": "systemd CPUQuota as granted core count times 100 percent",
                    "host_memory_allocation": "systemd MemoryMax in bytes",
                    "profile_limits": "stricter of registered profile limit and granted allocation",
                    "gpu_memory": "accounting only; no hardware memory limit claimed",
                },
            ),
            AdapterCapability(
                "supervised-lease-job-binding",
                True,
                EnforcementLevel.ADVISORY,
                {
                    **detail,
                    "input": "the execution context from the current active lease fence",
                    "cleanup": "exact unit invocation and cgroup cleanup receipt",
                    "supported_resources": ["cpu", "memory"],
                    "unmanaged_processes": "unsupported; same-user cgroups are not hostile-code isolation",
                },
            ),
            AdapterCapability(
                "grant-bound-cuda-visible-devices",
                True,
                EnforcementLevel.ADVISORY,
                {
                    **detail,
                    "source": "allocated NVIDIA resource UUIDs",
                    "cpu_only": "CUDA_VISIBLE_DEVICES is empty",
                    "limitation": "cooperative visibility, not hostile-code isolation",
                },
            ),
            AdapterCapability(
                "finite-job-runtime-limit",
                True,
                EnforcementLevel.ADVISORY,
                {**detail, "property": "RuntimeMaxSec", "applies_to": "job-purpose launches only"},
            ),
            AdapterCapability(
                "bounded-output-observation",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {**detail, "source": "systemd-journal", "max_bytes": self._max_output_bytes},
            ),
        )

    def _call(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        max_output_bytes: int | None = None,
    ) -> CommandResult:
        try:
            return self._runner(
                list(argv),
                timeout=self._command_timeout if timeout is None else timeout,
                max_output_bytes=max_output_bytes,
            )
        except FileNotFoundError as exc:
            raise ExecutionAdapterError(f"required command is unavailable: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExecutionAdapterError(f"command timed out: {Path(argv[0]).name}") from exc
        except OSError as exc:
            raise ExecutionAdapterError(f"command failed: {Path(argv[0]).name} ({type(exc).__name__})") from exc

    def _new_handle_data(
        self,
        attempt_id: str,
        *,
        invocation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        unit_name = attempt_unit_name(attempt_id)
        return {
            "kind": "systemd-user-unit",
            "version": 1,
            "attempt_id": attempt_id,
            "unit_name": unit_name,
            "description": f"Device Scheduler attempt {attempt_id}",
            "invocation_id": invocation_id,
            "metadata": dict(metadata or {}),
        }

    def _model_handle(self, data: Mapping[str, Any]) -> ExecutionHandle:
        return ExecutionHandle(
            adapter_id=self.adapter_id,
            external_id=str(data["unit_name"]),
            identity=str(data["attempt_id"]),
            started_at=datetime.now(timezone.utc),
            metadata={
                **dict(data.get("metadata") or {}),
                "handle_version": data.get("version", 1),
                "description": data.get("description"),
                "invocation_id": data.get("invocation_id"),
                "control_group": data.get("control_group") or (data.get("metadata") or {}).get("control_group"),
            },
        )

    def _as_handle_data(self, handle: ExecutionHandle | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(handle, ExecutionHandle):
            if handle.adapter_id != self.adapter_id:
                return {"kind": "unsupported", "version": 0}
            metadata = dict(handle.metadata)
            return {
                "kind": "systemd-user-unit",
                "version": metadata.get("handle_version", 1),
                "attempt_id": handle.identity,
                "unit_name": handle.external_id,
                "description": metadata.get("description", f"Device Scheduler attempt {handle.identity}"),
                "invocation_id": metadata.get("invocation_id"),
                "control_group": metadata.get("control_group"),
                "metadata": metadata,
            }
        if handle.get("kind") == "systemd-user-unit":
            return dict(handle)
        # The durable ledger stores the core ExecutionHandle DTO, not this
        # adapter's private representation. Convert that public shape back to
        # the exact unit and invocation identity captured in metadata.
        if handle.get("adapter_id") == self.adapter_id:
            metadata = dict(handle.get("metadata") or {})
            identity = handle.get("identity")
            return {
                "kind": "systemd-user-unit",
                "version": metadata.get("handle_version", 1),
                "attempt_id": identity,
                "unit_name": handle.get("external_id"),
                "description": metadata.get("description", f"Device Scheduler attempt {identity}"),
                "invocation_id": metadata.get("invocation_id"),
                "metadata": metadata,
            }
        return dict(handle)

    def _spec_argv(self, spec: Any) -> list[str]:
        argv = _value(spec, "argv")
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
            raise ValueError("execution spec argv must be a non-empty argument sequence")
        if any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("execution arguments must be NUL-free strings")
        return list(argv)

    def _unit_properties(
        self,
        spec: Any,
        *,
        runtime_max_seconds: float | None = None,
        termination_grace_seconds: float | None = None,
    ) -> list[str]:
        stop_timeout = termination_grace_seconds
        if stop_timeout is None:
            stop_timeout = _value(spec, "termination_grace_seconds")
        if stop_timeout is None:
            stop_timeout = self._stop_timeout_seconds
        if (
            isinstance(stop_timeout, bool)
            or not isinstance(stop_timeout, (int, float))
            or not math.isfinite(stop_timeout)
            or stop_timeout <= 0
        ):
            raise ValueError("termination_grace_seconds must be finite and positive")
        props = [
            "RemainAfterExit=yes",
            "Restart=no",
            "KillMode=control-group",
            f"TimeoutStopSec={stop_timeout:g}s",
            "StandardOutput=journal",
            "StandardError=journal",
        ]
        if runtime_max_seconds is not None:
            if (
                isinstance(runtime_max_seconds, bool)
                or not isinstance(runtime_max_seconds, (int, float))
                or not math.isfinite(runtime_max_seconds)
                or runtime_max_seconds <= 0
            ):
                raise ValueError("runtime_max_seconds must be finite and positive")
            props.append(f"RuntimeMaxSec={runtime_max_seconds:g}s")
        limits = _value(spec, "limits", {}) or {}
        cpu = _value(limits, "cpu_quota_percent")
        memory = _value(limits, "memory_max_bytes")
        tasks = _value(limits, "tasks_max")
        if cpu is not None:
            try:
                cpu_value = float(cpu)
            except (TypeError, ValueError) as exc:
                raise ValueError("cpu_quota_percent must be a positive number") from exc
            if not math.isfinite(cpu_value) or cpu_value <= 0:
                raise ValueError("cpu_quota_percent must be a finite positive number")
            props.append(f"CPUQuota={cpu_value:g}%")
        if memory is not None:
            if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
                raise ValueError("memory_max_bytes must be a positive integer")
            props.append(f"MemoryMax={memory}")
        if tasks is not None:
            if isinstance(tasks, bool) or not isinstance(tasks, int) or tasks <= 0:
                raise ValueError("tasks_max must be a positive integer")
            props.append(f"TasksMax={tasks}")
        return props

    def _environment(self, spec: Any) -> list[str]:
        environment = _value(spec, "environment", {}) or {}
        if not isinstance(environment, Mapping):
            raise ValueError("execution environment must be a mapping")
        result: list[str] = []
        for key, value in sorted(environment.items()):
            if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"environment variable {key!r} must have a NUL-free string value")
            result.append(f"--setenv={key}={value}")
        return result

    def _execution_spec(self, spec: JobSpec | Any) -> Mapping[str, Any]:
        profile_id = _value(spec, "profile_id")
        if profile_id is not None:
            if self._profile_resolver is None:
                raise ExecutionAdapterError(
                    f"no registered execution profile resolver for {profile_id!r}"
                )
            resolved = self._profile_resolver(profile_id)
            if not isinstance(resolved, Mapping):
                raise ExecutionAdapterError("execution profile resolver returned a non-object")
            return resolved
        return {
            "argv": _value(spec, "argv"),
            "working_directory": _value(spec, "working_directory"),
            "environment": _value(spec, "environment", {}),
            "limits": _value(spec, "limits", {}),
        }

    def _local_host_id(self) -> str:
        try:
            value = self._host_id() if callable(self._host_id) else self._host_id
        except OSError as exc:
            raise ExecutionAdapterError("local host identity is unavailable") from exc
        if not isinstance(value, str) or not value.strip():
            raise ExecutionAdapterError("local host identity is unavailable")
        return value

    @staticmethod
    def _execution_context(value: ExecutionContext | Mapping[str, Any] | None) -> ExecutionContext:
        if value is None:
            # Missing context confers no resource access. In particular, it
            # must never inherit all host GPUs from the caller environment.
            return ExecutionContext()
        if isinstance(value, ExecutionContext):
            return value
        if isinstance(value, Mapping):
            try:
                return ExecutionContext.from_dict(value)
            except (TypeError, ValueError) as exc:
                raise ExecutionAdapterError(f"invalid execution context: {exc}") from exc
        raise ExecutionAdapterError("execution_context must be an ExecutionContext object")

    @staticmethod
    def _byte_quantity(key: str, amount: float) -> int | None:
        factors = {
            "bytes": 1,
            "memory_bytes": 1,
            "host_memory_bytes": 1,
            "memory_mib": 1024**2,
            "memory_gib": 1024**3,
        }
        factor = factors.get(key.casefold())
        return None if factor is None else math.floor(amount * factor)

    @staticmethod
    def _capacity_bytes(resource: Resource) -> int | None:
        for key in ("bytes", "memory_bytes", "host_memory_bytes"):
            value = resource.capacities.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                return math.floor(value)
        return None

    @staticmethod
    def _positive_limit(value: Any, name: str, *, integer: bool = False) -> float | int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExecutionAdapterError(f"registered {name} must be finite and positive")
        if integer and not isinstance(value, int):
            raise ExecutionAdapterError(f"registered {name} must be an integer byte count")
        return int(value) if integer else float(value)

    def _bind_execution_context(
        self,
        spec: JobSpec,
        resolved: Mapping[str, Any],
        execution_context: ExecutionContext | Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], ExecutionContext]:
        context = self._execution_context(execution_context)
        if context.allocations and (context.claim_id is None or context.fencing_token is None):
            raise ExecutionAdapterError("resource allocations require a claim ID and fencing token")
        resources = {resource.resource_id: resource for resource in context.resources}
        local_host = self._local_host_id() if context.resources else None
        gpu_uuids: list[str] = []
        cpu_cores = 0.0
        host_memory_bytes = 0
        for allocation in context.allocations:
            resource = resources.get(allocation.resource_id)
            if resource is None:
                raise ExecutionAdapterError("allocated resource descriptor is missing")
            if resource.host_id != local_host:
                raise ExecutionAdapterError("systemd execution cannot bind a resource from another host")
            kind = resource.kind.casefold()
            if kind in _GPU_KINDS:
                uuid = resource.attributes.get("uuid")
                if not isinstance(uuid, str) or not _GPU_UUID.fullmatch(uuid):
                    raise ExecutionAdapterError("allocated NVIDIA GPU has no valid stable UUID")
                if uuid in gpu_uuids:
                    raise ExecutionAdapterError("execution context repeats an NVIDIA GPU UUID")
                gpu_uuids.append(uuid)
                # GPU memory allocations control admission only; no hardware
                # VRAM enforcement is claimed by this adapter.
            elif kind in _CPU_KINDS:
                for key, amount in allocation.quantities.items():
                    if key.casefold() != "cores":
                        raise ExecutionAdapterError(f"unsupported CPU allocation quantity: {key}")
                    capacity = resource.capacities.get("cores")
                    if (
                        isinstance(capacity, bool)
                        or not isinstance(capacity, (int, float))
                        or not math.isfinite(capacity)
                        or amount > capacity
                    ):
                        raise ExecutionAdapterError("CPU allocation exceeds its observed resource capacity")
                    cpu_cores += amount
            elif kind in _MEMORY_KINDS:
                capacity_bytes = self._capacity_bytes(resource)
                if capacity_bytes is None:
                    raise ExecutionAdapterError("host memory resource has no byte-valued capacity")
                allocated_bytes = 0
                for key, amount in allocation.quantities.items():
                    converted = self._byte_quantity(key, amount)
                    if converted is None:
                        raise ExecutionAdapterError(f"unsupported host memory allocation quantity: {key}")
                    allocated_bytes += converted
                if allocated_bytes > capacity_bytes:
                    raise ExecutionAdapterError("host memory allocation exceeds its observed resource capacity")
                host_memory_bytes += allocated_bytes

        gpu_uuids.sort()
        profile_environment = resolved.get("environment", {}) or {}
        job_environment = _value(spec, "environment", {}) or {}
        if not isinstance(profile_environment, Mapping) or not isinstance(job_environment, Mapping):
            raise ExecutionAdapterError("execution environment must be a mapping")
        merged_environment = dict(profile_environment)
        merged_environment.update(job_environment)
        expected_visible_devices = ",".join(gpu_uuids)
        requested_visible_devices = merged_environment.get("CUDA_VISIBLE_DEVICES")
        if requested_visible_devices is not None and requested_visible_devices != expected_visible_devices:
            raise ExecutionAdapterError(
                "requested CUDA_VISIBLE_DEVICES conflicts with the scheduler-granted GPU UUIDs"
            )
        merged_environment["CUDA_VISIBLE_DEVICES"] = expected_visible_devices

        limits = resolved.get("limits", {}) or {}
        if not isinstance(limits, Mapping):
            raise ExecutionAdapterError("registered execution limits must be a mapping")
        effective_limits = dict(limits)
        if cpu_cores > 0:
            cpu_quota = cpu_cores * 100.0
            profile_cpu_quota = effective_limits.get("cpu_quota_percent")
            if profile_cpu_quota is not None:
                profile_cpu_quota = self._positive_limit(profile_cpu_quota, "CPU quota")
                cpu_quota = min(cpu_quota, float(profile_cpu_quota))
            effective_limits["cpu_quota_percent"] = cpu_quota
        if host_memory_bytes > 0:
            profile_memory = effective_limits.get("memory_max_bytes")
            if profile_memory is not None:
                profile_memory = self._positive_limit(profile_memory, "memory limit", integer=True)
                host_memory_bytes = min(host_memory_bytes, int(profile_memory))
            if host_memory_bytes < 1:
                raise ExecutionAdapterError("granted host memory is below the enforceable systemd limit")
            effective_limits["memory_max_bytes"] = host_memory_bytes

        bound = dict(resolved)
        bound["environment"] = merged_environment
        bound["limits"] = effective_limits
        return bound, context

    def start(
        self,
        spec: JobSpec,
        *,
        attempt_id: str,
        execution_root: Path,
        execution_context: ExecutionContext | Mapping[str, Any] | None = None,
    ) -> ExecutionHandle:
        """Start one deterministic unit and return its invocation-bound handle."""
        resolved, context = self._bind_execution_context(
            spec, self._execution_spec(spec), execution_context
        )
        handle = self._new_handle_data(
            attempt_id,
            metadata={
                "execution_root": str(execution_root),
                "profile_id": _value(spec, "profile_id"),
                "profile_version": _value(resolved, "profile_version"),
            },
        )
        argv = self._spec_argv(resolved)
        cwd = _value(resolved, "working_directory", _value(resolved, "cwd"))
        if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
            raise StartNotSubmitted("execution spec requires an explicit working directory")
        working_directory = Path(cwd).expanduser()
        if not working_directory.is_absolute():
            raise StartNotSubmitted("working directory must be absolute")
        try:
            working_directory = working_directory.resolve()
        except (OSError, RuntimeError) as exc:
            raise StartNotSubmitted("working directory path cannot be resolved") from exc
        cwd = str(working_directory)
        argv = self._absolute_path_executable(argv, cwd)
        command = [
            "systemd-run",
            "--user",
            f"--unit={handle['unit_name']}",
            f"--description={handle['description']}",
            "--service-type=exec",
        ]
        runtime_limit = (
            float(spec.kill_timeout_seconds if spec.kill_timeout_seconds is not None else spec.timeout_seconds)
            if context.purpose == ExecutionPurpose.JOB else None
        )
        for prop in self._unit_properties(
            resolved,
            runtime_max_seconds=runtime_limit,
            termination_grace_seconds=spec.termination_grace_seconds,
        ):
            command.append(f"--property={prop}")
        command.append(f"--working-directory={cwd}")
        command.extend(self._environment(resolved))
        command.extend(["--", *argv])

        try:
            result = self._call(command)
        except ExecutionAdapterError as exc:
            raise StartUncertain(str(exc), handle) from exc
        props = _result_attr(result, "stdout", "")
        return_code = _result_attr(result, "returncode", 1)
        observation = self._show(handle["unit_name"])
        identity_matches = (
            observation.get("Id") == handle["unit_name"]
            and observation.get("Description") == handle["description"]
        )
        invocation_id = observation.get("InvocationID")
        if identity_matches and invocation_id:
            handle["invocation_id"] = invocation_id
            handle["control_group"] = observation.get("ControlGroup")
            handle["metadata"]["control_group"] = observation.get("ControlGroup")
            if return_code == 0:
                return self._model_handle(handle)
            # A lost response after systemd accepted the unique unit is an
            # idempotently inspectable launch, not a reason to submit it again.
            if observation.get("LoadState") == "loaded":
                return self._model_handle(handle)
        if return_code != 0:
            detail = _result_attr(result, "stderr", "") or props
            raise ExecutionAdapterError(
                f"systemd-run rejected attempt {attempt_id}: {str(detail).strip()[:1000]}",
                safe_diagnostic="systemd-run returned a nonzero status and no invocation could be verified",
            )
        raise StartUncertain(
            "systemd accepted the start request but its unit invocation could not be verified",
            handle,
        )

    @staticmethod
    def _absolute_path_executable(argv: list[str], cwd: str) -> list[str]:
        """Validate slash-containing executable paths while preserving argv[0].

        systemd's ExecStart parser accepts an absolute path or a bare command
        name; a relative path containing a slash is not resolved against
        WorkingDirectory. Resolve the target strictly for validation, but pass
        the normalized invocation path to systemd. In particular, resolving a
        virtual-environment interpreter symlink changes Python's environment
        discovery and can run the base interpreter instead.
        """
        executable = argv[0]
        if not executable:
            raise StartNotSubmitted("executable path is empty")
        if "/" not in executable:
            return argv
        path = Path(executable)
        if not path.is_absolute():
            path = Path(cwd) / path
        try:
            invocation_path = Path(os.path.abspath(path))
            resolved_target = invocation_path.resolve(strict=True)
            resolved_target.stat()
        except (OSError, RuntimeError) as exc:
            raise StartNotSubmitted("executable path does not resolve to an existing file") from exc
        if not resolved_target.is_file() or not os.access(resolved_target, os.X_OK):
            raise StartNotSubmitted("executable path is not a runnable regular file")
        return [str(invocation_path), *argv[1:]]

    def fence_missing_attempt(self, *, attempt_id: str) -> Mapping[str, Any]:
        """Fence a handle-less attempt and prove its systemd unit is stopped.

        A not-found lookup alone is not proof: a delayed systemd-run request
        could create the deterministic unit after reconciliation. A runtime
        mask acknowledged by the manager blocks that later start. This method
        never stops a unit by name: a raced active unit remains unknown (or is
        rebound to its invocation identity). Cleanup is accepted only after
        the mask, inactive unit state, and derived unit cgroup are verified.
        """
        unit_name = attempt_unit_name(attempt_id)
        expected_description = f"Device Scheduler attempt {attempt_id}"
        before_load = self._unit_load_state(unit_name)
        if before_load is None:
            return {"state": "unknown", "cleanup_verified": False, "reason": "unit load state is ambiguous", "unit_name": unit_name}
        masked_states = {"masked", "masked-runtime", "masked-transient"}
        before = self._show(unit_name) if before_load == "loaded" else {}
        if before_load == "loaded" and not self._expected_unit_identity(before, unit_name, expected_description):
            return {"state": "unknown", "cleanup_verified": False, "reason": "deterministic unit identity mismatch", "unit_name": unit_name}
        control_group = before.get("ControlGroup") or self._attempt_control_group(unit_name)
        if not control_group:
            return {"state": "unknown", "cleanup_verified": False, "reason": "unit cgroup path cannot be safely derived", "unit_name": unit_name}
        try:
            # Mask first, without stopping by name. A raced active unit is
            # left running and unknown; this operation only blocks a later
            # launch after the manager acknowledges the tombstone.
            result = self._call(["systemctl", "--user", "mask", "--runtime", unit_name])
        except ExecutionAdapterError as exc:
            return {"state": "unknown", "cleanup_verified": False, "reason": str(exc), "unit_name": unit_name}
        if _result_attr(result, "returncode", 1) != 0:
            detail = _result_attr(result, "stderr", "") or _result_attr(result, "stdout", "")
            return {"state": "unknown", "cleanup_verified": False, "reason": f"unit fence failed: {str(detail).strip()[:500]}", "unit_name": unit_name}
        after_load = self._unit_load_state(unit_name)
        if after_load not in masked_states:
            return {"state": "unknown", "cleanup_verified": False, "reason": "systemd did not confirm the runtime mask", "unit_name": unit_name}
        after = self._show(unit_name)
        if after.get("Description") not in {None, expected_description, unit_name}:
            return {"state": "unknown", "cleanup_verified": False, "reason": "racing unit identity cannot be verified", "unit_name": unit_name, "observation": after}
        active_state = after.get("ActiveState")
        substate = after.get("SubState")
        if active_state not in {"inactive", "failed"}:
            invocation_id = after.get("InvocationID")
            if self._expected_unit_identity(after, unit_name, expected_description) and invocation_id:
                active_handle = self._new_handle_data(
                    attempt_id,
                    invocation_id=invocation_id,
                    metadata={"fenced": True, "control_group": control_group},
                )
                return {
                    "state": "running",
                    "cleanup_verified": False,
                    "fenced": True,
                    "execution_handle": self._model_handle(active_handle).to_dict(),
                    "unit_name": unit_name,
                    "control_group": control_group,
                    "observation": after,
                }
            return {"state": "unknown", "cleanup_verified": False, "reason": "masked unit is not confirmed inactive", "unit_name": unit_name, "observation": after}
        if not self._cgroup_empty(control_group, active_state=active_state, substate=substate or ""):
            return {"state": "unknown", "cleanup_verified": False, "reason": "unit cgroup is not confirmed empty", "unit_name": unit_name, "control_group": control_group, "observation": after}
        return {
            "state": "cancelled",
            "cleanup_verified": True,
            "execution_may_have_started": True,
            "fenced": True,
            "fence": "systemd-runtime-mask",
            "unit_name": unit_name,
            "control_group": control_group,
            "observation": after,
        }

    def _unit_load_state(self, unit_name: str) -> str | None:
        try:
            result = self._call(["systemctl", "--user", "show", unit_name, "--property=LoadState"])
        except ExecutionAdapterError:
            return None
        if _result_attr(result, "returncode", 1) != 0:
            return None
        return _parse_properties(_result_attr(result, "stdout", "") or "").get("LoadState")

    @staticmethod
    def _expected_unit_identity(properties: Mapping[str, str], unit_name: str, description: str) -> bool:
        return properties.get("Id") == unit_name and properties.get("Description") == description

    def _attempt_control_group(self, unit_name: str) -> str | None:
        """Derive this user manager's app.slice path from our own cgroup."""
        try:
            hierarchy = next(
                line.split(":", 2)[2].strip()
                for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
                if line.startswith("0::")
            )
        except (OSError, StopIteration):
            return None
        parts = Path(hierarchy).parts
        manager_index = next(
            (index for index, item in enumerate(parts) if item.startswith("user@") and item.endswith(".service")),
            None,
        )
        if manager_index is None or parts[manager_index] != f"user@{os.getuid()}.service":
            return None
        relative_app_slice = Path(*parts[1:manager_index + 1]) / "app.slice"
        try:
            root = self._cgroup_root.resolve(strict=True)
            app_slice = (root / relative_app_slice).resolve(strict=True)
            app_slice.relative_to(root)
        except (OSError, ValueError):
            return None
        if app_slice != root / relative_app_slice or not (app_slice / "cgroup.events").is_file() or not (app_slice / "cgroup.procs").is_file():
            return None
        return str(Path("/") / relative_app_slice / unit_name)

    def _show(self, unit_name: str) -> dict[str, str]:
        properties = (
            "Id",
            "Description",
            "LoadState",
            "ActiveState",
            "SubState",
            "Result",
            "ExecMainCode",
            "ExecMainStatus",
            "ControlGroup",
            "InvocationID",
            "NRestarts",
            "Restart",
        )
        command = ["systemctl", "--user", "show", unit_name]
        for prop in properties:
            command.append(f"--property={prop}")
        try:
            result = self._call(command)
        except ExecutionAdapterError:
            return {}
        if _result_attr(result, "returncode", 1) != 0:
            return {}
        return _parse_properties(_result_attr(result, "stdout", "") or "")

    def resolve(self, *, attempt_id: str) -> ExecutionHandle | None:
        """Resolve a durable attempt identity to its current systemd invocation.

        This is a read-only reconciliation helper.  A missing unit returns
        ``None``; command failures raise so callers can preserve uncertainty
        instead of treating an unavailable systemd manager as proof of absence.
        """
        if not isinstance(attempt_id, str) or not attempt_id.strip():
            raise ValueError("attempt_id must be a non-empty string")
        expected_description = f"Device Scheduler attempt {attempt_id}"
        unit_name = attempt_unit_name(attempt_id)
        properties = self._show(unit_name)
        if not properties:
            # _show intentionally maps query errors to empty observations for
            # inspect(); reconciliation needs to distinguish that from absence.
            command = ["systemctl", "--user", "show", unit_name, "--property=LoadState"]
            try:
                result = self._call(command)
            except ExecutionAdapterError:
                raise
            if _result_attr(result, "returncode", 1) != 0:
                raise ExecutionAdapterError("systemd unit lookup failed; attempt state is unknown")
            retry_props = _parse_properties(_result_attr(result, "stdout", "") or "")
            if retry_props.get("LoadState") in {"not-found", "masked"}:
                return None
            raise ExecutionAdapterError("systemd unit lookup returned ambiguous state")
        if properties.get("LoadState") in {"not-found", "masked"}:
            return None
        if properties.get("LoadState") != "loaded":
            raise ExecutionAdapterError("systemd unit load state is ambiguous")
        if properties.get("Id") != unit_name:
            raise ExecutionAdapterError("systemd unit ID does not match the deterministic attempt unit")
        if properties.get("Description") != expected_description:
            raise ExecutionAdapterError("deterministic unit name is owned by a different description")
        invocation_id = properties.get("InvocationID")
        if not invocation_id:
            raise ExecutionAdapterError("systemd unit has no invocation identity")
        return self._model_handle(
            self._new_handle_data(
                attempt_id,
                invocation_id=invocation_id,
                metadata={
                    "reconciled": True,
                    "control_group": properties.get("ControlGroup"),
                },
            )
        )

    def _verify_handle(self, handle: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
        if handle.get("kind") != "systemd-user-unit" or handle.get("version") != 1:
            return {}, "unsupported execution handle"
        if handle.get("adapter_id", self.adapter_id) != self.adapter_id:
            return {}, "execution handle belongs to a different adapter"
        attempt_id = handle.get("attempt_id")
        unit_name = handle.get("unit_name")
        if not isinstance(attempt_id, str) or unit_name != attempt_unit_name(attempt_id):
            return {}, "execution handle identity is invalid"
        props = self._show(unit_name)
        if props.get("LoadState") != "loaded":
            return props, "unit is missing or unavailable"
        if props.get("Id") != unit_name:
            return props, "systemd unit identity does not match the deterministic attempt unit"
        if props.get("Description") != handle.get("description"):
            return props, "unit description does not match the attempt identity"
        current_invocation = props.get("InvocationID")
        recorded_invocation = handle.get("invocation_id")
        if not current_invocation:
            return props, "unit invocation identity is missing"
        if not recorded_invocation:
            return props, "execution handle is not bound to a recorded invocation identity"
        if current_invocation != recorded_invocation:
            return props, "unit invocation identity changed"
        if props.get("Restart", "no") != "no":
            return props, "unit restart policy is not disabled"
        return props, None

    def _cgroup_empty(
        self, control_group: str | None, *, active_state: str, substate: str
    ) -> bool:
        if not control_group:
            return False
        relative = Path(control_group.lstrip("/"))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        try:
            root = self._cgroup_root.resolve()
            cgroup_dir = (root / relative).resolve()
        except OSError:
            return False
        try:
            cgroup_dir.relative_to(root)
        except ValueError:
            return False
        if cgroup_dir.name != relative.name or not relative.name.endswith(".service"):
            return False
        try:
            cgroup_dir.stat()
        except FileNotFoundError:
            # A vanished unit cgroup is evidence that systemd has reaped its
            # complete control group. An existing cgroup without cgroup.events
            # is not understood and must remain unknown (for example cgroup v1).
            return active_state in {"inactive", "failed"} or (
                active_state == "active" and substate == "exited"
            )
        except OSError:
            return False
        events = cgroup_dir / "cgroup.events"
        try:
            contents = events.read_text(encoding="ascii")
        except FileNotFoundError:
            return False
        except OSError:
            return False
        for line in contents.splitlines():
            key, _, value = line.partition(" ")
            if key == "populated" and value in {"0", "1"}:
                return value == "0"
        return False

    def _logs(self, unit_name: str, invocation_id: str) -> tuple[str, bool, str | None]:
        command = [
            "journalctl",
            "--user",
            f"--unit={unit_name}",
            "--output=cat",
            "--no-pager",
            f"--lines={self._max_output_lines}",
            f"_SYSTEMD_INVOCATION_ID={invocation_id}",
        ]
        try:
            result = self._runner(
                command,
                timeout=self._command_timeout,
                max_output_bytes=self._max_output_bytes,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "", True, "journal query failed"
        return_code = _result_attr(result, "returncode", 1)
        if return_code != 0:
            return "", True, f"journal query returned exit status {return_code}"
        output = _result_attr(result, "stdout", "") or ""
        encoded = output.encode("utf-8", errors="replace")
        truncated = bool(_result_attr(result, "truncated", False)) or len(encoded) > self._max_output_bytes
        if len(encoded) > self._max_output_bytes:
            encoded = encoded[-self._max_output_bytes :]
        return encoded.decode("utf-8", errors="replace"), truncated, None

    def inspect(self, handle: ExecutionHandle | Mapping[str, Any]) -> dict[str, Any]:
        """Reconcile the unit without launching or signaling any process."""
        handle = self._as_handle_data(handle)
        props, problem = self._verify_handle(handle)
        invocation_id = props.get("InvocationID")
        if problem:
            return {
                "state": "unknown",
                "reason": problem,
                "cleanup_verified": False,
                "unit_name": handle.get("unit_name"),
                "invocation_id": invocation_id,
            }

        active = props.get("ActiveState", "")
        substate = props.get("SubState", "")
        result = props.get("Result", "")
        main_code = props.get("ExecMainCode", "")
        try:
            exit_status = int(props.get("ExecMainStatus", "0"))
        except ValueError:
            exit_status = -1
        if active == "active" and substate not in {"exited", "dead"}:
            state = "running"
        elif active == "failed" or result not in {"", "success"} or exit_status != 0:
            state = "failed"
        elif result == "success" and main_code in {"1", "exited"}:
            state = "succeeded"
        elif active == "inactive" and substate in {"dead", "failed"}:
            state = "unknown"
        else:
            state = "unknown"

        cleanup_verified = self._cgroup_empty(
            props.get("ControlGroup") or (handle.get("metadata") or {}).get("control_group"),
            active_state=active,
            substate=substate,
        )
        logs, truncated, logs_error = self._logs(handle["unit_name"], invocation_id or "")
        return {
            "state": state,
            "exit_code": exit_status if state in {"succeeded", "failed"} else None,
            "cleanup_verified": cleanup_verified,
            "unit_name": handle["unit_name"],
            "invocation_id": invocation_id,
            "active_state": active,
            "substate": substate,
            "result": result,
            "logs": logs,
            "logs_truncated": truncated,
            "logs_error": logs_error,
            "evidence": {
                "adapter": self.version,
                "unit_description": props.get("Description"),
                "invocation_id": invocation_id,
                "control_group": props.get("ControlGroup") or (handle.get("metadata") or {}).get("control_group"),
                "main_code": main_code,
            },
        }

    def cancel(
        self,
        handle: ExecutionHandle | Mapping[str, Any],
        *,
        operation_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Stop only a verified scheduler-owned unit invocation and confirm cleanup."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("cancel requires a non-empty operation_id")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        handle = self._as_handle_data(handle)
        props, problem = self._verify_handle(handle)
        if problem:
            return {
                "state": "unknown",
                "operation_id": operation_id,
                "reason": problem,
                "cleanup_verified": False,
                "unit_name": handle.get("unit_name"),
                "invocation_id": props.get("InvocationID"),
            }
        unit_name = str(handle["unit_name"])
        active = props.get("ActiveState")
        stop_succeeded = False
        if active == "active" and props.get("SubState") not in {"exited", "dead"}:
            try:
                response = self._call(
                    ["systemctl", "--user", "stop", unit_name],
                    timeout=max(0.1, timeout_seconds),
                )
            except ExecutionAdapterError as exc:
                return {
                    "state": "unknown",
                    "operation_id": operation_id,
                    "reason": str(exc),
                    "cleanup_verified": False,
                    "unit_name": unit_name,
                    "invocation_id": props.get("InvocationID"),
                }
            if _result_attr(response, "returncode", 1) != 0:
                detail = _result_attr(response, "stderr", "")
                return {
                    "state": "unknown",
                    "operation_id": operation_id,
                    "reason": f"systemctl stop did not complete: {str(detail).strip()[:1000]}",
                    "cleanup_verified": False,
                    "unit_name": unit_name,
                    "invocation_id": props.get("InvocationID"),
                }
            stop_succeeded = True

        # A transient unit can be unloaded from the user manager immediately
        # after a successful stop.  In that case inspect() can no longer bind
        # the unit to its InvocationID, even though this call started from an
        # invocation-verified handle and systemd acknowledged stopping it.
        # Verify against the original cgroup path and invocation instead of
        # converting a confirmed stop into an ambiguous unit lookup.
        post_props, post_problem = self._verify_handle(handle)
        if not post_problem:
            receipt = self.inspect(handle)
            cleanup_verified = bool(receipt.get("cleanup_verified"))
        elif stop_succeeded and self._unit_is_not_found(unit_name):
            cleanup_verified = self._cgroup_empty(
                props.get("ControlGroup") or (handle.get("metadata") or {}).get("control_group"),
                active_state="inactive",
                substate="dead",
            )
            receipt = {
                "state": "unknown",
                "reason": "unit was unloaded after systemd acknowledged stop",
                "unit_name": unit_name,
                "invocation_id": props.get("InvocationID"),
                "evidence": {
                    "adapter": self.version,
                    "unit_description": props.get("Description"),
                    "invocation_id": props.get("InvocationID"),
                    "control_group": props.get("ControlGroup") or (handle.get("metadata") or {}).get("control_group"),
                    "post_stop_unit_state": "not-found",
                },
            }
        else:
            cleanup_verified = False
            receipt = {
                "state": "unknown",
                "reason": post_problem or "post-stop invocation could not be verified",
                "unit_name": unit_name,
                "invocation_id": props.get("InvocationID"),
            }
        return {
            **receipt,
            "operation_id": operation_id,
            "stop_requested": True,
            "cleanup_verified": cleanup_verified,
        }

    def _unit_is_not_found(self, unit_name: str) -> bool:
        """Distinguish an unloaded unit from an unavailable systemd manager."""
        try:
            result = self._call(
                ["systemctl", "--user", "show", unit_name, "--property=LoadState"]
            )
        except ExecutionAdapterError:
            return False
        if _result_attr(result, "returncode", 1) != 0:
            return False
        return _parse_properties(_result_attr(result, "stdout", "") or "").get("LoadState") in {
            "not-found",
            "masked",
        }

    def _cgroup_process_ids(self, control_group: str | None) -> tuple[list[int] | None, str | None]:
        """Read PIDs only from a safe systemd unit cgroup subtree on cgroup v2."""
        if not control_group:
            return None, "systemd unit has no recorded control group"
        try:
            root = self._cgroup_root.resolve(strict=True)
            if not (root / "cgroup.controllers").is_file():
                return None, "cgroup v2 is unavailable"
            relative = Path(control_group.lstrip("/"))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                return None, "systemd control group path is invalid"
            base = root / relative
            base_resolved = base.resolve(strict=True)
            base_resolved.relative_to(root)
            if base_resolved != base:
                return None, "systemd control group path traverses a symbolic link"
            if base_resolved.name != relative.name or not relative.name.endswith(".service"):
                return None, "systemd control group identity is invalid"
        except (OSError, ValueError):
            return None, "systemd control group cannot be verified"

        pids: set[int] = set()
        try:
            for current, directories, _files in os.walk(base_resolved, topdown=True, followlinks=False):
                current_path = Path(current)
                if current_path.resolve(strict=True) != current_path:
                    return None, "systemd control group path traverses a symbolic link"
                current_path.relative_to(base_resolved)
                for name in list(directories):
                    child = current_path / name
                    if child.is_symlink():
                        return None, "systemd control group contains a symbolic link"
                    child.resolve(strict=True).relative_to(base_resolved)
                proc_file = current_path / "cgroup.procs"
                contents = proc_file.read_text(encoding="ascii")
                for line in contents.splitlines():
                    if not line.isdecimal() or int(line) <= 0:
                        return None, "cgroup.procs contains an invalid PID"
                    pids.add(int(line))
        except (OSError, ValueError):
            return None, "systemd control group process membership is unavailable or ambiguous"
        return sorted(pids), None

    def managed_process_ids(
        self, handles: Sequence[ExecutionHandle | Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Return PIDs proven to belong to supplied invocation-verified units.

        This is a point-in-time identity observation, not a process killer or a
        global census. Callers must supply all relevant active attempt handles;
        PIDs not returned remain unmanaged. Unknown handles are never used to
        classify process ownership.
        """
        if isinstance(handles, (str, bytes)) or not isinstance(handles, Sequence):
            raise ValueError("handles must be a sequence of execution handles")
        attempts: list[dict[str, Any]] = []
        all_pids: set[int] = set()
        uncertain = False
        seen_attempts: set[str] = set()
        for raw_handle in handles:
            if not isinstance(raw_handle, (ExecutionHandle, Mapping)):
                uncertain = True
                attempts.append({"state": "unknown", "reason": "unsupported execution handle"})
                continue
            handle = self._as_handle_data(raw_handle)
            attempt_id = handle.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in seen_attempts:
                uncertain = True
                attempts.append({"state": "unknown", "reason": "missing or duplicate attempt identity"})
                continue
            seen_attempts.add(attempt_id)
            props, problem = self._verify_handle(handle)
            unit_name = handle.get("unit_name")
            invocation_id = props.get("InvocationID")
            control_group = props.get("ControlGroup") or (handle.get("metadata") or {}).get("control_group")
            recorded_group = (handle.get("metadata") or {}).get("control_group")
            if not problem and recorded_group and props.get("ControlGroup") and recorded_group != props.get("ControlGroup"):
                problem = "systemd control group changed for the recorded invocation"
            if problem:
                uncertain = True
                attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "unit_name": unit_name,
                        "invocation_id": invocation_id,
                        "state": "unknown",
                        "process_ids": [],
                        "control_group": control_group,
                        "reason": problem,
                    }
                )
                continue
            pids, cgroup_error = self._cgroup_process_ids(control_group)
            if pids is None:
                uncertain = True
                attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "unit_name": unit_name,
                        "invocation_id": invocation_id,
                        "state": "unknown",
                        "process_ids": [],
                        "control_group": control_group,
                        "reason": cgroup_error,
                    }
                )
                continue
            duplicate_pids = all_pids.intersection(pids)
            if duplicate_pids:
                uncertain = True
                attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "unit_name": unit_name,
                        "invocation_id": invocation_id,
                        "state": "unknown",
                        "process_ids": [],
                        "control_group": control_group,
                        "reason": "PID appeared in multiple managed unit cgroups",
                    }
                )
                continue
            all_pids.update(pids)
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "unit_name": unit_name,
                    "invocation_id": invocation_id,
                    "state": "observed",
                    "process_ids": pids,
                    "control_group": control_group,
                }
            )
        return {
            "state": "unknown" if uncertain else "observed",
            "process_ids": sorted(all_pids),
            "attempts": attempts,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "scope": "invocation-verified-systemd-unit-cgroups",
            "coverage": "caller-supplied-handles",
        }

    def stop(
        self,
        handle: ExecutionHandle | Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility alias for callers that describe cancellation as stop."""
        return self.cancel(
            handle,
            operation_id=operation_id or f"compat-stop:{self._as_handle_data(handle).get('attempt_id', 'unknown')}",
            timeout_seconds=self._stop_timeout_seconds if timeout_seconds is None else timeout_seconds,
        )

    def collect(self, handle: ExecutionHandle | Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the bounded log and terminal receipt for an inspected attempt."""
        observation = self.inspect(handle)
        return {
            "adapter": self.version,
            "state": observation.get("state"),
            "exit_code": observation.get("exit_code"),
            "cleanup_verified": observation.get("cleanup_verified", False),
            "stdout": observation.get("logs", ""),
            "logs_truncated": observation.get("logs_truncated", False),
            "logs_error": observation.get("logs_error"),
            "evidence": observation.get("evidence", {}),
        }
