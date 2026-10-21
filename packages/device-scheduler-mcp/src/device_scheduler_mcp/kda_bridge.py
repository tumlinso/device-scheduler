"""Narrow compatibility transport for trusted KDA/KernelBender callers.

This bridge submits one bounded raw-argv job through the authenticated
Device Scheduler SDK. Its configured principal, execution roots and resource
scope are operator-owned settings; fields in the submitted request do not
grant additional authority. The bridge never contacts the legacy controller.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

from device_scheduler_core import (
    AllocationMode,
    CapabilityPrecondition,
    GPU_EXCLUSIVE_TAG,
    JobSpec,
    ResourceRequirement,
    ValidationError,
)

from .client import DeviceSchedulerClient, DeviceSchedulerClientError


class BridgeRefusal(ValueError):
    """A request cannot be represented with the configured scheduler scope."""


def _read_object(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BridgeRefusal(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise BridgeRefusal(f"expected a JSON object: {path}")
    return value


def _controller_request(spec_path: Path, spec: Mapping[str, Any], scheduler_config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the subset of legacy controller specs this adapter can honor."""
    if spec.get("schema_version") != 1:
        raise BridgeRefusal("legacy controller spec schema_version must be 1")
    argv = spec.get("argv")
    cwd = _require_text(spec.get("command_cwd"), "controller spec command_cwd")
    timeout = _positive_finite(spec.get("timeout"), "controller spec timeout", maximum=86_400)
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
        raise BridgeRefusal("legacy controller spec argv must be a non-empty array")
    legacy_resources = spec.get("resources", {})
    if not isinstance(legacy_resources, Mapping):
        raise BridgeRefusal("legacy controller resources must be an object")
    required_gpu_count = legacy_resources.get("gpus")
    if isinstance(required_gpu_count, bool) or not isinstance(required_gpu_count, int) or required_gpu_count < 1:
        raise BridgeRefusal("legacy controller request must specify a positive GPU count")
    configured = _configured_requirements(scheduler_config.get("resource_requirements"))
    configured_uuids = [item.selectors.get("uuid") for item in configured if item.kind.casefold() in {"gpu", "nvidia-gpu"}]
    if len(configured_uuids) != required_gpu_count:
        raise BridgeRefusal("Device Scheduler resource UUID count differs from the legacy request")
    legacy_uuids = legacy_resources.get("gpu_uuids")
    if legacy_uuids is not None:
        if (not isinstance(legacy_uuids, Sequence) or isinstance(legacy_uuids, (str, bytes))
                or any(not isinstance(item, str) for item in legacy_uuids)):
            raise BridgeRefusal("legacy controller gpu_uuids must be an array")
        if sorted(legacy_uuids) != sorted(configured_uuids):
            raise BridgeRefusal("Device Scheduler GPU UUID selectors differ from the legacy request")
    source_identity = _digest({"spec_path": str(spec_path.resolve()), "spec": dict(spec)})
    environment = spec.get("environment", {})
    if not isinstance(environment, Mapping):
        raise BridgeRefusal("legacy controller environment must be an object")
    result: dict[str, Any] = {
        "schema_version": 1,
        "request_id": f"controller-spec:{source_identity}",
        "argv": list(argv),
        "cwd": cwd,
        "timeout_seconds": timeout,
        "environment": dict(environment),
        "caller_contract": {
            key: spec.get(key)
            for key in ("campaign_id", "recipe", "toolchain", "binary_paths", "paths", "diagnostic_policy")
            if key in spec
        },
    }
    if "quiescence" in spec:
        result["quiescence"] = spec["quiescence"]
    return result


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BridgeRefusal(f"{name} must be a non-empty string")
    return value


def _positive_finite(value: Any, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise BridgeRefusal(f"{name} must be finite and positive")
    result = float(value)
    if maximum is not None and result > maximum:
        raise BridgeRefusal(f"{name} exceeds {maximum}")
    return result


def _configured_requirements(value: Any) -> tuple[ResourceRequirement, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise BridgeRefusal("device_scheduler.resource_requirements must be a non-empty array")
    try:
        requirements = tuple(
            item if isinstance(item, ResourceRequirement) else ResourceRequirement.from_dict(item)
            for item in value
        )
    except (TypeError, ValidationError) as exc:
        raise BridgeRefusal(f"invalid device-scheduler resource requirement: {exc}") from exc
    gpu_requirements = [item for item in requirements if item.kind.casefold() in {"gpu", "nvidia-gpu"}]
    if not gpu_requirements:
        raise BridgeRefusal("this CUDA bridge requires an explicit GPU resource requirement")
    normalized: list[ResourceRequirement] = []
    for requirement in requirements:
        if requirement.kind.casefold() not in {"gpu", "nvidia-gpu"}:
            normalized.append(requirement)
            continue
        quantities = dict(requirement.quantities)
        unsupported = set(quantities) - {"gpu_memory_bytes", "memory_bytes"}
        if unsupported:
            raise BridgeRefusal(
                "the configured NVIDIA adapter only supports byte-valued GPU memory requests; "
                f"unsupported quantities: {', '.join(sorted(unsupported))}"
            )
        if "gpu_memory_bytes" in quantities:
            if "memory_bytes" in quantities:
                raise BridgeRefusal("GPU requests cannot specify both gpu_memory_bytes and memory_bytes")
            # The NVIDIA resource descriptor publishes the canonical capacity
            # key `memory_bytes`; normalize the legacy bridge spelling before
            # the allocator performs exact quantity-to-capacity matching.
            quantities["memory_bytes"] = quantities.pop("gpu_memory_bytes")
            requirement = replace(requirement, quantities=quantities)
        if "memory_bytes" not in requirement.quantities:
            raise BridgeRefusal("GPU requests require an explicit byte-valued GPU memory estimate")
        uuid = requirement.selectors.get("uuid")
        if not isinstance(uuid, str) or not uuid.strip():
            raise BridgeRefusal("GPU requests require a stable UUID selector")
        normalized.append(requirement)
    return tuple(normalized)


def _preconditions(value: Any) -> tuple[CapabilityPrecondition, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise BridgeRefusal("quiescence must be an object")
    allowed = {"timeout_seconds", "interval_seconds", "consecutive_idle_samples"}
    unknown = set(value) - allowed
    if unknown:
        raise BridgeRefusal(f"unsupported legacy quiescence fields: {', '.join(sorted(unknown))}")
    if not value:
        raise BridgeRefusal("an empty quiescence request is ambiguous")
    timeout = _positive_finite(value.get("timeout_seconds", 60), "quiescence.timeout_seconds", maximum=600)
    interval = _positive_finite(value.get("interval_seconds", 1), "quiescence.interval_seconds", maximum=60)
    samples = value.get("consecutive_idle_samples", 3)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 1000:
        raise BridgeRefusal("quiescence.consecutive_idle_samples must be an integer from 1 through 1000")
    return (CapabilityPrecondition("gpu.quiescence", {
        "timeout_seconds": timeout,
        "interval_seconds": interval,
        "consecutive_samples": samples,
        "max_utilization_percent": 0,
    }),)


def _scope(config: Mapping[str, Any], cwd: Path) -> tuple[str, tuple[Path, ...], dict[str, Any]]:
    owner_id = _require_text(config.get("owner_id"), "device_scheduler.owner_id")
    roots_raw = config.get("execution_roots")
    if not isinstance(roots_raw, Sequence) or isinstance(roots_raw, (str, bytes)) or not roots_raw:
        raise BridgeRefusal("device_scheduler.execution_roots must be an explicit non-empty array")
    roots: list[Path] = []
    for item in roots_raw:
        root = Path(_require_text(item, "execution root")).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise BridgeRefusal(f"execution root is not a directory: {root}")
        roots.append(root)
    if not any(cwd == root or root in cwd.parents for root in roots):
        raise BridgeRefusal("job working directory is outside configured execution_roots")
    maximum_priority = config.get("max_priority")
    if isinstance(maximum_priority, bool) or not isinstance(maximum_priority, int) or not 0 <= maximum_priority <= 100:
        raise BridgeRefusal("device_scheduler.max_priority must be an integer from 0 through 100")
    scope_binding = {
        "principal_id": owner_id,
        "actions": ["jobs.submit", "jobs.read", "jobs.control"],
        "execution_roots": [str(item) for item in roots],
        "max_priority": maximum_priority,
        "resource_requirements": config.get("resource_requirements"),
        "binding_source": "operator_configuration",
    }
    return owner_id, tuple(roots), scope_binding


def _normalize_job(config: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[JobSpec, dict[str, Any]]:
    if request.get("schema_version") != 1:
        raise BridgeRefusal("launcher request schema_version must be 1")
    request_id = _require_text(request.get("request_id"), "request_id")
    argv = request.get("argv")
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv or any(
        not isinstance(arg, str) or "\x00" in arg for arg in argv
    ):
        raise BridgeRefusal("worker argv must be a non-empty string array")
    if not argv[0].strip():
        raise BridgeRefusal("worker executable must not be empty")
    cwd = Path(_require_text(request.get("cwd"), "cwd")).expanduser().resolve(strict=True)
    if not cwd.is_dir():
        raise BridgeRefusal("worker cwd must be a directory")
    owner_id, _roots, binding = _scope(config, cwd)
    outer_timeout = _positive_finite(request.get("timeout_seconds"), "timeout_seconds", maximum=86_400)
    priority_class = config.get("priority_class", "normal")
    if priority_class not in {"interactive", "normal", "background"}:
        raise BridgeRefusal("device_scheduler.priority_class is invalid")
    priority = config.get("priority", 50)
    max_priority = binding["max_priority"]
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= max_priority:
        raise BridgeRefusal("configured job priority exceeds its explicit operator grant")
    environment = config.get("environment", {})
    request_environment = request.get("environment", {})
    if not isinstance(environment, Mapping) or not isinstance(request_environment, Mapping) or any(
        not isinstance(key, str) or not key or "\x00" in key or
        not isinstance(value, str) or "\x00" in value for key, value in environment.items()
    ) or any(
        not isinstance(key, str) or not key or "\x00" in key or
        not isinstance(value, str) or "\x00" in value for key, value in request_environment.items()
    ):
        raise BridgeRefusal("device_scheduler.environment must be a string mapping")
    sensitive_name = re.compile(r"(?:secret|password|credential|token|api[_-]?key|private[_-]?key)", re.IGNORECASE)
    if any(sensitive_name.search(key) for key in (*environment, *request_environment)):
        raise BridgeRefusal("device_scheduler.environment cannot contain secret-like variable names")
    merged_environment = dict(environment)
    merged_environment.update(request_environment)
    raw_tags = config.get("tags", ())
    if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)) or any(not isinstance(tag, str) for tag in raw_tags):
        raise BridgeRefusal("device_scheduler.tags must be an array of strings")
    tags = set(raw_tags)
    requirements = _configured_requirements(config.get("resource_requirements"))
    if any(item.mode is AllocationMode.EXCLUSIVE for item in requirements):
        tags.add(GPU_EXCLUSIVE_TAG)
    preconditions = _preconditions(request.get("quiescence", config.get("quiescence")))
    if preconditions and GPU_EXCLUSIVE_TAG not in tags:
        raise BridgeRefusal("GPU quiescence requires an explicit whole-device exclusivity request")
    cleanup_timeout = _positive_finite(config.get("cleanup_timeout_seconds", 15), "cleanup_timeout_seconds", maximum=30)
    deadline_margin = _positive_finite(config.get("deadline_margin_seconds", 5), "deadline_margin_seconds", maximum=30)
    work_timeout = outer_timeout - cleanup_timeout - deadline_margin
    if work_timeout <= 0:
        raise BridgeRefusal("launcher timeout must leave time for cancellation and verified cleanup")
    spec = JobSpec(
        owner_id=owner_id,
        argv=tuple(argv),
        working_directory=str(cwd),
        environment=merged_environment,
        resources=requirements,
        tags=frozenset(tags),
        priority_class=priority_class,
        priority=priority,
        timeout_seconds=work_timeout,
        max_attempts=1,
        idempotency_key=f"kda:{request_id}",
        preconditions=preconditions,
    )
    caller_contract = request.get("caller_contract")
    if caller_contract is not None:
        if not isinstance(caller_contract, Mapping):
            raise BridgeRefusal("caller_contract must be an object")
        binding["caller_contract_digest"] = _digest(caller_contract)
    binding["scope_digest"] = _digest(binding)
    return spec, {"request_id": request_id, "scope_binding": binding}


def _digest(value: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _job_from(response: Mapping[str, Any]) -> Mapping[str, Any]:
    job = response.get("job", response)
    if not isinstance(job, Mapping):
        raise BridgeRefusal("scheduler response did not include a job")
    return job


async def run_request(
    config: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    client: Any | None = None,
    stop_event: asyncio.Event | None = None,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    """Submit once and wait for verified completion; unknown remains failure."""
    endpoint = _require_text(config.get("base_url", "http://127.0.0.1:8770"), "base_url")
    token_file = Path(_require_text(config.get("token_file"), "device_scheduler.token_file")).expanduser()
    if not token_file.is_absolute():
        raise BridgeRefusal("device_scheduler.token_file must be an absolute host-managed path")
    poll = _positive_finite(poll_interval_seconds, "poll_interval_seconds", maximum=30)
    cleanup_timeout = _positive_finite(config.get("cleanup_timeout_seconds", 15), "cleanup_timeout_seconds", maximum=30)
    if client is None:
        client = DeviceSchedulerClient(endpoint, token_file, timeout_seconds=min(30.0, max(5.0, poll + 3)))
    spec, envelope = _normalize_job(config, request)
    timeout = _positive_finite(request.get("timeout_seconds"), "timeout_seconds", maximum=86_400)
    budget = time.monotonic() + timeout
    submitted = await client.submit_job(spec)
    job = _job_from(submitted)
    job_id = _require_text(job.get("job_id", job.get("id")), "scheduler job_id")
    if job.get("owner", job.get("owner_id")) != spec.owner_id:
        raise BridgeRefusal("scheduler accepted the job under an unexpected owner")
    state = str(job.get("state", "unknown"))
    was_cancelled = False
    try:
        while state not in {"succeeded", "failed", "cancelled", "unknown"}:
            if stop_event is not None and stop_event.is_set():
                was_cancelled = True
                break
            remaining = budget - time.monotonic()
            if remaining <= max(2.0, cleanup_timeout) + 2:
                was_cancelled = True
                break
            waited = await client.wait_job(job_id, timeout_seconds=min(poll, remaining - 1))
            job = _job_from(waited)
            if waited.get("timed_out") is not True:
                state = str(job.get("state", "unknown"))
    except Exception:
        # A broken wait transport is not evidence that the job ended. Make one
        # explicit cancellation attempt and preserve unknown if it cannot be
        # reconciled.
        was_cancelled = True
    if was_cancelled or state not in {"succeeded", "failed", "cancelled"}:
        try:
            cancelled = await client.control_job(job_id, "cancel", reason="kda_bridge_deadline_or_signal")
            job = _job_from(cancelled)
            state = str(job.get("state", state))
        except Exception:
            # A failed cancellation request does not prove the execution stopped.
            pass
        cleanup_deadline = time.monotonic() + max(2.0, cleanup_timeout)
        while state not in {"succeeded", "failed", "cancelled", "unknown"} and time.monotonic() < cleanup_deadline:
            waited = await client.wait_job(job_id, timeout_seconds=min(poll, cleanup_deadline - time.monotonic()))
            job = _job_from(waited)
            state = str(job.get("state", "unknown"))
        if state not in {"succeeded", "failed", "cancelled"}:
            try:
                job = _job_from(await client.get_job(job_id))
                state = str(job.get("state", "unknown"))
            except Exception:
                state = "unknown"
    attempt = job.get("attempt") if isinstance(job.get("attempt"), Mapping) else {}
    cleanup_verified = job.get("cleanup_verified") is True or attempt.get("cleanup_verified") is True
    attempt_id = job.get("attempt_id") or attempt.get("attempt_id")
    result: dict[str, Any] = {
        "schema_version": 1,
        "request_id": envelope["request_id"],
        "status": "ok" if state == "succeeded" and cleanup_verified else "failed",
        "exit_code": 0 if state == "succeeded" and cleanup_verified else 1,
        "device_scheduler": {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "state": state,
            "cleanup_verified": cleanup_verified,
            "owner_id": spec.owner_id,
            "scope_binding": envelope["scope_binding"],
            "result": job.get("result"),
            "error": job.get("blocked_reason") or job.get("error"),
            "cancel_requested": was_cancelled,
        },
    }
    try:
        logs = await client.job_logs(job_id)
        result["device_scheduler"]["logs"] = logs
    except Exception as exc:
        result["device_scheduler"]["logs_error"] = type(exc).__name__
    return result


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _persist_job_logs(result: dict[str, Any], receipt_path: Path) -> None:
    details = result.get("device_scheduler")
    if not isinstance(details, dict):
        return
    logs = details.get("logs")
    if not isinstance(logs, Mapping):
        return
    execution: dict[str, Any] = {}
    for name in ("stdout", "stderr"):
        value = logs.get(name, "")
        if isinstance(value, Mapping):
            content = value.get("text", "")
            truncated = value.get("truncated", False)
        else:
            content, truncated = value, False
        if not isinstance(content, str):
            continue
        path = receipt_path.with_name(f"{receipt_path.stem}.{name}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        import hashlib

        execution[f"{name}_path"] = str(path)
        execution[f"{name}_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        execution[f"{name}_truncated"] = bool(truncated)
    if execution:
        details["execution"] = execution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="site config containing device_scheduler")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--request", type=Path, help="one KDA launcher request JSON")
    input_group.add_argument("--controller-spec", type=Path, help="translate a bounded legacy controller job spec")
    parser.add_argument("--result", type=Path, required=True, help="new launcher receipt path")
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        print("device-scheduler bridge: refusing to overwrite launcher receipt", file=sys.stderr)
        return 1
    stop_event = asyncio.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    old_handlers: dict[int, Any] = {}
    if __import__("threading").current_thread() is __import__("threading").main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
    try:
        config = _read_object(args.config.resolve(strict=True))
        scheduler_config = config.get("device_scheduler")
        if not isinstance(scheduler_config, Mapping):
            raise BridgeRefusal("selected site config has no device_scheduler object")
        request = (
            _read_object(args.request.resolve(strict=True))
            if args.request is not None else
            _controller_request(
                args.controller_spec.resolve(strict=True),
                _read_object(args.controller_spec.resolve(strict=True)),
                scheduler_config,
            )
        )
        try:
            result = asyncio.run(run_request(scheduler_config, request, stop_event=stop_event))
        except (DeviceSchedulerClientError, OSError, TimeoutError) as exc:
            request_id = request.get("request_id")
            result = {
                "schema_version": 1,
                "request_id": request_id,
                "status": "failed",
                "exit_code": 1,
                "device_scheduler": {"state": "unknown", "error_type": type(exc).__name__},
            }
        _persist_job_logs(result, args.result.resolve())
        _atomic_write(args.result.resolve(), result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result.get("status") == "ok" else 1
    except (BridgeRefusal, KeyError, OSError, ValidationError, ValueError) as exc:
        print(f"device-scheduler bridge: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
