#!/usr/bin/env python3
"""Opt-in CPU-only qualification for the systemd user-unit adapter.

By default this script prints its plan and performs no systemd operations.  A
real run requires both ``--run`` and ``--confirm-user-unit``.  The run creates
one uniquely named, bounded transient user service, checks its resource-control
properties and cgroup containment, reconciles it from a fresh Python process,
then stops it and verifies that its process cgroup is empty.

This is an adapter/host qualification, not a scheduler-daemon crash test.  It
does not exercise GPU/model processes and must be coordinated before use on a
shared host.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _source in (
    REPO_ROOT / "packages/device-scheduler-core/src",
    REPO_ROOT / "packages/device-scheduler-adapters/src",
):
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from device_scheduler_adapters.systemd import (  # noqa: E402
    StartUncertain,
    SystemdExecutionAdapter,
    attempt_unit_name,
)
from device_scheduler_core.models import ExecutionHandle, JobSpec  # noqa: E402


MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
TASK_LIMIT = 32
CPU_QUOTA_PERCENT = 100
STOP_TIMEOUT_SECONDS = 2
CHILD_LIFETIME_SECONDS = 45
REQUIRED_CHECKS = frozenset(
    {
        "deterministic_unit_identity",
        "invocation_identity_bound",
        "cpu_quota_applied",
        "memory_limit_applied",
        "task_limit_applied",
        "whole_tree_cgroup_matches",
        "fresh_process_observed_running_unit",
        "fresh_process_resolved_same_invocation",
        "systemd_stop_acknowledged",
        "cgroup_cleanup_verified",
        "child_process_tree_gone",
    }
)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _preflight() -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("qualification requires Linux")
    if shutil.which("systemctl") is None or shutil.which("systemd-run") is None:
        raise RuntimeError("systemctl and systemd-run must be installed")
    controllers_path = Path("/sys/fs/cgroup/cgroup.controllers")
    try:
        controllers = set(controllers_path.read_text(encoding="ascii").split())
    except OSError as exc:
        raise RuntimeError("unified cgroup v2 controller list is unavailable") from exc
    required = {"cpu", "memory", "pids"}
    missing = sorted(required - controllers)
    if missing:
        raise RuntimeError(f"required cgroup v2 controllers are unavailable: {', '.join(missing)}")
    try:
        manager = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"user systemd manager preflight failed: {type(exc).__name__}") from exc
    if manager.returncode != 0:
        raise RuntimeError("the current user systemd manager is unavailable")
    return {
        "platform": sys.platform,
        "uid": os.getuid(),
        "systemctl": shutil.which("systemctl"),
        "systemd_run": shutil.which("systemd-run"),
        "cgroup_v2_controllers": sorted(controllers),
        "user_manager": "reachable",
    }


def _child_code() -> str:
    return (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep({CHILD_LIFETIME_SECONDS})"
    )


def _parent_code() -> str:
    child = repr(_child_code())
    return (
        "import subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{child}]); "
        "print('CHILD_PID='+str(child.pid),flush=True); "
        "child.wait()"
    )


def _process_record(pid: int) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    try:
        stat = (proc / "stat").read_text(encoding="ascii")
        command_end = stat.rfind(")")
        if command_end < 0:
            return {"state": "unknown", "reason": "malformed_proc_stat"}
        state = stat[command_end + 2 :].split(maxsplit=1)[0]
        cgroups = (proc / "cgroup").read_text(encoding="ascii").splitlines()
        return {"state": state, "cgroups": cgroups}
    except FileNotFoundError:
        return {"state": "not_present", "cgroups": []}
    except OSError as exc:
        return {"state": "unknown", "reason": type(exc).__name__}


def _wait_for_child(adapter: SystemdExecutionAdapter, handle: ExecutionHandle, timeout: float) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = adapter.inspect(handle)
        logs = latest.get("logs", "")
        for line in logs.splitlines():
            if line.startswith("CHILD_PID="):
                try:
                    return int(line.split("=", 1)[1]), latest
                except ValueError:
                    break
        if latest.get("state") != "running":
            break
        time.sleep(0.2)
    raise RuntimeError(f"unit did not report its child PID; last observation={latest.get('state')}")


def _manager_properties(unit_name: str) -> dict[str, str]:
    properties = ("Id", "Description", "InvocationID", "ControlGroup", "CPUQuotaPerSecUSec", "MemoryMax", "TasksMax")
    command = ["systemctl", "--user", "show", unit_name]
    command.extend(f"--property={name}" for name in properties)
    result = subprocess.run(command, text=True, capture_output=True, timeout=8, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"systemd could not report unit properties: {result.stderr.strip()[:500]}")
    output: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            output[name] = value
    return output


def _quota_seconds(value: str) -> float | None:
    value = value.strip()
    units = (("us", 1e-6), ("ms", 1e-3), ("s", 1.0), ("min", 60.0))
    for suffix, scale in units:
        if value.endswith(suffix):
            try:
                return float(value[: -len(suffix)]) * scale
            except ValueError:
                return None
    try:
        return float(value) / 1_000_000
    except ValueError:
        return None


def _restart_reconcile(state_path: Path, receipt_path: Path) -> dict[str, Any]:
    """Run read-only resolution in another interpreter, like a new client."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--inspect-state",
        str(state_path),
        "--receipt-path",
        str(receipt_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=25, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"fresh-process reconciliation failed: {result.stderr.strip()[:1000]}")
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _inspect_state(state_path: Path, receipt_path: Path) -> int:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    attempt_id = state.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise RuntimeError("state file has no valid attempt_id")
    adapter = SystemdExecutionAdapter()
    handle = adapter.resolve(attempt_id=attempt_id)
    observation: dict[str, Any]
    if handle is None:
        observation = {"state": "unknown", "cleanup_verified": False, "reason": "unit_absent"}
    else:
        observation = adapter.inspect(handle)
    receipt = {
        "schema_version": 1,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "fresh_process_reconciliation": True,
        "scheduler_daemon_restarted": False,
        "execution_handle": handle.to_dict() if handle else None,
        "observation": observation,
    }
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _run_qualification(output_parent: Path | None) -> tuple[int, Path]:
    if output_parent is None:
        output_parent = Path(tempfile.gettempdir())
    output_parent = output_parent.expanduser().resolve(strict=True)
    if not output_parent.is_dir():
        raise RuntimeError("output parent must be a directory")
    output_dir = Path(tempfile.mkdtemp(prefix="device-scheduler-systemd-qualification-", dir=output_parent))
    output_dir.chmod(0o700)
    attempt_id = f"qualification-{uuid.uuid4()}"
    state_path = output_dir / "intent.json"
    receipt_path = output_dir / "receipt.json"
    profile = {
        "argv": [sys.executable, "-c", _parent_code()],
        "working_directory": str(output_dir),
        "environment": {},
        "limits": {
            "cpu_quota_percent": CPU_QUOTA_PERCENT,
            "memory_max_bytes": MEMORY_LIMIT_BYTES,
            "tasks_max": TASK_LIMIT,
        },
        "profile_version": "systemd-qualification-v1",
    }
    profile_id = "systemd-qualification"
    adapter = SystemdExecutionAdapter(
        stop_timeout_seconds=STOP_TIMEOUT_SECONDS,
        profile_resolver=lambda requested_id: profile if requested_id == profile_id else {},
    )
    handle: ExecutionHandle | None = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "qualification": "systemd-user-unit-cpu-only",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "attempt_id": attempt_id,
        "unit_name": attempt_unit_name(attempt_id),
        "output_directory": str(output_dir),
        "scheduler_daemon_restarted": False,
        "scope_limitations": [
            "fresh Python-process adapter reconciliation only; scheduler daemon restart not exercised",
            "CPU-only finite helper; no GPU/model or malicious-workload qualification",
            "same-user cgroup controls do not establish hostile-tenant isolation",
        ],
        "checks": {},
    }
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "unit_name": attempt_unit_name(attempt_id),
            "intent_persisted_before_start": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        result["preflight"] = _preflight()
        spec = JobSpec(owner_id="host-qualification", profile_id=profile_id)
        try:
            handle = adapter.start(spec, attempt_id=attempt_id, execution_root=output_dir / "exec")
        except StartUncertain:
            # Resolve the deterministic identity before deciding whether a
            # systemd-run timeout actually left a unit behind.
            handle = adapter.resolve(attempt_id=attempt_id)
            if handle is None:
                raise
        result["execution_handle"] = handle.to_dict()
        _write_json(
            state_path,
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "unit_name": attempt_unit_name(attempt_id),
                "execution_handle": handle.to_dict(),
                "intent_persisted_before_start": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        child_pid, first_observation = _wait_for_child(adapter, handle, timeout=12)
        control_group = first_observation.get("evidence", {}).get("control_group")
        process_before_stop = _process_record(child_pid)
        unified_cgroup = next(
            (line.partition("::")[2] for line in process_before_stop.get("cgroups", []) if "::" in line),
            None,
        )
        properties = _manager_properties(handle.external_id)
        quota_seconds = _quota_seconds(properties.get("CPUQuotaPerSecUSec", ""))
        result["manager_properties"] = properties
        result["checks"].update(
            {
                "deterministic_unit_identity": properties.get("Id") == handle.external_id,
                "invocation_identity_bound": properties.get("InvocationID") == handle.metadata.get("invocation_id"),
                "cpu_quota_applied": quota_seconds is not None and abs(quota_seconds - 1.0) <= 0.01,
                "memory_limit_applied": properties.get("MemoryMax") == str(MEMORY_LIMIT_BYTES),
                "task_limit_applied": properties.get("TasksMax") == str(TASK_LIMIT),
                "whole_tree_cgroup_matches": bool(control_group and unified_cgroup == control_group),
                "fresh_process_observed_running_unit": first_observation.get("state") == "running",
            }
        )
        result["child"] = {
            "pid": child_pid,
            "before_stop": process_before_stop,
            "cgroup": unified_cgroup,
            "systemd_control_group": control_group,
        }

        fresh_path = output_dir / "fresh-process-reconciliation.json"
        fresh = _restart_reconcile(state_path, fresh_path)
        fresh_observation = fresh.get("observation", {})
        result["fresh_process_reconciliation"] = fresh
        result["checks"]["fresh_process_resolved_same_invocation"] = (
            fresh_observation.get("state") == "running"
            and fresh_observation.get("invocation_id") == handle.metadata.get("invocation_id")
        )

        stop_receipt = adapter.cancel(
            handle,
            operation_id=f"qualification-stop:{attempt_id}",
            timeout_seconds=STOP_TIMEOUT_SECONDS + 10,
        )
        result["stop_receipt"] = stop_receipt
        process_after_stop = _process_record(child_pid)
        result["child"]["after_stop"] = process_after_stop
        state_after = process_after_stop.get("state")
        result["checks"].update(
            {
                "systemd_stop_acknowledged": stop_receipt.get("stop_requested") is True,
                "cgroup_cleanup_verified": stop_receipt.get("cleanup_verified") is True,
                "child_process_tree_gone": state_after == "not_present" or state_after == "Z",
            }
        )
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        if handle is not None:
            try:
                cleanup = adapter.cancel(
                    handle,
                    operation_id=f"qualification-cleanup:{attempt_id}",
                    timeout_seconds=STOP_TIMEOUT_SECONDS + 10,
                )
                result["cleanup_after_error"] = cleanup
            except Exception as cleanup_exc:  # report evidence; never signal a raw PID
                result["cleanup_after_error"] = {
                    "state": "unknown",
                    "cleanup_verified": False,
                    "error": type(cleanup_exc).__name__,
                }
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    checks = result.get("checks", {})
    result["required_checks"] = sorted(REQUIRED_CHECKS)
    passed = (
        "error" not in result
        and REQUIRED_CHECKS.issubset(checks)
        and all(checks[name] is True for name in REQUIRED_CHECKS)
    )
    result["verdict"] = "pass" if passed else "incomplete"
    _write_json(receipt_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (0 if passed else 1), output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="create one real transient user service")
    parser.add_argument(
        "--confirm-user-unit",
        action="store_true",
        help="acknowledge that --run creates and stops a transient systemd user unit",
    )
    parser.add_argument("--output-parent", type=Path, help="directory for a new private receipt folder")
    parser.add_argument("--inspect-state", type=Path, help="read-only fresh-process reconciliation of an intent file")
    parser.add_argument("--receipt-path", type=Path, help="receipt destination for --inspect-state")
    args = parser.parse_args(argv)
    if args.inspect_state is not None:
        if args.run or args.confirm_user_unit or args.receipt_path is None:
            parser.error("--inspect-state requires --receipt-path and cannot be combined with --run")
        return _inspect_state(args.inspect_state, args.receipt_path)
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "side_effects": False,
                    "required_for_live_run": ["--run", "--confirm-user-unit", "root coordination"],
                    "qualification": "CPU-only bounded transient unit, resource controls, cgroup tree cleanup, fresh-process reconciliation",
                    "systemd_daemon_restart": "not exercised by this adapter-level script",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm_user_unit:
        parser.error("--run requires --confirm-user-unit")
    code, output_dir = _run_qualification(args.output_parent)
    print(f"receipt_directory={output_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
