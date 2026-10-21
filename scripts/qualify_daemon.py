#!/usr/bin/env python3
"""Qualify real HTTP/MCP and a CPU unit surviving a scheduler crash.

Default invocation prints the scope without starting anything. --run launches
an isolated daemon and one finite CPU helper; no GPU allocations are requested.
Private credentials and the diagnostic ledger stay in a fresh /tmp directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
import uuid

import httpx

from device_scheduler_adapters.systemd import SystemdExecutionAdapter
from device_scheduler_core.models import ExecutionHandle
from device_scheduler_mcp.client import DeviceSchedulerClient


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "health", "ready", "mcp_auth_denied", "sdk_submit", "cli_same_job", "running_before_crash",
    "same_attempt_after_crash", "same_invocation_after_crash", "completed", "one_physical_launch",
    "cleanup_verified", "cpu_only_visibility", "grant_limits", "source_unchanged",
}


def production_hashes() -> dict[str, str]:
    paths = [path for package in (ROOT / "packages").iterdir() if package.is_dir()
             for path in (package / "src").rglob("*.py")]
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def private_write(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)


async def qualify(receipt_path: Path) -> int:
    source_hashes = production_hashes()
    work = Path(tempfile.mkdtemp(prefix="device-scheduler-daemon-qualification-"))
    work.chmod(0o700)
    tokens = work / "tokens"
    tokens.mkdir(mode=0o700)
    bearer = secrets.token_urlsafe(48)
    token_file = work / "client.token"
    private_write(token_file, bearer + "\n")
    private_write(tokens / "qualification.toml", (
        '[principal]\nprincipal_id="qualification"\nactions=["*"]\nmax_priority=100\n'
        f'execution_roots=[{json.dumps(str(work))}]\nbackend_ids=[]\n'
        f'token_sha256="{hashlib.sha256(bearer.encode()).hexdigest()}"\n'
    ))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = work / "service.toml"
    private_write(config, (
        f'[server]\nhost="127.0.0.1"\nport={port}\ntick_interval_seconds=0.2\n'
        f'[paths]\nstate_directory={json.dumps(str(work / "state"))}\n'
        f'[auth]\ntoken_directory={json.dumps(str(tokens))}\ntoken_file={json.dumps(str(token_file))}\n'
    ))
    origin = f"http://127.0.0.1:{port}"
    client = DeviceSchedulerClient(origin, token_file, timeout_seconds=10)
    command = [sys.executable, "-c", "import faulthandler,signal; faulthandler.register(signal.SIGUSR1,all_threads=True); from device_scheduler_mcp.cli import main; raise SystemExit(main())", "--config", str(config), "serve", "--log-level", "error"]
    daemon: subprocess.Popen | None = None
    job_id = None
    handle = None
    report = {"schema_version": 1, "scope": "live-http-mcp-and-cpu-daemon-crash", "checks": {},
              "receipt_id": "daemon-qualification-" + uuid.uuid4().hex,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "production_source_hashes": source_hashes,
              "diagnostic_directory": str(work), "gpu_allocations_requested": False,
              "limitations": ["one cooperative finite CPU helper; not GPU or model qualification"]}

    async def start() -> subprocess.Popen:
        with (work / "daemon.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
        deadline = time.monotonic() + 25
        async with httpx.AsyncClient(timeout=2) as http:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("qualification daemon exited; inspect private daemon.log")
                try:
                    ready = await http.get(origin + "/readyz")
                    if ready.status_code == 200 and ready.json().get("status") == "ready":
                        return process
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        process.terminate()
        process.wait(timeout=10)
        raise RuntimeError("qualification daemon did not become ready")

    try:
        daemon = await start()
        async with httpx.AsyncClient(timeout=5) as http:
            health = await http.get(origin + "/healthz")
            ready = await http.get(origin + "/readyz")
            denied = await http.post(origin + "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                     headers={"Accept": "application/json, text/event-stream"})
        report["checks"]["health"] = health.status_code == 200
        report["checks"]["ready"] = ready.status_code == 200 and ready.json().get("status") == "ready"
        report["checks"]["mcp_auth_denied"] = denied.status_code == 401
        code = (
            "from pathlib import Path; import os,time; "
            f"p=Path({str(work / 'launches')!r}); "
            f"Path({str(work / 'gpu-visibility')!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES','MISSING')); "
            "f=p.open('a'); f.write(str(os.getpid())+'\\n'); f.flush(); os.fsync(f.fileno()); f.close(); "
            "time.sleep(12); print('qualification-complete',flush=True)"
        )
        submitted = await client.call("jobs.submit", {"idempotency_key": "daemon-crash-qualification-v1", "spec": {
            "argv": [sys.executable, "-c", code], "working_directory": str(work), "timeout_seconds": 45,
            "resources": [{"kind": "cpu", "quantities": {"cores": 1}},
                          {"kind": "memory", "quantities": {"bytes": 256 * 1024**2}}],
        }})
        job_id = submitted["id"]
        report["job_id"] = job_id
        report["checks"]["sdk_submit"] = submitted["state"] == "queued"
        deadline = time.monotonic() + 20
        running = None
        while time.monotonic() < deadline:
            running = await client.call("jobs.get", {"job_id": job_id})
            if running["state"] == "running" and (running.get("attempt") or {}).get("execution_handle"):
                break
            await asyncio.sleep(0.1)
        report["checks"]["running_before_crash"] = running["state"] == "running" and bool((running.get("attempt") or {}).get("execution_handle"))
        if not report["checks"]["running_before_crash"]:
            raise RuntimeError("CPU job did not reach running")
        handle = running["attempt"]["execution_handle"]
        attempt_id = running["attempt_id"]
        invocation = handle["metadata"]["invocation_id"]
        unit = handle["external_id"]
        properties = await asyncio.to_thread(subprocess.run,
            ["systemctl", "--user", "show", unit, "--property=CPUQuotaPerSecUSec,MemoryMax,RuntimeMaxUSec"],
            capture_output=True, text=True, timeout=10, check=True)
        values = dict(line.split("=", 1) for line in properties.stdout.splitlines())
        report["unit_limits"] = values
        report["checks"]["grant_limits"] = values == {
            "CPUQuotaPerSecUSec": "1s", "MemoryMax": str(256 * 1024**2), "RuntimeMaxUSec": "45s"}
        cli = await asyncio.to_thread(subprocess.run, [sys.executable, "-m", "device_scheduler_mcp", "--config", str(config),
            "jobs", "get", job_id], capture_output=True, text=True, timeout=10)
        report["checks"]["cli_same_job"] = cli.returncode == 0 and json.loads(cli.stdout)["attempt_id"] == attempt_id
        daemon.kill()
        daemon.wait(timeout=10)
        daemon = await start()
        reconciled = await client.call("jobs.get", {"job_id": job_id})
        report["checks"]["same_attempt_after_crash"] = reconciled["attempt_id"] == attempt_id
        report["checks"]["same_invocation_after_crash"] = reconciled["attempt"]["execution_handle"]["metadata"]["invocation_id"] == invocation
        deadline = time.monotonic() + 45
        final = reconciled
        while time.monotonic() < deadline:
            final = await client.call("jobs.get", {"job_id": job_id})
            if final["state"] in {"succeeded", "failed", "cancelled", "unknown"}:
                break
            await asyncio.sleep(0.25)
        report["checks"]["completed"] = final["state"] == "succeeded"
        report["checks"]["cleanup_verified"] = final.get("cleanup_verified") is True
        report["checks"]["one_physical_launch"] = len((work / "launches").read_text().splitlines()) == 1
        report["checks"]["cpu_only_visibility"] = (work / "gpu-visibility").read_text() == ""
        report["attempt_id"] = attempt_id
        report["invocation_id"] = invocation
        report["final_state"] = final["state"]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        diagnostic = "".join(traceback.format_exception(exc)).replace(bearer, "[redacted]")
        private_write(work / "client-error.log", diagnostic)
        if daemon is not None and daemon.poll() is None:
            daemon.send_signal(signal.SIGUSR1)
            await asyncio.sleep(0.2)
    finally:
        if daemon is not None and daemon.poll() is None:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
        if handle is not None:
            try:
                report["cleanup"] = SystemdExecutionAdapter().cancel(ExecutionHandle.from_dict(handle), operation_id="qualification-final-cleanup", timeout_seconds=10)
            except Exception as exc:
                report["cleanup"] = {"cleanup_verified": False, "error": type(exc).__name__}
                report.setdefault("error", {"type": type(exc).__name__, "message": "qualification unit cleanup unresolved"})
        report["checks"]["source_unchanged"] = production_hashes() == source_hashes
        report["observed_at"] = datetime.now(timezone.utc).isoformat()
        report["verdict"] = "pass" if "error" not in report and REQUIRED <= report["checks"].keys() and all(report["checks"].values()) else "incomplete"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--receipt", type=Path, default=ROOT / "planning/qualification-v1/daemon-receipt.json")
    args = parser.parse_args()
    if not args.run:
        print("Plan: isolated daemon, authenticated HTTP/MCP/CLI, one finite CPU job, SIGKILL daemon, reconcile same unit; no GPU allocations.")
        return 0
    if args.receipt.exists():
        parser.error("receipt already exists; select a new evidence path")
    return asyncio.run(qualify(args.receipt))


if __name__ == "__main__":
    raise SystemExit(main())
