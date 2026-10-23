#!/usr/bin/env python3
"""Opt-in smoke qualification for generic Device Scheduler V1.

This talks only to an already-running loopback daemon. Without --execute it
prints scope and does not connect. It never starts a daemon, requests a GPU,
controls a backend, or calls FrameShift. Mutations use fresh IDs tagged with a
single run marker; the only cleanup operation targets the schedule created by
this run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SOURCE_ROOT = Path(os.environ.get("DEVICE_SCHEDULER_SOURCE", "/home/tumlinson/device-scheduler")).resolve()
for _package in ("device-scheduler-core", "device-scheduler-mcp"):
    _src = SOURCE_ROOT / "packages" / _package / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from device_scheduler_mcp.client import DeviceSchedulerClient  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_identity() -> dict[str, Any]:
    paths = sorted(
        path for package in ("device-scheduler-core", "device-scheduler-adapters", "device-scheduler-mcp")
        for path in (SOURCE_ROOT / "packages" / package / "src").rglob("*.py")
    )
    hashes = {str(path.relative_to(SOURCE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths if path.is_file()}
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(SOURCE_ROOT), *args], check=True,
                                  capture_output=True, text=True, timeout=3).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "source_root": str(SOURCE_ROOT),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "source_hashes": hashes,
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
    }


def atomic_receipt(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("daemon URL must be HTTP loopback")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("daemon URL cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("provide daemon origin only; MCP path is configured separately")
    port = parsed.port or 80
    return value.rstrip("/"), parsed.hostname, port


def _write_client_config(path: Path, *, host: str, port: int, token_file: Path, mcp_path: str) -> None:
    def q(value: str) -> str:
        return json.dumps(value)
    text = (
        f'[server]\nhost = {q(host)}\nport = {port}\nmcp_path = {q(mcp_path)}\n'
        f'\n[auth]\ntoken_file = {q(str(token_file))}\n'
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _job_spec(marker: str, root: Path, code: str, *, timeout: float = 90.0,
              not_before: datetime | None = None, dependencies: list[str] | None = None,
              event_trigger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "argv": [sys.executable, "-c", code],
        "working_directory": str(root),
        "timeout_seconds": timeout,
        "resources": [
            {"kind": "cpu", "quantities": {"cores": 0.25}},
            {"kind": "memory", "quantities": {"bytes": 128 * 1024 * 1024}},
        ],
        "tags": [marker],
    }
    if not_before is not None:
        result["not_before"] = not_before.isoformat()
    if dependencies:
        result["dependencies"] = dependencies
    if event_trigger is not None:
        result["event_trigger"] = dict(event_trigger)
    return result


async def _stdio_read(config_path: Path, job_id: str) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "device_scheduler_mcp", "--config", str(config_path), "mcp-stdio"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in sys.path if p)},
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            if "jobs_get" not in names:
                raise RuntimeError("stdio bridge did not expose jobs_get")
            result = await session.call_tool("jobs_get", {"arguments": {"job_id": job_id}})
            if getattr(result, "isError", False):
                raise RuntimeError("stdio jobs_get returned an MCP error")
            value = getattr(result, "structuredContent", None)
            if isinstance(value, dict):
                return {"tool_count": len(names), "job": value}
            for block in getattr(result, "content", ()):
                raw = getattr(block, "text", None)
                if raw:
                    decoded = json.loads(raw)
                    if isinstance(decoded, dict):
                        return {"tool_count": len(names), "job": decoded}
            raise RuntimeError("stdio jobs_get returned no structured job")


async def _poll_job(client: DeviceSchedulerClient, job_id: str, timeout: float = 60.0,
                    overall_deadline: float | None = None) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    if overall_deadline is not None:
        deadline = min(deadline, overall_deadline)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = await client.get_job(job_id)
        if last.get("state") in {"succeeded", "failed", "cancelled", "unknown"}:
            return last
        await asyncio.sleep(0.15)
    return last


async def qualify(args: argparse.Namespace) -> int:
    origin, host, port = _origin(args.url)
    token_path = args.token_file.expanduser().resolve()
    execution_root = args.execution_root.expanduser().resolve()
    if not token_path.is_file():
        raise ValueError("operator token file is unavailable")
    if not execution_root.is_dir():
        raise ValueError("execution root must already exist")
    marker = "generic-v1-" + uuid.uuid4().hex[:12]
    work_dir = Path(tempfile.mkdtemp(prefix=marker + "-", dir=execution_root))
    work_dir.chmod(0o700)
    receipt_path = args.receipt.expanduser().resolve() if args.receipt else work_dir / "qualification-receipt.json"
    config_path = work_dir / "client-config.toml"
    _write_client_config(config_path, host=host, port=port, token_file=token_path, mcp_path=args.mcp_path)
    client = DeviceSchedulerClient(origin, token_path, timeout_seconds=args.timeout, mcp_path=args.mcp_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": marker,
        "scope": "generic-v1-existing-loopback-daemon",
        "started_at": utc_now(),
        "daemon_origin": origin,
        "daemon_config_path": None,
        "client_config_path": str(config_path),
        "token_path": str(token_path),
        "execution_root": str(work_dir),
        "mutations_confined_to_marker": marker,
        "forbidden_operations": ["GPU allocation", "backend lifecycle", "FrameShift", "other clients' jobs/schedules"],
        "source_runtime": source_identity(),
        "checks": {},
        "created": {"jobs": [], "schedules": [], "events": [], "leases": []},
        "uncertain": [],
        "limitations": ["single host and existing daemon only", "no GPU/backend/FrameShift qualification"],
    }
    atomic_receipt(receipt_path, report)
    overall_deadline = time.monotonic() + 300

    def require_time() -> None:
        if time.monotonic() >= overall_deadline:
            raise TimeoutError("generic V1 qualification exceeded its five-minute budget")

    async def call(operation: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return await client.call(operation, payload or {})

    async def submit(key: str, spec: Mapping[str, Any]) -> dict[str, Any]:
        require_time()
        return await call("jobs.submit", {"idempotency_key": key, "spec": dict(spec)})

    async def save() -> None:
        report["updated_at"] = utc_now()
        atomic_receipt(receipt_path, report)

    schedule_id: str | None = None
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as http:
            health = await http.get(origin + "/healthz")
            ready = await http.get(origin + "/readyz")
        report["health"] = health.json() if health.headers.get("content-type", "").startswith("application/json") else {"status_code": health.status_code}
        report["readiness"] = ready.json() if ready.headers.get("content-type", "").startswith("application/json") else {"status_code": ready.status_code}
        report["checks"]["health"] = health.status_code == 200 and report["health"].get("status") == "ok"
        report["checks"]["ready"] = ready.status_code == 200 and report["readiness"].get("status") == "ready"
        if not report["checks"]["health"] or not report["checks"]["ready"]:
            raise RuntimeError("existing daemon is not healthy and ready")

        status = await call("admin.status")
        report["daemon_status"] = {key: status.get(key) for key in ("status", "scheduler", "adapters")}
        await save()

        # Basic execution and duplicate-submit idempotency through the SDK's MCP transport.
        hello = f"print({(marker + '-sdk')!r})"
        sdk_key = marker + "-sdk-job"
        sdk_spec = _job_spec(marker, work_dir, hello)
        first = await submit(sdk_key, sdk_spec)
        report["created"]["jobs"].append(first.get("id"))
        duplicate = await submit(sdk_key, sdk_spec)
        report["checks"]["duplicate_submit_same_id"] = first.get("id") == duplicate.get("id")
        if first.get("id") != duplicate.get("id"):
            raise RuntimeError("idempotent duplicate submit created a second job")
        job_id = str(first["id"])
        report["sdk_job_id"] = job_id
        finished = await _poll_job(client, job_id, overall_deadline=overall_deadline)
        report["sdk_job_final"] = {key: finished.get(key) for key in ("id", "state", "attempt_id", "cleanup_verified", "decision_id")}
        report["checks"]["sdk_job_succeeded"] = finished.get("state") == "succeeded" and finished.get("cleanup_verified") is True
        if not report["checks"]["sdk_job_succeeded"]:
            report["uncertain"].append({"entity": "job", "id": job_id, "state": finished.get("state")})
            raise RuntimeError("SDK submitted job did not complete with verified cleanup")
        waited = await client.wait_job(job_id, timeout_seconds=0.2)
        report["checks"]["sdk_wait"] = waited.get("timed_out") is False and (waited.get("job") or {}).get("state") == "succeeded"
        if not report["checks"]["sdk_wait"]:
            raise RuntimeError("bounded SDK wait did not observe the terminal job state")
        logs = await call("jobs.logs", {"job_id": job_id})
        stdout = logs.get("stdout", {}).get("text", "") if isinstance(logs.get("stdout"), Mapping) else str(logs.get("stdout", ""))
        report["checks"]["job_logs"] = marker + "-sdk" in stdout
        report["sdk_logs"] = {"stdout": stdout[:1000], "cleanup_verified": logs.get("cleanup_verified")}
        if not report["checks"]["job_logs"]:
            raise RuntimeError("SDK job logs did not contain the expected marker")

        # Cross-surface reads: CLI and stdio proxy must see the exact canonical job.
        cli = await asyncio.to_thread(subprocess.run,
            [sys.executable, "-m", "device_scheduler_mcp", "--config", str(config_path), "jobs", "get", job_id],
            cwd=SOURCE_ROOT, capture_output=True, text=True, timeout=args.timeout, check=False)
        cli_value = json.loads(cli.stdout) if cli.returncode == 0 and cli.stdout.strip() else {}
        report["checks"]["cli_same_job"] = cli_value.get("id") == job_id and cli_value.get("attempt_id") == finished.get("attempt_id")
        report["cli_returncode"] = cli.returncode
        if cli.returncode != 0:
            report["cli_stderr"] = cli.stderr[-1000:]
        stdio = await _stdio_read(config_path, job_id)
        stdio_job = stdio["job"]
        # MCP structured output may wrap fields; accept the canonical shape only.
        report["checks"]["stdio_same_job"] = stdio_job.get("id") == job_id and stdio_job.get("attempt_id") == finished.get("attempt_id")
        report["stdio"] = {"tool_count": stdio["tool_count"], "job_id": stdio_job.get("id")}
        if not report["checks"]["cli_same_job"] or not report["checks"]["stdio_same_job"]:
            raise RuntimeError("CLI or stdio proxy did not read the canonical job record")

        # A dependency remains blocked while its parent is running, then proceeds.
        slow_code = "import time; print('" + marker + "-parent', flush=True); time.sleep(1.2)"
        parent = await submit(marker + "-parent", _job_spec(marker, work_dir, slow_code))
        report["created"]["jobs"].append(parent.get("id"))
        parent_id = str(parent["id"])
        report["parent_job_id"] = parent_id
        deadline = min(time.monotonic() + 20, overall_deadline)
        parent_seen_running = False
        while time.monotonic() < deadline:
            current = await client.get_job(parent_id)
            if current.get("state") == "running":
                parent_seen_running = True
                break
            if current.get("state") in {"failed", "unknown"}:
                break
            await asyncio.sleep(0.1)
        child = await submit(marker + "-dependent", _job_spec(marker, work_dir, f"print({(marker + '-child')!r})", dependencies=[parent_id]))
        report["created"]["jobs"].append(child.get("id"))
        child_id = str(child["id"])
        child_before = await client.get_job(child_id)
        report["checks"]["dependency_gate"] = parent_seen_running and child_before.get("state") == "blocked" and child_before.get("attempt_id") is None
        parent_final = await _poll_job(client, parent_id, overall_deadline=overall_deadline)
        child_final = await _poll_job(client, child_id, overall_deadline=overall_deadline)
        report["checks"]["dependency_wakeup"] = parent_final.get("state") == "succeeded" and child_final.get("state") == "succeeded"
        report["dependency"] = {"parent": parent_final.get("state"), "child_before": child_before.get("state"), "child_after": child_final.get("state")}
        if not report["checks"]["dependency_gate"] or not report["checks"]["dependency_wakeup"]:
            raise RuntimeError("dependency gate or wakeup did not match the expected job-result semantics")

        # Event-gated one-off job: no execution before publication, then exactly once.
        event_id = marker + "-event"
        trigger = {"source": marker, "event_type": "qualification.ready", "event_key": event_id}
        event_job = await submit(marker + "-event-job", _job_spec(marker, work_dir, f"print({(marker + '-event-result')!r})", event_trigger=trigger))
        report["created"]["jobs"].append(event_job.get("id"))
        event_job_id = str(event_job["id"])
        report["checks"]["event_initially_blocked"] = event_job.get("state") == "blocked" and event_job.get("attempt_id") is None
        published = await call("events.publish", {
            "source": marker,
            "event_id": event_id,
            "idempotency_key": marker + "-event-publish",
            "payload": {"event_type": "qualification.ready", "marker": marker},
        })
        report["created"]["events"].append({"source": marker, "event_id": event_id})
        event_duplicate = await call("events.publish", {
            "source": marker, "event_id": event_id, "idempotency_key": marker + "-event-publish-retry",
            "payload": {"event_type": "qualification.ready", "marker": marker},
        })
        event_final = await _poll_job(client, event_job_id, overall_deadline=overall_deadline)
        report["checks"]["event_duplicate_identity"] = published.get("event", {}).get("event_id") == event_duplicate.get("event", {}).get("event_id") == event_id
        report["checks"]["event_released_job"] = event_final.get("state") == "succeeded"
        report["event_job_final"] = event_final.get("state")
        if not report["checks"]["event_initially_blocked"] or not report["checks"]["event_duplicate_identity"] or not report["checks"]["event_released_job"]:
            raise RuntimeError("event gate, deduplication, or wakeup failed")

        # Reserve-only CPU/memory lease may be attached to one supervised job,
        # remains active after verified job cleanup, and can then be renewed and
        # explicitly released. No unmanaged process is launched under a lease.
        lease_requirements = [
            {"kind": "cpu", "quantities": {"cores": 0.25}},
            {"kind": "memory", "quantities": {"bytes": 128 * 1024 * 1024}},
        ]
        lease_record = await call("leases.request", {
            "idempotency_key": marker + "-supervised-lease",
            "request": {
                "request_id": marker + "-supervised-lease",
                "requirements": lease_requirements,
                "lease_ttl_seconds": 240,
                "usage": "supervised",
            },
        })
        lease_id = str(lease_record["lease_id"])
        fencing_token = int(lease_record["fencing_token"])
        report["created"]["leases"].append(lease_id)
        report["supervised_lease_id"] = lease_id
        lease_bound_spec = _job_spec(marker, work_dir, f"print({(marker + '-lease-bound')!r})")
        lease_bound_spec.pop("resources")
        leased_job = await call("jobs.submit", {
            "idempotency_key": marker + "-lease-bound-job",
            "lease_id": lease_id,
            "fencing_token": fencing_token,
            "spec": lease_bound_spec,
        })
        leased_job_id = str(leased_job["id"])
        report["created"]["jobs"].append(leased_job_id)
        leased_final = await _poll_job(client, leased_job_id, timeout=60, overall_deadline=overall_deadline)
        after_job_lease = await call("leases.get", {"lease_id": lease_id})
        report["checks"]["supervised_lease_job_cleanup"] = (
            leased_final.get("state") == "succeeded"
            and leased_final.get("cleanup_verified") is True
            and after_job_lease.get("state") == "active"
        )
        if not report["checks"]["supervised_lease_job_cleanup"]:
            report["uncertain"].append({"entity": "lease", "id": lease_id, "state": after_job_lease.get("state"), "job_state": leased_final.get("state")})
            raise RuntimeError("supervised lease was not retained after verified attached-job cleanup")
        renewed = await call("leases.renew", {
            "lease_id": lease_id, "fencing_token": after_job_lease["fencing_token"], "ttl_seconds": 240,
        })
        released = await call("leases.release", {
            "lease_id": lease_id, "fencing_token": renewed["fencing_token"],
        })
        report["checks"]["supervised_lease_renew_release"] = released.get("state") == "released"
        report["supervised_lease"] = {"after_job": after_job_lease.get("state"), "after_renew": renewed.get("state"), "after_release": released.get("state")}
        if not report["checks"]["supervised_lease_renew_release"]:
            report["uncertain"].append({"entity": "lease", "id": lease_id, "state": released.get("state")})
            raise RuntimeError("supervised CPU/memory lease could not be verified released")

        # Delayed work must have no attempt before its nominal time.
        due = datetime.now(timezone.utc) + timedelta(seconds=5)
        delayed = await submit(marker + "-delayed", _job_spec(marker, work_dir, f"print({(marker + '-delayed')!r})", not_before=due))
        report["created"]["jobs"].append(delayed.get("id"))
        delayed_id = str(delayed["id"])
        before_due = await client.get_job(delayed_id)
        report["checks"]["delayed_not_started_early"] = before_due.get("attempt_id") is None and before_due.get("state") in {"queued", "blocked"}
        delayed_final = await _poll_job(client, delayed_id, timeout=60, overall_deadline=overall_deadline)
        report["checks"]["delayed_eventually_ran"] = delayed_final.get("state") == "succeeded"
        report["delayed"] = {"nominal_at": due.isoformat(), "state_before": before_due.get("state"), "state_after": delayed_final.get("state")}
        if not report["checks"]["delayed_not_started_early"] or not report["checks"]["delayed_eventually_ran"]:
            raise RuntimeError("delayed job ran early or failed to run after its not_before time")

        # Recurring interval schedule: observe one occurrence, then cancel only this run's schedule.
        schedule_id = marker + "-interval"
        schedule_job = _job_spec(marker, work_dir, f"print({(marker + '-interval')!r})", timeout=90)
        schedule = await call("schedules.put", {
            "idempotency_key": marker + "-schedule-create",
            "spec": {"schedule_id": schedule_id, "job": schedule_job,
                     "interval_seconds": 2.0, "overlap": "queue", "misfire_policy": "coalesce"},
        })
        # Owner is authenticated principal; schedule.put defaults to it if omitted.
        report["created"]["schedules"].append(schedule_id)
        report["schedule_id"] = schedule_id
        due_deadline = min(time.monotonic() + 30, overall_deadline)
        occurrence_jobs: list[str] = []
        while time.monotonic() < due_deadline:
            current_schedule = await call("schedules.get", {"schedule_id": schedule_id})
            occurrence_jobs = [str(item) for item in current_schedule.get("active_job_ids", [])]
            if occurrence_jobs:
                break
            await asyncio.sleep(0.15)
        report["checks"]["interval_materialized"] = bool(occurrence_jobs)
        cleanup = await call("schedules.control", {"schedule_id": schedule_id, "action": "cancel", "idempotency_key": marker + "-schedule-cancel"})
        report["schedule_cleanup"] = {"id": schedule_id, "state": cleanup.get("state")}
        if occurrence_jobs:
            first_scheduled_id = occurrence_jobs[0]
            all_scheduled_ids = set(occurrence_jobs)
            stable_ids: set[str] | None = None
            stable_observations = 0
            settle_deadline = min(time.monotonic() + 60, overall_deadline)
            final_schedule: dict[str, Any] = {}
            while time.monotonic() < settle_deadline and stable_observations < 2:
                current_schedule = await call("schedules.get", {"schedule_id": schedule_id})
                final_schedule = current_schedule
                all_scheduled_ids.update(str(value) for value in current_schedule.get("active_job_ids", []))
                current_ids = {str(value) for value in current_schedule.get("active_job_ids", [])}
                if not current_schedule.get("submission_intents") and current_ids == stable_ids:
                    stable_observations += 1
                else:
                    stable_observations = 0
                stable_ids = current_ids
                if stable_observations < 2:
                    await asyncio.sleep(0.2)
            occurrence_jobs = sorted(all_scheduled_ids)
            report["created"]["jobs"].extend(occurrence_jobs)
            interval_final = await _poll_job(client, first_scheduled_id, timeout=60, overall_deadline=overall_deadline)
            report["checks"]["interval_job_succeeded"] = interval_final.get("state") == "succeeded" and interval_final.get("cleanup_verified") is True
            report["schedule_jobs"] = []
            for owned_id in occurrence_jobs:
                owned = await client.get_job(owned_id)
                spec_tags = (owned.get("spec") or {}).get("tags", [])
                if marker not in spec_tags:
                    raise RuntimeError("schedule active_job_ids included a job without this run's marker")
                if owned.get("state") not in {"succeeded", "failed", "cancelled"} and owned_id != first_scheduled_id:
                    owned = await call("jobs.control", {"job_id": owned_id, "action": "cancel", "reason": marker})
                    owned = await _poll_job(client, owned_id, timeout=30, overall_deadline=overall_deadline)
                elif owned.get("state") not in {"succeeded", "failed", "cancelled"}:
                    owned = await _poll_job(client, owned_id, timeout=60, overall_deadline=overall_deadline)
                if owned.get("state") == "unknown" or owned.get("cleanup_verified") is not True:
                    report["uncertain"].append({"entity": "scheduled_job", "id": owned_id, "state": owned.get("state")})
                report["schedule_jobs"].append({"id": owned_id, "state": owned.get("state"), "cleanup_verified": owned.get("cleanup_verified")})
            report["checks"]["interval_jobs_quiesced"] = (
                not final_schedule.get("submission_intents")
                and len(report["schedule_jobs"]) == len(occurrence_jobs)
                and all(item["state"] in {"succeeded", "failed", "cancelled"} and item["cleanup_verified"] is True
                        for item in report["schedule_jobs"])
            )
        else:
            report["checks"]["interval_job_succeeded"] = False
            report["checks"]["interval_jobs_quiesced"] = False
        if not report["checks"]["interval_materialized"] or not report["checks"]["interval_job_succeeded"] or not report["checks"]["interval_jobs_quiesced"]:
            raise RuntimeError("interval schedule did not materialize, complete one occurrence, and quiesce all of its jobs")

        # Explicit cancellation and clean result collection for one tagged helper.
        cancel_code = "import time; print('" + marker + "-cancel-helper', flush=True); time.sleep(60)"
        cancel_job = await submit(marker + "-cancel", _job_spec(marker, work_dir, cancel_code, timeout=120))
        report["created"]["jobs"].append(cancel_job.get("id"))
        cancel_id = str(cancel_job["id"])
        deadline = min(time.monotonic() + 30, overall_deadline)
        running = None
        while time.monotonic() < deadline:
            running = await client.get_job(cancel_id)
            if running.get("state") == "running":
                break
            if running.get("state") in {"failed", "unknown"}:
                break
            await asyncio.sleep(0.1)
        if running is None or running.get("state") != "running":
            report["uncertain"].append({"entity": "job", "id": cancel_id, "state": None if running is None else running.get("state")})
            raise RuntimeError("cancellation helper did not reach running state")
        cancelled = await call("jobs.control", {"job_id": cancel_id, "action": "cancel", "reason": marker})
        cancelled = await _poll_job(client, cancel_id, timeout=30, overall_deadline=overall_deadline)
        report["checks"]["cancel_cleanup_verified"] = cancelled.get("state") == "cancelled" and cancelled.get("cleanup_verified") is True
        report["cancelled_job"] = {"id": cancel_id, "state": cancelled.get("state"), "cleanup_verified": cancelled.get("cleanup_verified")}
        if not report["checks"]["cancel_cleanup_verified"]:
            report["uncertain"].append({"entity": "job", "id": cancel_id, "state": cancelled.get("state")})
            raise RuntimeError("cancellation did not verify execution cleanup")

        # Decision explanation and policy replay; replay must not alter canonical counts.
        explanation = await call("policies.explain", {"job_id": job_id})
        report["checks"]["policy_explanation"] = bool(explanation.get("policy_id") or explanation.get("decision_id"))
        before_jobs = (await call("jobs.list")).get("jobs", [])
        before_schedules = (await call("schedules.list")).get("schedules", [])
        before_leases = (await call("leases.list")).get("leases", [])
        now = utc_now()
        replay = await call("policies.simulate", {
            "observations": [
                {"id": marker + "-interactive", "owner_id": "fairness-heavy", "priority_class": "interactive", "priority": 100,
                 "created_at": now, "can_fit": True},
                {"id": marker + "-normal", "owner_id": "fairness-light", "priority_class": "normal", "priority": 50,
                 "created_at": now, "can_fit": True},
            ],
            "service_by_owner": {"fairness-heavy": 1000.0, "fairness-light": 0.0},
        })
        after_jobs = (await call("jobs.list")).get("jobs", [])
        after_schedules = (await call("schedules.list")).get("schedules", [])
        after_leases = (await call("leases.list")).get("leases", [])
        def owned_jobs(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return [item for item in items if marker in (item.get("spec") or {}).get("tags", [])]
        def owned_schedules(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return [item for item in items if str(item.get("schedule_id", "")).startswith(marker)]
        owned_lease_ids = set(report["created"]["leases"])
        def owned_leases(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
            return [item for item in items if item.get("lease_id") in owned_lease_ids]
        report["checks"]["policy_simulation_no_side_effects"] = (
            replay.get("resource_allocation_performed") is False
            and replay.get("execution_started") is False
            and owned_jobs(before_jobs) == owned_jobs(after_jobs)
            and owned_schedules(before_schedules) == owned_schedules(after_schedules)
            and owned_leases(before_leases) == owned_leases(after_leases)
        )
        report["policy"] = {"explanation": explanation, "simulation": replay}
        report["checks"]["policy_priority_and_fairness"] = replay.get("proposal", {}).get("request_id") == marker + "-normal"
        if not report["checks"]["policy_explanation"] or not report["checks"]["policy_simulation_no_side_effects"] or not report["checks"]["policy_priority_and_fairness"]:
            raise RuntimeError("policy explanation or side-effect-free simulation check failed")

        report["verdict"] = "pass"
    except BaseException as exc:
        report["verdict"] = "incomplete"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        report["traceback"] = "".join(traceback.format_exception(exc))[-4000:]
        # Recover only the uniquely named schedule from this invocation. Unknown
        # job state is recorded and left untouched for explicit operator review.
        if schedule_id is not None and schedule_id in report["created"]["schedules"]:
            try:
                current = await call("schedules.get", {"schedule_id": schedule_id})
                if current.get("state") not in {"cancelled", "completed"}:
                    cleaned = await call("schedules.control", {"schedule_id": schedule_id, "action": "cancel", "idempotency_key": marker + "-schedule-cancel-recovery"})
                    report["schedule_cleanup_recovery"] = cleaned.get("state")
            except Exception as cleanup_exc:
                report["uncertain"].append({"entity": "schedule", "id": schedule_id, "error": type(cleanup_exc).__name__})
        # Stop only canonical jobs returned to this invocation, and only after
        # checking their persisted marker. Unknown jobs are left for an operator.
        for owned_id in dict.fromkeys(str(value) for value in report["created"]["jobs"] if value):
            try:
                owned = await client.get_job(owned_id)
                if marker not in (owned.get("spec") or {}).get("tags", []):
                    report["uncertain"].append({"entity": "job", "id": owned_id, "reason": "run_marker_mismatch"})
                    continue
                if owned.get("state") == "unknown":
                    report["uncertain"].append({"entity": "job", "id": owned_id, "state": "unknown"})
                    continue
                if owned.get("state") not in {"succeeded", "failed", "cancelled"}:
                    await call("jobs.control", {"job_id": owned_id, "action": "cancel", "reason": marker + "-failure-cleanup"})
                    owned = await _poll_job(client, owned_id, timeout=30)
                if owned.get("state") not in {"succeeded", "failed", "cancelled"} or owned.get("cleanup_verified") is not True:
                    report["uncertain"].append({"entity": "job", "id": owned_id, "state": owned.get("state"), "cleanup_verified": owned.get("cleanup_verified")})
            except Exception as cleanup_exc:
                report["uncertain"].append({"entity": "job", "id": owned_id, "error": type(cleanup_exc).__name__})
        # A supervised lease with no active/uncertain attached attempt can be
        # released by its current fence. Never release around unknown execution.
        for lease_id in report["created"].get("leases", []):
            try:
                lease = await call("leases.get", {"lease_id": lease_id})
                attached = []
                for job_id in dict.fromkeys(str(value) for value in report["created"]["jobs"] if value):
                    candidate = await client.get_job(job_id)
                    if candidate.get("lease_id") == lease_id:
                        attached.append(candidate)
                safe = all(
                    item.get("state") in {"succeeded", "failed", "cancelled"}
                    and item.get("cleanup_verified") is True
                    for item in attached
                )
                if lease.get("state") == "active" and safe:
                    released = await call("leases.release", {
                        "lease_id": lease_id, "fencing_token": lease["fencing_token"],
                    })
                    report.setdefault("lease_cleanup_recovery", []).append({"id": lease_id, "state": released.get("state")})
                elif lease.get("state") not in {"released", "active"} or not safe:
                    report["uncertain"].append({"entity": "lease", "id": lease_id, "state": lease.get("state"), "attached_jobs": [item.get("state") for item in attached]})
            except Exception as cleanup_exc:
                report["uncertain"].append({"entity": "lease", "id": lease_id, "error": type(cleanup_exc).__name__})
    finally:
        try:
            final_status = await call("admin.status")
            report["final_daemon_status"] = final_status.get("status")
        except Exception as status_exc:
            report["final_daemon_status"] = "unavailable"
            report["uncertain"].append({"entity": "daemon_status", "error": type(status_exc).__name__})
        report["source_unchanged"] = source_identity()["source_hashes"] == report["source_runtime"]["source_hashes"]
        report["finished_at"] = utc_now()
        report["receipt_path"] = str(receipt_path)
        report["verdict"] = "pass" if report.get("verdict") == "pass" and report["source_unchanged"] else "incomplete"
        if not report["source_unchanged"]:
            report["uncertain"].append({"source": "production_source_changed_during_qualification"})
        atomic_receipt(receipt_path, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("verdict") == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="required to connect and create tagged test jobs/schedule/event")
    parser.add_argument("--url", default="http://127.0.0.1:8770", help="origin of the already-running loopback daemon")
    parser.add_argument("--token-file", type=Path, default=Path("~/.config/device-scheduler/operator.token"))
    parser.add_argument("--execution-root", type=Path, help="existing path covered by the operator grant; test files remain here")
    parser.add_argument("--receipt", type=Path, help="structured receipt path; default is inside the created execution directory")
    parser.add_argument("--mcp-path", default="/mcp")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "mode": "plan_only",
            "would_connect": False,
            "would_start_daemon": False,
            "scope": "existing loopback daemon; CPU/memory jobs only; no GPU/backend/FrameShift",
            "mutations": ["tagged one-off jobs", "tagged event", "supervised CPU/memory lease cycle", "tagged interval schedule, cancelled after first occurrence"],
            "read_surfaces": ["HTTP health/readiness", "SDK via MCP", "CLI", "stdio MCP proxy", "policy explain/simulation"],
        }, indent=2))
        return 0
    if args.execution_root is None:
        parser.error("--execution-root is required with --execute")
    if not (0.1 <= args.timeout <= 300):
        parser.error("--timeout must be in [0.1, 300]")
    return asyncio.run(qualify(args))


if __name__ == "__main__":
    raise SystemExit(main())
