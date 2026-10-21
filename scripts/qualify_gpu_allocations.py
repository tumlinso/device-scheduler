#!/usr/bin/env python3
"""Qualify shared and exclusive allocation on one Project Control GPU grant.

This is a destructive-capability harness, not a controller cutover. It is
plan-only by default. A live run requires ``--run`` and ``--parent-grant`` and
keeps that parent reservation owned by Project Control throughout cleanup.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for source_path in (
    SCRIPTS,
    ROOT / "packages" / "device-scheduler-core" / "src",
    ROOT / "packages" / "device-scheduler-adapters" / "src",
    ROOT / "packages" / "device-scheduler-mcp" / "src",
):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from qualify_gpu_handoff import (  # noqa: E402
    DEFAULT_PC_PYTHON,
    DEFAULT_PC_RUNTIME,
    ParentGrantGuard,
    QualificationError,
    _process_start,
    gpu_snapshot,
    private_write,
)
from qualification_gpu_probe import runtime_visible_device_uuid  # noqa: E402

from device_scheduler_adapters.nvidia import NvidiaAdapter  # noqa: E402
from device_scheduler_core.models import Principal  # noqa: E402
from device_scheduler_mcp.auth import StaticTokenStore, TokenGrant  # noqa: E402
from device_scheduler_mcp.client import DeviceSchedulerClient  # noqa: E402
from device_scheduler_mcp.config import ServiceConfig  # noqa: E402
from device_scheduler_mcp.server import build_default_scheduler, create_app  # noqa: E402


MIB = 1024 * 1024
MIN_BUFFER_BYTES = 4 * MIB
MAX_BUFFER_BYTES = 128 * MIB
SHARED_BUDGET_BYTES = 512 * MIB
GPU_UUID_RE = re.compile(r"^GPU-[A-Fa-f0-9-]{16,64}$")


def port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def exact_gpu_snapshot(snapshot: Mapping[str, Any], gpu_uuid: str) -> dict[str, Any]:
    """Narrow a real NVIDIA snapshot to one GPU and its related isolation rows."""
    if not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise QualificationError("GPU UUID must be a stable NVIDIA UUID")
    raw_resources = snapshot.get("resources")
    if not isinstance(raw_resources, (list, tuple)):
        raise QualificationError("NVIDIA snapshot did not contain a resource list")
    selected: list[Any] = []
    gpu_ids: set[str] = set()
    for item in raw_resources:
        data = item.to_dict() if hasattr(item, "to_dict") else item
        if not isinstance(data, Mapping):
            continue
        attrs = data.get("attributes") if isinstance(data.get("attributes"), Mapping) else {}
        kind = data.get("kind")
        if kind == "gpu" and attrs.get("uuid") == gpu_uuid:
            selected.append(item)
            gpu_ids.add(str(data.get("resource_id")))
    # Isolation rows may precede the target in adapter output.
    if gpu_ids:
        for item in raw_resources:
            data = item.to_dict() if hasattr(item, "to_dict") else item
            if not isinstance(data, Mapping) or data.get("kind") != "isolation":
                continue
            attrs = data.get("attributes") if isinstance(data.get("attributes"), Mapping) else {}
            members = attrs.get("member_resource_ids", ())
            if isinstance(members, (list, tuple)) and gpu_ids.intersection(str(m) for m in members):
                if item not in selected:
                    selected.append(item)
    if len(gpu_ids) != 1:
        raise QualificationError("NVIDIA snapshot did not identify exactly one granted GPU")
    result = dict(snapshot)
    result["resources"] = selected
    inventory = result.get("gpu_inventory")
    if isinstance(inventory, Mapping):
        narrowed = dict(inventory)
        devices = inventory.get("devices")
        if isinstance(devices, list):
            narrowed["devices"] = [item for item in devices if isinstance(item, Mapping) and item.get("uuid") == gpu_uuid]
        result["gpu_inventory"] = narrowed
    telemetry = result.get("process_telemetry")
    if isinstance(telemetry, Mapping):
        narrowed = dict(telemetry)
        rows = telemetry.get("processes")
        if isinstance(rows, list):
            narrowed["processes"] = [
                item for item in rows
                if isinstance(item, Mapping) and (item.get("gpu_uuid") == gpu_uuid or item.get("identity_unknown"))
            ]
        result["process_telemetry"] = narrowed
    return result


class GrantedGpuAdapter:
    """Filter NVIDIA discovery before it enters this private scheduler ledger."""

    adapter_id = "nvidia-smi"
    api_version = "1"

    def __init__(self, source: NvidiaAdapter, gpu_uuid: str) -> None:
        self.source = source
        self.gpu_uuid = gpu_uuid

    def capabilities(self) -> Any:
        return self.source.capabilities()

    def snapshot(self) -> dict[str, Any]:
        return exact_gpu_snapshot(self.source.snapshot(), self.gpu_uuid)

    def discover(self) -> list[Any]:
        snap = self.snapshot()
        if snap.get("gpu_inventory", {}).get("state") != "observed":
            return []
        return list(snap["resources"])


def exclusive_request_waiting(job: Mapping[str, Any]) -> bool:
    """True only when the exclusive job has not acquired a lease or attempt."""
    state = job.get("state")
    return state in {"queued", "blocked"} and job.get("attempt_id") is None and job.get("lease_id") is None


def _listed_resource_statuses(response: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Unwrap the public resources.list status envelopes without losing status."""
    rows = response.get("resources")
    if not isinstance(rows, list):
        raise QualificationError("scheduler resource listing has an invalid shape")
    normalized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise QualificationError("scheduler resource listing contains a non-object row")
        nested = row.get("resource")
        resource = nested if isinstance(nested, Mapping) else row
        if not isinstance(resource.get("resource_id"), str):
            raise QualificationError("scheduler resource listing row has no resource identity")
        normalized.append((dict(resource), dict(row)))
    return normalized


def _cuda_error(cuda: ctypes.CDLL, code: int, operation: str) -> None:
    if code == 0:
        return
    cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
    cuda.cudaGetErrorString.restype = ctypes.c_char_p
    message = cuda.cudaGetErrorString(code)
    raise RuntimeError(f"{operation} failed: {message.decode(errors='replace') if message else code}")


def worker(gpu_uuid: str, memory_bytes: int, receipt_path: Path, release_path: Path, timeout: float) -> dict[str, Any]:
    """Allocate, memset, verify, hold at a barrier, free, and persist a receipt."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != gpu_uuid or not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise QualificationError("worker visibility must name exactly the granted GPU UUID")
    if isinstance(memory_bytes, bool) or not MIN_BUFFER_BYTES <= memory_bytes <= MAX_BUFFER_BYTES:
        raise QualificationError("CUDA allocation must be between 4 and 128 MiB")
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    cuda = ctypes.CDLL(library)
    cuda.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda.cudaGetDeviceCount.restype = ctypes.c_int
    cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda.cudaMalloc.restype = ctypes.c_int
    cuda.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cuda.cudaMemset.restype = ctypes.c_int
    cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cuda.cudaMemcpy.restype = ctypes.c_int
    cuda.cudaFree.argtypes = [ctypes.c_void_p]
    cuda.cudaFree.restype = ctypes.c_int
    count = ctypes.c_int()
    _cuda_error(cuda, cuda.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
    if count.value != 1:
        raise QualificationError(f"expected one visible CUDA device; found {count.value}")
    actual_uuid = runtime_visible_device_uuid(cuda)
    if actual_uuid.casefold() != gpu_uuid.casefold():
        raise QualificationError("CUDA runtime UUID differs from the parent Project Control grant")
    device = ctypes.c_void_p()
    _cuda_error(cuda, cuda.cudaMalloc(ctypes.byref(device), memory_bytes), "cudaMalloc")
    record: dict[str, Any] = {
        "schema_version": 1,
        "gpu_uuid": actual_uuid,
        "pid": os.getpid(),
        "pid_birth": _process_start(os.getpid()),
        "memory_bytes": memory_bytes,
        "cuda_library": library,
        "allocation": "cudaMalloc",
        "initialization": "cudaMemset(0xA5)",
        "free_verified": False,
    }
    if record["pid_birth"] is None:
        _cuda_error(cuda, cuda.cudaFree(device), "cudaFree after missing PID birth identity")
        raise QualificationError("worker PID birth identity could not be established")
    try:
        _cuda_error(cuda, cuda.cudaMemset(device, 0xA5, memory_bytes), "cudaMemset")
        probe = ctypes.create_string_buffer(4096)
        _cuda_error(cuda, cuda.cudaMemcpy(probe, device, len(probe), 2), "cudaMemcpy verification")
        if probe.raw != bytes([0xA5]) * len(probe):
            raise QualificationError("cudaMemset result did not match the expected byte pattern")
        record["verification"] = "first-4096-bytes-matched"
        record["allocated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(receipt_path, {**record, "phase": "allocated"})
        deadline = time.monotonic() + timeout
        while not release_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("worker release barrier timed out")
            time.sleep(0.05)
    finally:
        _cuda_error(cuda, cuda.cudaFree(device), "cudaFree")
        record["free_verified"] = True
        record["freed_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "passed"
    _write_json_atomic(receipt_path, record)
    return record


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


async def _health(base_url: str, timeout: float = 20) -> None:
    import httpx
    until = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=1) as client:
        while time.monotonic() < until:
            try:
                response = await client.get(base_url + "/readyz")
                if response.status_code == 200 and response.json().get("status") in {"ready", "degraded"}:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
    raise QualificationError("private authenticated scheduler failed to become ready")


def _service(root: Path, gpu_uuid: str) -> tuple[Any, Any, Path, int]:
    token = secrets.token_urlsafe(36)
    token_file = root / "client.token"
    private_write(token_file, token + "\n")
    principal = Principal(
        principal_id="gpu-allocation-qualification",
        actions=frozenset({
            "resources.list", "resources.capabilities", "jobs.submit", "jobs.get", "jobs.list",
            "jobs.wait", "jobs.logs", "leases.get", "leases.list", "admin.status",
        }),
        execution_roots=(str(ROOT), str(root)),
        max_priority=50,
    )
    token_store = StaticTokenStore([TokenGrant(hashlib.sha256(token.encode()).hexdigest(), principal)])
    listen_port = port()
    config = ServiceConfig(
        host="127.0.0.1", port=listen_port, tick_interval_seconds=0.1,
        token_file=token_file, token_directory=root,
        state_directory=root / "state", ledger_path=root / "state" / "ledger.sqlite3",
        artifact_directory=root / "artifacts", gpu_observation_max_age_seconds=10,
    )
    scheduler = build_default_scheduler(config)
    original = next((adapter for adapter in scheduler.resource_adapters if getattr(adapter, "adapter_id", None) == "nvidia-smi"), None)
    if original is None:
        raise QualificationError("default composition did not include the NVIDIA adapter")
    from device_scheduler_adapters.host import LinuxHostAdapter
    scheduler.resource_adapters = (LinuxHostAdapter(), GrantedGpuAdapter(original, gpu_uuid))
    # Keep the private qualification daemon's scheduler tick under this
    # harness's control. NVIDIA discovery is synchronous; an independent
    # background tick racing resources.list can serialize two slow discovery
    # calls inside the ASGI event loop and make an otherwise healthy request
    # exceed the client timeout. The same production scheduler and adapters
    # are exercised by explicit ticks below.
    app = create_app(config, scheduler=scheduler, token_store=token_store, background_ticks=False)
    server_config = uvicorn.Config(app, host="127.0.0.1", port=listen_port, log_level="error", lifespan="on")
    server = uvicorn.Server(server_config)
    return server, scheduler, token_file, listen_port


def _job_spec(gpu_uuid: str, work: Path, label: str, buffer_bytes: int, *, exclusive: bool = False,
              isolation_id: str | None = None, memory_budget: int = SHARED_BUDGET_BYTES,
              parent_grant: Path | None = None, pc_python: Path = DEFAULT_PC_PYTHON,
              pc_runtime: Path = DEFAULT_PC_RUNTIME) -> dict[str, Any]:
    if parent_grant is None:
        raise QualificationError("every CUDA worker command must carry the live parent-grant guard")
    gpu_requirement: dict[str, Any] = {
        "kind": "gpu", "quantities": {"memory_bytes": memory_budget if not exclusive else 1},
        # Requirement selectors address Resource.attributes directly. The
        # public JSON resource representation nests these under `attributes`,
        # but the allocator selector key itself is simply `uuid`.
        "selectors": {"uuid": gpu_uuid}, "mode": "shared", "same_host": True,
    }
    resources = [gpu_requirement]
    tags: list[str] = []
    if exclusive:
        tags.append("gpu.exclusive")
    if isolation_id:
        resources.append({
            "kind": "isolation", "quantities": {"slots": 1},
            "selectors": {"resource_id": isolation_id}, "mode": "exclusive", "same_host": True,
        })
    return {
        "owner_id": "gpu-allocation-qualification",
        "argv": [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--gpu-uuid", gpu_uuid,
            "--memory-bytes", str(buffer_bytes), "--receipt", str(work / f"{label}.json"),
            "--release-file", str(work / f"{label}.release"), "--worker-timeout", "240",
            "--run", "--parent-grant", str(parent_grant),
            "--project-control-python", str(pc_python), "--project-control-runtime", str(pc_runtime),
        ],
        "working_directory": str(ROOT),
        "environment": {},
        "resources": resources,
        "tags": tags,
        "priority_class": "normal",
        "priority": 50,
        "timeout_seconds": 300,
        "max_attempts": 1,
        "interruption": {"cooperative_yield": False, "checkpoint_supported": False, "restart_safe": False, "cancellation_supported": True},
        "idempotency_key": f"gpu-qualification-{label}",
    }


async def _tick(scheduler: Any, report: dict[str, Any], reason: str) -> None:
    started = time.monotonic()
    await asyncio.to_thread(scheduler.tick)
    duration = time.monotonic() - started
    report.setdefault("scheduler_tick_durations_seconds", []).append({
        "reason": reason, "duration": round(duration, 6),
    })


async def _wait_job(client: DeviceSchedulerClient, scheduler: Any, report: dict[str, Any],
                    guard: ParentGrantGuard, job_id: str, acceptable: set[str], timeout: float,
                    *, check_parent: bool = True) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_parent:
            guard.verify()
        detail = await client.get_job(job_id)
        if detail.get("state") in acceptable:
            return detail
        if detail.get("state") in {"failed", "cancelled", "unknown"}:
            raise QualificationError(f"job {job_id} ended in {detail.get('state')}")
        await _tick(scheduler, report, f"wait:{job_id}")
        await asyncio.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not reach {sorted(acceptable)}")


async def _wait_receipt(path: Path, guard: ParentGrantGuard, timeout: float = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        guard.verify()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict) and value.get("phase") == "allocated":
            return value
        await asyncio.sleep(0.1)
    raise TimeoutError(f"worker did not publish allocation receipt: {path.name}")


async def _job_evidence(client: DeviceSchedulerClient, job_id: str) -> dict[str, Any]:
    detail = await client.get_job(job_id)
    evidence: dict[str, Any] = {"job": detail}
    lease_id = detail.get("lease_id")
    if isinstance(lease_id, str) and lease_id:
        evidence["lease"] = await client.get_lease(lease_id)
    evidence["logs"] = await client.job_logs(job_id)
    return evidence


async def _execute(args: argparse.Namespace, guard: ParentGrantGuard, grant: Mapping[str, Any]) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="device-scheduler-gpu-allocation-"))
    os.chmod(work, 0o700)
    base_url = ""
    server: uvicorn.Server | None = None
    server_task: asyncio.Task[Any] | None = None
    unmanaged: subprocess.Popen[bytes] | None = None
    release_files: list[Path] = []
    submitted_ids: list[str] = []
    report: dict[str, Any] = {
        "schema_version": 1, "status": "running", "gpu_uuid": grant["gpu_uuid"],
        "parent_grant": {k: grant[k] for k in ("owner_id", "resource_ids", "pid", "process_start", "state")},
        "started_at": datetime.now(timezone.utc).isoformat(), "source": "nvidia-smi plus authenticated Device Scheduler SDK",
        "claims": [], "jobs": [], "scenarios": [], "limitations": [
            "GPU memory budgets are scheduler admission accounting, not hardware-enforced limits",
            "process visibility follows installed nvidia-smi compute-process telemetry",
            "the Project Control parent reservation remains the outer authorization boundary",
        ],
    }
    try:
        guard.verify(force=True)
        baseline = gpu_snapshot(str(grant["gpu_uuid"]))
        if baseline["process_count"] != 0:
            raise QualificationError("the granted GPU must be process-empty before allocation qualification")
        report["source_snapshot_before"] = baseline
        server, scheduler, token_file, listen_port = _service(work / "scheduler", str(grant["gpu_uuid"]))
        base_url = f"http://127.0.0.1:{listen_port}"
        server_task = asyncio.create_task(asyncio.to_thread(server.run))
        await _health(base_url)
        client = DeviceSchedulerClient(base_url, token_file, timeout_seconds=30)
        # `baseline` is read from nvidia-smi and the adapter is hard-filtered
        # to this UUID before it can write the private ledger. Avoid an eager
        # resources.list RPC here: that endpoint performs synchronous NVIDIA
        # discovery, whose latency is itself under qualification. Lease
        # evidence below verifies every allocation uses this exact identity.
        resource_id = str(baseline["resource_id"])
        report["scheduler"] = {
            "base_url": base_url, "principal_id": "gpu-allocation-qualification",
            "resource_id": resource_id,
            "policy": (await client.status()).get("scheduler"),
        }
        free = int(baseline["memory_free_bytes"])
        budget = min(SHARED_BUDGET_BYTES, max(256 * MIB, free // 8))
        if free < budget * 2 + 128 * MIB:
            raise QualificationError("granted GPU lacks safe observed headroom for two shared budgets")

        async def submit(label: str, *, exclusive: bool = False, isolation_id: str | None = None) -> tuple[str, Path, Path]:
            receipt = work / f"{label}.json"
            release = work / f"{label}.release"
            release_files.append(release)
            result = await client.submit_job(_job_spec(
                str(grant["gpu_uuid"]), work, label, min(32 * MIB, MAX_BUFFER_BYTES),
                exclusive=exclusive, isolation_id=isolation_id, memory_budget=budget,
                parent_grant=args.parent_grant, pc_python=args.project_control_python,
                pc_runtime=args.project_control_runtime,
            ))
            job_id = str(result.get("job_id", result.get("id", "")))
            if not job_id:
                raise QualificationError("scheduler SDK submission returned no job identity")
            submitted_ids.append(job_id)
            await _tick(scheduler, report, f"admit:{label}")
            return job_id, receipt, release

        shared: list[tuple[str, Path, Path]] = []
        for index in (1, 2):
            shared.append(await submit(f"shared-{index}"))
        shared_receipts = [await _wait_receipt(item[1], guard) for item in shared]
        if len({item.get("pid") for item in shared_receipts}) != 2:
            raise QualificationError("two shared CUDA jobs did not hold distinct simultaneous process identities")
        if any(item.get("gpu_uuid") != grant["gpu_uuid"] for item in shared_receipts):
            raise QualificationError("shared worker observed a CUDA UUID outside the parent grant")
        shared_jobs = [await client.get_job(item[0]) for item in shared]
        if any(job.get("state") != "running" or not job.get("attempt_id") or not job.get("lease_id") for job in shared_jobs):
            raise QualificationError("both shared jobs must be running with durable attempts and leases while workers hold memory")
        shared_leases = [await client.get_lease(str(item["lease_id"])) for item in shared_jobs if item.get("lease_id")]
        if len(shared_leases) != 2 or any(item.get("state") != "active" for item in shared_leases):
            raise QualificationError("both concurrent shared jobs must hold active visible leases")
        for lease in shared_leases:
            allocations = lease.get("allocations", [])
            gpu_allocations = [item for item in allocations if item.get("resource_id") == resource_id]
            if len(gpu_allocations) != 1 or gpu_allocations[0].get("mode") != "shared":
                raise QualificationError("shared job lease did not contain one shared allocation on the granted GPU")
            amount = gpu_allocations[0].get("quantities", {}).get("memory_bytes")
            if amount != budget:
                raise QualificationError("shared GPU lease does not preserve the declared memory budget")
        report["scenarios"].append({
            "name": "two-shared-allocations-coexist", "passed": True,
            "configured_budget_bytes_each": budget, "worker_receipts": shared_receipts,
            "leases": shared_leases,
        })
        exclusive_id, exclusive_receipt_path, exclusive_release = await submit("exclusive-after-shared", exclusive=True)
        await asyncio.sleep(1.0)
        exclusive_wait = await client.get_job(exclusive_id)
        visible_leases = (await client.call("leases.list", {})).get("leases", [])
        if not exclusive_request_waiting(exclusive_wait):
            raise QualificationError("whole-device exclusive work acquired or attempted execution before shared cleanup")
        report["scenarios"].append({
            "name": "exclusive-waits-for-verified-shared-cleanup", "passed": True,
            "waiting_job": exclusive_wait,
            "active_shared_leases": [l for l in visible_leases if l.get("state") != "released"],
        })
        for _, _, release in shared:
            private_write(release, "release\n")
        completed_shared = []
        for job_id, _, _ in shared:
            detail = await _wait_job(client, scheduler, report, guard, job_id, {"succeeded"}, 120)
            if detail.get("cleanup_verified") is not True:
                raise QualificationError("shared job did not verify process cleanup before releasing capacity")
            completed_shared.append(detail)
        report["scenarios"][-1]["shared_cleanup_jobs"] = completed_shared
        exclusive_ready = await _wait_receipt(exclusive_receipt_path, guard)
        if exclusive_ready.get("gpu_uuid") != grant["gpu_uuid"]:
            raise QualificationError("exclusive worker observed a CUDA UUID outside the parent grant")
        private_write(exclusive_release, "release\n")
        exclusive_done = await _wait_job(client, scheduler, report, guard, exclusive_id, {"succeeded"}, 120)
        if exclusive_done.get("cleanup_verified") is not True:
            raise QualificationError("exclusive job did not verify cleanup")
        report["jobs"].extend([*completed_shared, exclusive_done])
        report["scenarios"].append({
            "name": "exclusive-runs-after-shared-reclamation", "passed": True,
            "worker_receipt": json.loads(exclusive_receipt_path.read_text(encoding="utf-8")),
            "job": exclusive_done,
        })

        # An unmanaged CUDA process is deliberately started outside the
        # scheduler's systemd units. Its finite barrier makes the live test
        # bounded and keeps every operation within the same parent grant.
        unmanaged_label = "unmanaged"
        unmanaged_receipt = work / f"{unmanaged_label}.json"
        unmanaged_release = work / f"{unmanaged_label}.release"
        release_files.append(unmanaged_release)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(grant["gpu_uuid"])
        unmanaged = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--gpu-uuid", str(grant["gpu_uuid"]),
             "--memory-bytes", str(8 * MIB), "--receipt", str(unmanaged_receipt),
             "--release-file", str(unmanaged_release), "--worker-timeout", "180",
             "--run", "--parent-grant", str(args.parent_grant),
             "--project-control-python", str(args.project_control_python),
             "--project-control-runtime", str(args.project_control_runtime)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        unmanaged_ready = await _wait_receipt(unmanaged_receipt, guard)
        unmanaged_exclusive_id, _, unmanaged_job_release = await submit("exclusive-blocked-by-unmanaged", exclusive=True)
        await asyncio.sleep(1.0)
        unmanaged_wait = await client.get_job(unmanaged_exclusive_id)
        current_leases = (await client.call("leases.list", {})).get("leases", [])
        if not exclusive_request_waiting(unmanaged_wait):
            raise QualificationError("exclusive request was granted despite an unmanaged visible CUDA PID")
        observed = gpu_snapshot(str(grant["gpu_uuid"]))
        observed_pids = {int(item["pid"]) for item in observed["processes"] if isinstance(item.get("pid"), int)}
        if unmanaged_ready.get("pid") not in observed_pids:
            raise QualificationError("NVIDIA telemetry did not observe the unmanaged worker PID")
        report["scenarios"].append({
            "name": "unmanaged-process-blocks-exclusive", "passed": True,
            "worker_receipt": unmanaged_ready, "nvidia_observation": observed,
            "waiting_job": unmanaged_wait,
        })
        private_write(unmanaged_release, "release\n")
        if unmanaged.wait(timeout=30) != 0:
            raise QualificationError("unmanaged CUDA allocation worker failed during cleanup")
        unmanaged = None
        await _wait_job(client, scheduler, report, guard, unmanaged_exclusive_id, {"running"}, 120)
        ready = await _wait_receipt(work / "exclusive-blocked-by-unmanaged.json", guard)
        private_write(unmanaged_job_release, "release\n")
        detail = await _wait_job(client, scheduler, report, guard, unmanaged_exclusive_id, {"succeeded"}, 120)
        if detail.get("cleanup_verified") is not True:
            raise QualificationError("exclusive job after unmanaged cleanup lacks verified release")
        report["jobs"].append(detail)
        report["scenarios"].append({
            "name": "exclusive-resumes-after-unmanaged-cleanup", "passed": True,
            "worker_receipt": ready, "job": detail,
        })

        listed = await client.list_resources()
        normalized_resources = _listed_resource_statuses(listed)
        gpu_rows = [(resource, status) for resource, status in normalized_resources
                    if resource.get("kind") == "gpu"]
        if len(gpu_rows) != 1:
            raise QualificationError("private scheduler inventory is not constrained to one granted GPU")
        gpu_resource, gpu_status = gpu_rows[0]
        if gpu_resource.get("resource_id") != resource_id or gpu_resource.get("attributes", {}).get("uuid") != grant["gpu_uuid"]:
            raise QualificationError("private scheduler inventory is not constrained to the exact parent GPU")
        if gpu_status.get("available") is not True:
            raise QualificationError("private scheduler reports the granted GPU unavailable after cleanup")
        status = await client.resource_capabilities()
        report["capabilities"] = status
        profiling = _profiling_domain(
            gpu_resource, [resource for resource, _ in normalized_resources], status,
        )
        if profiling["domain_resource_id"]:
            first_id, first_path, first_release = await submit(
                "profile-first", isolation_id=profiling["domain_resource_id"]
            )
            first_worker = await _wait_receipt(first_path, guard)
            second_id, second_path, second_release = await submit(
                "profile-second", isolation_id=profiling["domain_resource_id"]
            )
            await asyncio.sleep(1.0)
            second_wait = await client.get_job(second_id)
            if second_wait.get("state") not in {"queued", "blocked"} or second_wait.get("attempt_id"):
                raise QualificationError("profiler isolation domain did not serialize competing exclusive profiles")
            private_write(first_release, "release\n")
            first_done = await _wait_job(client, scheduler, report, guard, first_id, {"succeeded"}, 120)
            if first_done.get("cleanup_verified") is not True:
                raise QualificationError("first profiler job did not verify cleanup")
            second_worker = await _wait_receipt(second_path, guard)
            private_write(second_release, "release\n")
            second_done = await _wait_job(client, scheduler, report, guard, second_id, {"succeeded"}, 120)
            if second_done.get("cleanup_verified") is not True:
                raise QualificationError("second profiler job did not verify cleanup")
            profiling.update({
                "passed": True, "first_worker_receipt": first_worker,
                "second_waiting_job": second_wait, "second_worker_receipt": second_worker,
                "completed_jobs": [first_done, second_done],
            })
            report["jobs"].extend([first_done, second_done])
        report["scenarios"].append(profiling)
        report["source_snapshot_after"] = gpu_snapshot(str(grant["gpu_uuid"]))
        report["execution_evidence"] = [await _job_evidence(client, job_id) for job_id in submitted_ids]
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:600]}
    finally:
        for release in release_files:
            if not release.exists():
                try:
                    private_write(release, "release\n")
                except Exception:
                    pass
        if unmanaged is not None:
            try:
                unmanaged.wait(timeout=10)
            except subprocess.TimeoutExpired:
                unmanaged.terminate()
                try:
                    unmanaged.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    unmanaged.kill()
                    unmanaged.wait(timeout=5)
        if base_url and submitted_ids:
            try:
                client = DeviceSchedulerClient(base_url, work / "scheduler" / "client.token", timeout_seconds=30)
                for job_id in submitted_ids:
                    try:
                        detail = await _wait_job(
                            client, scheduler, report, guard, job_id, {"succeeded", "failed", "cancelled"}, 40,
                            check_parent=False,
                        )
                        if detail.get("cleanup_verified") is not True:
                            report.setdefault("cleanup_errors", []).append(f"job {job_id} cleanup was not verified")
                    except Exception as exc:
                        report.setdefault("cleanup_errors", []).append(f"job {job_id}: {type(exc).__name__}")
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(f"scheduler cleanup inspection: {type(exc).__name__}")
        if report.get("cleanup_errors"):
            report["status"] = "failed"
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            try:
                await asyncio.wait_for(server_task, timeout=20)
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(f"private scheduler stop: {type(exc).__name__}")
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["diagnostic_directory"] = str(work)
        try:
            report["parent_grant_rechecked"] = guard.verify(force=True)
        except Exception as exc:
            report["parent_grant_rechecked"] = {"state": "unverified", "error": type(exc).__name__}
            report["status"] = "failed"
        if report.get("cleanup_errors"):
            report["status"] = "failed"
        report["receipt"] = str(work / "qualification.json")
        _write_json_atomic(Path(report["receipt"]), report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _profiling_domain(gpu: Mapping[str, Any], resources: list[Mapping[str, Any]], capabilities: Mapping[str, Any]) -> dict[str, Any]:
    advertised = False
    for adapter in capabilities.get("adapters", []):
        for item in adapter.get("capabilities", []) if isinstance(adapter, Mapping) else []:
            if isinstance(item, Mapping) and item.get("name") == "topology-derived-profiler-isolation-domains" and item.get("supported"):
                advertised = True
    gpu_id = gpu.get("resource_id")
    domains = []
    for item in resources:
        attrs = item.get("attributes") if isinstance(item.get("attributes"), Mapping) else {}
        members = attrs.get("member_resource_ids", ())
        if item.get("kind") == "isolation" and isinstance(members, (list, tuple)) and gpu_id in members:
            domains.append(str(item.get("resource_id")))
    if not advertised:
        return {"name": "topology-profiler-conflict", "passed": None, "domain_resource_id": None, "skipped": "adapter does not advertise the capability"}
    if not domains:
        return {"name": "topology-profiler-conflict", "passed": None, "domain_resource_id": None, "skipped": "no topology isolation domain includes the granted GPU"}
    return {
        "name": "topology-profiler-conflict", "passed": None,
        "domain_resource_id": sorted(domains)[0], "available": True,
        "domain_resource_ids": sorted(domains),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the finite live GPU allocation scenarios")
    parser.add_argument("--parent-grant", type=Path, help="active Project Control parent-pool grant")
    parser.add_argument("--project-control-python", type=Path, default=DEFAULT_PC_PYTHON)
    parser.add_argument("--project-control-runtime", type=Path, default=DEFAULT_PC_RUNTIME)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--memory-bytes", type=int)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--release-file", type=Path)
    parser.add_argument("--worker-timeout", type=float, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        if not args.run or args.parent_grant is None:
            parser.error("CUDA worker mode requires --run and --parent-grant")
        worker_guard = ParentGrantGuard(args.parent_grant, args.project_control_python, args.project_control_runtime)
        try:
            worker_grant = worker_guard.verify(force=True)
        except QualificationError as exc:
            parser.error(str(exc))
        if args.gpu_uuid != worker_grant["gpu_uuid"]:
            parser.error("CUDA worker UUID does not match the active parent grant")
        if not all((args.gpu_uuid, args.memory_bytes, args.receipt, args.release_file)):
            parser.error("internal worker requires GPU UUID, memory size, receipt and release file")
        try:
            print(json.dumps(worker(args.gpu_uuid, args.memory_bytes, args.receipt, args.release_file, args.worker_timeout), sort_keys=True))
        except Exception as exc:
            print(json.dumps({"status": "failed", "type": type(exc).__name__, "message": str(exc)[:300]}), flush=True)
            return 1
        return 0
    if not args.run:
        if args.parent_grant is not None:
            parser.error("--parent-grant is accepted only with --run")
        print(json.dumps({
            "status": "plan-only", "gpu_started": False, "daemon_started": False,
            "scope": "one exact Project Control GPU grant; authenticated Device Scheduler MCP SDK; two shared CUDA allocations; exclusive cleanup gate; unmanaged-PID gate; optional topology isolation",
            "required_for_live_run": ["--run", "--parent-grant PATH"],
        }, indent=2, sort_keys=True))
        return 0
    if args.parent_grant is None:
        parser.error("live GPU execution requires --parent-grant together with --run")
    guard = ParentGrantGuard(args.parent_grant, args.project_control_python, args.project_control_runtime)
    try:
        grant = guard.verify(force=True)
    except QualificationError as exc:
        parser.error(str(exc))
    report = asyncio.run(_execute(args, guard, grant))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
