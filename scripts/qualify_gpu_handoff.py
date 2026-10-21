#!/usr/bin/env python3
"""Plan or execute one isolated llama.cpp / FrameShift GPU handoff journey.

The default is plan-only. Execution requires both an active Project Control
parent-pool grant and the two explicit live-run flags. The parent reservation
must remain held by ``qualification_pool.py`` for the entire run. This harness
is qualification scaffolding, not a controller cutover or production backend
stop procedure.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

import httpx

from device_scheduler_adapters.nvidia import NvidiaAdapter
from device_scheduler_adapters.systemd import SystemdExecutionAdapter
from device_scheduler_core.models import ExecutionHandle
from device_scheduler_mcp.client import DeviceSchedulerClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA = Path("/home/tumlinson/Software/llama.cpp/build/bin/llama-server")
DEFAULT_MODEL = Path(
    "/home/tumlinson/.local/share/core4/models/qwen3.6-35b-a3b-q4-k-m/"
    "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7/model.gguf"
)
DEFAULT_FRAMESHIFT = Path("/home/tumlinson/FrameShift/src")
DEFAULT_PC_PYTHON = Path("/home/tumlinson/.local/state/project-control/current/bin/python")
DEFAULT_PC_RUNTIME = Path("/tmp/codex-todo-orchestrator-1000")
APP_ID = "device-scheduler-gpu-qualification"
AGENT_TYPE_ID = "device-scheduler-gpu-handoff-agent"
MODEL_BINDING_ID = "device-scheduler-qualification-model"
TOOLKIT_ID = "device-scheduler-gpu-probe"
CONTINUATION_REF = "device-scheduler-qualification-frame"
CONSUMER_CONTRACT = "device-scheduler-qualification-turn-boundary-v1"
EXECUTION_PROFILE_ID = "qualification-llama-exec"
BACKEND_PROFILE_ID = "qualification-llama-backend"
BACKEND_CONTROL_REF = "frameshift-local"
PROBE_BYTES = 128 * 1024 * 1024
GIB = 1024**3
MIB = 1024**2
GPU_ENVELOPE_BYTES = 2 * GIB
GPU_KIND = "gpu"
TERMINAL_HANDOFFS = {"complete", "cancelled", "blocked", "unknown"}
GRANT_FIELDS = {
    "schema_version", "owner_id", "gpu_uuid", "resource_ids", "project_root",
    "pid", "process_start", "state", "created_at",
}


class QualificationError(RuntimeError):
    """A precondition, evidence assertion, or bounded operation failed."""


class QualificationCancelled(QualificationError):
    """The handoff reached a safe cancelled terminal state after a signal."""


def _process_start(pid: int) -> str | None:
    """Return Linux /proc start ticks without splitting the parenthesized name."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 1 :].strip().split()
        return tail[19]  # proc field 22 (starttime), after fields 1 and 2
    except (OSError, IndexError, ValueError):
        return None


def _process_is_same(pid: int, process_start: str) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return _process_start(pid) == process_start


def _parse_created_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise QualificationError("parent grant created_at must be an ISO timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QualificationError("parent grant created_at is not an ISO timestamp") from exc
    if stamp.tzinfo is None:
        raise QualificationError("parent grant created_at must include a timezone")
    return stamp.astimezone(timezone.utc)


def validate_parent_grant(
    path: Path | str,
    owner_reader: Callable[[str], Mapping[str, Any] | None],
    *,
    project_root: Path | str = ROOT,
    current_uid: int | None = None,
    now: datetime | None = None,
    process_check: Callable[[int, str], bool] = _process_is_same,
) -> dict[str, Any]:
    """Check the root wrapper's grant against the public PC owner API evidence."""
    grant_path = Path(path).expanduser()
    try:
        info = grant_path.lstat()
    except OSError as exc:
        raise QualificationError("parent grant file is unavailable") from exc
    uid = os.getuid() if current_uid is None else current_uid
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise QualificationError("parent grant must be a regular non-symlink file")
    if info.st_uid != uid or info.st_mode & 0o077:
        raise QualificationError("parent grant must be owned by the current user with mode 0600")
    try:
        grant = json.loads(grant_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError("parent grant is not readable JSON") from exc
    if not isinstance(grant, Mapping) or not GRANT_FIELDS <= set(grant):
        raise QualificationError("parent grant is missing required schema-version-1 fields")
    if grant.get("schema_version") != 1 or grant.get("state") != "active":
        raise QualificationError("parent grant must use schema version 1 and remain active")
    owner_id = grant.get("owner_id")
    gpu_uuid = grant.get("gpu_uuid")
    pid = grant.get("pid")
    start = grant.get("process_start")
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise QualificationError("parent grant owner_id is invalid")
    if not isinstance(gpu_uuid, str) or not re.fullmatch(r"GPU-[A-Fa-f0-9-]{16,64}", gpu_uuid):
        raise QualificationError("parent grant gpu_uuid is invalid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise QualificationError("parent grant pid is invalid")
    if not isinstance(start, str) or not start:
        raise QualificationError("parent grant process_start is invalid")
    created = _parse_created_at(grant.get("created_at"))
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if created > current_time + timedelta(seconds=60):
        raise QualificationError("parent grant timestamp is in the future")
    root = str(Path(project_root).expanduser().resolve())
    if str(Path(str(grant.get("project_root", ""))).expanduser().resolve()) != root:
        raise QualificationError("parent grant project_root does not identify this repository")
    resources_raw = grant.get("resource_ids")
    if isinstance(resources_raw, (str, bytes)) or not isinstance(resources_raw, (list, tuple)):
        raise QualificationError("parent grant resource_ids must be an array")
    resources = [str(item) for item in resources_raw]
    if not resources or len(resources) != len(set(resources)):
        raise QualificationError("parent grant resource_ids must be non-empty and unique")
    exact_gpu = f"accelerator:{gpu_uuid}"
    accelerator_ids = [item for item in resources if item.startswith("accelerator:")]
    if accelerator_ids != [exact_gpu]:
        raise QualificationError("parent grant must cover exactly its one declared accelerator UUID")
    if grant.get("independent_allocator_cutover") is True:
        raise QualificationError("parent grant must remain a subordinate qualification reservation")
    if not process_check(pid, start):
        raise QualificationError("parent reservation process is absent or its PID birth identity changed")
    owner = owner_reader(owner_id)
    if not isinstance(owner, Mapping):
        raise QualificationError("Project Control public owner lookup did not return an active owner")
    owner_resources = owner.get("resources")
    if isinstance(owner_resources, (str, bytes)) or not isinstance(owner_resources, (list, tuple)):
        raise QualificationError("Project Control owner record has no resource list")
    if (
        owner.get("state") != "active"
        or int(owner.get("pid", -1)) != pid
        or str(owner.get("process_start")) != start
        or str(Path(str(owner.get("project_root", ""))).expanduser().resolve()) != root
        or sorted(str(item) for item in owner_resources) != sorted(resources)
        or exact_gpu not in {str(item) for item in owner_resources}
    ):
        raise QualificationError("Project Control owner no longer exactly matches the supplied parent grant")
    return {
        "owner_id": owner_id,
        "gpu_uuid": gpu_uuid,
        "resource_ids": sorted(resources),
        "project_root": root,
        "pid": pid,
        "process_start": start,
        "state": "active",
        "created_at": created.isoformat(),
    }


def read_project_control_owner(
    owner_id: str,
    *,
    python: Path | str = DEFAULT_PC_PYTHON,
    runtime_dir: Path | str = DEFAULT_PC_RUNTIME,
) -> Mapping[str, Any] | None:
    """Use the deployed read-only HostResourceFacade owner lookup."""
    interpreter = Path(python).expanduser()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise QualificationError("deployed Project Control Python is unavailable")
    code = (
        "import json,sys; "
        "from project_control.host_resources import HostCoordinator,HostResourceFacade; "
        "owner=HostResourceFacade(HostCoordinator(create=False)).owner(sys.argv[1]); "
        "print(json.dumps(owner,default=str,separators=(',',':')))"
    )
    env = dict(os.environ)
    runtime = str(Path(runtime_dir).expanduser())
    env["PROJECT_CONTROL_HOST_RESOURCES_RUNTIME_DIR"] = runtime
    env["TODO_BACKGROUND_HOST_RUNTIME_DIR"] = runtime
    try:
        result = subprocess.run(
            [str(interpreter), "-c", code, owner_id], capture_output=True,
            text=True, timeout=10, check=False, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError("Project Control public owner API query failed") from exc
    if result.returncode != 0:
        raise QualificationError("Project Control public owner API query returned an error")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationError("Project Control public owner API returned malformed JSON") from exc
    return value if isinstance(value, Mapping) else None


class ParentGrantGuard:
    def __init__(self, path: Path, pc_python: Path, pc_runtime: Path) -> None:
        self.path = path
        self.pc_python = pc_python
        self.pc_runtime = pc_runtime
        self.grant: dict[str, Any] | None = None
        self.last_check = 0.0

    def verify(self, *, force: bool = False) -> dict[str, Any]:
        if not force and self.grant is not None and time.monotonic() - self.last_check < 30:
            return self.grant
        self.grant = validate_parent_grant(
            self.path,
            lambda owner_id: read_project_control_owner(
                owner_id, python=self.pc_python, runtime_dir=self.pc_runtime,
            ),
            project_root=ROOT,
        )
        self.last_check = time.monotonic()
        return self.grant


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _context_from_prompt(prompt: str) -> Mapping[str, Any]:
    marker = "FRAME_CONTEXT_JSON:\n"
    if marker not in prompt:
        raise ValueError("FrameShift prompt omitted its structured context")
    document = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(document, Mapping):
        raise ValueError("FrameShift context must be an object")
    return document


def _contains_handoff_result(messages: Any, handoff_id: str) -> bool:
    if isinstance(messages, Mapping):
        if messages.get("handoff_id") == handoff_id and isinstance(messages.get("jobs"), list):
            return True
        return any(_contains_handoff_result(value, handoff_id) for value in messages.values())
    if isinstance(messages, (list, tuple)):
        return any(_contains_handoff_result(value, handoff_id) for value in messages)
    return False


def _action_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("action"), str):
            found.append(value)
    return found


def decode_agent_action(text: str, request: Any, *, gpu_uuid: str, handoff_id: str) -> Any:
    """Strictly validate the actual model's first tool proposal and resumed result."""
    from frameshift.models import ActionKind, ProposedAction, TypedPayload

    context = _context_from_prompt(request.input.data)
    messages = context.get("messages", [])
    resumed = _contains_handoff_result(messages, handoff_id)
    candidates = _action_objects(text)
    if not candidates:
        raise ValueError("model response contained no JSON action object")
    action = candidates[-1]
    if not resumed:
        if action.get("action") != ActionKind.TOOL_REQUEST.value:
            raise ValueError("first model turn must request the declared finite GPU probe")
        if action.get("toolkit_id") != TOOLKIT_ID:
            raise ValueError("model requested an unregistered tool")
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping) or (
            arguments.get("operation") != "allocate_read_free"
            or arguments.get("gpu_uuid") != gpu_uuid
            or arguments.get("memory_bytes") != PROBE_BYTES
        ):
            raise ValueError("model tool request did not match the bounded granted-GPU probe contract")
        return ProposedAction(
            kind=ActionKind.TOOL_REQUEST,
            toolkit_id=TOOLKIT_ID,
            arguments=TypedPayload("json", "1", dict(arguments)),
        )
    if action.get("action") != ActionKind.COMPLETE.value:
        raise ValueError("resumed model turn must complete after the durable handoff result")
    payload = action.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("report"), str):
        raise ValueError("resumed model completion must include a report")
    return ProposedAction(
        kind=ActionKind.COMPLETE,
        payload=TypedPayload("json", "1", dict(payload)),
    )


def _prompt_context(context: Mapping[str, Any]) -> str:
    return (
        "You are the single FrameShift continuation in a device-scheduler qualification. "
        "Read the complete durable context below. Return exactly one JSON object and no markdown. "
        f"If there is no message whose handoff_id is {context.get('expected_handoff_id', '')!r}, "
        "request the registered toolkit with this exact shape: "
        '{"action":"tool_request","toolkit_id":"device-scheduler-gpu-probe",'
        f'"arguments":{{"operation":"allocate_read_free","gpu_uuid":"{context.get("gpu_uuid", "")}",'
        f'"memory_bytes":{PROBE_BYTES}}}}}. '
        "After a message contains the matching handoff_id and its jobs result, complete with "
        '{"action":"complete","payload":{"report":"briefly describe the observed result"}}. '
        "Never invent results.\nFRAME_CONTEXT_JSON:\n"
        + json.dumps(_jsonable(context), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _encode_prompt(context: Mapping[str, Any]) -> str:
    return _prompt_context(context)


def _encode_qualification_context(
    context: Mapping[str, Any], *, gpu_uuid: str, handoff_id: str
) -> str:
    """Add trusted fixed identifiers to each reconstructed FrameShift prompt."""
    if not isinstance(context, Mapping):
        raise ValueError("FrameShift context codec input must be an object")
    enriched = dict(context)
    enriched["gpu_uuid"] = gpu_uuid
    enriched["expected_handoff_id"] = handoff_id

    def wire(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): wire(item) for key, item in value.items()}
        if isinstance(value, (tuple, list, set, frozenset)):
            return [wire(item) for item in value]
        fields = getattr(value, "__dataclass_fields__", None)
        if fields is not None:
            return {name: wire(getattr(value, name)) for name in fields}
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (str, int, float, bool)):
            return enum_value
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return wire(value.to_dict())
        raise TypeError(f"unsupported FrameShift context value: {type(value).__name__}")

    return (
        "You are the single FrameShift continuation in a device-scheduler qualification. "
        "Read the complete durable context below. Return exactly one JSON object and no markdown. "
        f"If there is no message whose handoff_id is {handoff_id!r}, request the registered toolkit "
        "with this exact shape: "
        f'{{"action":"tool_request","toolkit_id":"{TOOLKIT_ID}",'
        f'"arguments":{{"operation":"allocate_read_free","gpu_uuid":"{gpu_uuid}",'
        f'"memory_bytes":{PROBE_BYTES}}}}}. '
        "After a message contains the matching handoff_id and its jobs result, complete with "
        '{"action":"complete","payload":{"report":"briefly describe the observed result"}}. '
        "Never invent results.\nFRAME_CONTEXT_JSON:\n"
        + json.dumps(wire(enriched), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _count_tokens(value: Any) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", str(value), flags=re.UNICODE)))


def private_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _toml_mapping(values: Mapping[str, str]) -> str:
    return "{ " + ", ".join(f"{key} = {_toml_string(value)}" for key, value in sorted(values.items())) + " }"


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_source_hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        result[relative] = _sha256_file(path)
    return result


def capture_source_identity(
    *, frameshift_source: Path, llama_binary: Path, model: Path
) -> dict[str, Any]:
    """Bind a run to actual production source and external runtime bytes."""
    production_roots = [
        ROOT / "packages" / "device-scheduler-core" / "src",
        ROOT / "packages" / "device-scheduler-adapters" / "src",
        ROOT / "packages" / "device-scheduler-mcp" / "src",
    ]
    production_hashes: dict[str, str] = {}
    for source_root in production_roots:
        for source_path in source_root.rglob("*.py"):
            relative = source_path.relative_to(ROOT).as_posix()
            production_hashes[relative] = _sha256_file(source_path)
    qualification_paths = [
        Path(__file__).resolve(), Path(__file__).with_name("qualification_gpu_probe.py").resolve(),
    ]
    qualification_hashes = _python_source_hashes(ROOT, qualification_paths)

    fs_root = frameshift_source.expanduser().resolve()
    frameshift_hashes = {
        path.relative_to(fs_root).as_posix(): _sha256_file(path)
        for path in sorted((fs_root / "frameshift").rglob("*.py"))
    }


def _source_identity_matches(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """Require unchanged code/runtime identities over the live daemon interval."""
    return dict(before) == dict(after)
    if not frameshift_hashes:
        raise QualificationError("FrameShift source identity is empty")
    try:
        revision = subprocess.run(
            ["git", "-C", str(fs_root.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        frameshift_head = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        frameshift_head = None

    binary_path = llama_binary.expanduser().resolve()
    binary_stat = binary_path.stat()
    model_path = model.expanduser().resolve()
    model_stat = model_path.stat()
    return {
        "production_source_hashes": dict(sorted(production_hashes.items())),
        "qualification_source_hashes": qualification_hashes,
        "external_frameshift": {
            "source_root": str(fs_root),
            "git_head": frameshift_head,
            "source_hashes": dict(sorted(frameshift_hashes.items())),
        },
        "llama_binary": {
            "path": str(binary_path),
            "sha256": _sha256_file(binary_path),
            "size_bytes": binary_stat.st_size,
        },
        "model_identity": {
            "path": str(model_path),
            "size_bytes": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
            "filename_digest_hint": model_path.parent.name,
            "content_sha256": None,
            "content_sha256_status": "not-computed-large-local-model",
        },
    }


def _resource_data(value: Any) -> dict[str, Any]:
    raw = value.to_dict() if hasattr(value, "to_dict") else value
    if not isinstance(raw, Mapping):
        raise QualificationError("NVIDIA adapter returned an invalid resource record")
    return dict(raw)


def gpu_snapshot(uuid_value: str) -> dict[str, Any]:
    snapshot = NvidiaAdapter().snapshot()
    inventory = snapshot.get("gpu_inventory")
    process_observation = snapshot.get("process_telemetry")
    if not isinstance(inventory, Mapping) or inventory.get("state") != "observed":
        raise QualificationError("NVIDIA UUID inventory is unknown")
    if not isinstance(process_observation, Mapping) or process_observation.get("state") != "observed":
        raise QualificationError("NVIDIA compute-process telemetry is unknown")
    matches = [
        _resource_data(item) for item in snapshot.get("resources", ())
        if _resource_data(item).get("attributes", {}).get("uuid") == uuid_value
    ]
    if len(matches) != 1:
        raise QualificationError("NVIDIA snapshot did not identify exactly one granted UUID")
    resource = matches[0]
    attrs = dict(resource.get("attributes") or {})
    total = int(float(resource.get("capacities", {}).get("memory_bytes", 0)))
    free_raw = attrs.get("memory_free_bytes_observation")
    if isinstance(free_raw, bool) or not isinstance(free_raw, (int, float)):
        raise QualificationError("granted GPU free-memory observation is unknown")
    if attrs.get("process_telemetry") != "compute-apps-visible":
        raise QualificationError("granted GPU process identity is not fully visible")
    processes = attrs.get("processes_observation")
    if not isinstance(processes, list):
        raise QualificationError("granted GPU process observations are unavailable")
    return {
        "resource_id": resource.get("resource_id"),
        "gpu_uuid": uuid_value,
        "name": attrs.get("name"),
        "pci_bus_id": attrs.get("pci_bus_id"),
        "memory_total_bytes": total,
        "memory_free_bytes": int(free_raw),
        "inventory_observed_at": attrs.get("inventory_observed_at"),
        "process_observed_at": attrs.get("process_observed_at"),
        "process_telemetry": attrs.get("process_telemetry"),
        "process_count": attrs.get("process_count_observation"),
        "processes": [dict(item) for item in processes if isinstance(item, Mapping)],
        "processes_observation": dict(process_observation),
        "resource": resource,
    }


def _managed_processes(handle: Mapping[str, Any]) -> list[int]:
    observed = SystemdExecutionAdapter().managed_process_ids([ExecutionHandle.from_dict(handle)])
    if observed.get("state") != "observed":
        raise QualificationError("systemd invocation process mapping is not verified")
    pids = observed.get("process_ids")
    if not isinstance(pids, list) or not pids:
        raise QualificationError("managed backend invocation has no observed process IDs")
    return sorted(int(pid) for pid in pids)


def _assert_gpu_processes(sample: Mapping[str, Any], pids: list[int], *, positive_memory: bool) -> int:
    rows = sample.get("processes", ())
    matched = [item for item in rows if item.get("pid") in pids]
    if not set(pids) & {item.get("pid") for item in matched}:
        raise QualificationError("NVIDIA process telemetry does not contain the managed backend PID")
    used_values = [item.get("used_memory_bytes") for item in matched]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in used_values):
        raise QualificationError("managed backend GPU memory use is not observed")
    used = sum(used_values)
    if positive_memory and used <= 0:
        raise QualificationError("model weights were not observed resident on the granted GPU")
    return used


def _load_frameshift(frameshift_source: Path) -> None:
    resolved = frameshift_source.expanduser().resolve()
    if not (resolved / "frameshift" / "__init__.py").is_file():
        raise QualificationError("FrameShift source checkout is unavailable")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    import frameshift

    module_path = Path(frameshift.__file__).resolve()
    if resolved not in module_path.parents:
        raise QualificationError("imported FrameShift package does not come from the requested source checkout")


def _frameshift_runtime(
    work: Path,
    *,
    backend_id: str,
    llama_port: int,
    gpu_uuid: str,
    handoff_id: str,
    frame_id: str,
    app_token: str,
    scheduler_token: str,
    timeout_seconds: float,
) -> tuple[Any, Any, Any, dict[str, str]]:
    from frameshift import (
        ActionKind, AgentType, CapabilityGrant, Engine, FrameBudget, Lifecycle,
        ModelBinding, ProposedAction, Registry, Toolkit, TypedPayload,
    )
    from frameshift.backends.llama import LlamaCppHTTPProvider
    from frameshift.service import FrameShiftService
    from frameshift.client import FrameShiftClient

    registry = Registry()
    registry.register_codec(
        __import__("frameshift.models", fromlist=["PayloadCodec"]).PayloadCodec(
            "qualification.prompt", "1", "text", count_tokens=_count_tokens,
            encode=functools.partial(
                _encode_qualification_context, gpu_uuid=gpu_uuid, handoff_id=handoff_id,
            ),
        )
    )
    registry.register_model_binding(ModelBinding(
        id=MODEL_BINDING_ID, version="1", model_id="qwen3.6-35b-a3b-q4-k-m",
        model_version="local-gguf-pinned", input_codecs=(("qualification.prompt", "1"),),
        output_codecs=(("json", "1"),), max_context_tokens=4096,
    ))
    registry.register_toolkit(Toolkit(
        id=TOOLKIT_ID, version="1", capabilities=frozenset({"gpu.probe"}),
        description="Request one scheduler-managed bounded CUDA allocation/copy/free job.",
    ))
    registry.register_agent_type(AgentType(
        id=AGENT_TYPE_ID, version="1", lifecycle=Lifecycle.PERSISTENT,
        input_codec=("qualification.prompt", "1"), output_codec=("json", "1"),
        message_codec=("json", "1"), model_binding_ids=(MODEL_BINDING_ID,),
        toolkit_ids=(TOOLKIT_ID,),
        allowed_actions=frozenset({ActionKind.TOOL_REQUEST, ActionKind.COMPLETE, ActionKind.FAIL}),
        max_context_tokens=4096, max_output_tokens=128,
    ))
    registry.register_application(APP_ID)
    registry.register_grant(CapabilityGrant(
        id="qualification-grant", app_id=APP_ID,
        allowed_agent_types=frozenset({AGENT_TYPE_ID}), allowed_toolkits=frozenset({TOOLKIT_ID}),
        capabilities=frozenset({"gpu.probe"}), max_frames=1, max_children_per_frame=0,
        max_turns=3, max_output_tokens_per_turn=128, max_context_tokens=4096,
    ))

    def decoder(text: str, request: Any) -> Any:
        return decode_agent_action(text, request, gpu_uuid=gpu_uuid, handoff_id=handoff_id)

    provider = LlamaCppHTTPProvider(
        f"http://127.0.0.1:{llama_port}", provider_id="qualification-llama-provider",
        backend_id=backend_id, model_id="qwen3.6-35b-a3b-q4-k-m", model_version="local-gguf-pinned",
        input_codecs=(("qualification.prompt", "1"),), output_codecs=(("json", "1"),),
        max_concurrency=1, request_timeout=timeout_seconds, output_decoder=decoder,
    )
    engine = Engine(str(work / "frameshift.sqlite3"), registry, [provider])
    service = FrameShiftService(
        engine, {APP_ID: app_token}, scheduler_tokens={"device-scheduler": scheduler_token},
    )
    server = service.create_server("127.0.0.1", 0, owner_lock_path=work / "frameshift.lock")
    server_thread = threading.Thread(target=server.serve_forever, name="qualification-frameshift-http", daemon=True)
    server_thread.start()
    port = server.server_address[1]
    client = FrameShiftClient(f"http://127.0.0.1:{port}", app_token, timeout=max(10, min(timeout_seconds, 60)))
    tokens = {"app": app_token, "scheduler": scheduler_token}
    return server, server_thread, client, {**tokens, "base_url": f"http://127.0.0.1:{port}"}


def _write_scheduler_configuration(
    work: Path,
    *,
    port: int,
    token: str,
    scheduler_token_file: Path,
    fs_url: str,
    fs_app_token_file: Path,
    frame_id: str,
    llama_binary: Path,
    model: Path,
    llama_port: int,
    gpu_uuid: str,
) -> tuple[Path, Path, Path, Path]:
    token_dir = work / "tokens"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    token_file = work / "client.token"
    private_write(token_file, token + "\n")
    digest = hashlib.sha256(token.encode()).hexdigest()
    principal = (
        "[principal]\n"
        'principal_id = "qualification"\n'
        'actions = ["*"]\n'
        f"execution_roots = {_toml_array([str(work), str(ROOT)])}\n"
        'backend_ids = ["*"]\n'
        "max_priority = 100\n"
        f"token_sha256 = {_toml_string(digest)}\n"
    )
    private_write(token_dir / "qualification.toml", principal)
    state_dir = work / "state"
    state_dir.mkdir(mode=0o700)
    state_dir.chmod(0o700)
    config = work / "service.toml"
    private_write(config, (
        "[server]\nhost = \"127.0.0.1\"\n"
        f"port = {port}\ntick_interval_seconds = 0.1\nhttp_timeout_seconds = 30.0\n\n"
        "[paths]\n"
        f"state_directory = {_toml_string(str(state_dir))}\n"
        f"token_directory = {_toml_string(str(token_dir))}\n"
        f"token_file = {_toml_string(str(token_file))}\n"
        f"profiles_path = {_toml_string(str(work / 'profiles.toml'))}\n"
        f"backend_profiles_path = {_toml_string(str(work / 'backend-profiles.toml'))}\n"
        f"continuations_path = {_toml_string(str(work / 'continuations.toml'))}\n"
    ))
    execution = work / "profiles.toml"
    argv = [
        str(llama_binary), "-m", str(model), "--n-gpu-layers", "1", "--ctx-size", "4096",
        "--batch-size", "32", "--ubatch-size", "32", "--parallel", "1", "--slots",
        "--host", "127.0.0.1", "--port", str(llama_port), "--alias", "device-scheduler-qualification",
    ]
    profile = (
        f"[profiles.{_toml_string(EXECUTION_PROFILE_ID)}]\n"
        f"profile_version = \"1\"\nargv = {_toml_array(argv)}\n"
        f"working_directory = {_toml_string(str(work))}\n"
        f"environment = {_toml_mapping({'CUDA_VISIBLE_DEVICES': gpu_uuid, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'})}\n"
        "limits = { cpu_quota_percent = 800.0, memory_max_bytes = 51539607552, tasks_max = 128 }\n"
    )
    private_write(execution, profile)
    backend_file = work / "backend-profiles.toml"
    profile_url = f"http://127.0.0.1:{llama_port}"
    backend_text = (
        f"[backend_profiles.{_toml_string(BACKEND_PROFILE_ID)}]\n"
        f"profile_id = {_toml_string(BACKEND_PROFILE_ID)}\nversion = \"1\"\n"
        f"execution_profile_id = {_toml_string(EXECUTION_PROFILE_ID)}\n"
        f"health_url = {_toml_string(profile_url + '/health')}\n"
        'expected_model_identity = "device-scheduler-qualification"\n'
        f"identity_url = {_toml_string(profile_url + '/props')}\n"
        'identity_json_path = "model_alias"\n'
        f"slots_url = {_toml_string(profile_url + '/slots')}\n"
        'access_mode = "frameshift-only"\n'
        f"frameshift_control_ref = {_toml_string(BACKEND_CONTROL_REF)}\n"
        'health_json_path = "status"\nexpected_health_value = "ok"\n'
        "readiness_interval_seconds = 0.25\nreadiness_timeout_seconds = 300.0\nstop_timeout_seconds = 30.0\n\n"
        f"[frameshift_controls.{_toml_string(BACKEND_CONTROL_REF)}]\n"
        f"base_url = {_toml_string(fs_url)}\n"
        f"token_file = {_toml_string(str(scheduler_token_file))}\n"
        "timeout_seconds = 5.0\n"
    )
    private_write(backend_file, backend_text)
    continuation_file = work / "continuations.toml"
    private_write(continuation_file, (
        f"[continuations.{_toml_string(CONTINUATION_REF)}]\n"
        f"base_url = {_toml_string(fs_url)}\n"
        f"frame_id = {_toml_string(frame_id)}\n"
        f"token_file = {_toml_string(str(fs_app_token_file))}\n"
    ))
    return config, token_file, token_dir, state_dir


async def _wait_http_ready(origin: str, daemon: subprocess.Popen[Any], *, timeout: float = 25) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2) as http:
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                raise QualificationError("private Device Scheduler daemon exited before readiness")
            try:
                response = await http.get(origin + "/readyz")
                if response.status_code == 200 and response.json().get("status") == "ready":
                    health = await http.get(origin + "/healthz")
                    if health.status_code != 200:
                        raise QualificationError("private Device Scheduler health check failed")
                    return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(0.2)
    raise QualificationError("private Device Scheduler daemon did not become ready")


async def _wait_frame_state(client: Any, frame_id: str, states: set[str], *, deadline: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    while time.monotonic() < deadline:
        frame, result = await asyncio.to_thread(client.inspect, frame_id), await asyncio.to_thread(client.result, frame_id)
        state = str(frame.get("state", "unknown"))
        if state in {"failed", "cancelled"}:
            raise QualificationError(f"FrameShift continuation entered terminal state {state}")
        if state in states and result is not None:
            return frame, result
        await asyncio.sleep(0.5)
    raise TimeoutError("FrameShift continuation did not reach the expected bounded state")


async def _cancel_handoff_until_terminal(
    client: Any,
    guard: ParentGrantGuard,
    handoff_id: str,
    *,
    timeout_seconds: float,
    reason: str = "qualification received SIGINT or SIGTERM",
) -> dict[str, Any]:
    """Request cancellation once and wait boundedly for the coordinator's cleanup result."""
    await client.control_handoff(
        handoff_id, "cancel", reason=reason,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        guard.verify()
        record = await client.get_handoff(handoff_id)
        state = str(record.get("state", "unknown"))
        if state in TERMINAL_HANDOFFS:
            return record
        await asyncio.sleep(0.5)
    raise TimeoutError(
        "handoff cancellation did not reach a safe terminal state; preserve diagnostics and parent grant"
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} response was not an object")
    return dict(value)


def _fs_result_kind(result: Mapping[str, Any]) -> str:
    action = result.get("action")
    if not isinstance(action, Mapping):
        raise QualificationError("FrameShift result omitted its proposed action")
    return str(action.get("kind", "unknown"))


def _llama_health(profile_url: str) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for key, route in (("health", "/health"), ("identity", "/props"), ("slots", "/slots")):
        request = Request(profile_url + route, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            if response.status != 200:
                raise QualificationError(f"llama.cpp {route} returned HTTP {response.status}")
            value = json.loads(response.read().decode("utf-8"))
        observed[key] = value
    if observed["health"].get("status") != "ok":
        raise QualificationError("llama.cpp health response was not ready")
    if observed["identity"].get("model_alias") != "device-scheduler-qualification":
        raise QualificationError("llama.cpp model identity did not match the pinned alias")
    slots = observed["slots"]
    if not isinstance(slots, list) or not slots or any(item.get("is_processing") is not False for item in slots):
        raise QualificationError("llama.cpp slots are not in a known idle state")
    return observed


async def execute_qualification(args: argparse.Namespace, guard: ParentGrantGuard) -> int:
    grant = guard.verify(force=True)
    gpu_uuid = grant["gpu_uuid"]
    work = Path(tempfile.mkdtemp(prefix="device-scheduler-gpu-handoff-"))
    work.chmod(0o700)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "single-host-real-llama-frameshift-device-scheduler-handoff",
        "status": "running",
        "parent_grant": {key: grant[key] for key in ("owner_id", "gpu_uuid", "resource_ids", "pid", "process_start")},
        "diagnostic_directory": str(work),
        "source": {
            "repository_root": str(ROOT),
            "frameshift_source": str(args.frameshift_source.resolve()),
            "llama_binary": str(args.llama_binary.resolve()),
            "model_path": str(args.model.resolve()),
        },
        "checks": {},
        "limitations": [
            "single-user cooperative host; CUDA_VISIBLE_DEVICES is routing, not hostile-code isolation",
            "one 128 MiB CUDA allocation/copy/free smoke; no kernel correctness suite or performance claim",
            "manual exact-unit fixture cleanup is qualification-only and is not a production backend stop API",
            "one fresh FrameShift frame and one local llama.cpp backend; no multi-host or restart matrix",
        ],
    }
    daemon: subprocess.Popen[Any] | None = None
    fs_server: Any = None
    fs_server_thread: threading.Thread | None = None
    ds_client: DeviceSchedulerClient | None = None
    frame_client: Any = None
    backend_id = "qualification-llama-" + uuid.uuid4().hex[:12]
    handoff_id = "qualification-handoff-" + uuid.uuid4().hex[:12]
    frame_id = str(uuid.uuid4())
    backend_handle: dict[str, Any] | None = None
    attached = False
    handoff_state: str | None = None
    cleanup_safe = False
    llama_url = ""
    stop_requested = False
    old_handlers: dict[int, Any] = {}
    source_identity_before: dict[str, Any] | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        report["signal"] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)
    try:
        _load_frameshift(args.frameshift_source)
        if not args.llama_binary.is_file() or not os.access(args.llama_binary, os.X_OK):
            raise QualificationError("local llama-server binary is missing or not executable")
        if not args.model.is_file() or args.model.stat().st_size <= 0:
            raise QualificationError("the pinned local GGUF model is missing or empty")
        source_identity_before = capture_source_identity(
            frameshift_source=args.frameshift_source,
            llama_binary=args.llama_binary,
            model=args.model,
        )
        report["source_identity_before"] = source_identity_before
        guard.verify(force=True)
        before = gpu_snapshot(gpu_uuid)
        if "v100" not in str(before.get("name", "")).casefold():
            raise QualificationError("the granted UUID is not observed as a V100 GPU")
        if int(before["memory_total_bytes"]) < 14 * GIB:
            raise QualificationError("the granted GPU does not expose the expected 16 GiB V100 class capacity")
        if int(before["memory_free_bytes"]) < 3 * GIB:
            raise QualificationError("insufficient fresh free-memory headroom for the 2 GiB managed envelope")
        if any(row.get("gpu_uuid") == gpu_uuid for row in before["processes"]):
            raise QualificationError("parent wrapper did not establish an empty granted GPU before the child run")
        report["gpu_before_backend"] = {key: before[key] for key in (
            "resource_id", "gpu_uuid", "name", "pci_bus_id", "memory_total_bytes", "memory_free_bytes",
            "inventory_observed_at", "process_observed_at", "process_count",
        )}
        report["checks"]["parent_grant_active"] = True
        report["checks"]["single_v100_uuid"] = True
        report["checks"]["pre_backend_quiescent"] = True

        # Create independent random application, scheduler and Device Scheduler credentials.
        ds_token = secrets.token_urlsafe(48)
        fs_app_token = secrets.token_urlsafe(48)
        fs_scheduler_token = secrets.token_urlsafe(48)
        fs_app_token_file = work / "frameshift-app.token"
        fs_scheduler_token_file = work / "frameshift-scheduler.token"
        private_write(fs_app_token_file, fs_app_token + "\n")
        private_write(fs_scheduler_token_file, fs_scheduler_token + "\n")
        ds_port = _port()
        llama_port = _port()
        llama_url = f"http://127.0.0.1:{llama_port}"
        server, server_thread, frame_client, fs_config = _frameshift_runtime(
            work, backend_id=backend_id, llama_port=llama_port, gpu_uuid=gpu_uuid,
            handoff_id=handoff_id, frame_id=frame_id, app_token=fs_app_token,
            scheduler_token=fs_scheduler_token, timeout_seconds=args.timeout_seconds,
        )
        fs_server, fs_server_thread = server, server_thread
        scheduler_config, ds_token_file, _, _ = _write_scheduler_configuration(
            work, port=ds_port, token=ds_token, scheduler_token_file=fs_scheduler_token_file,
            fs_url=fs_config["base_url"], fs_app_token_file=fs_app_token_file,
            frame_id=frame_id, llama_binary=args.llama_binary.resolve(), model=args.model.resolve(),
            llama_port=llama_port, gpu_uuid=gpu_uuid,
        )
        origin = f"http://127.0.0.1:{ds_port}"
        daemon_command = [
            sys.executable, "-m", "device_scheduler_mcp", "--config", str(scheduler_config),
            "serve", "--log-level", "error",
        ]
        daemon_env = dict(os.environ)
        package_paths = [str(ROOT / "packages" / name / "src") for name in (
            "device-scheduler-core", "device-scheduler-adapters", "device-scheduler-mcp",
        )]
        daemon_env["PYTHONPATH"] = os.pathsep.join(package_paths + ([daemon_env["PYTHONPATH"]] if daemon_env.get("PYTHONPATH") else []))
        with (work / "device-scheduler.log").open("ab") as log:
            daemon = subprocess.Popen(daemon_command, cwd=ROOT, env=daemon_env, stdout=log, stderr=log, start_new_session=True)
        await _wait_http_ready(origin, daemon)
        ds_client = DeviceSchedulerClient(origin, ds_token_file, timeout_seconds=30)
        report["checks"]["private_daemon_healthy"] = True

        resources_response = await ds_client.call("resources.list")
        resources = resources_response.get("resources", [])
        selected = [item for item in resources if item.get("attributes", {}).get("uuid") == gpu_uuid]
        if len(selected) != 1 or selected[0].get("kind", "").casefold() not in {"gpu", "nvidia_gpu", "cuda_gpu"}:
            raise QualificationError("Device Scheduler did not discover exactly the grant's GPU UUID")
        resource = selected[0]
        if resource.get("resource_id") != before["resource_id"]:
            raise QualificationError("Device Scheduler and NVIDIA snapshot disagree on the granted GPU identity")
        report["scheduler_gpu_resource"] = resource
        report["checks"]["scheduler_discovered_exact_uuid"] = True

        backend_spec = {
            "backend_id": backend_id,
            "host_id": resource["host_id"],
            "adapter_id": "managed-llama.cpp",
            "profile_id": BACKEND_PROFILE_ID,
            "configuration_ref": BACKEND_PROFILE_ID,
            "residency_requirements": [{
                "kind": GPU_KIND, "quantities": {"memory_bytes": GPU_ENVELOPE_BYTES},
                "selectors": {"uuid": gpu_uuid}, "mode": "shared",
            }],
            "attached_consumers": [],
            "eviction_contract": {
                "consumer_ack_required": True, "drain_timeout_seconds": 90,
                "model_eviction_requires_explicit_grant": True,
            },
        }
        guard.verify(force=True)
        registered = await ds_client.call("backends.register", {
            "spec": backend_spec, "idempotency_key": f"qualify-register-{backend_id}",
        })
        if registered.get("state") != "stopped":
            raise QualificationError("managed backend registration did not start in stopped state")
        report["backend_id"] = backend_id

        # Backend start itself establishes and journals the exact FrameShift
        # admission close before launching the managed service.
        from frameshift.client import FrameShiftSchedulerClient

        scheduler_client = FrameShiftSchedulerClient(fs_config["base_url"], fs_scheduler_token, timeout=10)
        guard.verify(force=True)
        started = await ds_client.call("backends.control", {"backend_id": backend_id, "action": "start"})
        if started.get("state") != "ready" or not isinstance(started.get("execution_handle"), Mapping):
            raise QualificationError("Device Scheduler did not return a ready backend and durable execution handle")
        backend_handle = dict(started["execution_handle"])
        initial_invocation = (backend_handle.get("metadata") or {}).get("invocation_id")
        if not initial_invocation:
            raise QualificationError("managed backend execution handle omitted its invocation identity")
        gate_open = await asyncio.to_thread(scheduler_client.backend_admission, backend_id)
        if gate_open.get("closed") is not False or gate_open.get("open") is not True:
            raise QualificationError("backend start did not reopen FrameShift admission")
        resident = await asyncio.to_thread(gpu_snapshot, gpu_uuid)
        resident_pids = await asyncio.to_thread(_managed_processes, backend_handle)
        resident_used = _assert_gpu_processes(resident, resident_pids, positive_memory=True)
        if resident["memory_free_bytes"] >= before["memory_free_bytes"]:
            raise QualificationError("V100 free memory did not decrease after llama.cpp loaded")
        report["initial_backend"] = {
            "execution_handle": backend_handle,
            "process_ids": resident_pids,
            "used_gpu_memory_bytes": resident_used,
            "gpu_observation": {key: resident[key] for key in ("memory_free_bytes", "process_count", "inventory_observed_at", "process_observed_at")},
            "frame_shift_gate_after_start": gate_open,
        }
        report["checks"]["backend_started_and_ready"] = True
        report["checks"]["backend_weights_resident"] = True

        frame = await asyncio.to_thread(frame_client.spawn, AGENT_TYPE_ID,
            "Request the one registered CUDA probe, then use its durable result to finish this qualification turn.",
            grant_id="qualification-grant", frame_id=frame_id,
            budget={"turns_remaining": 3, "max_output_tokens_per_turn": 128, "max_context_tokens": 4096},
        )
        attached = True  # set before the call so an uncertain attach is handled conservatively
        guard.verify(force=True)
        attached_result = await ds_client.call("backends.attach", {
            "backend_id": backend_id,
            "continuation_ref": CONTINUATION_REF,
            "contract_id": CONSUMER_CONTRACT,
            "idempotency_key": f"qualify-attach-{backend_id}-{frame_id}",
        })
        report["frame_id"] = frame_id
        report["frame_spawn"] = frame
        if CONTINUATION_REF not in (attached_result.get("spec") or {}).get("attached_consumers", attached_result.get("attached_consumers", [])):
            # Some SDK projections omit the nested spec; verify through the registered backend authority.
            attached_result = await ds_client.call("backends.get", {"backend_id": backend_id})
            if CONTINUATION_REF not in (attached_result.get("spec") or {}).get("attached_consumers", []):
                raise QualificationError("FrameShift continuation did not appear in the registered backend consumer set")
        server.start_scheduler(poll_interval=0.05, workers=1)
        first_frame, first_result = await _wait_frame_state(
            frame_client, frame_id, {"suspended"}, deadline=time.monotonic() + min(args.timeout_seconds, 600),
        )
        if _fs_result_kind(first_result) != "tool_request":
            raise QualificationError("actual model did not produce the declared first-turn tool request")
        first_action = _mapping(first_result["action"], "initial FrameShift action")
        first_args = _mapping(first_action.get("arguments"), "initial FrameShift tool arguments")
        first_payload = first_args.get("data", first_args)
        if not isinstance(first_payload, Mapping) or first_payload.get("gpu_uuid") != gpu_uuid or first_payload.get("memory_bytes") != PROBE_BYTES:
            raise QualificationError("durable first-turn action did not bind the requested probe to the grant UUID")

        paused_handle = await ds_client.call("backends.get", {"backend_id": backend_id})
        paused_handle = _mapping(paused_handle.get("execution_handle"), "backend handle after ordinary suspension")
        paused_invocation = (paused_handle.get("metadata") or {}).get("invocation_id")
        paused_snapshot = await asyncio.to_thread(gpu_snapshot, gpu_uuid)
        paused_pids = await asyncio.to_thread(_managed_processes, paused_handle)
        paused_used = _assert_gpu_processes(paused_snapshot, paused_pids, positive_memory=True)
        if paused_invocation != initial_invocation:
            raise QualificationError("ordinary FrameShift tool suspension unexpectedly replaced the backend invocation")
        report["ordinary_suspension"] = {
            "frame_state": first_frame.get("state"), "generation": first_result.get("generation"),
            "execution_handle": paused_handle, "process_ids": paused_pids,
            "used_gpu_memory_bytes": paused_used,
            "gpu_observation": {key: paused_snapshot[key] for key in ("memory_free_bytes", "process_count", "inventory_observed_at", "process_observed_at")},
        }
        report["checks"]["ordinary_pause_kept_weights_resident"] = True

        # Submit one bounded tool result as a handoff job. GPU memory is advisory;
        # the exact CUDA UUID is also pinned into the command environment.
        job = {
            "owner_id": "qualification", "argv": [sys.executable, str(Path(__file__).with_name("qualification_gpu_probe.py")),
                "--gpu-uuid", gpu_uuid, "--memory-bytes", str(PROBE_BYTES)],
            "working_directory": str(ROOT),
            "environment": {"CUDA_VISIBLE_DEVICES": gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            "resources": [{"kind": GPU_KIND, "quantities": {"memory_bytes": 256 * MIB}, "selectors": {"uuid": gpu_uuid}, "mode": "shared"}],
            "priority_class": "interactive", "priority": 90, "timeout_seconds": 180,
            "max_attempts": 1,
            "interruption": {"cooperative_yield": False, "checkpoint_supported": False,
                "restart_safe": False, "cancellation_supported": True},
        }
        deadline = datetime.now(timezone.utc) + timedelta(seconds=min(args.timeout_seconds, 2400))
        handoff_spec = {
            "handoff_id": handoff_id, "owner_id": "qualification", "jobs": [job],
            "backend_ids": [backend_id], "continuation_refs": [CONTINUATION_REF],
            "eviction_grants": [{
                "grant_id": "grant-" + uuid.uuid4().hex,
                "backend_id": backend_id, "granted_by": "qualification",
                "scope": f"backend:{backend_id}",
                "expires_at": (deadline + timedelta(minutes=2)).isoformat(),
            }],
            "resource_envelope": [{
                "kind": GPU_KIND, "quantities": {"memory_bytes": GPU_ENVELOPE_BYTES},
                "selectors": {"uuid": gpu_uuid}, "mode": "shared",
            }],
            "drain_timeout_seconds": 90, "run_timeout_seconds": 240,
            "restore_timeout_seconds": 360, "deadline": deadline.isoformat(),
        }
        guard.verify(force=True)
        submitted = await ds_client.submit_handoff({
            "spec": handoff_spec, "idempotency_key": f"qualify-{handoff_id}",
        })
        report["handoff_submission"] = submitted
        if submitted.get("handoff_id") != handoff_id:
            raise QualificationError("Device Scheduler did not persist the requested handoff identity")
        end = time.monotonic() + min(args.timeout_seconds, 2400)
        handoff: dict[str, Any] = submitted
        while time.monotonic() < end:
            if stop_requested:
                report["cancellation_requested_at"] = datetime.now(timezone.utc).isoformat()
                handoff = await _cancel_handoff_until_terminal(
                    ds_client, guard, handoff_id,
                    timeout_seconds=min(args.timeout_seconds, 600),
                )
                handoff_state = str(handoff.get("state", "unknown"))
                report["handoff"] = handoff
                if handoff_state != "cancelled":
                    raise QualificationError(
                        f"handoff cancellation settled as {handoff_state!r}; preserve diagnostics and inspect explicitly"
                    )
                report["checks"]["bounded_handoff_cancellation_and_recovery"] = True
                report["limitations"].append(
                    "signal cancellation leaves the private frame and backend ledger for explicit inspection; no automatic rerun or fixture stop is attempted"
                )
                raise QualificationCancelled(
                    "handoff reached its cancelled terminal state; inspect its durable receipt before releasing the parent grant"
                )
            guard.verify()
            handoff = await ds_client.get_handoff(handoff_id)
            handoff_state = str(handoff.get("state", "unknown"))
            if handoff_state in TERMINAL_HANDOFFS:
                break
            await asyncio.sleep(0.5)
        else:
            report["handoff_timeout_at"] = datetime.now(timezone.utc).isoformat()
            try:
                handoff = await _cancel_handoff_until_terminal(
                    ds_client, guard, handoff_id,
                    timeout_seconds=min(args.timeout_seconds, 600),
                    reason="qualification run deadline elapsed",
                )
            except Exception as exc:
                raise TimeoutError(
                    "handoff deadline elapsed and bounded cancellation could not be verified; preserve diagnostics and parent grant"
                ) from exc
            handoff_state = str(handoff.get("state", "unknown"))
            report["handoff"] = handoff
            if handoff_state == "cancelled":
                report["checks"]["bounded_handoff_cancellation_and_recovery"] = True
                report["limitations"].append(
                    "run deadline cancelled the handoff safely; preserve the private frame and backend ledger for explicit inspection"
                )
                raise QualificationCancelled(
                    "run deadline elapsed; handoff reached its cancelled terminal state without automatic replay"
                )
            raise TimeoutError(
                f"handoff deadline elapsed; coordinator settled as {handoff_state!r}, preserve diagnostics and parent grant"
            )
        report["handoff"] = handoff
        if handoff_state != "complete":
            raise QualificationError(f"handoff stopped in {handoff_state!r}; preserve diagnostics and do not retry automatically")
        report["checks"]["handoff_complete"] = True

        job_ids = handoff.get("job_ids")
        if not isinstance(job_ids, list) or len(job_ids) != 1:
            raise QualificationError("handoff did not record exactly one finite CUDA probe job")
        job_record = await ds_client.get_job(str(job_ids[0]))
        logs = await ds_client.job_logs(str(job_ids[0]))
        stdout = (logs.get("stdout") or {}).get("text", "")
        if not isinstance(stdout, str):
            raise QualificationError("job result has no bounded stdout evidence")
        probe_records = []
        for line in stdout.splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping) and candidate.get("test") == "finite-cuda-allocation-copy-free":
                probe_records.append(dict(candidate))
        if job_record.get("state") != "succeeded" or job_record.get("cleanup_verified") is not True:
            raise QualificationError("CUDA probe job lacks a succeeded state and verified systemd cleanup")
        if len(probe_records) != 1 or probe_records[0].get("status") != "passed" or probe_records[0].get("gpu_uuid") != gpu_uuid:
            raise QualificationError("finite CUDA probe receipt is missing, duplicated, or not bound to the grant UUID")
        report["probe_job"] = {"job": job_record, "logs": logs, "probe_receipt": probe_records[0]}
        report["checks"]["finite_cuda_allocation_copy_free"] = True
        effects = handoff.get("effects") or {}
        evictions = [
            value.get("receipt") for name, value in effects.items()
            if str(name).startswith(f"evict:{backend_id}") and isinstance(value, Mapping)
        ]
        if len(evictions) != 1 or evictions[0].get("state") != "evicted" or evictions[0].get("cleanup_verified") is not True:
            raise QualificationError("handoff ledger omitted one verified managed backend eviction receipt")
        reclamation = evictions[0].get("resource_reclamation")
        if not isinstance(reclamation, Mapping) or reclamation.get("state") != "reclaimed" or not reclamation.get("evidence"):
            raise QualificationError("handoff ledger omitted physical GPU reclamation evidence")
        report["eviction_receipt"] = evictions[0]
        report["checks"]["verified_backend_eviction_and_gpu_reclamation"] = True

        barrier = await asyncio.to_thread(frame_client.inspect_handoff_barrier, frame_id, handoff_id=handoff_id)
        if barrier.get("state") != "resumed" or barrier.get("safe_boundary") is not True or barrier.get("durable") is not True:
            raise QualificationError("FrameShift did not retain a durable resumed handoff barrier")
        event_rows = await asyncio.to_thread(frame_client.events, frame_id, limit=1000)
        handoff_events = [event for event in event_rows if event.get("payload", {}).get("handoff_id") == handoff_id]
        counts = Counter(event.get("kind") for event in handoff_events)
        if counts["handoff_result_delivered"] != 1 or counts["handoff_continuation_resumed"] != 1:
            raise QualificationError("FrameShift delivery/resumption evidence is missing or duplicated")
        final_result = await asyncio.to_thread(
            frame_client.await_result, frame_id, timeout=min(args.timeout_seconds, 600),
            after_generation=int(first_result.get("generation", 0)), poll_interval=0.25,
        )
        final_frame = await asyncio.to_thread(frame_client.inspect, frame_id)
        if final_result is None or final_frame.get("state") != "completed" or _fs_result_kind(final_result) != "complete":
            raise QualificationError("the real model did not complete a later generation after handoff result delivery")
        report["frame_shift_completion"] = {
            "barrier": barrier, "events": handoff_events, "event_counts": dict(counts),
            "final_frame": final_frame, "final_result": final_result,
        }
        report["checks"]["durable_result_delivered_once"] = True
        report["checks"]["resumed_model_completed"] = True

        restored = await ds_client.call("backends.get", {"backend_id": backend_id})
        restored_handle = _mapping(restored.get("execution_handle"), "restored backend execution handle")
        restored_invocation = (restored_handle.get("metadata") or {}).get("invocation_id")
        if restored.get("state") != "ready" or not restored_invocation or restored_invocation == initial_invocation:
            raise QualificationError("backend restoration did not create a fresh ready invocation")
        llama_status = await asyncio.to_thread(_llama_health, llama_url)
        restored_sample = await asyncio.to_thread(gpu_snapshot, gpu_uuid)
        restored_pids = await asyncio.to_thread(_managed_processes, restored_handle)
        restored_used = _assert_gpu_processes(restored_sample, restored_pids, positive_memory=True)
        report["restored_backend"] = {
            "backend": restored, "execution_handle": restored_handle,
            "process_ids": restored_pids, "used_gpu_memory_bytes": restored_used,
            "gpu_observation": {key: restored_sample[key] for key in ("memory_free_bytes", "process_count", "inventory_observed_at", "process_observed_at")},
            "llama_readiness": llama_status,
        }
        report["checks"]["pinned_backend_restored_ready"] = True

        guard.verify(force=True)
        revoked = await asyncio.to_thread(
            frame_client.revoke_handoff_consumer, frame_id,
            backend_id=backend_id, contract_id=CONSUMER_CONTRACT,
        )
        if revoked.get("revoked") is not True:
            raise QualificationError("FrameShift did not durably revoke the completed qualification consumer")
        attached = False
        report["manual_fixture_cleanup"] = {
            "authorized_scope": "cancel only the exact returned restored systemd unit after continuation detach",
            "production_stop_api_claimed": False,
            "consumer_revocation": revoked,
        }
        guard.verify(force=True)
        stop = await asyncio.to_thread(
            SystemdExecutionAdapter().cancel, ExecutionHandle.from_dict(restored_handle),
            operation_id=f"qualification-manual-stop:{backend_id}", timeout_seconds=30,
        )
        report["manual_fixture_cleanup"]["systemd_cancel_receipt"] = stop
        if stop.get("cleanup_verified") is not True:
            raise QualificationError("manual fixture cleanup did not verify the exact backend unit cleanup")
        after_stop = await asyncio.to_thread(gpu_snapshot, gpu_uuid)
        if set(restored_pids) & {row.get("pid") for row in after_stop["processes"]}:
            raise QualificationError("NVIDIA process telemetry still sees a restored backend PID after fixture cleanup")
        report["manual_fixture_cleanup"]["gpu_after_stop"] = {
            key: after_stop[key] for key in ("gpu_uuid", "memory_free_bytes", "process_count", "inventory_observed_at", "process_observed_at")
        }
        report["manual_fixture_cleanup"]["verified_gpu_processes_gone"] = True
        cleanup_safe = True
        report["checks"]["manual_fixture_unit_and_gpu_cleanup"] = True
        report["checks"]["parent_pool_still_active_after_cleanup"] = bool(guard.verify(force=True))
        report["status"] = "passed"
    except QualificationCancelled as exc:
        report["status"] = "cancelled"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        report["traceback"] = ""
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        report["traceback"] = "".join(traceback.format_exception(exc))[-8000:]
    finally:
        # Keep unknown/blocked external outcomes untouched. Only close the private
        # temporary services and stop a backend when its continuation was revoked
        # and the complete handoff restored it. Otherwise the parent operator keeps
        # the grant and diagnostic ledger for explicit recovery.
        if cleanup_safe and attached is False and handoff_state == "complete" and backend_handle is not None:
            report["manual_fixture_cleanup"]["ledger_backend_state_after_manual_stop"] = "ready (historical; no production stop route invoked)"
        if daemon is not None and daemon.poll() is None:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
        if fs_server is not None:
            try:
                fs_server.shutdown()
                fs_server.server_close()
                if fs_server_thread is not None:
                    fs_server_thread.join(timeout=10)
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(f"FrameShift private service: {type(exc).__name__}")
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        if source_identity_before is not None:
            try:
                source_identity_after = capture_source_identity(
                    frameshift_source=args.frameshift_source,
                    llama_binary=args.llama_binary,
                    model=args.model,
                )
                report["source_identity_after"] = source_identity_after
                unchanged = _source_identity_matches(source_identity_before, source_identity_after)
                report["checks"]["source_identity_unchanged_during_run"] = unchanged
                if not unchanged:
                    report["status"] = "failed"
                    report["error"] = {
                        "type": "QualificationError",
                        "message": "production, harness, FrameShift, llama binary, or model identity changed during the live run",
                    }
            except Exception as exc:
                report["source_identity_after_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "failed"
                report["error"] = {
                    "type": "QualificationError",
                    "message": "could not verify source/runtime identity after the live run",
                }
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["diagnostic_directory"] = str(work)
        receipt = work / "qualification.json"
        report["receipt"] = str(receipt)
        receipt.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    return 0 if report["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grant-file", type=Path, required=True, help="active parent pool grant emitted by qualification_pool.py")
    parser.add_argument("--project-control-python", type=Path, default=DEFAULT_PC_PYTHON)
    parser.add_argument("--project-control-runtime", type=Path, default=DEFAULT_PC_RUNTIME)
    parser.add_argument("--frameshift-source", type=Path, default=DEFAULT_FRAMESHIFT)
    parser.add_argument("--llama-binary", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--execute", action="store_true", help="perform the isolated live qualification")
    parser.add_argument("--confirm-live-gpu-handoff", action="store_true", help="explicitly authorize GPU backend eviction and restoration for this qualification only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 300 <= args.timeout_seconds <= 3600:
        parser.error("--timeout-seconds must be between 300 and 3600")
    guard = ParentGrantGuard(args.grant_file, args.project_control_python, args.project_control_runtime)
    try:
        grant = guard.verify(force=True)
        if not args.frameshift_source.is_dir():
            raise QualificationError("FrameShift source checkout is unavailable")
        if not args.llama_binary.is_file() or not args.model.is_file():
            raise QualificationError("the pinned local llama.cpp binary or GGUF model is unavailable")
    except QualificationError as exc:
        parser.error(str(exc))
    if not args.execute:
        if args.confirm_live_gpu_handoff:
            parser.error("--confirm-live-gpu-handoff requires --execute")
        print(json.dumps({
            "status": "plan-only", "gpu_started": False, "daemon_started": False,
            "model_started": False, "parent_grant": {
                "owner_id": grant["owner_id"], "gpu_uuid": grant["gpu_uuid"],
                "resource_ids": grant["resource_ids"], "state": grant["state"],
            },
            "scope": "one V100 llama.cpp FrameShift tool-request handoff with finite CUDA allocation/copy/free, restore and resumed completion",
            "preconditions": [
                "qualification_pool.py remains alive and retains this exact Project Control owner",
                "no independent allocator or other user may use the granted pool",
                "one 128 MiB CUDA allocation only; no kernel design, download, or performance claim",
                "manual exact-unit cleanup is qualification-only; unresolved outcomes keep the parent grant held",
            ],
        }, indent=2, sort_keys=True))
        return 0
    if not args.confirm_live_gpu_handoff:
        parser.error("live execution requires --confirm-live-gpu-handoff as well as --execute")
    return asyncio.run(execute_qualification(args, guard))


if __name__ == "__main__":
    raise SystemExit(main())
