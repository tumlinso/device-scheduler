"""Authenticated MCP daemon composition and lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import fcntl
import stat
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from .auth import StaticTokenStore, StaticTokenVerifier, principal_from_request
from .config import ServiceConfig, load_backend_configuration, load_named_profiles, load_profiles
from .policies import load_policy
from .registry import Operation, operation_names, operations
from .service import SchedulerDispatch, SchedulerService

_log = logging.getLogger("device_scheduler_mcp.server")


class DaemonOwnership:
    """Hold one private host-local lock for the process lifetime."""

    def __init__(self, state_directory: str | os.PathLike[str]) -> None:
        self.state_directory = os.path.abspath(os.path.expanduser(os.fspath(state_directory)))
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError("daemon ownership is already held")
        os.makedirs(self.state_directory, mode=0o700, exist_ok=True)
        directory = os.lstat(self.state_directory)
        if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
            raise RuntimeError("scheduler state directory must be a real directory")
        if directory.st_uid != os.getuid() or directory.st_mode & 0o077:
            raise RuntimeError("scheduler state directory must be private and owned by the service user")
        path = os.path.join(self.state_directory, "daemon.lock")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("daemon lock must be a private regular file owned by the service user")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another device-scheduler daemon already owns this state directory") from exc
            self._fd = fd
        except Exception:
            os.close(fd)
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class TickWorker:
    """One serial worker keeps durable delayed and background work moving."""

    def __init__(self, service: SchedulerService, interval_seconds: float) -> None:
        self.service = service
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="device-scheduler-tick", daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        thread = self._thread
        if thread is None:
            return True
        self._stop.set()
        thread.join(timeout=max(5.0, self.interval_seconds * 4))
        if thread.is_alive():
            _log.error("scheduler tick did not stop; daemon ownership will remain held")
            return False
        self._thread = None
        return True

    def _run(self) -> None:
        # Production performs a synchronous reconcile before accepting work.
        # Delay this loop's first tick to avoid racing that startup gate.
        next_tick = time.monotonic() + self.interval_seconds
        while not self._stop.is_set():
            delay = max(0.0, next_tick - time.monotonic())
            if self._stop.wait(delay):
                return
            try:
                self.service.tick()
            except Exception as exc:
                # Adapter messages can include workload arguments or environment values.
                _log.warning("background scheduler tick failed (%s)", type(exc).__name__)
            next_tick = time.monotonic() + self.interval_seconds
            # Avoid a tight catch-up loop after an unusually slow reconciliation tick.
            if next_tick < time.monotonic() - self.interval_seconds:
                next_tick = time.monotonic()


def create_mcp_server(
    scheduler: SchedulerDispatch | SchedulerService,
    token_store: StaticTokenStore,
    config: ServiceConfig | None = None,
) -> FastMCP:
    """Create the protocol server; process lifecycle belongs to the ASGI app.

    FastMCP invokes its protocol lifespan for stateless HTTP requests, so it
    must never start workers, hold process locks, or run durable reconciliation.
    """
    selected_config = config or ServiceConfig.default()
    service = scheduler if isinstance(scheduler, SchedulerService) else SchedulerService(scheduler)

    @asynccontextmanager
    async def protocol_lifespan(_server: FastMCP):
        yield {}

    issuer_host = f"[{selected_config.host}]" if ":" in selected_config.host else selected_config.host
    server = FastMCP(
        "device-scheduler",
        instructions=(
            "Device Scheduler is the authenticated authority for resources and supervised work. "
            "All jobs and handoffs are durable after accepted submission. Ordinary suspension does "
            "not unload shared model weights; eviction requires an explicit authorized handoff."
        ),
        host=selected_config.host,
        port=selected_config.port,
        streamable_http_path=selected_config.mcp_path,
        stateless_http=True,
        json_response=True,
        log_level="ERROR",
        auth=AuthSettings(
            issuer_url=f"http://{issuer_host}:{selected_config.port}",
            resource_server_url=None,
            required_scopes=[],
        ),
        token_verifier=StaticTokenVerifier(token_store),
        lifespan=protocol_lifespan,
    )
    register_operation_tools(server, service)

    @server.custom_route("/healthz", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "device-scheduler", "api_version": 1})

    @server.custom_route("/readyz", methods=["GET"])
    async def readiness(_request: Request) -> JSONResponse:
        try:
            state = service.status()
        except Exception as exc:
            _log.warning("scheduler readiness check failed (%s)", type(exc).__name__)
            return JSONResponse({"status": "unavailable"}, status_code=503)
        status = state.get("status") if isinstance(state, dict) else None
        ready = status in {"healthy", "ready", "ok"}
        return JSONResponse({
            "status": "ready" if ready else "degraded",
            "service": "device-scheduler",
            "operations": list(operation_names()),
            "scheduler": state,
        }, status_code=200 if ready else 503)

    return server


def register_operation_tools(server: FastMCP, service: SchedulerService) -> tuple[str, ...]:
    """Register the canonical core operation registry as structured MCP tools."""
    registered: list[str] = []
    for operation in operations():
        annotations = ToolAnnotations(
            readOnlyHint=operation.read_only,
            destructiveHint=not operation.read_only,
            idempotentHint=operation.read_only,
            openWorldHint=False,
        )

        def handler_factory(contract: Operation):
            async def handler(arguments: dict[str, Any], context: Context) -> dict[str, Any]:
                principal = principal_from_request(context.request_context.request)
                return await asyncio.to_thread(service.dispatch, contract.name, arguments, principal)

            handler.__name__ = contract.tool_name
            return handler

        server.add_tool(
            handler_factory(operation),
            name=operation.tool_name,
            description=operation.description,
            annotations=annotations,
            structured_output=True,
        )
        # FastMCP infers ``arguments`` as an untyped object. Replace only the
        # advertised parameters with the shared canonical operation contract;
        # dispatch continues to receive the same nested ``arguments`` object.
        server._tool_manager._tools[operation.tool_name].parameters = {
            "type": "object",
            "properties": {"arguments": operation.input_schema},
            "required": ["arguments"],
            "additionalProperties": False,
        }
        registered.append(operation.tool_name)
    return tuple(registered)


def create_asgi_app(
    scheduler: SchedulerDispatch | SchedulerService,
    token_store: StaticTokenStore,
    config: ServiceConfig | None = None,
    *,
    background_ticks: bool = True,
    startup_reconcile: bool = False,
    ownership: DaemonOwnership | None = None,
) -> Any:
    """Create a fake-injectable ASGI application used by tests and daemon startup."""
    selected_config = config or ServiceConfig.default()
    service = scheduler if isinstance(scheduler, SchedulerService) else SchedulerService(scheduler)
    ticker = TickWorker(service, selected_config.tick_interval_seconds) if background_ticks else None
    server = create_mcp_server(service, token_store, selected_config)
    app = server.streamable_http_app()
    # FastMCP's protocol lifespan runs for each stateless HTTP request; keep
    # process-wide reconciliation and workers in the ASGI lifespan only.
    transport_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(application: Any):
        try:
            async with transport_lifespan(application):
                if startup_reconcile:
                    summary = service.tick()
                    if summary.get("status") != "ok":
                        raise RuntimeError("startup reconciliation failed; daemon will not accept work")
                if ticker is not None:
                    ticker.start()
                yield {}
        finally:
            stopped = True
            if ticker is not None:
                stopped = ticker.stop()
            if ownership is not None and stopped:
                ownership.release()

    app.router.lifespan_context = combined_lifespan
    # Keep explicit references available to health/diagnostic tooling without opening adapters.
    app.state.device_scheduler_server = server
    app.state.device_scheduler_operation_names = operation_names()
    return app


def create_app(
    config: ServiceConfig | None = None,
    *,
    scheduler: SchedulerDispatch | SchedulerService | None = None,
    token_store: StaticTokenStore | None = None,
    background_ticks: bool = True,
) -> Any:
    """Create the production app or an injected test app.

    Default composition creates the SQLite ledger and configured host, NVIDIA,
    systemd, continuation, and managed-backend adapters. Importing or starting
    the daemon does not start a model backend or reserve a GPU.
    """
    selected_config = config or ServiceConfig.default()
    ownership: DaemonOwnership | None = None
    if scheduler is None:
        ownership = DaemonOwnership(selected_config.state_directory)
        ownership.acquire()
        try:
            scheduler = build_default_scheduler(selected_config)
        except Exception:
            ownership.release()
            raise
    if token_store is None:
        try:
            token_store = StaticTokenStore.from_directory(selected_config.token_directory)
        except Exception:
            if ownership is not None:
                ownership.release()
            raise
    try:
        return create_asgi_app(
            scheduler, token_store, selected_config, background_ticks=background_ticks,
            startup_reconcile=ownership is not None, ownership=ownership,
        )
    except BaseException:
        if ownership is not None:
            ownership.release()
        raise


def build_default_scheduler(config: ServiceConfig) -> SchedulerDispatch:
    """Compose the host-local ledger and resource/execution adapters."""
    from device_scheduler_adapters.host import LinuxHostAdapter
    from device_scheduler_adapters.nvidia import NvidiaAdapter
    from device_scheduler_adapters.sqlite import Ledger
    from device_scheduler_adapters.systemd import SystemdExecutionAdapter
    from device_scheduler_core.allocator import ResourceAllocator
    from device_scheduler_core.application import Scheduler
    from device_scheduler_core.application import SchedulerConfig

    ledger = Ledger(config.resolved_ledger_path)
    nvidia_adapter = NvidiaAdapter()
    resource_adapters = (LinuxHostAdapter(), nvidia_adapter)
    allocator = ResourceAllocator(
        ledger,
        gpu_headroom_fraction=config.gpu_headroom_fraction,
        gpu_observation_max_age_seconds=config.gpu_observation_max_age_seconds,
    )
    policy = load_policy(config)
    profiles = load_profiles(config.profiles_path)
    backend_profiles, backend_controls = load_backend_configuration(config.backend_profiles_path)
    continuations = _load_continuations(config.continuations_path)
    execution_adapter = SystemdExecutionAdapter(
        profile_resolver=lambda profile_id: profiles[profile_id],
    )
    from device_scheduler_adapters.backends import (
        BackendProfile,
        FrameShiftBackendAdmissionClient,
        ManagedLlamaBackendAdapter,
    )
    from device_scheduler_adapters.frameshift import FrameShiftContinuationAdapter
    from device_scheduler_core.models import BackendSpec, HandoffSpec

    continuation_adapter = FrameShiftContinuationAdapter(continuations) if continuations else None
    backend_profile_models: dict[str, BackendProfile] = {}
    for profile_id, value in backend_profiles.items():
        try:
            profile = BackendProfile(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid backend profile {profile_id!r}: {type(exc).__name__}") from exc
        if profile.profile_id != profile_id:
            raise ValueError(f"backend profile key {profile_id!r} does not match its profile_id")
        if profile.execution_profile_id not in profiles:
            raise ValueError(f"backend profile {profile_id!r} references an unregistered execution profile")
        backend_profile_models[profile_id] = profile

    from .auth import read_client_token

    frameshift_controls: dict[str, FrameShiftBackendAdmissionClient] = {}
    for control_ref, control in backend_controls.items():
        base_url = control.get("base_url")
        token_file = control.get("token_file")
        timeout = control.get("timeout_seconds", 3.0)
        if not isinstance(base_url, str) or not base_url or not isinstance(token_file, str) or not token_file:
            raise ValueError(f"FrameShift controller {control_ref!r} requires base_url and token_file")
        try:
            frameshift_controls[control_ref] = FrameShiftBackendAdmissionClient(
                base_url, read_client_token(token_file), timeout_seconds=timeout,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid FrameShift controller {control_ref!r}: {type(exc).__name__}") from exc

    def resolve_backend_handle(spec: BackendSpec) -> Any:
        record = ledger.get("backend", spec.backend_id)
        if isinstance(record, dict) and isinstance(record.get("execution_handle"), dict):
            return record["execution_handle"]
        # A process may have launched before its backend row received the
        # receipt. Consult its stable start intent, but never issue a new start.
        operation = ledger.operation(f"backend-start:{spec.backend_id}") if callable(getattr(ledger, "operation", None)) else None
        receipts = operation.get("receipts", ()) if isinstance(operation, dict) else ()
        for item in reversed(receipts if isinstance(receipts, list) else list(receipts)):
            receipt = item.get("receipt") if isinstance(item, dict) else None
            handle = receipt.get("execution_handle") if isinstance(receipt, dict) else None
            if isinstance(handle, dict):
                return handle
        return None

    def authorize_eviction(spec: BackendSpec, operation_id: str) -> dict[str, Any]:
        handoff = ledger.get("handoff", operation_id)
        if not isinstance(handoff, dict) or handoff.get("handoff_id") != operation_id:
            return {"authorized": False, "reason": "persisted handoff not found"}
        try:
            request = HandoffSpec.from_dict(handoff["spec"])
        except (KeyError, TypeError, ValueError):
            return {"authorized": False, "reason": "persisted handoff specification is invalid"}
        grant = next((item for item in request.eviction_grants if item.backend_id == spec.backend_id), None)
        now = datetime.now(timezone.utc)
        if (
            grant is None
            or request.handoff_id != operation_id
            or spec.backend_id not in request.backend_ids
            or grant.scope != f"backend:{spec.backend_id}"
            or grant.expires_at <= now
        ):
            return {"authorized": False, "reason": "handoff has no exact, live backend eviction grant"}
        return {
            "authorized": True,
            "grant_id": grant.grant_id,
            "granted_by": grant.granted_by,
            "scope": grant.scope,
            "expires_at": grant.expires_at.isoformat(),
            "handoff_id": request.handoff_id,
        }

    def capture_resource_reclamation_baseline(
        spec: BackendSpec, capture_operation_id: str
    ) -> dict[str, Any]:
        """Bind a pre-stop NVIDIA sample to the exact backend systemd invocation."""
        def unknown(reason: str) -> dict[str, Any]:
            return {
                "state": "unknown",
                "backend_id": spec.backend_id,
                "operation_id": capture_operation_id,
                "reason": reason,
            }

        backend = ledger.get("backend", spec.backend_id)
        if not isinstance(backend, dict):
            return unknown("managed backend record is unavailable")
        lease_id = backend.get("transferred_lease_id") or backend.get("lease_id")
        lease = ledger.get("lease", str(lease_id)) if lease_id else None
        if not isinstance(lease, dict):
            return unknown("backend residency lease record is unavailable")
        handle = resolve_backend_handle(spec)
        if not isinstance(handle, (dict,)) and not hasattr(handle, "to_dict"):
            return unknown("backend systemd execution handle is unavailable")
        try:
            handle_data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
            metadata = handle_data.get("metadata") or {}
            unit_name = handle_data.get("external_id") or handle_data.get("unit_name")
            invocation_id = metadata.get("invocation_id") if isinstance(metadata, Mapping) else None
            invocation_id = handle_data.get("invocation_id") or invocation_id
            control_group = handle_data.get("control_group")
            control_group = control_group or (metadata.get("control_group") if isinstance(metadata, Mapping) else None)
            execution_identity = {
                "unit_name": unit_name,
                "invocation_id": invocation_id,
                "control_group": control_group,
            }
            if any(not isinstance(value, str) or not value.strip() for value in execution_identity.values()):
                return unknown("backend execution identity is incomplete")
            ownership = execution_adapter.managed_process_ids([handle])
        except Exception as exc:
            return unknown(f"managed process identity lookup failed: {type(exc).__name__}")
        if not isinstance(ownership, Mapping) or ownership.get("state") != "observed":
            return unknown("managed process identity mapping is not fully observed")
        attempts = ownership.get("attempts")
        if not isinstance(attempts, Sequence) or len(attempts) != 1 or not isinstance(attempts[0], Mapping):
            return unknown("backend execution identity did not resolve to one systemd invocation")
        attempt = attempts[0]
        if (
            attempt.get("state") != "observed"
            or attempt.get("unit_name") != execution_identity["unit_name"]
            or attempt.get("invocation_id") != execution_identity["invocation_id"]
            or attempt.get("control_group") != execution_identity["control_group"]
        ):
            return unknown("managed process mapping does not match the persisted backend invocation")
        backend_pids = ownership.get("process_ids")
        if (
            isinstance(backend_pids, (str, bytes))
            or not isinstance(backend_pids, Sequence)
            or not backend_pids
            or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in backend_pids)
        ):
            return unknown("backend invocation has no positively observed process IDs")
        backend_pid_set = set(backend_pids)

        try:
            snapshot = nvidia_adapter.snapshot()
        except Exception as exc:
            return unknown(f"NVIDIA telemetry failed: {type(exc).__name__}")
        process_telemetry = snapshot.get("process_telemetry") if isinstance(snapshot, Mapping) else None
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("gpu_inventory", {}).get("state") != "observed"
            or not isinstance(process_telemetry, Mapping)
            or process_telemetry.get("state") != "observed"
        ):
            return unknown("NVIDIA inventory and compute-process telemetry must both be observed")
        inventory_at = snapshot.get("inventory_observed_at")
        processes_at = process_telemetry.get("observed_at")
        resources = snapshot.get("resources")
        if isinstance(resources, (str, bytes)) or not isinstance(resources, Sequence):
            return unknown("NVIDIA resource inventory has an invalid shape")
        resource_map: dict[str, Mapping[str, Any]] = {}
        for resource in resources:
            data = resource.to_dict() if hasattr(resource, "to_dict") else resource
            if isinstance(data, Mapping) and isinstance(data.get("resource_id"), str):
                resource_map[data["resource_id"]] = data

        gpu_allocations: list[tuple[str, str]] = []
        for allocation in lease.get("allocations", ()):
            if not isinstance(allocation, Mapping) or not isinstance(allocation.get("resource_id"), str):
                continue
            resource_id = allocation["resource_id"]
            stored_resource = ledger.get("resource", resource_id)
            if not isinstance(stored_resource, Mapping) or str(stored_resource.get("kind", "")).casefold() not in {
                "gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device",
            }:
                continue
            attrs = stored_resource.get("attributes") or {}
            gpu_uuid = attrs.get("uuid") if isinstance(attrs, Mapping) else None
            if not isinstance(gpu_uuid, str) or not gpu_uuid:
                return unknown("backend residency lease contains a GPU without a stable UUID")
            gpu_allocations.append((resource_id, gpu_uuid))
        if not gpu_allocations:
            return unknown("backend residency lease does not identify a GPU")

        gpu_evidence: list[dict[str, Any]] = []
        for resource_id, gpu_uuid in gpu_allocations:
            data = resource_map.get(resource_id)
            if data is None or str(data.get("kind", "")).casefold() not in {
                "gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device",
            }:
                return unknown(f"fresh NVIDIA inventory omitted backend GPU {resource_id}")
            attrs = data.get("attributes") or {}
            if not isinstance(attrs, Mapping) or attrs.get("uuid") != gpu_uuid:
                return unknown(f"fresh NVIDIA inventory identity changed for backend GPU {resource_id}")
            free_bytes = attrs.get("memory_free_bytes_observation")
            rows = attrs.get("processes_observation")
            if (
                isinstance(free_bytes, bool)
                or not isinstance(free_bytes, int)
                or free_bytes < 0
                or isinstance(rows, (str, bytes))
                or not isinstance(rows, Sequence)
            ):
                return unknown(f"GPU memory or process observation is incomplete for {gpu_uuid}")
            selected: list[Mapping[str, Any]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    return unknown(f"GPU process observation is malformed for {gpu_uuid}")
                if row.get("identity_unknown") is True:
                    return unknown(f"an NVIDIA-visible process has unknown device identity on {gpu_uuid}")
                if row.get("gpu_uuid") == gpu_uuid:
                    selected.append(row)
            gpu_pids: set[int] = set()
            used_bytes = 0
            for row in selected:
                pid, used = row.get("pid"), row.get("used_memory_bytes")
                if (
                    isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                    or isinstance(used, bool) or not isinstance(used, int) or used <= 0
                ):
                    return unknown(f"GPU process memory identity is incomplete for {gpu_uuid}")
                if pid in gpu_pids:
                    return unknown(f"GPU process identity is duplicated for {gpu_uuid}")
                gpu_pids.add(pid)
                used_bytes += used
            attributed = gpu_pids & backend_pid_set
            if not attributed or attributed != gpu_pids or used_bytes <= 0:
                return unknown(f"visible GPU processes are not exactly attributable to backend {spec.backend_id}")
            gpu_evidence.append({
                "resource_id": resource_id,
                "gpu_uuid": gpu_uuid,
                "memory_free_bytes": free_bytes,
                "used_gpu_memory_bytes": used_bytes,
                "inventory_observed_at": attrs.get("inventory_observed_at"),
                "process_observed_at": attrs.get("process_observed_at"),
                "process_ids": sorted(gpu_pids),
                "backend_process_ids": sorted(attributed),
            })
        return {
            "state": "observed",
            "backend_id": spec.backend_id,
            "operation_id": capture_operation_id,
            "profile_id": spec.profile_id,
            "profile_version": spec.pinned_configuration.get("profile_version"),
            "observed_at": processes_at,
            "source": "nvidia-smi+invocation-verified-systemd-unit-cgroups",
            "execution_identity": execution_identity,
            "process_mapping_state": "observed",
            "gpus": gpu_evidence,
            "evidence": {
                "nvidia_inventory_observed_at": inventory_at,
                "nvidia_processes_observed_at": processes_at,
                "systemd_scope": ownership.get("scope"),
                "systemd_process_observed_at": ownership.get("observed_at"),
            },
            "limitations": [
                "NVIDIA process visibility and memory values are observational and are not enforcement guarantees",
                "same-user processes outside registered systemd units are outside the isolation boundary",
            ],
        }

    def verify_resource_reclamation(
        spec: BackendSpec, operation_id: str, validated_baseline: Mapping[str, Any]
    ) -> dict[str, Any]:
        from math import ceil

        handoff = ledger.get("handoff", operation_id)
        if not isinstance(handoff, dict) or handoff.get("handoff_id") != operation_id:
            return {"state": "unknown", "reason": "persisted handoff is unavailable"}
        persisted_baseline = handoff.get("reclamation_baselines", {}).get(spec.backend_id)
        if not isinstance(persisted_baseline, Mapping) or dict(persisted_baseline) != dict(validated_baseline):
            return {"state": "unknown", "reason": "validated baseline does not match the durable handoff receipt"}
        snapshot = handoff.get("backend_snapshots", {}).get(spec.backend_id) if isinstance(handoff, dict) else None
        lease_id = snapshot.get("lease_id") if isinstance(snapshot, dict) else None
        source_lease = ledger.get("lease", str(lease_id)) if lease_id else None
        if not isinstance(source_lease, dict):
            return {"state": "unknown", "reason": "backend residency lease snapshot is unavailable"}
        try:
            handoff_spec = HandoffSpec.from_dict(handoff["spec"])
        except (KeyError, TypeError, ValueError):
            return {"state": "unknown", "reason": "persisted handoff specification is invalid"}
        gpu_requests = [
            item for item in handoff_spec.resource_envelope
            if item.kind.casefold() in {"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"}
        ]
        requested_bytes = [float(item.quantities.get("memory_bytes", 0.0)) for item in gpu_requests]
        if not requested_bytes or max(requested_bytes) <= 0:
            return {"state": "unknown", "reason": "handoff has no explicit GPU memory envelope to verify"}
        targets: list[tuple[str, str]] = []
        for allocation in source_lease.get("allocations", ()):
            if not isinstance(allocation, dict):
                continue
            resource_id = allocation.get("resource_id")
            resource = ledger.get("resource", str(resource_id)) if resource_id else None
            if not isinstance(resource, dict) or str(resource.get("kind", "")).casefold() not in {"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"}:
                continue
            uuid = (resource.get("attributes") or {}).get("uuid")
            if isinstance(uuid, str) and uuid:
                targets.append((str(resource_id), uuid))
        if not targets:
            return {"state": "unknown", "reason": "backend residency lease does not identify an observed GPU UUID"}
        threshold = ceil(max(requested_bytes))
        observations = []
        baseline_gpus = {str(item["resource_id"]): item for item in validated_baseline.get("gpus", ()) if isinstance(item, Mapping)}
        for resource_id, gpu_uuid in targets:
            baseline_gpu = baseline_gpus.get(resource_id)
            if not isinstance(baseline_gpu, Mapping) or baseline_gpu.get("gpu_uuid") != gpu_uuid:
                return {"state": "unknown", "reason": f"durable baseline is missing backend GPU {resource_id}"}
            baseline_free = baseline_gpu.get("memory_free_bytes")
            backend_used = baseline_gpu.get("used_gpu_memory_bytes")
            baseline_pids = set(baseline_gpu.get("backend_process_ids", ()))
            if (
                isinstance(baseline_free, bool) or not isinstance(baseline_free, int)
                or isinstance(backend_used, bool) or not isinstance(backend_used, int)
                or not baseline_pids
            ):
                return {"state": "unknown", "reason": f"durable baseline for {gpu_uuid} is invalid"}
            minimum_free = max(threshold, baseline_free + backend_used)
            try:
                observation = nvidia_adapter.wait_memory_reclaimed(
                    gpu_uuid, minimum_free, consecutive_samples=3, interval_seconds=0.25, timeout_seconds=10.0,
                )
            except Exception as exc:
                observation = {"state": "unknown", "reason": type(exc).__name__}
            for sample in observation.get("samples", ()) if isinstance(observation, Mapping) else ():
                process_doc = sample.get("processes") if isinstance(sample, Mapping) else None
                process_rows = process_doc.get("processes") if isinstance(process_doc, Mapping) else None
                if isinstance(process_rows, Sequence) and not isinstance(process_rows, (str, bytes)):
                    observed_pids = {
                        row.get("pid") for row in process_rows if isinstance(row, Mapping)
                    }
                    if baseline_pids & observed_pids:
                        observation = {
                            "state": "unknown",
                            "reason": "a pre-eviction backend process remains visible after stop",
                            "samples": observation.get("samples", []),
                        }
                        break
            observations.append({
                "resource_id": resource_id,
                "gpu_uuid": gpu_uuid,
                "baseline_operation_id": validated_baseline.get("operation_id"),
                "baseline_memory_free_bytes": baseline_free,
                "baseline_backend_memory_bytes": backend_used,
                "minimum_free_bytes": minimum_free,
                "baseline_backend_process_ids": sorted(baseline_pids),
                **observation,
            })
        if any(item.get("state") != "reclaimed" for item in observations):
            return {
                "state": "unknown",
                "reason": "one or more backend GPUs did not meet the observed memory floor",
                "evidence": observations,
                "memory_enforcement": "accounting-only-observation",
            }
        return {
            "state": "reclaimed",
            "operation_id": operation_id,
            "evidence": observations,
            "minimum_free_bytes": threshold,
            "memory_enforcement": "accounting-only-observation",
            "reclamation_semantics": "post-stop free-memory floor compared with durable pre-stop telemetry; observational, not enforced",
            "scope": "NVIDIA post-stop telemetry for GPUs in the transferred backend residency lease",
        }

    backend_adapter = None
    if backend_profile_models:
        backend_adapter = ManagedLlamaBackendAdapter(
            execution_adapter,
            profile_resolver=lambda spec: backend_profile_models[spec.profile_id],
            handle_resolver=resolve_backend_handle,
            frameshift_controls=frameshift_controls,
            eviction_authorizer=authorize_eviction,
            resource_reclamation_baseline_capturer=capture_resource_reclamation_baseline,
            resource_reclamation_verifier=verify_resource_reclamation,
            execution_root=str(config.state_directory / "backends"),
        )
    return Scheduler(
        ledger=ledger,
        allocator=allocator,
        executor=execution_adapter,
        resource_adapters=resource_adapters,
        backend_adapter=backend_adapter,
        continuation_adapter=continuation_adapter,
        config=SchedulerConfig(
            state_root=config.state_directory,
            artifact_root=config.resolved_artifact_directory,
            profiles=profiles,
            backend_profiles=backend_profiles,
            policy=policy,
            fairness_window_seconds=config.fairness_window_seconds,
        ),
    )


def _load_continuations(path: str | os.PathLike[str] | None) -> dict[str, dict[str, str]]:
    """Resolve operator-configured FrameShift token-file references in memory."""
    records = load_named_profiles(path, "continuations")
    result: dict[str, dict[str, str]] = {}
    from .auth import read_client_token

    for ref, value in records.items():
        base_url, frame_id, token_file = value.get("base_url"), value.get("frame_id"), value.get("token_file")
        if not all(isinstance(item, str) and item for item in (base_url, frame_id, token_file)):
            raise ValueError("each continuation profile requires base_url, frame_id and token_file")
        result[ref] = {
            "base_url": base_url,
            "frame_id": frame_id,
            "token": read_client_token(token_file),
        }
    return result
