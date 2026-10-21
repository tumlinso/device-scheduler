from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

import httpx
import pytest
from device_scheduler_core.errors import PermissionDeniedError
from device_scheduler_core.models import HandoffSpec, Principal
from device_scheduler_mcp.auth import StaticTokenStore, TokenGrant
from device_scheduler_mcp.client import DeviceSchedulerClient, DeviceSchedulerClientError
from device_scheduler_mcp.config import ServiceConfig
from device_scheduler_mcp.registry import operation_names, operations
from device_scheduler_mcp.server import DaemonOwnership, TickWorker, build_default_scheduler, create_asgi_app
from device_scheduler_mcp.service import SchedulerService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], Principal]] = []

    def dispatch(self, operation: str, args: dict[str, object], principal: Principal) -> dict[str, object]:
        if "*" not in principal.actions and operation not in principal.actions:
            raise PermissionDeniedError("operation is outside the principal grant")
        self.calls.append((operation, args, principal))
        return {"operation": operation, "arguments": args, "principal_id": principal.principal_id}

    def tick(self) -> dict[str, object]:
        return {"status": "idle"}

    def status(self) -> dict[str, object]:
        return {"status": "ready", "ledger": "fixture"}


def _token_store(token: str, principal: Principal) -> StaticTokenStore:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return StaticTokenStore([TokenGrant(digest, principal)])


def _private_token(path: Path) -> str:
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return token


@pytest.mark.anyio
async def test_authenticated_streamable_http_uses_core_dispatch_and_tool_registry(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    token = _private_token(token_path)
    principal = Principal("test-agent", actions=frozenset({"resources.list"}))
    scheduler = FakeScheduler()
    config = ServiceConfig(token_file=token_path, token_directory=tmp_path)
    app = create_asgi_app(scheduler, _token_store(token, principal), config, background_ticks=False)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        factory = lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs)
        client = DeviceSchedulerClient("http://127.0.0.1:8770", token_path, http_client_factory=factory)
        result = await client.list_resources()
        assert result == {"operation": "resources.list", "arguments": {}, "principal_id": "test-agent"}

        tools = await app.state.device_scheduler_server.list_tools()
        assert {tool.name for tool in tools} == {item.tool_name for item in operations()}
        assert set(operation_names()) == {item.name for item in operations()}
        assert all(tool.annotations is not None for tool in tools)
        annotations = {tool.name: tool.annotations for tool in tools}
        assert annotations["resources_list"].readOnlyHint is True
        assert annotations["jobs_submit"].destructiveHint is True

        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as anonymous:
            health = await anonymous.get("/healthz")
            readiness = await anonymous.get("/readyz")
            unauthenticated_mcp = await anonymous.get("/mcp")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        assert readiness.status_code == 200 and readiness.json()["scheduler"]["status"] == "ready"
        assert unauthenticated_mcp.status_code == 401

    assert [(name, caller.principal_id) for name, _args, caller in scheduler.calls] == [
        ("resources.list", "test-agent"),
    ]


@pytest.mark.anyio
async def test_sdk_serializes_typed_specs_and_surfaces_permission_errors(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    token = _private_token(token_path)
    principal = Principal("submitter", actions=frozenset({"jobs.submit"}))
    scheduler = FakeScheduler()
    config = ServiceConfig(token_file=token_path, token_directory=tmp_path)
    app = create_asgi_app(scheduler, _token_store(token, principal), config, background_ticks=False)
    transport = httpx.ASGITransport(app=app)
    client = DeviceSchedulerClient(
        "http://127.0.0.1:8770/mcp",
        token_path,
        http_client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    spec = {
        "owner_id": "submitter",
        "argv": ["pytest", "-q"],
        "working_directory": "/tmp/project",
        "idempotency_key": "request-1",
    }

    async with app.router.lifespan_context(app):
        submitted = await client.submit_job(spec)
        assert submitted["operation"] == "jobs.submit"
        assert scheduler.calls[0][1] == {"spec": spec}

        with pytest.raises(DeviceSchedulerClientError) as error:
            await client.list_resources()
        assert error.value.code == "permission_denied"
        assert "test-agent" not in json.dumps(error.value.details)


@pytest.mark.anyio
async def test_handoff_submission_transport_preserves_one_typed_spec_envelope(tmp_path: Path) -> None:
    class HandoffCaptureScheduler(FakeScheduler):
        submitted_spec: HandoffSpec | None = None

        def dispatch(self, operation, args, principal):
            result = super().dispatch(operation, args, principal)
            if operation == "handoffs.submit":
                self.submitted_spec = HandoffSpec.from_dict(args["spec"])
            return result

    token_path = tmp_path / "client.token"
    token = _private_token(token_path)
    principal = Principal("submitter", actions=frozenset({"handoffs.submit"}))
    scheduler = HandoffCaptureScheduler()
    config = ServiceConfig(token_file=token_path, token_directory=tmp_path)
    app = create_asgi_app(scheduler, _token_store(token, principal), config, background_ticks=False)
    transport = httpx.ASGITransport(app=app)
    client = DeviceSchedulerClient(
        "http://127.0.0.1:8770",
        token_path,
        http_client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    spec = HandoffSpec.from_dict({
        "handoff_id": "transport-handoff",
        "owner_id": "submitter",
        "jobs": [{
            "owner_id": "submitter",
            "argv": ["python", "-c", "pass"],
            "working_directory": "/tmp",
        }],
        "backend_ids": ["backend-a"],
        "continuation_refs": [],
        "eviction_grants": [{
            "grant_id": "grant-a",
            "backend_id": "backend-a",
            "granted_by": "submitter",
            "scope": "backend:backend-a",
            "expires_at": "2030-01-01T00:00:00+00:00",
        }],
        "resource_envelope": [{
            "kind": "gpu",
            "quantities": {"memory_bytes": 1},
        }],
        "drain_timeout_seconds": 10,
        "run_timeout_seconds": 10,
        "restore_timeout_seconds": 10,
    })

    async with app.router.lifespan_context(app):
        submitted = await client.call("handoffs.submit", {
            "spec": spec.to_dict(),
            "idempotency_key": "transport-handoff-request",
        })

    assert submitted["operation"] == "handoffs.submit"
    assert scheduler.submitted_spec == spec
    assert scheduler.calls[0][1] == {
        "spec": spec.to_dict(),
        "idempotency_key": "transport-handoff-request",
    }


@pytest.mark.anyio
async def test_sdk_maps_invalid_bearer_token_to_unauthorized(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    _private_token(token_path)
    accepted_token = secrets.token_urlsafe(32)
    scheduler = FakeScheduler()
    app = create_asgi_app(
        scheduler,
        _token_store(accepted_token, Principal("valid", actions=frozenset({"resources.list"}))),
        ServiceConfig(token_file=token_path),
        background_ticks=False,
    )
    client = DeviceSchedulerClient(
        "http://127.0.0.1:8770",
        token_path,
        http_client_factory=lambda **kwargs: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), **kwargs
        ),
    )
    async with app.router.lifespan_context(app):
        with pytest.raises(DeviceSchedulerClientError) as error:
            await client.list_resources()
    assert error.value.code == "unauthorized"


@pytest.mark.anyio
async def test_health_and_readiness_do_not_discover_or_start_resources(tmp_path: Path) -> None:
    token = secrets.token_urlsafe(32)
    scheduler = FakeScheduler()
    app = create_asgi_app(
        scheduler,
        _token_store(token, Principal("health-only", actions=frozenset())),
        ServiceConfig(token_directory=tmp_path),
        background_ticks=False,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            assert (await client.get("/healthz")).status_code == 200
            assert (await client.get("/readyz")).status_code == 200
    assert scheduler.calls == []


@pytest.mark.anyio
async def test_readiness_fails_closed_for_unknown_scheduler_health(tmp_path: Path) -> None:
    class UnknownHealthScheduler(FakeScheduler):
        def status(self) -> dict[str, object]:
            return {"status": "unknown"}

    app = create_asgi_app(
        UnknownHealthScheduler(),
        _token_store("unused-token", Principal("health-only", actions=frozenset())),
        ServiceConfig(token_directory=tmp_path),
        background_ticks=False,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            readiness = await client.get("/readyz")
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "degraded"


def test_background_tick_progresses_while_client_dispatch_is_blocked() -> None:
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    tick_entered = threading.Event()
    overlap_lock = threading.Lock()
    active = 0
    max_active = 0

    class BlockingScheduler:
        def _enter(self) -> None:
            nonlocal active, max_active
            with overlap_lock:
                active += 1
                max_active = max(max_active, active)

        def _exit(self) -> None:
            nonlocal active
            with overlap_lock:
                active -= 1

        def dispatch(self, operation: str, args: dict[str, object], principal: Principal) -> dict[str, object]:
            self._enter()
            dispatch_started.set()
            release_dispatch.wait(timeout=2)
            self._exit()
            return {"status": "accepted"}

        def tick(self) -> dict[str, object]:
            self._enter()
            tick_entered.set()
            self._exit()
            return {"status": "processed"}

        def status(self) -> dict[str, object]:
            return {"status": "ready"}

    service = SchedulerService(BlockingScheduler())
    principal = Principal("test-agent", actions=frozenset({"resources.list"}))
    dispatch_thread = threading.Thread(
        target=lambda: service.dispatch("resources.list", {}, principal), daemon=True,
    )
    dispatch_thread.start()
    assert dispatch_started.wait(timeout=1)
    ticker = TickWorker(service, 0.05)
    ticker.start()
    try:
        time.sleep(0.02)
        assert tick_entered.wait(timeout=1), "slow client dispatch starved the scheduler tick"
        release_dispatch.set()
    finally:
        release_dispatch.set()
        ticker.stop()
        dispatch_thread.join(timeout=1)
    assert not dispatch_thread.is_alive()
    assert max_active == 2


def test_slow_inventory_tick_does_not_delay_client_submission_dispatch() -> None:
    tick_started = threading.Event()
    release_tick = threading.Event()
    submitted = threading.Event()

    class SlowInventoryScheduler:
        def dispatch(self, operation: str, args: dict[str, object], principal: Principal) -> dict[str, object]:
            assert operation == "jobs.submit"
            submitted.set()
            return {"state": "queued"}

        def tick(self) -> dict[str, object]:
            tick_started.set()
            release_tick.wait(timeout=2)
            return {"status": "ok"}

        def status(self) -> dict[str, object]:
            return {"status": "ready"}

    service = SchedulerService(SlowInventoryScheduler())
    principal = Principal("submitter", actions=frozenset({"jobs.submit"}))
    tick_thread = threading.Thread(target=service.tick, daemon=True)
    tick_thread.start()
    assert tick_started.wait(timeout=1)
    result: list[dict[str, object]] = []
    dispatch_thread = threading.Thread(
        target=lambda: result.append(service.dispatch("jobs.submit", {"spec": {}}, principal)),
        daemon=True,
    )
    dispatch_thread.start()
    try:
        assert submitted.wait(timeout=0.5), "slow inventory tick held up durable submission dispatch"
        assert result == [{"state": "queued"}]
        assert not release_tick.is_set()
    finally:
        release_tick.set()
        tick_thread.join(timeout=1)
        dispatch_thread.join(timeout=1)
    assert not tick_thread.is_alive()
    assert not dispatch_thread.is_alive()


def test_wait_dispatch_does_not_block_tick_that_completes_the_job() -> None:
    completed = threading.Event()

    class WaitingScheduler:
        def dispatch(self, operation: str, args: dict[str, object], principal: Principal) -> dict[str, object]:
            assert operation == "jobs.wait"
            assert completed.wait(timeout=1.0)
            return {"status": "succeeded"}

        def tick(self) -> dict[str, object]:
            completed.set()
            return {"status": "ok"}

        def status(self) -> dict[str, object]:
            return {"status": "healthy"}

    service = SchedulerService(WaitingScheduler())
    principal = Principal("waiter", actions=frozenset({"jobs.wait"}))
    result: list[dict[str, object]] = []
    thread = threading.Thread(
        target=lambda: result.append(service.dispatch("jobs.wait", {"job_id": "job-1"}, principal)),
        daemon=True,
    )
    thread.start()
    ticker = TickWorker(service, 0.02)
    ticker.start()
    try:
        thread.join(timeout=0.5)
        assert not thread.is_alive(), "job wait blocked the tick required to complete it"
        assert result == [{"status": "succeeded"}]
    finally:
        ticker.stop()
        thread.join(timeout=1)


def test_daemon_ownership_excludes_a_second_process_owner(tmp_path: Path) -> None:
    first = DaemonOwnership(tmp_path / "state")
    first.acquire()
    second = DaemonOwnership(tmp_path / "state")
    try:
        with pytest.raises(RuntimeError, match="another device-scheduler daemon"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


@pytest.mark.anyio
async def test_production_lifespan_reconciles_before_ready_and_releases_lock(tmp_path: Path) -> None:
    class StartupScheduler(FakeScheduler):
        def __init__(self, tick_status: str = "ok") -> None:
            super().__init__()
            self.tick_status = tick_status
            self.ticks = 0

        def tick(self) -> dict[str, object]:
            self.ticks += 1
            return {"status": self.tick_status}

    owner = DaemonOwnership(tmp_path / "state")
    owner.acquire()
    scheduler = StartupScheduler()
    app = create_asgi_app(
        scheduler,
        _token_store("unused-token", Principal("test", actions=frozenset({"*"}))),
        ServiceConfig(state_directory=tmp_path / "state"),
        background_ticks=False,
        startup_reconcile=True,
        ownership=owner,
    )
    async with app.router.lifespan_context(app):
        assert scheduler.ticks == 1
        assert owner._fd is not None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            ready = await client.get("/readyz")
        assert ready.status_code == 200
    assert owner._fd is None


@pytest.mark.anyio
async def test_stateless_mcp_requests_do_not_repeat_daemon_startup_or_tick(tmp_path: Path) -> None:
    class CountingScheduler(FakeScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.ticks = 0

        def tick(self) -> dict[str, object]:
            self.ticks += 1
            return {"status": "ok"}

    token = secrets.token_urlsafe(32)
    scheduler = CountingScheduler()
    principal = Principal("test", actions=frozenset({"resources.list"}))
    app = create_asgi_app(
        scheduler,
        _token_store(token, principal),
        ServiceConfig(tick_interval_seconds=60),
        background_ticks=True,
        startup_reconcile=True,
    )
    async with app.router.lifespan_context(app):
        assert scheduler.ticks == 1
        token_path = tmp_path / "client.token"
        token_path.write_text(token + "\n", encoding="utf-8")
        token_path.chmod(0o600)
        client = DeviceSchedulerClient(
            "http://127.0.0.1:8770",
            token_path,
            http_client_factory=lambda **kwargs: httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), **kwargs
            ),
        )
        await client.list_resources()
        assert scheduler.ticks == 1


@pytest.mark.anyio
async def test_failed_startup_reconciliation_never_opens_http_service(tmp_path: Path) -> None:
    class StartupScheduler(FakeScheduler):
        def tick(self) -> dict[str, object]:
            return {"status": "degraded", "errors": [{"component": "job_reconciliation"}]}

    owner = DaemonOwnership(tmp_path / "state")
    owner.acquire()
    app = create_asgi_app(
        StartupScheduler(),
        _token_store("unused-token", Principal("test", actions=frozenset({"*"}))),
        ServiceConfig(state_directory=tmp_path / "state"),
        background_ticks=False,
        startup_reconcile=True,
        ownership=owner,
    )
    with pytest.raises(BaseExceptionGroup) as startup_error:
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail before accepting HTTP")
    assert "startup reconciliation failed" in repr(startup_error.value)
    assert owner._fd is None


def test_default_composition_uses_configured_ledger_and_has_no_startup_execution(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.toml"
    profiles.write_text(
        '[profiles."cuda-tests"]\n'
        'argv = ["python", "-m", "pytest"]\n'
        'working_directory = "/tmp/project"\n',
        encoding="utf-8",
    )
    profiles.chmod(0o600)
    config = ServiceConfig(state_directory=tmp_path / "state", profiles_path=profiles)

    scheduler = build_default_scheduler(config)

    assert scheduler.config.state_root == config.state_directory.resolve()
    assert scheduler.config.resolved_artifact_root == config.resolved_artifact_directory.resolve()
    assert scheduler.ledger.path == config.resolved_ledger_path
    assert scheduler.status()["adapters"]["adapters"]
    assert scheduler.executor._execution_spec({"profile_id": "cuda-tests"}) == {
        "argv": ["python", "-m", "pytest"],
        "working_directory": "/tmp/project",
    }


def test_create_app_releases_daemon_lock_if_asgi_composition_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from device_scheduler_mcp import server as server_module

    config = ServiceConfig(state_directory=tmp_path / "state")
    monkeypatch.setattr(server_module, "build_default_scheduler", lambda _config: FakeScheduler())

    def fail_asgi(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("app construction failed")

    monkeypatch.setattr(server_module, "create_asgi_app", fail_asgi)
    with pytest.raises(RuntimeError, match="app construction failed"):
        server_module.create_app(
            config,
            token_store=_token_store("unused-token", Principal("operator", actions=frozenset({"*"}))),
        )

    owner = DaemonOwnership(config.state_directory)
    owner.acquire()
    owner.release()


def test_default_composition_wires_only_operator_configured_frameshift_backend(tmp_path: Path) -> None:
    execution = tmp_path / "profiles.toml"
    execution.write_text(
        '[profiles."llama-process"]\n'
        'argv = ["llama-server", "--slots"]\n'
        f'working_directory = "{tmp_path}"\n',
        encoding="utf-8",
    )
    execution.chmod(0o600)
    scheduler_token = tmp_path / "frameshift.token"
    _private_token(scheduler_token)
    backend_profiles = tmp_path / "backends.toml"
    backend_profiles.write_text(
        "[backend_profiles.llama-main]\n"
        "profile_id = 'llama-main'\n"
        "version = 'llama-v1'\n"
        "execution_profile_id = 'llama-process'\n"
        "health_url = 'http://127.0.0.1:8080/health'\n"
        "slots_url = 'http://127.0.0.1:8080/slots'\n"
        "access_mode = 'frameshift-only'\n"
        "frameshift_control_ref = 'frameshift-local'\n"
        "[frameshift_controls.frameshift-local]\n"
        "base_url = 'http://127.0.0.1:9010'\n"
        f"token_file = '{scheduler_token}'\n",
        encoding="utf-8",
    )
    backend_profiles.chmod(0o600)
    continuation_token = tmp_path / "frameshift-continuation.token"
    _private_token(continuation_token)
    continuations = tmp_path / "continuations.toml"
    continuations.write_text(
        "[continuations.agent-frame]\n"
        "base_url = 'http://127.0.0.1:9010'\n"
        "frame_id = 'agent-frame'\n"
        f"token_file = '{continuation_token}'\n",
        encoding="utf-8",
    )
    continuations.chmod(0o600)
    scheduler = build_default_scheduler(ServiceConfig(
        state_directory=tmp_path / "state",
        profiles_path=execution,
        backend_profiles_path=backend_profiles,
        continuations_path=continuations,
        gpu_headroom_fraction=0.2,
        gpu_observation_max_age_seconds=12,
        fairness_window_seconds=600,
        policy_parameters={"age_quantum_seconds": 90},
    ))

    from device_scheduler_adapters.backends import ManagedLlamaBackendAdapter
    from device_scheduler_adapters.frameshift import FrameShiftContinuationAdapter

    assert isinstance(scheduler.backend_adapter, ManagedLlamaBackendAdapter)
    assert isinstance(scheduler.continuation_adapter, FrameShiftContinuationAdapter)
    assert scheduler.config.backend_profiles["llama-main"]["version"] == "llama-v1"
    assert scheduler.allocator.gpu_headroom_fraction == 0.2
    assert scheduler.allocator.gpu_observation_max_age_seconds == 12
    assert scheduler.config.fairness_window_seconds == 600
    assert scheduler.policy.age_quantum_seconds == 90
    assert scheduler.policy.configuration_snapshot["fairness_window_seconds"] == 600

    # Exercise the production callback wiring with controlled evidence. No
    # systemd unit, llama server, HTTP endpoint or GPU is contacted by this test.
    from datetime import datetime, timedelta, timezone
    from device_scheduler_core.models import (
        BackendSpec,
        EvictionGrant,
        ExecutionHandle,
        HandoffSpec,
        JobSpec,
        ResourceRequirement,
    )

    backend_id = "llama-main-instance"
    resource_id = "gpu:test-host:GPU-test"
    lease_id = "backend-lease"
    observed_at = datetime.now(timezone.utc).isoformat()
    backend_spec = BackendSpec(
        backend_id=backend_id,
        host_id="test-host",
        adapter_id="managed-llama.cpp",
        profile_id="llama-main",
        configuration_ref="llama-main",
        residency_requirements=(ResourceRequirement("gpu", {"memory_bytes": 4 * 1024**2}),),
        pinned_configuration={"profile_version": "llama-v1"},
    )
    handle = ExecutionHandle(
        adapter_id="systemd-user-unit",
        external_id="device-scheduler-attempt-test.service",
        identity="backend-attempt",
        metadata={
            "handle_version": 1,
            "invocation_id": "invocation-test",
            "control_group": "/user.slice/device-scheduler-attempt-test.service",
        },
    )
    scheduler.ledger.put("resource", resource_id, {
        "resource_id": resource_id,
        "host_id": "test-host",
        "kind": "gpu",
        "capacities": {"memory_bytes": float(16 * 1024**2)},
        "attributes": {"uuid": "GPU-test"},
    })
    scheduler.ledger.put("lease", lease_id, {
        "lease_id": lease_id,
        "allocations": [{"resource_id": resource_id, "quantities": {"memory_bytes": float(4 * 1024**2)}}],
    })
    scheduler.ledger.put("backend", backend_id, {
        "id": backend_id,
        "backend_id": backend_id,
        "spec": backend_spec.to_dict(),
        "state": "ready",
        "lease_id": lease_id,
        "execution_handle": handle.to_dict(),
    })

    scheduler.executor.managed_process_ids = lambda _handles: {
        "state": "observed",
        "process_ids": [4321],
        "attempts": [{
            "attempt_id": "backend-attempt",
            "unit_name": handle.external_id,
            "invocation_id": "invocation-test",
            "control_group": handle.metadata["control_group"],
            "state": "observed",
            "process_ids": [4321],
        }],
        "observed_at": observed_at,
        "scope": "invocation-verified-systemd-unit-cgroups",
    }
    nvidia = scheduler.resource_adapters[1]
    nvidia.snapshot = lambda: {
        "gpu_inventory": {"state": "observed"},
        "inventory_observed_at": observed_at,
        "process_telemetry": {
            "state": "observed",
            "observed_at": observed_at,
            "processes": [{
                "gpu_uuid": "GPU-test", "pid": 4321,
                "used_memory_bytes": 1024**2, "identity_unknown": False,
            }],
        },
        "resources": [{
            "resource_id": resource_id,
            "host_id": "test-host",
            "kind": "gpu",
            "attributes": {
                "uuid": "GPU-test",
                "memory_free_bytes_observation": 8 * 1024**2,
                "inventory_observed_at": observed_at,
                "process_observed_at": observed_at,
                "processes_observation": [{
                    "gpu_uuid": "GPU-test", "pid": 4321,
                    "used_memory_bytes": 1024**2, "identity_unknown": False,
                }],
            },
        }],
    }

    baseline_operation = "handoff-baseline-test:baseline:0"
    baseline = scheduler.backend_adapter.capture_reclamation_baseline(
        backend_spec, operation_id=baseline_operation,
    )
    assert baseline["state"] == "observed"
    assert baseline["execution_identity"]["invocation_id"] == "invocation-test"
    assert baseline["gpus"][0]["backend_process_ids"] == [4321]
    contaminated = nvidia.snapshot()
    contaminated["resources"][0]["attributes"]["processes_observation"].append({
        "gpu_uuid": "GPU-test", "pid": 9876,
        "used_memory_bytes": 512, "identity_unknown": False,
    })
    nvidia.snapshot = lambda: contaminated
    refused_baseline = scheduler.backend_adapter.capture_reclamation_baseline(
        backend_spec, operation_id="handoff-baseline-test:baseline:1",
    )
    assert refused_baseline["state"] == "unknown"
    nvidia.snapshot = lambda: {
        "gpu_inventory": {"state": "observed"},
        "inventory_observed_at": observed_at,
        "process_telemetry": {
            "state": "observed",
            "observed_at": observed_at,
            "processes": [{
                "gpu_uuid": "GPU-test", "pid": 4321,
                "used_memory_bytes": 1024**2, "identity_unknown": False,
            }],
        },
        "resources": [{
            "resource_id": resource_id,
            "host_id": "test-host",
            "kind": "gpu",
            "attributes": {
                "uuid": "GPU-test",
                "memory_free_bytes_observation": 8 * 1024**2,
                "inventory_observed_at": observed_at,
                "process_observed_at": observed_at,
                "processes_observation": [{
                    "gpu_uuid": "GPU-test", "pid": 4321,
                    "used_memory_bytes": 1024**2, "identity_unknown": False,
                }],
            },
        }],
    }

    handoff_id = "handoff-baseline-test"
    handoff_spec = HandoffSpec(
        handoff_id=handoff_id,
        owner_id="operator",
        jobs=(JobSpec(owner_id="operator", profile_id="llama-process"),),
        backend_ids=(backend_id,),
        continuation_refs=(),
        eviction_grants=(EvictionGrant(
            grant_id="grant-test", backend_id=backend_id, granted_by="operator",
            scope=f"backend:{backend_id}", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        ),),
        resource_envelope=(ResourceRequirement("gpu", {"memory_bytes": 2 * 1024**2}),),
        drain_timeout_seconds=10,
        run_timeout_seconds=10,
        restore_timeout_seconds=10,
    )
    scheduler.ledger.put("handoff", handoff_id, {
        "handoff_id": handoff_id,
        "spec": handoff_spec.to_dict(),
        "backend_snapshots": {backend_id: {"lease_id": lease_id}},
        "reclamation_baselines": {backend_id: dict(baseline)},
    })
    requested_floors: list[int] = []

    def reclaimed(_uuid: str, minimum_free_bytes: int, **_kwargs: object) -> dict[str, object]:
        requested_floors.append(minimum_free_bytes)
        return {
            "state": "reclaimed",
            "samples": [{"processes": {"state": "empty", "processes": []}}],
            "minimum_free_bytes": minimum_free_bytes,
            "evidence": {"source": "fixture"},
        }

    nvidia.wait_memory_reclaimed = reclaimed
    verifier = scheduler.backend_adapter._resource_reclamation_verifier
    receipt = verifier(backend_spec, handoff_id, baseline)
    assert receipt["state"] == "reclaimed"
    assert receipt["operation_id"] == handoff_id
    assert requested_floors == [9 * 1024**2]

    nvidia.wait_memory_reclaimed = lambda _uuid, minimum_free_bytes, **_kwargs: {
        "state": "reclaimed",
        "samples": [{"processes": {"state": "busy", "processes": [{"pid": 4321}]}}],
        "minimum_free_bytes": minimum_free_bytes,
        "evidence": {"source": "fixture"},
    }
    lingering_pid = verifier(backend_spec, handoff_id, baseline)
    assert lingering_pid["state"] == "unknown"
