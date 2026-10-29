from __future__ import annotations

import asyncio
import hashlib
import secrets
from pathlib import Path

import anyio
import httpx
import pytest
from mcp import ClientSession
from types import SimpleNamespace

from device_scheduler_core.models import JobSpec, Principal, ResourceRequest, ResourceRequirement, ScheduleSpec
from device_scheduler_mcp.auth import StaticTokenStore, TokenGrant
from device_scheduler_mcp.client import DeviceSchedulerClient, DeviceSchedulerClientError, SupervisedReservation
from device_scheduler_mcp.registry import operations
from device_scheduler_mcp.server import create_asgi_app
from device_scheduler_mcp.stdio import create_stdio_mcp


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((operation, arguments))
        return {"operation": operation, "arguments": arguments}


def _token_file(path: Path) -> None:
    path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    path.chmod(0o600)


class FakeScheduler:
    def dispatch(self, operation: str, args: dict[str, object], principal: Principal) -> dict[str, object]:
        if operation not in principal.actions:
            raise PermissionError("ungranted operation")
        return {"operation": operation, "principal_id": principal.principal_id, "arguments": args}

    def tick(self) -> dict[str, object]:
        return {"status": "idle"}

    def status(self) -> dict[str, object]:
        return {"status": "ready"}


def test_sdk_typed_helpers_use_core_models_and_canonical_operations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://localhost:8770", token_path)
    recorded: list[tuple[str, dict[str, object]]] = []

    async def call(operation: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        payload = dict(arguments or {})
        recorded.append((operation, payload))
        return {"ok": True}

    monkeypatch.setattr(client, "call", call)
    job = JobSpec(owner_id="agent", profile_id="cuda-tests", idempotency_key="job-1")
    request = ResourceRequest(
        request_id="lease-request-1",
        owner_id="agent",
        requirements=(ResourceRequirement("gpu", {"memory_mib": 512}),),
        lease_ttl_seconds=60,
    )
    schedule = ScheduleSpec(
        schedule_id="nightly", owner_id="agent", job=job, interval_seconds=86400
    )

    async def run() -> None:
        await client.submit_job(job)
        await client.request_lease(request)
        await client.put_schedule(schedule)
        await client.wait_job("job-1", timeout_seconds=12)

    asyncio.run(run())
    assert [operation for operation, _args in recorded] == [
        "jobs.submit", "leases.request", "schedules.put", "jobs.wait",
    ]
    assert recorded[0][1] == {"spec": job.to_dict()}
    assert recorded[1][1] == {"request": request.to_dict()}
    assert recorded[2][1] == {"spec": schedule.to_dict()}
    assert recorded[3][1] == {"job_id": "job-1", "timeout_seconds": 12}


def test_sdk_priority_demand_and_forecast_helpers_use_public_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://localhost:8770", token_path)
    recorded: list[tuple[str, dict[str, object]]] = []

    async def call(operation: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        payload = dict(arguments or {})
        recorded.append((operation, payload))
        return {"state": "pending"}

    monkeypatch.setattr(client, "call", call)
    request = {
        "requirements": [{"kind": "gpu", "quantities": {"gpu_memory_bytes": 1}, "mode": "exclusive"}],
        "lease_ttl_seconds": 120,
    }

    async def run() -> None:
        await client.request_priority_lease_demand(request, "demand-1", wait_seconds=2)
        await client.get_priority_lease_demand("demand-1")
        await client.wait_priority_lease_demand("demand-1", wait_seconds=5)
        await client.cancel_priority_lease_demand("demand-1")
        await client.forecast_resources(request)

    asyncio.run(run())
    assert recorded == [
        ("leases.demand", {"action": "request", "demand_id": "demand-1", "request": request, "wait_seconds": 2}),
        ("leases.demand", {"action": "get", "demand_id": "demand-1"}),
        ("leases.demand", {"action": "wait", "demand_id": "demand-1", "wait_seconds": 5}),
        ("leases.demand", {"action": "cancel", "demand_id": "demand-1"}),
        ("resources.forecast", {"request": request, "horizon_seconds": 600, "max_candidates": 64}),
    ]


def test_sdk_rejects_non_loopback_and_unbounded_waits(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    with pytest.raises(ValueError, match="loopback"):
        DeviceSchedulerClient("http://scheduler.internal:8770", token_path)

    client = DeviceSchedulerClient("http://127.0.0.1:8770", token_path)
    with pytest.raises(ValueError, match="0 through 300"):
        asyncio.run(client.wait_job("job-1", timeout_seconds=301))


def test_mcp_stdio_bridge_registers_and_forwards_the_same_operation_set() -> None:
    target = RecordingClient()
    server = create_stdio_mcp(target)  # type: ignore[arg-type]
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {operation.tool_name for operation in operations()}

    asyncio.run(server.call_tool("jobs_submit", {"arguments": {"spec": {"owner_id": "agent"}}}))
    assert target.calls == [("jobs.submit", {"spec": {"owner_id": "agent"}})]
    schemas = {tool.name: tool.inputSchema for tool in tools}
    jobs_args = schemas["jobs_submit"]["properties"]["arguments"]
    spec = jobs_args["properties"]["spec"]
    assert jobs_args["required"] == ["spec"]
    assert jobs_args["properties"]["fencing_token"]["type"] == "integer"
    assert "owner_id" in spec["required"]
    assert {"type": "array", "items": {"type": "string"}} in spec["properties"]["argv"]["anyOf"]
    assert spec["properties"]["priority_class"]["enum"] == ["interactive", "normal", "background"]
    assert spec["properties"]["resources"]["items"]["properties"]["quantities"]["type"] == "object"
    assert spec["properties"]["resources"]["items"]["required"] == ["kind", "quantities"]
    assert spec["properties"]["resources"]["items"]["properties"]["mode"]["enum"] == ["shared", "exclusive"]
    assert {"type": "number"} in spec["properties"]["expected_duration_seconds"]["anyOf"]
    assert {"type": "number"} in spec["properties"]["kill_timeout_seconds"]["anyOf"]
    assert spec["properties"]["preemption_action"]["enum"] == ["restart", "abandon"]
    request = schemas["leases_request"]["properties"]["arguments"]["properties"]["request"]
    assert {"request_id", "owner_id", "requirements", "lease_ttl_seconds"}.issubset(request["required"])
    assert request["properties"]["requirements"]["items"]["properties"]["quantities"]["type"] == "object"
    assert schemas["leases_renew"]["properties"]["arguments"]["required"] == ["lease_id", "fencing_token", "ttl_seconds"]
    assert "leases_demand" in schemas and "resources_forecast" in schemas
    tool_map = {tool.name: tool for tool in tools}
    assert tool_map["resources_forecast"].annotations.readOnlyHint is True
    assert "history_compact" in schemas and "history_status" in schemas


def test_sdk_one_off_job_preserves_handle_when_bounded_wait_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://localhost:8770", token_path)
    recorded: list[tuple[str, object]] = []

    async def submit(spec, **kwargs):
        recorded.append(("submit", kwargs))
        return {"id": "durable-job-1", "state": "queued"}

    async def wait(job_id, *, timeout_seconds):
        recorded.append(("wait", timeout_seconds))
        return {"job": {"id": job_id, "state": "running"}, "timed_out": True}

    async def logs(job_id, **kwargs):
        recorded.append(("logs", job_id))
        return {"job_id": job_id, "stdout": ""}

    monkeypatch.setattr(client, "submit_job", submit)
    monkeypatch.setattr(client, "wait_job", wait)
    monkeypatch.setattr(client, "job_logs", logs)
    result = asyncio.run(client.run_job({"argv": ["true"]}, wait_seconds=0.1))
    assert result["job_id"] == "durable-job-1" and result["timed_out"] is True
    assert result["logs_available"] is False
    assert [name for name, _value in recorded] == ["submit", "wait", "logs"]


def test_sdk_job_runner_keeps_nested_or_direct_status_if_log_fetch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://localhost:8770", token_path)

    async def submit(_spec, **_kwargs):
        return {"job": {"id": "job-2", "state": "queued"}}

    async def wait(_job_id, *, timeout_seconds):
        return {"id": "job-2", "state": "queued", "timed_out": True}

    async def logs(_job_id, **_kwargs):
        raise DeviceSchedulerClientError("not_found", "attempt logs are not available yet")

    monkeypatch.setattr(client, "submit_job", submit)
    monkeypatch.setattr(client, "wait_job", wait)
    monkeypatch.setattr(client, "job_logs", logs)
    result = asyncio.run(client.run_job({"argv": ["true"]}, wait_seconds=0))
    assert result["job_id"] == "job-2"
    assert result["job"] == {"id": "job-2", "state": "queued", "timed_out": True}
    assert result["logs"] is None and result["logs_available"] is False
    assert result["logs_error"]["code"] == "not_found"


def test_supervised_reservation_tracks_latest_fence_and_uncertain_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://localhost:8770", token_path)
    sent: list[tuple[str, dict[str, object]]] = []

    async def call(operation, arguments=None):
        args = dict(arguments or {})
        sent.append((operation, args))
        if operation == "leases.renew":
            return {"lease_id": "lease-1", "fencing_token": 8, "state": "active"}
        if operation == "leases.release":
            return {"lease_id": "lease-1", "fencing_token": 9, "state": "unknown", "cleanup_verified": False}
        return {"lease_id": "lease-1", "fencing_token": 7, "state": "active"}

    monkeypatch.setattr(client, "call", call)

    async def run():
        reservation = await client.reserve({"request_id": "r", "owner_id": "agent", "requirements": [], "lease_ttl_seconds": 30, "idempotency_key": "r"})
        await reservation.renew(45)
        await reservation.submit_job({"argv": ["true"]})
        result = await reservation.release()
        return reservation, result

    reservation, result = asyncio.run(run())
    assert reservation.fencing_token == 9 and reservation.released is False
    assert result["cleanup_verified"] is False
    assert sent[0][1]["request"]["usage"] == "supervised"
    assert sent[-1][1]["fencing_token"] == 8


def test_stdio_session_bridge_forwards_to_authenticated_http_daemon(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    token = secrets.token_urlsafe(32)
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    principal = Principal("stdio-agent", actions=frozenset({"resources.list"}))
    store = StaticTokenStore([TokenGrant(hashlib.sha256(token.encode()).hexdigest(), principal)])
    app = create_asgi_app(FakeScheduler(), store, background_ticks=False)
    transport = httpx.ASGITransport(app=app)
    remote = DeviceSchedulerClient(
        "http://127.0.0.1:8770",
        token_path,
        http_client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    proxy = create_stdio_mcp(remote)

    async def journey() -> None:
        async with app.router.lifespan_context(app):
            client_to_server_send, server_read = anyio.create_memory_object_stream(0)
            server_to_client_send, client_read = anyio.create_memory_object_stream(0)
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    proxy._mcp_server.run,
                    server_read,
                    server_to_client_send,
                    proxy._mcp_server.create_initialization_options(),
                )
                try:
                    async with ClientSession(client_read, client_to_server_send) as session:
                        await session.initialize()
                        names = {tool.name for tool in (await session.list_tools()).tools}
                        assert names == {operation.tool_name for operation in operations()}
                        result = await session.call_tool("resources_list", {"arguments": {"kind": "gpu"}})
                        assert result.structuredContent == {
                            "operation": "resources.list",
                            "principal_id": "stdio-agent",
                            "arguments": {"kind": "gpu"},
                        }
                finally:
                    task_group.cancel_scope.cancel()

    anyio.run(journey, backend="asyncio")


def test_sdk_decodes_structured_scheduler_errors() -> None:
    result = SimpleNamespace(
        isError=False,
        structuredContent={"error": {"code": "permission_denied", "message": "denied", "details": {"scope": "jobs.submit"}}},
        content=[],
    )
    from device_scheduler_mcp.client import _decode_result

    with pytest.raises(DeviceSchedulerClientError) as error:
        _decode_result(result)
    assert error.value.code == "permission_denied"
    assert error.value.details == {"scope": "jobs.submit"}



def test_sdk_job_submission_preserves_explicit_lease_fence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token_path = tmp_path / "client.token"
    _token_file(token_path)
    client = DeviceSchedulerClient("http://127.0.0.1:8770", token_path)
    recorded = []
    async def call(operation, arguments):
        recorded.append((operation, arguments))
        return {"ok": True}
    monkeypatch.setattr(client, "call", call)
    asyncio.run(client.submit_job({"argv": ["/bin/true"]}, lease_id="lease:test", fencing_token=42))
    assert recorded == [("jobs.submit", {"spec": {"argv": ["/bin/true"]}, "lease_id": "lease:test", "fencing_token": 42})]
