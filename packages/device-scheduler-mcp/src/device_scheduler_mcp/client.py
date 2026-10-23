"""Async Python client for the authenticated Device Scheduler MCP daemon."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from device_scheduler_core.models import HandoffSpec, JobSpec, JsonModel, ResourceRequest, ScheduleSpec

from .auth import read_client_token
from .registry import tool_for_operation

TModel = TypeVar("TModel", bound=JsonModel)


class DeviceSchedulerClientError(RuntimeError):
    """Stable client error decoded from a scheduler MCP response."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class DeviceSchedulerProtocolError(DeviceSchedulerClientError):
    def __init__(self, message: str) -> None:
        super().__init__("protocol_error", message)


class DeviceSchedulerClient:
    """A small async SDK that uses the same MCP operations as the CLI and tools."""

    def __init__(
        self,
        base_url: str,
        token_file: Path | str,
        *,
        timeout_seconds: float = 30.0,
        mcp_path: str = "/mcp",
        http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
    ) -> None:
        self.endpoint = _mcp_endpoint(base_url, mcp_path)
        if not 0.1 <= timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 0.1 and 300")
        self.token_file = Path(token_file).expanduser()
        self.timeout_seconds = float(timeout_seconds)
        self._http_client_factory = http_client_factory or httpx.AsyncClient

    async def call(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one canonical operation through its registered MCP tool."""
        tool_name = tool_for_operation(operation)
        payload = dict(arguments or {})
        token = read_client_token(self.token_file)
        timeout = self.timeout_seconds
        if operation == "jobs.wait":
            wait_s = payload.get("timeout_seconds", payload.get("wait_seconds", payload.get("timeout", 0)))
            if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)) or not 0 <= wait_s <= 300:
                raise ValueError("job wait timeout must be from 0 through 300 seconds")
            timeout = max(timeout, float(wait_s) + 10)

        http_client = self._http_client_factory(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout),
        )
        try:
            async with streamable_http_client(self.endpoint, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, {"arguments": payload})
            return _decode_result(result)
        except DeviceSchedulerClientError:
            raise
        except Exception as exc:
            status = _http_status(exc)
            if status == 401:
                raise DeviceSchedulerClientError("unauthorized", "scheduler authentication failed") from exc
            if status == 403:
                raise DeviceSchedulerClientError("permission_denied", "scheduler denied this operation") from exc
            raise DeviceSchedulerClientError("transport_error", "could not complete the scheduler request") from exc
        finally:
            await http_client.aclose()

    async def list_resources(self) -> dict[str, Any]:
        return await self.call("resources.list")

    async def resource_capabilities(self) -> dict[str, Any]:
        return await self.call("resources.capabilities")

    async def request_lease(self, request: ResourceRequest | Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("leases.request", {"request": _as_mapping(request)})

    async def get_lease(self, lease_id: str) -> dict[str, Any]:
        return await self.call("leases.get", {"lease_id": lease_id})

    async def renew_lease(self, lease_id: str, **changes: Any) -> dict[str, Any]:
        return await self.call("leases.renew", {"lease_id": lease_id, **changes})

    async def release_lease(self, lease_id: str, **options: Any) -> dict[str, Any]:
        return await self.call("leases.release", {"lease_id": lease_id, **options})

    async def submit_job(
        self, spec: JobSpec | Mapping[str, Any], *,
        lease_id: str | None = None, fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Submit durable work, optionally using an owned supervised lease."""
        arguments: dict[str, Any] = {"spec": _as_mapping(spec)}
        if lease_id is not None:
            arguments["lease_id"] = lease_id
        if fencing_token is not None:
            arguments["fencing_token"] = fencing_token
        return await self.call("jobs.submit", arguments)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self.call("jobs.get", {"job_id": job_id})

    async def wait_job(self, job_id: str, *, timeout_seconds: float = 0) -> dict[str, Any]:
        return await self.call("jobs.wait", {"job_id": job_id, "timeout_seconds": timeout_seconds})

    async def job_logs(self, job_id: str, **options: Any) -> dict[str, Any]:
        return await self.call("jobs.logs", {"job_id": job_id, **options})

    async def control_job(self, job_id: str, action: str, **options: Any) -> dict[str, Any]:
        return await self.call("jobs.control", {"job_id": job_id, "action": action, **options})

    async def submit_handoff(self, spec: HandoffSpec | Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("handoffs.submit", {"spec": _as_mapping(spec)})

    async def get_handoff(self, handoff_id: str) -> dict[str, Any]:
        return await self.call("handoffs.get", {"handoff_id": handoff_id})

    async def control_handoff(self, handoff_id: str, action: str, **options: Any) -> dict[str, Any]:
        return await self.call("handoffs.control", {"handoff_id": handoff_id, "action": action, **options})

    async def put_schedule(self, spec: ScheduleSpec | Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("schedules.put", {"spec": _as_mapping(spec)})

    async def publish_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("events.publish", {"event": dict(event)})

    async def explain_policy(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("policies.explain", dict(request))

    async def simulate_policy(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return await self.call("policies.simulate", dict(request))

    async def status(self) -> dict[str, Any]:
        return await self.call("admin.status")


def _as_mapping(value: TModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, JsonModel):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("typed SDK input must be a core model or mapping")


def _mcp_endpoint(base_url: str, mcp_path: str = "/mcp") -> str:
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a loopback HTTP URL")
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an http loopback URL without user info, query, or fragment")
    host = parsed.hostname.lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("base_url must point to loopback")
    if not isinstance(mcp_path, str) or not mcp_path.startswith("/") or mcp_path.endswith("/"):
        raise ValueError("mcp_path must be an absolute route without a trailing slash")
    path = parsed.path.rstrip("/")
    if path and path != mcp_path:
        raise ValueError("base_url path must be empty or match mcp_path")
    endpoint_path = path or mcp_path
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def _decode_result(result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        detail = _content_payload(result)
        error = detail.get("error") if isinstance(detail, dict) else None
        if isinstance(error, Mapping):
            raise DeviceSchedulerClientError(
                str(error.get("code", "operation_failed")),
                str(error.get("message", "scheduler operation failed")),
                error.get("details") if isinstance(error.get("details"), Mapping) else {},
            )
        raise DeviceSchedulerClientError("operation_failed", "scheduler operation failed")
    payload = getattr(result, "structuredContent", None)
    if not isinstance(payload, dict):
        payload = _content_payload(result)
    if not isinstance(payload, dict):
        raise DeviceSchedulerProtocolError("scheduler returned a non-object result")
    error = payload.get("error")
    if isinstance(error, Mapping):
        raise DeviceSchedulerClientError(
            str(error.get("code", "operation_failed")),
            str(error.get("message", "scheduler operation failed")),
            error.get("details") if isinstance(error.get("details"), Mapping) else {},
        )
    return payload


def _content_payload(result: Any) -> Any:
    for block in getattr(result, "content", ()):
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (TypeError, json.JSONDecodeError):
                return {"message": "scheduler returned an unstructured response"}
    return None


def _http_status(error: BaseException) -> int | None:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            status = _http_status(child)
            if status is not None:
                return status
    return None
