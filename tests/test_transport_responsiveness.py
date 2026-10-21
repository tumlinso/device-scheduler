from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from device_scheduler_core.models import Principal
from device_scheduler_mcp.auth import StaticTokenStore
from device_scheduler_mcp.config import ServiceConfig
from device_scheduler_mcp.server import create_mcp_server


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_slow_mcp_dispatch_does_not_block_the_mcp_event_loop() -> None:
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    event_loop_thread = threading.get_ident()
    dispatch_thread: list[int] = []
    calls: list[tuple[str, dict[str, object], Principal]] = []

    class SlowInventoryScheduler:
        def dispatch(
            self, operation: str, args: dict[str, object], principal: Principal
        ) -> dict[str, object]:
            dispatch_thread.append(threading.get_ident())
            calls.append((operation, args, principal))
            dispatch_started.set()
            release_dispatch.wait(timeout=2)
            return {"resources": []}

        def tick(self) -> dict[str, object]:
            return {"status": "idle"}

        def status(self) -> dict[str, object]:
            return {"status": "ready"}

    principal = Principal("test-agent", actions=frozenset({"resources.list"}))
    server = create_mcp_server(
        SlowInventoryScheduler(),
        StaticTokenStore([]),
        ServiceConfig(),
    )
    tool = server._tool_manager.get_tool("resources_list")
    assert tool is not None
    assert tool.is_async is True
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            request=SimpleNamespace(
                user=SimpleNamespace(
                    access_token=SimpleNamespace(
                        claims={"device_scheduler_principal": principal.to_dict()}
                    )
                )
            )
        )
    )

    started_at = time.monotonic()
    request = asyncio.create_task(tool.run({"arguments": {}}, context=context, convert_result=True))
    try:
        await asyncio.sleep(0.05)
        elapsed = time.monotonic() - started_at
        assert elapsed < 0.5, "synchronous scheduler dispatch blocked the ASGI event loop"
        assert dispatch_started.is_set()
        assert dispatch_thread[0] != event_loop_thread
        assert not request.done()
    finally:
        release_dispatch.set()

    result = await asyncio.wait_for(request, timeout=1)
    content, structured = result
    assert structured == {"resources": []}
    assert len(content) == 1
    assert calls == [("resources.list", {}, principal)]
