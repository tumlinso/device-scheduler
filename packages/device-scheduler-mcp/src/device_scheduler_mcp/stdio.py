"""MCP stdio client bridge; it forwards every request to the daemon ledger."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .client import DeviceSchedulerClient
from .registry import Operation, operations


def create_stdio_mcp(client: DeviceSchedulerClient) -> FastMCP:
    """Build a local stdio surface whose tools are all remote daemon calls."""
    server = FastMCP(
        "device-scheduler-stdio",
        instructions=(
            "This stdio server is a thin bridge to the authenticated Device Scheduler daemon. "
            "Accepted operations remain durable after this client disconnects."
        ),
        log_level="ERROR",
    )
    for operation in operations():
        def handler_factory(contract: Operation):
            async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
                return await client.call(contract.name, arguments)

            handler.__name__ = contract.tool_name
            return handler

        server.add_tool(
            handler_factory(operation),
            name=operation.tool_name,
            description=operation.description,
            annotations=ToolAnnotations(
                readOnlyHint=operation.read_only,
                destructiveHint=False,
                idempotentHint=operation.read_only,
                openWorldHint=False,
            ),
            structured_output=True,
        )
    return server


def run_stdio_proxy(client: DeviceSchedulerClient) -> None:
    """Run the official MCP stdio transport against the daemon-backed bridge."""
    create_stdio_mcp(client).run(transport="stdio")
