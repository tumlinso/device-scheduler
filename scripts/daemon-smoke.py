#!/usr/bin/env python3
"""Read-only health and authenticated MCP smoke for a running local daemon."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from device_scheduler_mcp.client import DeviceSchedulerClient, DeviceSchedulerClientError


async def _run(base_url: str, token_file: Path, timeout: float) -> dict[str, object]:
    root = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as http:
        health = await http.get(f"{root}/healthz")
        readiness = await http.get(f"{root}/readyz")
    if health.status_code != 200 or health.json().get("status") != "ok":
        raise RuntimeError("healthz did not report healthy")
    if readiness.status_code != 200 or readiness.json().get("status") != "ready":
        raise RuntimeError("readyz did not report ready")

    client = DeviceSchedulerClient(root, token_file, timeout_seconds=timeout)
    capabilities = await client.call("resources.capabilities")
    resources = await client.call("resources.list")
    status = await client.call("admin.status")
    return {
        "health": health.json(),
        "readiness": readiness.json(),
        "mcp": "ok",
        "capability_adapter_count": len(capabilities.get("adapters", [])),
        "resource_count": len(resources.get("resources", [])),
        "scheduler_status": status.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770", help="loopback daemon origin")
    parser.add_argument("--token-file", type=Path, default=Path("~/.config/device-scheduler/client.token"))
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    try:
        result = asyncio.run(_run(args.url, args.token_file.expanduser(), args.timeout))
    except (httpx.HTTPError, DeviceSchedulerClientError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"daemon-smoke: check failed ({type(exc).__name__})") from None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
