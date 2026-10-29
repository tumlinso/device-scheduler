"""Command-line interface backed by the same authenticated MCP operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn

from .auth import AuthConfigurationError
from .client import DeviceSchedulerClient, DeviceSchedulerClientError
from .config import ConfigurationError, ServiceConfig, load_json_or_toml
from .server import create_app
from .stdio import run_stdio_proxy


_GROUP_COMMANDS: dict[str, dict[str, tuple[str, str | None, str | None]]] = {
    "resources": {
        "list": ("resources.list", None, None),
        "capabilities": ("resources.capabilities", None, None),
        "forecast": ("resources.forecast", None, "request"),
    },
    "leases": {
        "request": ("leases.request", None, "request"),
        "demand": ("leases.demand", None, None),
        "get": ("leases.get", "lease_id", None),
        "list": ("leases.list", None, None),
        "renew": ("leases.renew", "lease_id", None),
        "release": ("leases.release", "lease_id", None),
    },
    "jobs": {
        "submit": ("jobs.submit", None, "spec"),
        "get": ("jobs.get", "job_id", None),
        "list": ("jobs.list", None, None),
        "wait": ("jobs.wait", "job_id", None),
        "logs": ("jobs.logs", "job_id", None),
        "control": ("jobs.control", "job_id", None),
    },
    "backends": {
        "register": ("backends.register", None, "spec"),
        "get": ("backends.get", "backend_id", None),
        "list": ("backends.list", None, None),
        "control": ("backends.control", "backend_id", None),
        "attach": ("backends.attach", "backend_id", "consumer"),
    },
    "handoffs": {
        "submit": ("handoffs.submit", None, "spec"),
        "get": ("handoffs.get", "handoff_id", None),
        "list": ("handoffs.list", None, None),
        "control": ("handoffs.control", "handoff_id", None),
    },
    "schedules": {
        "put": ("schedules.put", None, "spec"),
        "get": ("schedules.get", "schedule_id", None),
        "list": ("schedules.list", None, None),
        "control": ("schedules.control", "schedule_id", None),
    },
    "events": {"publish": ("events.publish", None, "event")},
    "history": {"compact": ("history.compact", None, None), "status": ("history.status", None, None)},
    "policies": {
        "explain": ("policies.explain", None, None),
        "simulate": ("policies.simulate", None, None),
    },
    "admin": {
        "status": ("admin.status", None, None),
        "reconcile": ("admin.reconcile", None, None),
        "backup": ("admin.backup", None, None),
    },
}
_GROUP_ALIASES = {
    "resources": "resource",
    "leases": "lease",
    "jobs": "job",
    "backends": "backend",
    "handoffs": "handoff",
    "schedules": "schedule",
    "events": "event",
    "history": "history",
    "policies": "policy",
    "admin": "admin",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="device-scheduler", description="Device Scheduler daemon and client")
    parser.add_argument("--config", type=Path, help="host-local JSON or TOML configuration file")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the single scheduler daemon")
    serve.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"), default="info")
    commands.add_parser("mcp-stdio", aliases=["mcpstdio"], help="run an MCP stdio bridge to the configured daemon")

    call = commands.add_parser("call", help="invoke one canonical operation with a JSON argument object")
    call.add_argument("operation")
    _add_arguments_input(call)

    for family, actions in _GROUP_COMMANDS.items():
        family_name = _GROUP_ALIASES[family]
        aliases = [family] if family != family_name else []
        family_parser = commands.add_parser(family_name, aliases=aliases, help=f"{family} operations")
        action_parsers = family_parser.add_subparsers(dest="action_name", required=True)
        for action_name, (operation, id_field, payload_field) in actions.items():
            command_parser = action_parsers.add_parser(action_name, help=operation)
            command_parser.set_defaults(operation=operation, id_field=id_field, payload_field=payload_field)
            if id_field:
                command_parser.add_argument("entity_id")
            if operation == "jobs.wait":
                command_parser.add_argument("--timeout-seconds", type=float, default=0)
            if operation.endswith(".control"):
                command_parser.add_argument("control_action")
            _add_arguments_input(command_parser, payload_required=payload_field is not None)
        if family == "jobs":
            runner = action_parsers.add_parser("run", help="submit once, wait briefly, and collect logs")
            runner.add_argument("--spec-file", required=True, help="JSON/TOML job specification, or - for JSON on stdin")
            runner.add_argument("--wait-seconds", type=float, default=30, help="bounded wait from 0 through 300 seconds")
            runner.add_argument("--log-options-json", help="JSON object of jobs.logs options")
    return parser


def _add_arguments_input(parser: argparse.ArgumentParser, *, payload_required: bool = False) -> None:
    parser.add_argument("--arguments-file", type=Path, help="JSON or TOML object containing operation arguments")
    parser.add_argument("--arguments-json", help="JSON object containing operation arguments")
    if payload_required:
        parser.add_argument("--payload-file", type=Path, required=True, help="JSON or TOML specification or event")


def _read_arguments(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "arguments_file", None) and getattr(args, "arguments_json", None):
        raise ConfigurationError("choose only one of --arguments-file and --arguments-json")
    if getattr(args, "arguments_file", None):
        value = load_json_or_toml(args.arguments_file)
    elif getattr(args, "arguments_json", None) is not None:
        try:
            value = json.loads(args.arguments_json)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("--arguments-json must contain valid JSON") from exc
    else:
        value = {}
    if not isinstance(value, dict):
        raise ConfigurationError("operation arguments must be a JSON object")
    return dict(value)


def _command_arguments(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    operation = args.operation
    payload = _read_arguments(args)
    id_field = getattr(args, "id_field", None)
    if id_field:
        payload[id_field] = args.entity_id
    if operation == "jobs.wait":
        payload["timeout_seconds"] = args.timeout_seconds
    if operation.endswith(".control"):
        payload["action"] = args.control_action
    payload_field = getattr(args, "payload_field", None)
    if payload_field:
        source = load_json_or_toml(args.payload_file)
        if not isinstance(source, dict):
            raise ConfigurationError("payload file must contain an object")
        payload[payload_field] = source
    return operation, payload


def _read_payload(path: str) -> dict[str, Any]:
    if path == "-":
        try:
            value = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("job specification on stdin must be valid JSON") from exc
    else:
        value = load_json_or_toml(Path(path))
    if not isinstance(value, dict):
        raise ConfigurationError("job specification must be an object")
    return value


def _load_service_config(path: Path | None) -> ServiceConfig:
    if path is not None:
        return ServiceConfig.from_file(path)
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    installed = config_root / "device-scheduler" / "config.toml"
    return ServiceConfig.from_file(installed) if installed.exists() else ServiceConfig.default()


async def _run_client_command(config: ServiceConfig, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    client = DeviceSchedulerClient(
        f"http://{_url_host(config.host)}:{config.port}",
        config.token_file,
        timeout_seconds=config.http_timeout_seconds,
        mcp_path=config.mcp_path,
    )
    return await client.call(operation, arguments)


async def _run_job_command(config: ServiceConfig, args: argparse.Namespace) -> dict[str, Any]:
    client = DeviceSchedulerClient(
        f"http://{_url_host(config.host)}:{config.port}", config.token_file,
        timeout_seconds=config.http_timeout_seconds, mcp_path=config.mcp_path,
    )
    log_options = json.loads(args.log_options_json) if args.log_options_json else {}
    if not isinstance(log_options, dict):
        raise ConfigurationError("--log-options-json must contain a JSON object")
    return await client.run_job(_read_payload(args.spec_file), wait_seconds=args.wait_seconds, log_options=log_options)


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_service_config(args.config)
        if args.command == "serve":
            app = create_app(config)
            uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level)
            return 0
        if args.command in {"mcp-stdio", "mcpstdio"}:
            client = DeviceSchedulerClient(
                f"http://{_url_host(config.host)}:{config.port}",
                config.token_file,
                timeout_seconds=config.http_timeout_seconds,
                mcp_path=config.mcp_path,
            )
            run_stdio_proxy(client)
            return 0
        if args.command in {"jobs", "job"} and getattr(args, "action_name", None) == "run":
            result = asyncio.run(_run_job_command(config, args))
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
            return 0
        operation, arguments = (
            (args.operation, _read_arguments(args)) if args.command == "call" else _command_arguments(args)
        )
        result = asyncio.run(_run_client_command(config, operation, arguments))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except DeviceSchedulerClientError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": str(exc), "details": exc.details}}, indent=2), file=sys.stderr)
        return 2
    except (AuthConfigurationError, ConfigurationError, OSError, ValueError) as exc:
        print(f"device-scheduler: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
