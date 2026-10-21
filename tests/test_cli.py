from __future__ import annotations

import json
from pathlib import Path

from device_scheduler_mcp import cli
from device_scheduler_mcp.client import DeviceSchedulerClientError


class FakeClient:
    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(
        self, base_url: str, token_file: Path, *, timeout_seconds: float, mcp_path: str = "/mcp"
    ) -> None:
        self.base_url = base_url
        self.token_file = token_file
        self.timeout_seconds = timeout_seconds
        self.mcp_path = mcp_path

    async def call(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((operation, arguments))
        return {"operation": operation, "arguments": arguments}


def _config_file(tmp_path: Path) -> Path:
    config = tmp_path / "client.toml"
    config.write_text(
        "[server]\nhost = '127.0.0.1'\nport = 8770\n"
        "[auth]\ntoken_file = '/tmp/client.token'\n",
        encoding="utf-8",
    )
    return config


def test_grouped_cli_commands_map_to_canonical_operation(monkeypatch, tmp_path: Path, capsys) -> None:
    FakeClient.calls.clear()
    monkeypatch.setattr(cli, "DeviceSchedulerClient", FakeClient)
    config = _config_file(tmp_path)
    payload = tmp_path / "job.json"
    spec = {
        "owner_id": "agent",
        "profile_id": "cuda-tests",
        "idempotency_key": "cli-job-1",
    }
    payload.write_text(json.dumps(spec), encoding="utf-8")

    result = cli.main(["--config", str(config), "job", "submit", "--payload-file", str(payload)])

    assert result == 0
    assert FakeClient.calls == [("jobs.submit", {"spec": spec})]
    assert json.loads(capsys.readouterr().out) == {"operation": "jobs.submit", "arguments": {"spec": spec}}


def test_generic_cli_call_accepts_json_arguments(monkeypatch, tmp_path: Path, capsys) -> None:
    FakeClient.calls.clear()
    monkeypatch.setattr(cli, "DeviceSchedulerClient", FakeClient)
    config = _config_file(tmp_path)

    result = cli.main([
        "--config", str(config), "call", "policies.simulate", "--arguments-json", '{"policy_id":"fair"}',
    ])

    assert result == 0
    assert FakeClient.calls == [("policies.simulate", {"policy_id": "fair"})]
    assert json.loads(capsys.readouterr().out)["operation"] == "policies.simulate"


def test_cli_errors_are_machine_readable_and_exit_nonzero(monkeypatch, tmp_path: Path, capsys) -> None:
    class DeniedClient(FakeClient):
        async def call(self, operation: str, arguments: dict[str, object]) -> dict[str, object]:
            raise DeviceSchedulerClientError("permission_denied", "operation denied", {"operation": operation})

    monkeypatch.setattr(cli, "DeviceSchedulerClient", DeniedClient)
    config = _config_file(tmp_path)
    result = cli.main(["--config", str(config), "resources", "list"])

    assert result == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": {
            "code": "permission_denied",
            "message": "operation denied",
            "details": {"operation": "resources.list"},
        }
    }


def test_serve_and_stdio_are_explicit_transport_entrypoints(monkeypatch, tmp_path: Path) -> None:
    config = _config_file(tmp_path)
    made: list[object] = []
    served: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(cli, "create_app", lambda cfg: made.append(cfg) or "asgi-app")
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: served.append((app, kwargs)))
    assert cli.main(["--config", str(config), "serve"]) == 0
    assert made and served[0][0] == "asgi-app"
    assert served[0][1]["host"] == "127.0.0.1"
    assert served[0][1]["port"] == 8770

    forwarded: list[object] = []
    monkeypatch.setattr(cli, "run_stdio_proxy", lambda client: forwarded.append(client))
    monkeypatch.setattr(cli, "DeviceSchedulerClient", FakeClient)
    assert cli.main(["--config", str(config), "mcpstdio"]) == 0
    assert len(forwarded) == 1
