from __future__ import annotations

import hashlib
import json
import runpy
import secrets
import sys
from pathlib import Path

import pytest

from device_scheduler_core.models import Principal
from device_scheduler_mcp.auth import (
    AuthConfigurationError,
    StaticTokenStore,
    TokenGrant,
    read_client_token,
)
from device_scheduler_mcp.config import ConfigurationError, ServiceConfig, load_profiles


def _private_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def test_token_manifest_resolves_digest_to_scoped_principal(tmp_path: Path) -> None:
    token_dir = tmp_path / "tokens"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    token = secrets.token_urlsafe(32)
    manifest = "\n".join([
        "[principal]",
        f'token_sha256 = "{hashlib.sha256(token.encode()).hexdigest()}"',
        'principal_id = "cuda-agent"',
        'actions = ["jobs.submit", "jobs.get"]',
        'execution_roots = ["/tmp/device-scheduler-tests"]',
        'backend_ids = ["llama-main"]',
        "max_priority = 65",
        "",
    ])
    _private_file(token_dir / "cuda-agent.toml", manifest)

    store = StaticTokenStore.from_directory(token_dir)

    principal = store.authenticate(token)
    assert len(store) == 1
    assert principal == Principal(
        principal_id="cuda-agent",
        actions=frozenset({"jobs.submit", "jobs.get"}),
        execution_roots=("/tmp/device-scheduler-tests",),
        backend_ids=frozenset({"llama-main"}),
        max_priority=65,
    )
    assert store.authenticate(token + "x") is None


def test_private_token_file_is_read_without_exposing_it_in_config(tmp_path: Path) -> None:
    token_path = tmp_path / "client.token"
    token = secrets.token_urlsafe(32)
    _private_file(token_path, token + "\n")
    assert read_client_token(token_path) == token

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "server": {"host": "127.0.0.1", "port": 9000},
        "auth": {"token_directory": str(tmp_path), "token_file": str(token_path)},
    }), encoding="utf-8")
    config = ServiceConfig.from_file(config_path)
    assert config.host == "127.0.0.1"
    assert config.token_file == token_path
    assert token not in config_path.read_text(encoding="utf-8")

    malformed = tmp_path / "malformed.token"
    _private_file(malformed, " " + secrets.token_urlsafe(32))
    with pytest.raises(AuthConfigurationError, match="opaque value"):
        read_client_token(malformed)


def test_unsafe_permissions_and_symlinks_are_rejected(tmp_path: Path) -> None:
    token_dir = tmp_path / "tokens"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    insecure = token_dir / "client.toml"
    insecure.write_text("[principal]\nprincipal_id='client'\n", encoding="utf-8")
    insecure.chmod(0o644)
    with pytest.raises(AuthConfigurationError, match="mode 0600"):
        StaticTokenStore.from_directory(token_dir)

    token_path = tmp_path / "token"
    token_path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    token_path.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(token_path)
    with pytest.raises(AuthConfigurationError, match="regular files"):
        read_client_token(link)


def test_token_directory_and_grant_set_fail_closed(tmp_path: Path) -> None:
    open_dir = tmp_path / "open"
    open_dir.mkdir(mode=0o755)
    open_dir.chmod(0o755)
    with pytest.raises(AuthConfigurationError, match="mode 0700"):
        StaticTokenStore.from_directory(open_dir)

    principal = Principal("same-client", actions=frozenset({"jobs.get"}))
    digest = "a" * 64
    with pytest.raises(AuthConfigurationError, match="duplicate token digest"):
        StaticTokenStore([TokenGrant(digest, principal), TokenGrant(digest, Principal("other"))])


def test_token_provisioner_writes_a_private_client_token_and_hashed_grant(tmp_path: Path, monkeypatch, capsys) -> None:
    config_root = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    monkeypatch.setattr(sys, "argv", [
        "provision-token.py", "--principal-id", "smoke", "--actions", "resources.list", "admin.status",
    ])
    provisioner = runpy.run_path("scripts/provision-token.py", run_name="provision_token_test")

    assert provisioner["main"]() == 0
    token_path = config_root / "device-scheduler" / "client.token"
    token_dir = config_root / "device-scheduler" / "tokens"
    token = read_client_token(token_path)
    store = StaticTokenStore.from_directory(token_dir)
    assert store.authenticate(token) == Principal(
        "smoke", actions=frozenset({"resources.list", "admin.status"}),
    )
    output = capsys.readouterr().out
    assert token not in output
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert token_dir.stat().st_mode & 0o777 == 0o700


def test_token_provisioner_rejects_path_traversal_in_principal_id(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    monkeypatch.setattr(sys, "argv", [
        "provision-token.py", "--principal-id", "../../outside", "--actions", "resources.list",
    ])
    provisioner = runpy.run_path("scripts/provision-token.py", run_name="provision_token_test")

    with pytest.raises(ValueError, match="principal ID"):
        provisioner["main"]()
    assert not (tmp_path / "outside.toml").exists()


def test_json_and_toml_profiles_reject_inline_secret_values(tmp_path: Path) -> None:
    good = tmp_path / "profiles.toml"
    good.write_text('[profiles.cuda-tests]\nargv = ["pytest"]\nsecret_ref = "vault://cuda-test"\n', encoding="utf-8")
    good.chmod(0o600)
    assert load_profiles(good)["cuda-tests"]["secret_ref"] == "vault://cuda-test"

    bad = tmp_path / "profiles.json"
    bad.write_text('{"profiles":{"cuda-tests":{"environment":{"HF_TOKEN":"raw-secret"}}}}', encoding="utf-8")
    bad.chmod(0o600)
    with pytest.raises(ConfigurationError, match="secret references"):
        load_profiles(bad)

    config = tmp_path / "remote.json"
    config.write_text('{"server":{"host":"0.0.0.0"}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="loopback"):
        ServiceConfig.from_file(config)

