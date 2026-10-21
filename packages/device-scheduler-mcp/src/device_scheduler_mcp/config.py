"""Host-local configuration loading for the daemon and its clients."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Configuration is missing, malformed, or unsafe for local operation."""


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8770
    tick_interval_seconds: float = 0.25
    token_directory: Path = Path("~/.config/device-scheduler/tokens")
    token_file: Path = Path("~/.config/device-scheduler/client.token")
    state_directory: Path = Path("~/.local/state/device-scheduler")
    ledger_path: Path | None = None
    artifact_directory: Path | None = None
    profiles_path: Path | None = None
    backend_profiles_path: Path | None = None
    continuations_path: Path | None = None
    gpu_headroom_fraction: float = 0.10
    gpu_observation_max_age_seconds: float = 30.0
    policy_extension: str = "builtin.weighted-fair"
    policy_parameters: Mapping[str, Any] = field(default_factory=dict)
    policy_options: Mapping[str, Any] = field(default_factory=dict)
    fairness_window_seconds: float = 86400.0
    mcp_path: str = "/mcp"
    http_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _validate_loopback_host(self.host)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be an integer from 1 through 65535")
        if not isinstance(self.tick_interval_seconds, (int, float)) or not 0.05 <= self.tick_interval_seconds <= 60:
            raise ConfigurationError("tick_interval_seconds must be between 0.05 and 60")
        if not isinstance(self.http_timeout_seconds, (int, float)) or not 0.1 <= self.http_timeout_seconds <= 300:
            raise ConfigurationError("http_timeout_seconds must be between 0.1 and 300")
        if (
            isinstance(self.gpu_headroom_fraction, bool)
            or not isinstance(self.gpu_headroom_fraction, (int, float))
            or not math.isfinite(self.gpu_headroom_fraction)
            or not 0 <= self.gpu_headroom_fraction < 1
        ):
            raise ConfigurationError("gpu_headroom_fraction must be finite and in [0, 1)")
        if (
            isinstance(self.gpu_observation_max_age_seconds, bool)
            or not isinstance(self.gpu_observation_max_age_seconds, (int, float))
            or not math.isfinite(self.gpu_observation_max_age_seconds)
            or self.gpu_observation_max_age_seconds <= 0
        ):
            raise ConfigurationError("gpu_observation_max_age_seconds must be finite and positive")
        if (
            isinstance(self.fairness_window_seconds, bool)
            or not isinstance(self.fairness_window_seconds, (int, float))
            or not math.isfinite(self.fairness_window_seconds)
            or self.fairness_window_seconds <= 0
        ):
            raise ConfigurationError("fairness_window_seconds must be finite and positive")
        if not isinstance(self.policy_extension, str) or not self.policy_extension.strip():
            raise ConfigurationError("policy extension must be a non-empty installed entry-point name")
        if not isinstance(self.policy_parameters, Mapping):
            raise ConfigurationError("built-in policy parameters must be an object")
        if not isinstance(self.policy_options, Mapping):
            raise ConfigurationError("installed policy options must be an object")
        if self.policy_extension == "builtin.weighted-fair" and self.policy_options:
            raise ConfigurationError("policy options apply only to an installed policy entry point")
        if self.policy_extension != "builtin.weighted-fair" and self.policy_parameters:
            raise ConfigurationError("built-in policy parameters cannot be combined with an installed policy")
        if _contains_secret_key(self.policy_options):
            raise ConfigurationError("installed policy options may contain secret references, not secret values")
        object.__setattr__(self, "gpu_headroom_fraction", float(self.gpu_headroom_fraction))
        object.__setattr__(self, "gpu_observation_max_age_seconds", float(self.gpu_observation_max_age_seconds))
        object.__setattr__(self, "fairness_window_seconds", float(self.fairness_window_seconds))
        object.__setattr__(self, "policy_parameters", dict(self.policy_parameters))
        object.__setattr__(self, "policy_options", dict(self.policy_options))
        if not self.mcp_path.startswith("/") or self.mcp_path.endswith("/"):
            raise ConfigurationError("mcp_path must be an absolute route without a trailing slash")
        for name in ("token_directory", "token_file", "state_directory"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        for name in ("ledger_path", "artifact_directory", "profiles_path", "backend_profiles_path", "continuations_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())

    @property
    def resolved_ledger_path(self) -> Path:
        return self.ledger_path or self.state_directory / "ledger.sqlite3"

    @property
    def resolved_artifact_directory(self) -> Path:
        return self.artifact_directory or self.state_directory / "artifacts"

    @classmethod
    def from_file(cls, path: Path | str) -> "ServiceConfig":
        config_path = Path(path).expanduser()
        raw = load_json_or_toml(config_path)
        if not isinstance(raw, Mapping):
            raise ConfigurationError("configuration root must be an object")
        allowed = {"server", "paths", "auth", "execution", "allocation", "policy"}
        if set(raw) - allowed:
            raise ConfigurationError("configuration contains unknown sections")
        server = _section(raw, "server")
        paths = _section(raw, "paths")
        auth = _section(raw, "auth")
        execution = _section(raw, "execution")
        allocation = _section(raw, "allocation")
        policy = _section(raw, "policy")
        if set(server) - {"host", "port", "tick_interval_seconds", "mcp_path", "http_timeout_seconds"}:
            raise ConfigurationError("server section contains unknown settings")
        if set(paths) - {
            "state_directory", "ledger_path", "artifact_directory", "profiles_path",
            "backend_profiles_path", "continuations_path",
        }:
            raise ConfigurationError("paths section contains unknown settings")
        if set(auth) - {"token_directory", "token_file"}:
            raise ConfigurationError("auth section contains unknown settings")
        if set(execution) - {"profiles_path"}:
            raise ConfigurationError("execution section contains unknown settings")
        if set(allocation) - {"gpu_headroom_fraction", "gpu_observation_max_age_seconds"}:
            raise ConfigurationError("allocation section contains unknown settings")
        if "profiles_path" in execution and "profiles_path" in paths:
            raise ConfigurationError("profiles_path must be configured in one section")
        extension = policy.pop("extension", "builtin.weighted-fair")
        fairness_window_seconds = policy.pop("fairness_window_seconds", 86400.0)
        options = policy.pop("options", {})
        if not isinstance(options, Mapping):
            raise ConfigurationError("policy.options must be an object")
        if extension == "builtin.weighted-fair":
            policy_parameters = policy
        else:
            if policy:
                raise ConfigurationError("only fairness_window_seconds and options apply to an installed policy")
            policy_parameters = {}
        values: dict[str, Any] = {
            **server,
            **paths,
            **auth,
            **allocation,
            "policy_extension": extension,
            "policy_parameters": policy_parameters,
            "policy_options": options,
            "fairness_window_seconds": fairness_window_seconds,
        }
        if "profiles_path" in execution:
            values["profiles_path"] = execution["profiles_path"]
        try:
            return cls(**values)
        except TypeError as exc:
            raise ConfigurationError("configuration contains an unsupported setting") from exc

    @classmethod
    def default(cls) -> "ServiceConfig":
        return cls()


def load_json_or_toml(path: Path | str) -> Any:
    """Read a JSON or TOML config/profile file based on its suffix."""
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read configuration file: {source}") from exc
    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(text)
        if suffix == ".toml":
            return tomllib.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"invalid {suffix[1:].upper()} in {source}") from exc
    raise ConfigurationError("configuration and profile files must use .json or .toml")


def load_profiles(path: Path | str | None) -> dict[str, Any]:
    """Load named host execution profiles without accepting inline credentials."""
    if path is None:
        return {}
    source = Path(path).expanduser()
    _require_private_profile_file(source)
    data = load_json_or_toml(source)
    if not isinstance(data, Mapping) or set(data) != {"profiles"} or not isinstance(data["profiles"], Mapping):
        raise ConfigurationError("profile file must contain only a [profiles] object")
    profiles = dict(data["profiles"])
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name or not isinstance(profile, Mapping):
            raise ConfigurationError("profile names and values must be objects")
        if _contains_secret_key(profile):
            raise ConfigurationError("profile files may contain secret references, not secret values")
    return profiles


def load_named_profiles(path: Path | str | None, section: str) -> dict[str, Any]:
    """Load a private operator profile file containing one named section."""
    if path is None:
        return {}
    if section not in {"backend_profiles", "continuations"}:
        raise ValueError("unsupported profile section")
    source = Path(path).expanduser()
    _require_private_profile_file(source)
    data = load_json_or_toml(source)
    if not isinstance(data, Mapping) or set(data) != {section} or not isinstance(data[section], Mapping):
        raise ConfigurationError(f"profile file must contain only a [{section}] object")
    values = dict(data[section])
    for name, profile in values.items():
        if not isinstance(name, str) or not name or not isinstance(profile, Mapping):
            raise ConfigurationError(f"{section} names and values must be objects")
        if _contains_secret_key(profile):
            raise ConfigurationError(f"{section} files may contain secret references, not secret values")
    return values


def load_backend_configuration(
    path: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load llama backend profiles and named FrameShift admission controls."""
    if path is None:
        return {}, {}
    source = Path(path).expanduser()
    _require_private_profile_file(source)
    data = load_json_or_toml(source)
    if (
        not isinstance(data, Mapping)
        or not set(data).issubset({"backend_profiles", "frameshift_controls"})
        or "backend_profiles" not in data
        or not isinstance(data["backend_profiles"], Mapping)
    ):
        raise ConfigurationError(
            "backend profile file must contain [backend_profiles] and optional [frameshift_controls] tables"
        )
    profiles = dict(data["backend_profiles"])
    controls_raw = data.get("frameshift_controls", {})
    if not isinstance(controls_raw, Mapping):
        raise ConfigurationError("frameshift_controls must be an object")
    controls = dict(controls_raw)
    for section_name, section in (("backend_profiles", profiles), ("frameshift_controls", controls)):
        for name, value in section.items():
            if not isinstance(name, str) or not name or not isinstance(value, Mapping):
                raise ConfigurationError(f"{section_name} names and values must be objects")
            if _contains_secret_key(value):
                raise ConfigurationError(f"{section_name} may contain secret file references, not secret values")
    for control_ref, control in controls.items():
        if set(control) - {"base_url", "token_file", "timeout_seconds"}:
            raise ConfigurationError(f"FrameShift controller {control_ref!r} contains unknown settings")
    return profiles, controls


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                normalized = key.casefold().replace("_", "").replace("-", "")
                contains_secret = any(word in normalized for word in ("password", "secret", "token", "apikey", "privatekey", "credential"))
                if contains_secret and not normalized.endswith("ref") and normalized not in {"tokenfile", "secretfile", "apikeyfile"}:
                    return True
            if _contains_secret_key(child):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_secret_key(child) for child in value)
    return False


def _require_private_profile_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ConfigurationError(f"cannot read profile file: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ConfigurationError("profile configuration must be a regular file")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ConfigurationError("profile configuration must be owned by the service user with mode 0600")


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} section must be an object")
    return dict(value)


def _validate_loopback_host(host: str) -> None:
    if not isinstance(host, str) or not host:
        raise ConfigurationError("host must be a loopback address")
    if host.lower() == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ConfigurationError("server host must be a loopback address") from exc
    if not address.is_loopback:
        raise ConfigurationError("server host must be a loopback address")
