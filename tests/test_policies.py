from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from device_scheduler_mcp.config import ConfigurationError, ServiceConfig
from device_scheduler_mcp.policies import POLICY_ENTRYPOINT_GROUP, ConfiguredPolicy, load_policy


def test_builtin_policy_uses_configured_parameters_and_records_snapshot() -> None:
    config = ServiceConfig(
        gpu_headroom_fraction=0.2,
        fairness_window_seconds=1234,
        policy_parameters={
            "age_quantum_seconds": 45,
            "max_age_boost": 5,
            "large_request_protection_seconds": 90,
            "fairness_penalty": 0.05,
            "class_weights": {"interactive": 5, "normal": 2, "background": 1},
            "owner_weights": {"project-a": 1.5},
        },
    )

    policy = load_policy(config)

    assert isinstance(policy, ConfiguredPolicy)
    assert policy.policy_id == "builtin.weighted-fair"
    assert policy.api_version == "1"
    assert policy.age_quantum_seconds == 45
    assert policy.max_age_boost == 5
    assert policy.large_request_protection_seconds == 90
    assert policy.fairness_penalty == 0.05
    assert policy.configuration_snapshot == {
        "api_version": "1",
        "policy_id": "builtin.weighted-fair",
        "version": "1",
        "source": {"kind": "builtin", "name": "builtin.weighted-fair"},
        "parameters": {
            "class_weights": {"interactive": 5.0, "normal": 2.0, "background": 1.0},
            "owner_weights": {"project-a": 1.5},
            "age_quantum_seconds": 45.0,
            "max_age_boost": 5.0,
            "large_request_protection_seconds": 90.0,
            "fairness_penalty": 0.05,
        },
        "fairness_window_seconds": 1234.0,
    }


class _InstalledPolicy:
    api_version = "1"
    policy_id = "org.example.installed"
    version = "2026.10"

    def __init__(self, window: int = 0) -> None:
        self.window = window

    def choose_next(self, requests: Any, **_kwargs: Any) -> Any:
        return requests[0] if requests else None


def test_installed_policy_resolves_only_registered_entry_point_and_binds_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    entry = SimpleNamespace(
        name="example.fair",
        load=lambda: _InstalledPolicy,
        dist=SimpleNamespace(name="example-scheduler-policy", version="2.3.4"),
    )

    def entries(*, group: str) -> list[object]:
        calls.append(group)
        return [entry]

    monkeypatch.setattr("device_scheduler_mcp.policies.entry_points", entries)
    policy = load_policy(ServiceConfig(
        policy_extension="example.fair",
        policy_options={"window": 19},
        fairness_window_seconds=600,
    ))

    assert calls == [POLICY_ENTRYPOINT_GROUP]
    assert policy.policy_id == "org.example.installed"
    assert policy.version == "2026.10"
    assert policy.window == 19
    assert policy.choose_next(["candidate"]) == "candidate"
    assert policy.configuration_snapshot["source"] == {
        "kind": "entry_point",
        "group": POLICY_ENTRYPOINT_GROUP,
        "name": "example.fair",
        "distribution": "example-scheduler-policy",
        "distribution_version": "2.3.4",
    }
    assert policy.configuration_snapshot["parameters"] == {"window": 19}


@pytest.mark.parametrize(
    ("api_version", "version", "with_choose"),
    [("2", "1.0", True), ("1", "", True), ("1", "1.0", False)],
)
def test_installed_policy_requires_supported_api_identity_and_contract(
    monkeypatch: pytest.MonkeyPatch,
    api_version: str,
    version: str,
    with_choose: bool,
) -> None:
    attrs: dict[str, Any] = {"api_version": api_version, "policy_id": "org.example.invalid", "version": version}
    if with_choose:
        attrs["choose_next"] = lambda *_args, **_kwargs: None
    invalid_policy = type("InvalidPolicy", (), attrs)
    entry = SimpleNamespace(name="invalid", load=lambda: invalid_policy, dist=None)
    monkeypatch.setattr("device_scheduler_mcp.policies.entry_points", lambda **_kwargs: [entry])

    with pytest.raises(ConfigurationError):
        load_policy(ServiceConfig(policy_extension="invalid"))


def test_installed_policy_must_resolve_to_exactly_one_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("device_scheduler_mcp.policies.entry_points", lambda **_kwargs: [])
    with pytest.raises(ConfigurationError, match="exactly one installed"):
        load_policy(ServiceConfig(policy_extension="missing"))


def test_builtin_policy_rejects_constructor_parameter_typos() -> None:
    with pytest.raises(ConfigurationError, match="invalid built-in policy settings"):
        load_policy(ServiceConfig(policy_parameters={"aging_seconds": 10}))
