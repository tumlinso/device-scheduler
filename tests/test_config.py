from __future__ import annotations

from pathlib import Path

import pytest

from device_scheduler_mcp.config import (
    ConfigurationError,
    ServiceConfig,
    load_backend_configuration,
    load_named_profiles,
)


def test_service_default_port_and_backend_profile_paths(tmp_path: Path) -> None:
    assert ServiceConfig.default().port == 8770
    config_path = tmp_path / "service.toml"
    config_path.write_text(
        "[server]\nport = 8770\n"
        "[paths]\nstate_directory = '/tmp/ds-state'\n"
        "backend_profiles_path = '/tmp/backends.toml'\n"
        "continuations_path = '/tmp/continuations.toml'\n",
        encoding="utf-8",
    )
    config = ServiceConfig.from_file(config_path)
    assert config.port == 8770
    assert config.backend_profiles_path == Path("/tmp/backends.toml")
    assert config.continuations_path == Path("/tmp/continuations.toml")


def test_service_config_loads_allocation_and_builtin_policy_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "service.toml"
    config_path.write_text(
        "[allocation]\n"
        "gpu_headroom_fraction = 0.15\n"
        "gpu_observation_max_age_seconds = 12\n"
        "[policy]\n"
        "extension = 'builtin.weighted-fair'\n"
        "fairness_window_seconds = 43200\n"
        "age_quantum_seconds = 90\n"
        "max_age_boost = 4\n"
        "large_request_protection_seconds = 240\n"
        "fairness_penalty = 0.03\n"
        "class_weights = { interactive = 4, normal = 2, background = 1 }\n"
        "owner_weights = { project_a = 2 }\n",
        encoding="utf-8",
    )
    config = ServiceConfig.from_file(config_path)
    assert config.gpu_headroom_fraction == 0.15
    assert config.gpu_observation_max_age_seconds == 12
    assert config.fairness_window_seconds == 43200
    assert config.policy_extension == "builtin.weighted-fair"
    assert config.policy_parameters["age_quantum_seconds"] == 90
    assert config.policy_parameters["class_weights"]["interactive"] == 4
    assert config.policy_parameters["owner_weights"]["project_a"] == 2


def test_service_config_loads_only_named_installed_policy_options(tmp_path: Path) -> None:
    config_path = tmp_path / "service.toml"
    config_path.write_text(
        "[policy]\n"
        "extension = 'org.example.fair-policy'\n"
        "fairness_window_seconds = 1200\n"
        "[policy.options]\n"
        "window = 17\n",
        encoding="utf-8",
    )
    config = ServiceConfig.from_file(config_path)
    assert config.policy_extension == "org.example.fair-policy"
    assert config.policy_parameters == {}
    assert config.policy_options == {"window": 17}
    assert config.fairness_window_seconds == 1200


@pytest.mark.parametrize("value", [-0.01, 1.0, float("nan")])
def test_service_config_rejects_invalid_gpu_headroom(value: float) -> None:
    with pytest.raises(ConfigurationError, match="gpu_headroom_fraction"):
        ServiceConfig(gpu_headroom_fraction=value)


def test_service_config_rejects_builtin_parameters_for_an_installed_policy() -> None:
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        ServiceConfig(policy_extension="org.example.policy", policy_parameters={"max_age_boost": 2})


def test_continuation_profile_accepts_a_token_file_reference_only(tmp_path: Path) -> None:
    path = tmp_path / "continuations.toml"
    path.write_text(
        "[continuations.local-frame]\n"
        "base_url = 'http://127.0.0.1:9010'\n"
        "frame_id = 'agent-frame'\n"
        "token_file = '~/.config/frameshift/token'\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    assert load_named_profiles(path, "continuations")["local-frame"]["frame_id"] == "agent-frame"

    path.write_text(
        "[continuations.local-frame]\n"
        "base_url = 'http://127.0.0.1:9010'\n"
        "frame_id = 'agent-frame'\n"
        "token = 'inline-secret'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="secret references"):
        load_named_profiles(path, "continuations")


def test_profile_files_must_be_private_regular_files(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text("[backend_profiles.test]\nprofile_id='test'\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mode 0600"):
        load_named_profiles(path, "backend_profiles")


def test_backend_configuration_keeps_controller_credentials_as_file_references(tmp_path: Path) -> None:
    path = tmp_path / "backends.toml"
    path.write_text(
        "[backend_profiles.llama-main]\n"
        "profile_id = 'llama-main'\n"
        "version = 'llama-v1'\n"
        "execution_profile_id = 'llama-process'\n"
        "health_url = 'http://127.0.0.1:8080/health'\n"
        "slots_url = 'http://127.0.0.1:8080/slots'\n"
        "access_mode = 'frameshift-only'\n"
        "frameshift_control_ref = 'frameshift-local'\n"
        "[frameshift_controls.frameshift-local]\n"
        "base_url = 'http://127.0.0.1:9010'\n"
        "token_file = '~/.config/frameshift/scheduler.token'\n"
        "timeout_seconds = 3.0\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    profiles, controls = load_backend_configuration(path)
    assert profiles["llama-main"]["access_mode"] == "frameshift-only"
    assert controls["frameshift-local"]["token_file"] == "~/.config/frameshift/scheduler.token"

    path.write_text(
        "[backend_profiles.llama-main]\nprofile_id = 'llama-main'\n"
        "[frameshift_controls.frameshift-local]\nbase_url = 'http://127.0.0.1:9010'\n"
        "token = 'not-allowed-inline'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="secret file references"):
        load_backend_configuration(path)
