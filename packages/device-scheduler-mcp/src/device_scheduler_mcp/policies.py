"""Load only the built-in policy or an operator-installed policy entry point."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Mapping

from device_scheduler_core.policy import WeightedFairPolicy

from .config import ConfigurationError, ServiceConfig


POLICY_ENTRYPOINT_GROUP = "device_scheduler.policies"
POLICY_API_VERSION = "1"
BUILTIN_POLICY = "builtin.weighted-fair"


class ConfiguredPolicy:
    """Expose an immutable loader snapshot while delegating policy methods."""

    __slots__ = ("_policy", "_snapshot")

    def __init__(self, policy: Any, snapshot: Mapping[str, Any]) -> None:
        self._policy = policy
        self._snapshot = deepcopy(dict(snapshot))

    @property
    def configuration_snapshot(self) -> dict[str, Any]:
        return deepcopy(self._snapshot)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def api_version(self) -> str:
        return self._policy.api_version

    @property
    def policy_id(self) -> str:
        return self._policy.policy_id

    @property
    def version(self) -> str:
        return self._policy.version

    def choose_next(self, *args: Any, **kwargs: Any) -> Any:
        return self._policy.choose_next(*args, **kwargs)


def load_policy(config: ServiceConfig) -> Any:
    """Construct the configured trusted policy and attach its exact snapshot.

    Job specifications cannot name modules or provide policy code. Extensions
    resolve exclusively through the Python installation's declared entry points.
    """
    if config.policy_extension == BUILTIN_POLICY:
        try:
            policy = WeightedFairPolicy(**dict(config.policy_parameters))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid built-in policy settings: {type(exc).__name__}") from exc
        parameters = {
            "class_weights": dict(policy.class_weights),
            "owner_weights": dict(policy.owner_weights),
            "age_quantum_seconds": policy.age_quantum_seconds,
            "max_age_boost": policy.max_age_boost,
            "large_request_protection_seconds": policy.large_request_protection_seconds,
            "fairness_penalty": policy.fairness_penalty,
        }
        source = {"kind": "builtin", "name": BUILTIN_POLICY}
    else:
        policy, distribution = _load_installed_policy(config.policy_extension, config.policy_options)
        parameters = dict(config.policy_options)
        source = {
            "kind": "entry_point",
            "group": POLICY_ENTRYPOINT_GROUP,
            "name": config.policy_extension,
            **distribution,
        }

    api_version = getattr(policy, "api_version", None)
    policy_id = getattr(policy, "policy_id", None)
    version = getattr(policy, "version", None)
    if api_version != POLICY_API_VERSION:
        raise ConfigurationError(
            f"policy {config.policy_extension!r} declares unsupported API version {api_version!r}"
        )
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ConfigurationError("configured policy must expose a non-empty policy_id")
    if not isinstance(version, str) or not version.strip():
        raise ConfigurationError("configured policy must expose a non-empty string version")
    if not callable(getattr(policy, "choose_next", None)):
        raise ConfigurationError("configured policy does not implement choose_next")

    # The application copies this into durable decision and health evidence.
    snapshot = {
        "api_version": POLICY_API_VERSION,
        "policy_id": policy_id,
        "version": version,
        "source": source,
        "parameters": parameters,
        "fairness_window_seconds": config.fairness_window_seconds,
    }
    return ConfiguredPolicy(policy, snapshot)


def _load_installed_policy(name: str, options: Mapping[str, Any]) -> tuple[Any, dict[str, str]]:
    try:
        candidates = [item for item in entry_points(group=POLICY_ENTRYPOINT_GROUP) if item.name == name]
    except Exception as exc:
        raise ConfigurationError(f"cannot inspect installed scheduler policies: {type(exc).__name__}") from exc
    if len(candidates) != 1:
        raise ConfigurationError(
            f"policy entry point {name!r} must resolve to exactly one installed {POLICY_ENTRYPOINT_GROUP} entry"
        )
    selected = candidates[0]
    if not isinstance(selected, EntryPoint):
        # Entry-point-like test doubles and alternative Python implementations
        # may expose the standard attributes without using the concrete class.
        if not all(hasattr(selected, field) for field in ("name", "load")):
            raise ConfigurationError("installed policy entry point is malformed")
    try:
        factory = selected.load()
        if not callable(factory):
            raise TypeError("entry point must load a policy factory")
        policy = factory(**dict(options))
    except Exception as exc:
        raise ConfigurationError(f"cannot construct installed policy {name!r}: {type(exc).__name__}") from exc
    distribution: dict[str, str] = {}
    dist = getattr(selected, "dist", None)
    if dist is not None:
        dist_name = getattr(dist, "name", None)
        dist_version = getattr(dist, "version", None)
        if isinstance(dist_name, str) and dist_name:
            distribution["distribution"] = dist_name
        if isinstance(dist_version, str) and dist_version:
            distribution["distribution_version"] = dist_version
    return policy, distribution
