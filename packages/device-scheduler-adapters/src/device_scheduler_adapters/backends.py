"""Managed llama.cpp backend lifecycle with pinned profiles and explicit eviction.

This adapter provides process/readiness effects for the handoff coordinator.  It
does not decide whether a handoff is useful: admission closure is called only
with a persisted operation ID, eviction requires an explicit authorization
lookup, and restoration starts the exact pinned profile.  Continuation state
and durable handoff phase transitions remain owned by the core/coordinator.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import quote as urlquote
from urllib.request import Request, urlopen


_GPU_RESOURCE_KINDS = {"gpu", "nvidia_gpu", "cuda_gpu", "cuda_device", "nvidia_cuda_device"}

from device_scheduler_core.models import (
    AdapterCapability,
    BackendSpec,
    EnforcementLevel,
    ExecutionContext,
    ExecutionPurpose,
    ExecutionHandle,
    JobSpec,
)


class BackendAdapterError(RuntimeError):
    """A requested backend operation could not be completed safely."""


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Operator-registered launch and readiness profile for one model server."""

    profile_id: str
    version: str
    execution_profile_id: str
    health_url: str
    expected_model_identity: str | None = None
    inference_base_url: str | None = None
    inference_model_id: str | None = None
    identity_url: str | None = None
    identity_json_path: str | None = None
    slots_url: str | None = None
    access_mode: str | None = None
    frameshift_control_ref: str | None = None
    health_json_path: str | None = "status"
    expected_health_value: Any = "ok"
    readiness_interval_seconds: float = 0.25
    readiness_timeout_seconds: float = 120.0
    stop_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in ("profile_id", "version", "execution_profile_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be a non-empty string")
        self._validate_loopback_url(self.health_url, "health_url")
        if self.identity_url is not None:
            self._validate_loopback_url(self.identity_url, "identity_url")
        if self.slots_url is not None:
            self._validate_loopback_url(self.slots_url, "slots_url")
        if (self.inference_base_url is None) != (self.inference_model_id is None):
            raise ValueError("inference_base_url and inference_model_id must be configured together")
        if self.inference_base_url is not None:
            self._validate_inference_url(self.inference_base_url)
            if self.expected_model_identity is None:
                raise ValueError("an inference endpoint requires a pinned expected_model_identity")
            if (
                not isinstance(self.inference_model_id, str)
                or not self.inference_model_id.strip()
                or "\x00" in self.inference_model_id
            ):
                raise ValueError("inference_model_id must be a non-empty NUL-free string")
        if self.access_mode not in {None, "frameshift-only"}:
            raise ValueError("access_mode must be frameshift-only when configured")
        if self.frameshift_control_ref is not None and (
            not isinstance(self.frameshift_control_ref, str)
            or not self.frameshift_control_ref.strip()
            or "\x00" in self.frameshift_control_ref
        ):
            raise ValueError("frameshift_control_ref must be a non-empty reference")
        if self.access_mode == "frameshift-only" and not self.frameshift_control_ref:
            raise ValueError("frameshift_control_ref is required for frameshift-only access")
        if self.expected_model_identity is not None and not self.identity_json_path:
            raise ValueError("identity_json_path is required when model identity is pinned")
        for name, value, allow_zero in (
            ("readiness_interval_seconds", self.readiness_interval_seconds, False),
            ("readiness_timeout_seconds", self.readiness_timeout_seconds, True),
            ("stop_timeout_seconds", self.stop_timeout_seconds, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                bound = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a finite {bound} number")

    @staticmethod
    def _validate_loopback_url(value: str, name: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{name} must be an HTTP(S) URL")
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = parsed.hostname.lower() == "localhost"
        if not is_loopback:
            raise ValueError(f"{name} must target loopback")

    @staticmethod
    def _validate_inference_url(value: str) -> None:
        parsed = urlparse(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("inference_base_url must be a credential-free HTTP(S) URL without query or fragment")
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = parsed.hostname.lower() == "localhost"
        if not is_loopback:
            raise ValueError("inference_base_url must target loopback")


def _member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _handle_mapping(handle: ExecutionHandle | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    if isinstance(handle, ExecutionHandle):
        return handle.to_dict()
    if isinstance(handle, Mapping):
        return dict(handle)
    return None


def _call_json(reader: Callable[..., Any], url: str, timeout: float) -> tuple[int | None, Any, str | None]:
    try:
        response = reader(url, timeout=timeout)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return None, None, f"health request failed: {type(exc).__name__}"
    if isinstance(response, Mapping):
        status = response.get("status", response.get("status_code"))
        body = response.get("json", response.get("body"))
        error = response.get("error")
        try:
            parsed_status = int(status) if status is not None else None
        except (TypeError, ValueError):
            parsed_status = None
        return parsed_status, body, str(error) if error else None
    status = getattr(response, "status_code", getattr(response, "status", None))
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    body = getattr(response, "json", None)
    if callable(body):
        try:
            body = body()
        except (ValueError, OSError) as exc:
            return status, None, f"health response is not valid JSON: {type(exc).__name__}"
    return status, body, None


def _default_health_reader(url: str, *, timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read(1024 * 1024)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        return {"status": exc.code, "json": body}
    with response:
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return {"status": response.status, "json": None, "error": "health response exceeds 1 MiB"}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        return {"status": response.status, "json": body}


def _json_path(document: Any, path: str | None) -> Any:
    if not path:
        return document
    value = document
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            return None
    return value


def _as_profile(value: BackendProfile | Mapping[str, Any]) -> BackendProfile:
    if isinstance(value, BackendProfile):
        return value
    if not isinstance(value, Mapping):
        raise BackendAdapterError("backend profile resolver returned a non-object")
    try:
        return BackendProfile(**dict(value))
    except (TypeError, ValueError) as exc:
        raise BackendAdapterError(f"invalid registered backend profile: {exc}") from exc


class FrameShiftBackendAdmissionClient:
    """Authenticated client for FrameShift's scheduler-only backend gate."""

    api_version = "1"

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        *,
        timeout_seconds: float = 3.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("FrameShift controller base_url must be HTTP(S)")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
        if not loopback or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("FrameShift controller base_url must be credential-free loopback origin")
        if (
            not isinstance(bearer_token, str)
            or not bearer_token.strip()
            or any(char in bearer_token for char in "\x00\r\n")
        ):
            raise ValueError("FrameShift scheduler bearer token is required")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._timeout = float(timeout_seconds)
        self._opener = opener

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._bearer_token}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            response = self._opener(request, timeout=self._timeout)
            try:
                status = int(getattr(response, "status", getattr(response, "code", 0)))
                raw = response.read(1024 * 1024 + 1)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise BackendAdapterError(f"FrameShift admission request failed: {type(exc).__name__}") from exc
        if len(raw) > 1024 * 1024:
            raise BackendAdapterError("FrameShift admission response exceeds 1 MiB")
        if status != 200:
            raise BackendAdapterError(f"FrameShift admission endpoint returned HTTP {status}")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendAdapterError("FrameShift admission response is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise BackendAdapterError("FrameShift admission response must be a JSON object")
        return dict(document)

    @staticmethod
    def _backend_path(backend_id: str) -> str:
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id is required")
        return f"/v1/backends/{urlquote(backend_id, safe='')}/admission"

    @staticmethod
    def _epoch(value: Any) -> str | int:
        if isinstance(value, bool) or not (
            (isinstance(value, str) and value.strip())
            or (isinstance(value, int) and value >= 0)
        ):
            raise BackendAdapterError("FrameShift admission response has no valid epoch")
        return value

    @staticmethod
    def _generation_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BackendAdapterError("FrameShift admission response has no valid active-generation count")
        return value

    def close_backend_admission(self, backend_id: str, operation_id: str) -> dict[str, Any]:
        """Close FrameShift generation admission and fence offers/leases."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id is required")
        document = self._request(
            "POST",
            self._backend_path(backend_id) + "/close",
            {"operation_id": operation_id},
        )
        if (
            document.get("closed") is not True
            or document.get("backend_id") != backend_id
            or document.get("operation_id") != operation_id
        ):
            raise BackendAdapterError("FrameShift did not verify the requested backend admission close")
        document["epoch"] = self._epoch(document.get("epoch"))
        document["active_generations"] = self._generation_count(document.get("active_generations"))
        if document.get("active_generations_source") != "frameshift-provider-leases-upper-bound":
            raise BackendAdapterError("FrameShift close receipt does not identify the provider-lease upper bound")
        return document

    def backend_admission_state(self, backend_id: str) -> dict[str, Any]:
        """Read gate state and FrameShift provider-lease upper bound."""
        document = self._request("GET", self._backend_path(backend_id))
        if document.get("backend_id") != backend_id or not isinstance(document.get("closed"), bool):
            raise BackendAdapterError("FrameShift returned invalid backend admission state")
        document["epoch"] = self._epoch(document.get("epoch"))
        document["active_generations"] = self._generation_count(document.get("active_generations"))
        if document.get("active_generations_source") != "frameshift-provider-leases-upper-bound":
            raise BackendAdapterError("FrameShift state does not identify the provider-lease upper bound")
        return document

    def open_backend_admission(self, backend_id: str, operation_id: str, epoch: str | int) -> dict[str, Any]:
        """Reopen only the exact close epoch that this handoff fenced."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id is required")
        epoch = self._epoch(epoch)
        document = self._request(
            "POST",
            self._backend_path(backend_id) + "/open",
            {"operation_id": operation_id, "epoch": epoch},
        )
        if (
            document.get("opened") is not True
            or document.get("backend_id") != backend_id
            or document.get("operation_id") != operation_id
            or self._epoch(document.get("epoch")) != epoch
        ):
            raise BackendAdapterError("FrameShift did not verify reopening the requested backend epoch")
        return document


class ManagedLlamaBackendAdapter:
    """Coordinate process execution, loopback readiness and authorized eviction."""

    api_version = "1"
    adapter_id = "managed-llama.cpp"

    def __init__(
        self,
        execution: Any,
        *,
        profile_resolver: Callable[[BackendSpec], BackendProfile | Mapping[str, Any]],
        handle_resolver: Callable[[BackendSpec], ExecutionHandle | Mapping[str, Any] | None],
        frameshift_controls: Mapping[str, FrameShiftBackendAdmissionClient] | None = None,
        eviction_authorizer: Callable[[BackendSpec, str], Mapping[str, Any]] | None = None,
        resource_reclamation_baseline_capturer: Callable[[BackendSpec, str], Mapping[str, Any]] | None = None,
        resource_reclamation_verifier: Callable[[BackendSpec, str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        reclamation_baseline_max_age_seconds: float = 30.0,
        admission_state_reader: Callable[[BackendSpec], Mapping[str, Any]] | None = None,
        health_reader: Callable[..., Any] = _default_health_reader,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        execution_root: str = "~/.local/state/device-scheduler/backends",
        health_request_timeout: float = 2.0,
    ) -> None:
        if (
            isinstance(health_request_timeout, bool)
            or not isinstance(health_request_timeout, (int, float))
            or not math.isfinite(health_request_timeout)
            or health_request_timeout <= 0
        ):
            raise ValueError("health_request_timeout must be a finite positive number")
        if (
            isinstance(reclamation_baseline_max_age_seconds, bool)
            or not isinstance(reclamation_baseline_max_age_seconds, (int, float))
            or not math.isfinite(reclamation_baseline_max_age_seconds)
            or reclamation_baseline_max_age_seconds <= 0
        ):
            raise ValueError("reclamation_baseline_max_age_seconds must be finite and positive")
        self.execution = execution
        self._profile_resolver = profile_resolver
        self._handle_resolver = handle_resolver
        self._frameshift_controls = dict(frameshift_controls or {})
        self._eviction_authorizer = eviction_authorizer
        self._resource_reclamation_baseline_capturer = resource_reclamation_baseline_capturer
        self._resource_reclamation_verifier = resource_reclamation_verifier
        self._reclamation_baseline_max_age_seconds = float(reclamation_baseline_max_age_seconds)
        self._admission_state_reader = admission_state_reader
        self._health_reader = health_reader
        self._monotonic = monotonic
        self._sleep = sleeper
        self._execution_root = execution_root
        self._health_request_timeout = health_request_timeout

    def capabilities(self) -> Sequence[AdapterCapability]:
        return (
            AdapterCapability(
                "managed-llama.cpp-launch",
                True,
                EnforcementLevel.ADVISORY,
                {"backend_kind": "llama.cpp-compatible-http-service", "source": "registered-execution-profile"},
            ),
            AdapterCapability(
                "bounded-backend-readiness",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "loopback-http", "requires_execution_running": True},
            ),
            AdapterCapability(
                "pinned-model-identity-check",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"conditional": "profile.expected_model_identity is configured"},
            ),
            AdapterCapability(
                "explicit-grant-model-eviction",
                self._eviction_authorizer is not None,
                EnforcementLevel.ENFORCED if self._eviction_authorizer is not None else None,
                {"requires": "core eviction-grant verifier"},
            ),
            AdapterCapability(
                "admission-close-and-reopen",
                bool(self._frameshift_controls),
                EnforcementLevel.ENFORCED if self._frameshift_controls else None,
                {"source": "authenticated FrameShift scheduler-admission API", "standing_contract": "frameshift-only"},
            ),
            AdapterCapability(
                "llama-active-slot-observation",
                bool(self._frameshift_controls),
                EnforcementLevel.OBSERVATIONAL if self._frameshift_controls else None,
                {"source": "loopback llama.cpp GET /slots; requires --slots", "combined_with": "FrameShift provider-lease upper bound"},
            ),
            AdapterCapability(
                "resource-reclamation-verification",
                self._resource_reclamation_verifier is not None,
                EnforcementLevel.OBSERVATIONAL if self._resource_reclamation_verifier is not None else None,
                {"does_not_infer_reclamation_from_process_exit": True},
            ),
            AdapterCapability(
                "pre-eviction-resource-baseline",
                self._resource_reclamation_baseline_capturer is not None,
                EnforcementLevel.OBSERVATIONAL if self._resource_reclamation_baseline_capturer is not None else None,
                {"requires": "fresh per-GPU memory and exact backend cgroup PID evidence"},
            ),
        )

    def _profile(self, backend: BackendSpec) -> BackendProfile:
        if backend.adapter_id != self.adapter_id:
            raise BackendAdapterError("backend adapter_id does not match this adapter")
        try:
            resolved = self._profile_resolver(backend)
        except (LookupError, OSError, TimeoutError) as exc:
            raise BackendAdapterError(f"registered backend profile lookup failed: {type(exc).__name__}") from exc
        profile = _as_profile(resolved)
        if profile.profile_id != backend.profile_id:
            raise BackendAdapterError("resolved profile ID does not match backend registration")
        pinned_version = backend.pinned_configuration.get("profile_version")
        if pinned_version is None:
            raise BackendAdapterError("backend has no pinned profile_version")
        if profile.version != pinned_version:
            raise BackendAdapterError(
                f"registered backend profile version {profile.version!r} does not match pinned {pinned_version!r}"
            )
        pinned_identity = backend.pinned_configuration.get("model_identity")
        if pinned_identity is not None and profile.expected_model_identity != pinned_identity:
            raise BackendAdapterError("registered profile model identity does not match pinned configuration")
        return profile

    def _control_for(self, profile: BackendProfile) -> FrameShiftBackendAdmissionClient:
        if profile.access_mode != "frameshift-only":
            raise BackendAdapterError("backend profile must declare the frameshift-only access contract")
        if not profile.slots_url:
            raise BackendAdapterError("backend profile must configure the loopback llama /slots endpoint")
        control_ref = profile.frameshift_control_ref
        control = self._frameshift_controls.get(control_ref or "")
        if control is None:
            raise BackendAdapterError("backend profile FrameShift scheduler controller is unavailable")
        return control

    def _admission_observation(
        self, backend: BackendSpec, profile: BackendProfile
    ) -> dict[str, Any]:
        try:
            if self._admission_state_reader is not None:
                document = self._admission_state_reader(backend)
            else:
                document = self._control_for(profile).backend_admission_state(backend.backend_id)
            if not isinstance(document, Mapping) or document.get("backend_id") != backend.backend_id:
                raise BackendAdapterError("FrameShift admission state identity is invalid")
            closed = document.get("closed")
            if not isinstance(closed, bool):
                raise BackendAdapterError("FrameShift admission state is unknown")
            epoch = document.get("epoch")
            if isinstance(epoch, bool) or not (
                (isinstance(epoch, str) and epoch.strip())
                or (isinstance(epoch, int) and epoch >= 0)
            ):
                raise BackendAdapterError("FrameShift admission epoch is unknown")
            operation_id = document.get("operation_id")
            if operation_id is not None and (not isinstance(operation_id, str) or not operation_id.strip()):
                raise BackendAdapterError("FrameShift admission operation identity is invalid")
            active = document.get("active_generations")
            if isinstance(active, bool) or not isinstance(active, int) or active < 0:
                raise BackendAdapterError("FrameShift provider lease count is unknown")
            active_source = document.get("active_generations_source")
            if active_source != "frameshift-provider-leases-upper-bound":
                raise BackendAdapterError("FrameShift provider lease count has no trusted source label")
            return {
                "state": "observed",
                "admission_state": "closed" if closed else "open",
                "admission_closed": closed,
                "admission_operation_id": operation_id,
                "admission_epoch": epoch,
                "active_generations": active,
                "active_generations_source": active_source,
                "document": dict(document),
            }
        except (BackendAdapterError, OSError, TimeoutError, LookupError) as exc:
            return {
                "state": "unknown",
                "admission_state": "unknown",
                "admission_closed": None,
                "admission_operation_id": None,
                "admission_epoch": None,
                "active_generations": None,
                "reason": str(exc) if isinstance(exc, BackendAdapterError) else type(exc).__name__,
            }

    def _active_generations(self, backend: BackendSpec) -> Mapping[str, Any]:
        """Combine FrameShift's lease upper bound with llama's physical slot view."""
        admission_observation: dict[str, Any] | None = None
        try:
            profile = self._profile(backend)
            admission_observation = self._admission_observation(backend, profile)
            if admission_observation.get("state") != "observed":
                raise BackendAdapterError("FrameShift backend admission state is unknown")
            provider_count = admission_observation.get("active_generations")
            if isinstance(provider_count, bool) or not isinstance(provider_count, int) or provider_count < 0:
                raise BackendAdapterError("FrameShift active-generation upper bound is unknown")
            status, slots, error = _call_json(
                self._health_reader, profile.slots_url, self._health_request_timeout
            )
            if error or status != 200:
                raise BackendAdapterError(error or f"llama /slots returned HTTP {status}")
            if not isinstance(slots, list) or not slots:
                raise BackendAdapterError("llama /slots response must be a non-empty array")
            slot_ids: set[int] = set()
            processing = 0
            for slot in slots:
                if not isinstance(slot, Mapping):
                    raise BackendAdapterError("llama /slots contains a non-object entry")
                slot_id = slot.get("id")
                is_processing = slot.get("is_processing")
                if (
                    isinstance(slot_id, bool)
                    or not isinstance(slot_id, int)
                    or slot_id < 0
                    or slot_id in slot_ids
                    or not isinstance(is_processing, bool)
                ):
                    raise BackendAdapterError("llama /slots contains malformed or duplicate slot identity/state")
                slot_ids.add(slot_id)
                processing += int(is_processing)
            return {
                "state": "observed",
                "active_generations": max(provider_count, processing),
                "frameshift_provider_lease_upper_bound": provider_count,
                "frameshift_state": admission_observation,
                "admission_observation": admission_observation,
                "llama_slots": {
                    "total": len(slots),
                    "processing": processing,
                    "url": profile.slots_url,
                },
                "combination": "max(provider-lease-upper-bound, physical-non-idle-slot-count)",
                "limitations": [
                    "FrameShift count covers only registered provider users under the frameshift-only standing contract",
                    "llama /slots is observational and requires the server to start with --slots",
                    "same-administrative-domain access is not hostile-client isolation",
                ],
            }
        except (BackendAdapterError, OSError, TimeoutError) as exc:
            return {
                "state": "unknown",
                "active_generations": None,
                "admission_observation": admission_observation
                or {
                    "state": "unknown",
                    "admission_state": "unknown",
                    "admission_closed": None,
                    "admission_operation_id": None,
                    "admission_epoch": None,
                },
                "reason": str(exc) if isinstance(exc, BackendAdapterError) else type(exc).__name__,
            }

    def _attempt_id(self, backend: BackendSpec, operation_id: str, action: str) -> str:
        digest = hashlib.sha256(
            f"{backend.backend_id}\0{operation_id}\0{action}".encode("utf-8")
        ).hexdigest()[:32]
        return f"backend-{action}-{digest}"

    @staticmethod
    def _parse_observation_time(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _execution_identity(handle: ExecutionHandle | Mapping[str, Any]) -> dict[str, str] | None:
        """Extract the exact systemd invocation identity from a durable handle."""
        if isinstance(handle, ExecutionHandle):
            metadata = handle.metadata
            unit_name = handle.external_id
        elif isinstance(handle, Mapping):
            metadata = handle.get("metadata") or {}
            unit_name = handle.get("unit_name") or handle.get("external_id")
        else:
            return None
        if not isinstance(metadata, Mapping):
            return None
        invocation_id = handle.get("invocation_id") if isinstance(handle, Mapping) else None
        invocation_id = invocation_id or metadata.get("invocation_id")
        control_group = handle.get("control_group") if isinstance(handle, Mapping) else None
        control_group = control_group or metadata.get("control_group")
        if any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in (unit_name, invocation_id, control_group)):
            return None
        if not unit_name.endswith(".service") or not control_group.startswith("/"):
            return None
        if ".." in control_group.split("/"):
            return None
        return {
            "unit_name": unit_name,
            "invocation_id": invocation_id,
            "control_group": control_group,
        }

    def _validate_reclamation_baseline(
        self,
        raw: Mapping[str, Any],
        *,
        backend: BackendSpec,
        profile: BackendProfile,
        expected_operation_id: str | None = None,
        expected_execution_identity: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Validate a persisted baseline without treating unknown evidence as zero."""
        if (
            raw.get("state") != "observed"
            or raw.get("backend_id") != backend.backend_id
            or not isinstance(raw.get("operation_id"), str)
            or not raw.get("operation_id", "").strip()
            or (expected_operation_id is not None and raw.get("operation_id") != expected_operation_id)
            or raw.get("profile_id") != profile.profile_id
            or raw.get("profile_version") != profile.version
            or raw.get("source") != "nvidia-smi+invocation-verified-systemd-unit-cgroups"
        ):
            raise ValueError("baseline receipt identity or evidence source is not verified")
        execution_identity = raw.get("execution_identity")
        if not isinstance(execution_identity, Mapping) or any(
            not isinstance(execution_identity.get(field), str)
            or not execution_identity[field].strip()
            or "\x00" in execution_identity[field]
            for field in ("unit_name", "invocation_id", "control_group")
        ):
            raise ValueError("baseline receipt must identify the exact managed execution")
        if not str(execution_identity["unit_name"]).endswith(".service") or not str(
            execution_identity["control_group"]
        ).startswith("/") or ".." in str(execution_identity["control_group"]).split("/"):
            raise ValueError("baseline execution identity contains an invalid unit or cgroup")
        if raw.get("process_mapping_state") != "observed":
            raise ValueError("baseline process mapping is not positively observed")
        normalized_identity = {
            field: str(execution_identity[field])
            for field in ("unit_name", "invocation_id", "control_group")
        }
        if expected_execution_identity is not None and normalized_identity != dict(expected_execution_identity):
            raise ValueError("baseline does not match the current managed execution invocation")

        captured_at = self._parse_observation_time(raw.get("observed_at"))
        now = datetime.now(timezone.utc)
        if captured_at is None or (now - captured_at).total_seconds() < 0 or (
            now - captured_at
        ).total_seconds() > self._reclamation_baseline_max_age_seconds:
            raise ValueError("baseline receipt is stale or has no valid timestamp")
        raw_gpus = raw.get("gpus")
        if isinstance(raw_gpus, (str, bytes)) or not isinstance(raw_gpus, Sequence) or not raw_gpus:
            raise ValueError("baseline receipt has no GPU observations")
        gpus: list[dict[str, Any]] = []
        resource_ids: set[str] = set()
        gpu_uuids: set[str] = set()
        for gpu in raw_gpus:
            if not isinstance(gpu, Mapping):
                raise ValueError("baseline GPU observation is not an object")
            resource_id = gpu.get("resource_id")
            gpu_uuid = gpu.get("gpu_uuid")
            free = gpu.get("memory_free_bytes")
            used = gpu.get("used_gpu_memory_bytes")
            if (
                not isinstance(resource_id, str)
                or not resource_id.strip()
                or resource_id in resource_ids
                or not isinstance(gpu_uuid, str)
                or not gpu_uuid.strip()
                or gpu_uuid in gpu_uuids
                or isinstance(free, bool)
                or not isinstance(free, int)
                or free < 0
                or isinstance(used, bool)
                or not isinstance(used, int)
                or used <= 0
            ):
                raise ValueError("baseline GPU identity or memory observation is invalid")
            inventory_at = self._parse_observation_time(gpu.get("inventory_observed_at"))
            process_at = self._parse_observation_time(gpu.get("process_observed_at"))
            if inventory_at is None or process_at is None or any(
                (now - stamp).total_seconds() < 0
                or (now - stamp).total_seconds() > self._reclamation_baseline_max_age_seconds
                for stamp in (inventory_at, process_at)
            ):
                raise ValueError("baseline GPU telemetry is stale or has no valid timestamp")
            raw_pids = gpu.get("process_ids")
            backend_pids = gpu.get("backend_process_ids")
            if any(
                isinstance(items, (str, bytes))
                or not isinstance(items, Sequence)
                or not items
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in items)
                or len(set(items)) != len(items)
                for items in (raw_pids, backend_pids)
            ):
                raise ValueError("baseline GPU process identities are invalid")
            if set(raw_pids) != set(backend_pids):
                raise ValueError("unmanaged or unattributed GPU processes prevent a reclamation baseline")
            resource_ids.add(resource_id)
            gpu_uuids.add(gpu_uuid)
            gpus.append(
                {
                    "resource_id": resource_id,
                    "gpu_uuid": gpu_uuid,
                    "memory_free_bytes": free,
                    "used_gpu_memory_bytes": used,
                    "backend_process_ids": sorted(backend_pids),
                    "process_ids": sorted(raw_pids),
                    "inventory_observed_at": inventory_at.isoformat(),
                    "process_observed_at": process_at.isoformat(),
                }
            )
        evidence = raw.get("evidence")
        if not isinstance(evidence, (Mapping, Sequence)) or isinstance(evidence, (str, bytes)) or not evidence:
            raise ValueError("baseline receipt has no structured evidence")
        limitations = raw.get("limitations", ())
        if isinstance(limitations, (str, bytes)) or not isinstance(limitations, Sequence):
            raise ValueError("baseline limitations must be a sequence")
        return {
            "state": "observed",
            "backend_id": backend.backend_id,
            "operation_id": raw["operation_id"],
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "observed_at": captured_at.isoformat(),
            "source": "nvidia-smi+invocation-verified-systemd-unit-cgroups",
            "execution_identity": normalized_identity,
            "process_mapping_state": "observed",
            "gpus": gpus,
            "evidence": evidence,
            "limitations": list(limitations),
        }

    def capture_reclamation_baseline(
        self, backend: BackendSpec, *, operation_id: str
    ) -> Mapping[str, Any]:
        """Capture trusted, fresh GPU evidence before an authorized backend stop."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("baseline capture requires a non-empty operation_id")
        if self._resource_reclamation_baseline_capturer is None:
            return {
                "state": "unsupported",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "reason": "no resource reclamation baseline capturer is configured",
            }
        try:
            profile = self._profile(backend)
            raw = self._resource_reclamation_baseline_capturer(backend, operation_id)
        except (BackendAdapterError, LookupError, OSError, TimeoutError) as exc:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "reason": str(exc) if isinstance(exc, BackendAdapterError) else type(exc).__name__,
            }
        if not isinstance(raw, Mapping):
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "reason": "baseline capturer returned a non-object receipt",
            }
        handle, handle_error = self._resolve_handle(backend)
        actual_identity = self._execution_identity(handle) if handle is not None else None
        if actual_identity is None:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "reason": handle_error or "current execution identity is not verifiable",
            }
        try:
            return self._validate_reclamation_baseline(
                raw,
                backend=backend,
                profile=profile,
                expected_operation_id=operation_id,
                expected_execution_identity=actual_identity,
            )
        except (TypeError, ValueError) as exc:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "reason": f"baseline receipt validation failed: {exc}",
            }

    def _resolve_handle(self, backend: BackendSpec) -> tuple[ExecutionHandle | Mapping[str, Any] | None, str | None]:
        try:
            handle = self._handle_resolver(backend)
        except (LookupError, OSError, TimeoutError) as exc:
            return None, f"execution handle lookup failed: {type(exc).__name__}"
        if isinstance(handle, Mapping) and "adapter_id" in handle:
            try:
                handle = ExecutionHandle.from_dict(handle)
            except (TypeError, ValueError) as exc:
                return None, f"execution handle is invalid: {type(exc).__name__}"
        return handle, None if handle is not None else "backend execution handle is unavailable"

    def _health_once(self, profile: BackendProfile, *, timeout_seconds: float) -> dict[str, Any]:
        status, health_document, health_error = _call_json(
            self._health_reader, profile.health_url, min(timeout_seconds, self._health_request_timeout)
        )
        if health_error:
            return {"ready": False, "state": "unknown", "reason": health_error, "identity_verified": False}
        if status != 200:
            return {
                "ready": False,
                "state": "starting" if status is not None and status < 500 else "unknown",
                "reason": f"health endpoint returned HTTP {status}",
                "http_status": status,
                "identity_verified": False,
            }
        if profile.health_json_path and _json_path(health_document, profile.health_json_path) != profile.expected_health_value:
            return {
                "ready": False,
                "state": "starting",
                "reason": "health response does not report the configured healthy value",
                "http_status": status,
                "identity_verified": False,
            }

        if profile.expected_model_identity is None:
            return {
                "ready": True,
                "state": "ready",
                "http_status": status,
                "identity_verified": False,
                "identity_reason": "model identity is not configured for this profile",
                "health": health_document,
            }
        identity_url = profile.identity_url or profile.health_url
        if identity_url == profile.health_url:
            identity_status, identity_document, identity_error = status, health_document, None
        else:
            identity_status, identity_document, identity_error = _call_json(
                self._health_reader,
                identity_url,
                min(timeout_seconds, self._health_request_timeout),
            )
        if identity_error or identity_status != 200:
            return {
                "ready": False,
                "state": "unknown",
                "reason": identity_error or f"identity endpoint returned HTTP {identity_status}",
                "identity_verified": False,
                "http_status": status,
            }
        actual_identity = _json_path(identity_document, profile.identity_json_path)
        if actual_identity != profile.expected_model_identity:
            return {
                "ready": False,
                "state": "failed",
                "reason": "served model identity does not match the pinned backend profile",
                "expected_model_identity": profile.expected_model_identity,
                "observed_model_identity": actual_identity,
                "identity_verified": False,
                "http_status": status,
            }
        return {
            "ready": True,
            "state": "ready",
            "http_status": status,
            "identity_verified": True,
            "model_identity": actual_identity,
            "health": health_document,
        }

    def inspect(self, backend: BackendSpec) -> Mapping[str, Any]:
        """Inspect the persisted service handle, health and active generations."""
        try:
            profile = self._profile(backend)
        except BackendAdapterError as exc:
            return {"state": "unknown", "backend_id": backend.backend_id, "reason": str(exc)}
        handle, handle_error = self._resolve_handle(backend)
        if handle is None:
            admission = self._admission_observation(backend, profile)
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "reason": handle_error,
                "active_generations": None,
                "execution_handle": None,
                "admission_state": admission.get("admission_state", "unknown"),
                "admission_closed": admission.get("admission_closed"),
                "admission_operation_id": admission.get("admission_operation_id"),
                "admission_epoch": admission.get("admission_epoch"),
                "admission_observation": admission,
            }
        try:
            execution_observation = self.execution.inspect(handle)
        except (OSError, TimeoutError) as exc:
            admission = self._admission_observation(backend, profile)
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "reason": f"execution inspection failed: {type(exc).__name__}",
                "active_generations": None,
                "execution_handle": _handle_mapping(handle),
                "admission_state": admission.get("admission_state", "unknown"),
                "admission_closed": admission.get("admission_closed"),
                "admission_operation_id": admission.get("admission_operation_id"),
                "admission_epoch": admission.get("admission_epoch"),
                "admission_observation": admission,
            }
        state = _member(execution_observation, "state", "unknown")
        health = None
        if state == "running":
            health = self._health_once(profile, timeout_seconds=0.1)
            endpoint = self._endpoint(profile, health)
            if endpoint is not None:
                health = {**health, "endpoint": endpoint}
            if not health.get("ready"):
                state = health.get("state", "unknown")
        generations = self._active_generations(backend)
        admission = generations.get("admission_observation", {})
        return {
            "state": state,
            "backend_id": backend.backend_id,
            "profile_id": backend.profile_id,
            "profile_version": backend.pinned_configuration.get("profile_version"),
            "execution_handle": _handle_mapping(handle),
            "execution": execution_observation,
            "readiness": health,
            "active_generations": generations.get("active_generations"),
            "generation_state": generations.get("state", "unknown"),
            "generation_observation": dict(generations),
            "admission_state": admission.get("admission_state", "unknown"),
            "admission_closed": admission.get("admission_closed"),
            "admission_operation_id": admission.get("admission_operation_id"),
            "admission_epoch": admission.get("admission_epoch"),
            "admission_observation": dict(admission),
            "evidence": {
                "adapter": self.adapter_id,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def drain(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Close new backend admission; continuation suspension is coordinated separately."""
        try:
            profile = self._profile(backend)
            control = self._control_for(profile)
        except BackendAdapterError as exc:
            return {
                "state": "blocked",
                "operation_id": operation_id,
                "admission_closed": False,
                "reason": str(exc),
            }
        if (
            not isinstance(operation_id, str)
            or not operation_id.strip()
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("drain requires operation_id and a positive timeout")
        try:
            receipt = control.close_backend_admission(backend.backend_id, operation_id)
        except (BackendAdapterError, OSError, TimeoutError) as exc:
            return {
                "state": "unknown",
                "operation_id": operation_id,
                "admission_closed": False,
                "reason": f"admission closure failed: {type(exc).__name__}",
            }
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("closed") is not True
            or receipt.get("backend_id") != backend.backend_id
            or receipt.get("operation_id") != operation_id
        ):
            return {
                "state": "unknown",
                "operation_id": operation_id,
                "admission_closed": False,
                "close_receipt": dict(receipt) if isinstance(receipt, Mapping) else None,
                "reason": "admission controller did not verify closed state",
            }
        return {
            "state": "admission_closed",
            "operation_id": operation_id,
            "admission_closed": True,
            "close_receipt": dict(receipt),
            "continuations_suspended": False,
            "evidence": {
                "adapter": self.adapter_id,
                "note": "coordinator must suspend registered consumers and inspect active generations before eviction",
            },
        }

    def prepare_initial_start(
        self, backend: BackendSpec, *, operation_id: str
    ) -> Mapping[str, Any]:
        """Fence a registered FrameShift gate before the first backend launch.

        Backends without a standing FrameShift contract may still be launched
        and checked for readiness, but this adapter will not claim that they can
        be drained or evicted safely.
        """
        expected_operation_id = f"backend-start:{backend.backend_id}"
        if operation_id != expected_operation_id:
            raise ValueError(f"initial start requires operation_id={expected_operation_id!r}")
        try:
            profile = self._profile(backend)
        except BackendAdapterError as exc:
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "admission_closed": False,
                "reason": str(exc),
            }
        if profile.access_mode != "frameshift-only":
            return {
                "state": "not_required",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "admission_supported": False,
                "eviction_supported": False,
                "admission_closed": False,
                "active_generations": None,
                "reason": "profile has no standing FrameShift-only consumer contract",
            }
        try:
            control = self._control_for(profile)
            before = control.backend_admission_state(backend.backend_id)
            if not isinstance(before, Mapping) or before.get("backend_id") != backend.backend_id:
                raise BackendAdapterError("FrameShift startup admission state has the wrong backend identity")
            if not isinstance(before.get("closed"), bool):
                raise BackendAdapterError("FrameShift startup admission state is unknown")
            before_operation = before.get("operation_id")
            if before_operation is not None and (
                not isinstance(before_operation, str) or not before_operation.strip()
            ):
                raise BackendAdapterError("FrameShift startup operation identity is invalid")
            epoch = before.get("epoch")
            if isinstance(epoch, bool) or not (
                (isinstance(epoch, str) and epoch.strip())
                or (isinstance(epoch, int) and epoch >= 0)
            ):
                raise BackendAdapterError("FrameShift startup admission epoch is unknown")

            if before.get("closed") is True:
                if before_operation not in {None, operation_id}:
                    return {
                        "state": "blocked",
                        "backend_id": backend.backend_id,
                        "operation_id": operation_id,
                        "admission_closed": True,
                        "admission_epoch": epoch,
                        "current_operation_id": before_operation,
                        "reason": "a different durable operation already owns the closed backend gate",
                    }
                if before_operation is None:
                    # An unconfigured FrameShift gate is initially closed. Ask
                    # FrameShift to bind that state to the durable startup
                    # operation so the later open is fenced to this epoch.
                    close_receipt = control.close_backend_admission(
                        backend.backend_id, operation_id
                    )
                else:
                    close_receipt = {
                        "closed": True,
                        "backend_id": backend.backend_id,
                        "operation_id": operation_id,
                        "epoch": epoch,
                    }
            else:
                if before_operation == operation_id:
                    return {
                        "state": "blocked",
                        "backend_id": backend.backend_id,
                        "operation_id": operation_id,
                        "admission_closed": False,
                        "admission_epoch": epoch,
                        "reason": "startup operation already opened this backend gate; reconcile the prior start",
                    }
                close_receipt = control.close_backend_admission(
                    backend.backend_id, operation_id
                )

            if (
                not isinstance(close_receipt, Mapping)
                or close_receipt.get("closed") is not True
                or close_receipt.get("backend_id") != backend.backend_id
                or close_receipt.get("operation_id") != operation_id
            ):
                return {
                    "state": "unknown",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "admission_closed": False,
                    "close_receipt": dict(close_receipt) if isinstance(close_receipt, Mapping) else None,
                    "reason": "FrameShift did not verify the startup admission close",
                }
            closed_epoch = close_receipt.get("epoch")
            if isinstance(closed_epoch, bool) or not (
                (isinstance(closed_epoch, str) and closed_epoch.strip())
                or (isinstance(closed_epoch, int) and closed_epoch >= 0)
            ):
                raise BackendAdapterError("FrameShift close receipt has no valid startup epoch")

            after = control.backend_admission_state(backend.backend_id)
            if (
                not isinstance(after, Mapping)
                or after.get("backend_id") != backend.backend_id
                or after.get("closed") is not True
                or after.get("operation_id") != operation_id
                or after.get("epoch") != closed_epoch
            ):
                raise BackendAdapterError("FrameShift startup gate changed before verification")
            active = after.get("active_generations")
            if (
                isinstance(active, bool)
                or not isinstance(active, int)
                or active < 0
                or after.get("active_generations_source")
                != "frameshift-provider-leases-upper-bound"
            ):
                raise BackendAdapterError("FrameShift startup provider lease count is unknown")
            if active != 0:
                return {
                    "state": "blocked",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "admission_closed": True,
                    "admission_epoch": closed_epoch,
                    "active_generations": active,
                    "active_generations_source": "frameshift-provider-leases-upper-bound",
                    "close_receipt": dict(close_receipt),
                    "reason": "registered FrameShift consumers are still active",
                }
            return {
                "state": "admission_closed",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "admission_supported": True,
                "eviction_supported": True,
                "admission_closed": True,
                "admission_epoch": closed_epoch,
                "active_generations": 0,
                "active_generations_source": "frameshift-provider-leases-upper-bound",
                "close_receipt": dict(close_receipt),
                "evidence": {"source": "authenticated-frameshift-admission-api", "rechecked": True},
            }
        except (BackendAdapterError, OSError, TimeoutError, LookupError) as exc:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "admission_closed": False,
                "reason": str(exc) if isinstance(exc, BackendAdapterError) else type(exc).__name__,
            }

    @staticmethod
    def _backend_execution_context(
        backend: BackendSpec,
        value: ExecutionContext | Mapping[str, Any] | None,
    ) -> tuple[ExecutionContext | None, str | None]:
        if value is None:
            return None, "backend restoration requires an actual protected resource allocation context"
        if isinstance(value, ExecutionContext):
            context = value
        elif isinstance(value, Mapping):
            try:
                context = ExecutionContext.from_dict(value)
            except (TypeError, ValueError) as exc:
                return None, f"backend execution context is invalid: {exc}"
        else:
            return None, "backend execution context must be a typed object"
        if context.purpose != ExecutionPurpose.BACKEND:
            return None, "backend restoration requires execution purpose 'backend'"
        if not context.claim_id or context.fencing_token is None:
            return None, "backend restoration requires a protected claim ID and fencing token"
        if not context.allocations:
            return None, "backend restoration has no granted allocations"
        resources = {resource.resource_id: resource for resource in context.resources}
        allocations_by_kind: dict[str, list[Any]] = {}
        for allocation in context.allocations:
            resource = resources.get(allocation.resource_id)
            if resource is None or resource.host_id != backend.host_id:
                return None, "backend allocation descriptors must belong to the registered host"
            allocations_by_kind.setdefault(resource.kind.casefold(), []).append(allocation)

        def matching_allocations(kind: str) -> list[Any]:
            normalized = kind.casefold()
            if normalized in _GPU_RESOURCE_KINDS:
                return [
                    allocation
                    for candidate_kind, allocations in allocations_by_kind.items()
                    if candidate_kind in _GPU_RESOURCE_KINDS
                    for allocation in allocations
                ]
            return allocations_by_kind.get(normalized, [])

        for requirement in backend.residency_requirements:
            candidates = matching_allocations(requirement.kind)
            if not candidates:
                return None, f"backend reservation does not cover required resource kind {requirement.kind}"
            if requirement.mode.value == "exclusive" and not any(
                allocation.mode.value == "exclusive" for allocation in candidates
            ):
                return None, f"backend reservation does not cover exclusive {requirement.kind} access"
            for quantity, needed in requirement.quantities.items():
                granted = sum(
                    allocation.quantities.get(quantity, 0.0) for allocation in candidates
                )
                if granted < needed:
                    return None, f"backend reservation does not cover required {requirement.kind} {quantity}"
        return context, None

    @staticmethod
    def _endpoint(profile: BackendProfile, readiness: Mapping[str, Any]) -> dict[str, str] | None:
        if (
            not readiness.get("ready")
            or readiness.get("identity_verified") is not True
            or profile.inference_base_url is None
            or profile.inference_model_id is None
        ):
            return None
        return {
            "server_kind": "llama.cpp",
            "protocol": "openai-compatible",
            "base_url": profile.inference_base_url,
            "model_id": profile.inference_model_id,
        }

    def evict(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        reclamation_baseline: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Stop an owned backend only after a core grant and report reclamation evidence."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("evict requires a non-empty operation_id")
        try:
            profile = self._profile(backend)
        except BackendAdapterError as exc:
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "reason": str(exc),
            }
        if not isinstance(reclamation_baseline, Mapping):
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "reason": "a persisted pre-eviction resource reclamation baseline is required",
            }
        handle, handle_error = self._resolve_handle(backend)
        actual_identity = self._execution_identity(handle) if handle is not None else None
        if actual_identity is None:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "reason": handle_error or "current execution identity is not verifiable",
            }
        try:
            validated_baseline = self._validate_reclamation_baseline(
                reclamation_baseline,
                backend=backend,
                profile=profile,
                expected_execution_identity=actual_identity,
            )
        except (TypeError, ValueError) as exc:
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "reason": f"pre-eviction reclamation baseline validation failed: {exc}",
            }
        if backend.eviction_contract.model_eviction_requires_explicit_grant:
            if self._eviction_authorizer is None:
                return {
                    "state": "blocked",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "cleanup_verified": False,
                    "reason": "no core eviction-grant verifier is configured",
                }
            try:
                authorization = self._eviction_authorizer(backend, operation_id)
            except (LookupError, OSError) as exc:
                return {
                    "state": "unknown",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "cleanup_verified": False,
                    "reason": f"eviction grant lookup failed: {type(exc).__name__}",
                }
            if not isinstance(authorization, Mapping) or authorization.get("authorized") is not True or not authorization.get("grant_id"):
                return {
                    "state": "blocked",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "cleanup_verified": False,
                    "authorization": dict(authorization) if isinstance(authorization, Mapping) else None,
                    "reason": "no valid explicit model-eviction grant was confirmed",
                }
        else:
            authorization = {"authorized": True, "basis": "backend eviction contract"}

        try:
            control = self._control_for(profile)
        except BackendAdapterError as exc:
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "reason": str(exc),
            }
        try:
            closed = control.close_backend_admission(backend.backend_id, operation_id)
        except (BackendAdapterError, OSError, TimeoutError) as exc:
            closed = {"closed": False, "reason": f"admission closure failed: {type(exc).__name__}"}
        if (
            not isinstance(closed, Mapping)
            or closed.get("closed") is not True
            or closed.get("backend_id") != backend.backend_id
            or closed.get("operation_id") != operation_id
        ):
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "close_receipt": dict(closed) if isinstance(closed, Mapping) else None,
                "reason": "backend admission is not verified closed",
            }
        active = self._active_generations(backend)
        active_count = active.get("active_generations") if isinstance(active, Mapping) else None
        active_state = active.get("state") if isinstance(active, Mapping) else None
        admission = active.get("admission_observation") if isinstance(active, Mapping) else None
        if (
            active_state not in {"observed", "ready", "idle"}
            or isinstance(active_count, bool)
            or not isinstance(active_count, int)
            or active_count != 0
            or not isinstance(admission, Mapping)
            or admission.get("admission_closed") is not True
            or admission.get("admission_operation_id") != operation_id
        ):
            return {
                "state": "blocked" if isinstance(active_count, int) and active_count > 0 else "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "close_receipt": dict(closed),
                "active_generation_observation": dict(active) if isinstance(active, Mapping) else None,
                "reason": "active generations are nonzero or cannot be verified absent",
            }

        try:
            execution_receipt = self.execution.cancel(
                handle,
                operation_id=operation_id,
                timeout_seconds=profile_stop_timeout(profile),
            )
        except Exception as exc:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "cleanup_verified": False,
                "close_receipt": dict(closed),
                "active_generation_observation": dict(active),
                "reason": f"backend stop requires reconciliation: {type(exc).__name__}",
            }
        cleanup_verified = bool(_member(execution_receipt, "cleanup_verified", False))
        if not cleanup_verified:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "authorization": dict(authorization),
                "execution_receipt": dict(execution_receipt),
                "close_receipt": dict(closed),
                "active_generation_observation": dict(active),
                "cleanup_verified": False,
                "resource_reclamation": {"state": "unknown", "reason": "execution tree cleanup not verified"},
            }

        if self._resource_reclamation_verifier is None:
            reclamation: Mapping[str, Any] = {
                "state": "unknown",
                "reason": "no resource reclamation verifier is configured",
                "enforcement": "not established",
            }
        else:
            try:
                reclamation = self._resource_reclamation_verifier(
                    backend, operation_id, validated_baseline
                )
            except (OSError, TimeoutError) as exc:
                reclamation = {
                    "state": "unknown",
                    "reason": f"resource reclamation check failed: {type(exc).__name__}",
                }
        reclamation_state = reclamation.get("state") if isinstance(reclamation, Mapping) else None
        reclamation_evidence = reclamation.get("evidence") if isinstance(reclamation, Mapping) else None
        if isinstance(reclamation_evidence, Mapping):
            evidence_present = bool(reclamation_evidence)
        elif isinstance(reclamation_evidence, str):
            evidence_present = bool(reclamation_evidence.strip())
        elif isinstance(reclamation_evidence, Sequence) and not isinstance(reclamation_evidence, (str, bytes)):
            evidence_present = bool(reclamation_evidence)
        else:
            evidence_present = False
        state = (
            "evicted"
            if reclamation_state == "reclaimed"
            and reclamation.get("operation_id") == operation_id
            and evidence_present
            else "unknown"
        )
        return {
            "state": state,
            "backend_id": backend.backend_id,
            "operation_id": operation_id,
            "authorization": dict(authorization),
            "execution_receipt": dict(execution_receipt),
            "close_receipt": dict(closed),
            "active_generation_observation": dict(active),
            "cleanup_verified": True,
            "resource_reclamation": dict(reclamation),
            "reclamation_baseline": validated_baseline,
            "evidence": {
                "adapter": self.adapter_id,
                "profile_id": profile.profile_id,
                "profile_version": profile.version,
            },
        }

    def restore(
        self,
        backend: BackendSpec,
        *,
        operation_id: str,
        execution_context: ExecutionContext | Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Start the exact registered profile under an idempotent attempt identity."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("restore requires a non-empty operation_id")
        profile = self._profile(backend)
        context, context_error = self._backend_execution_context(backend, execution_context)
        if context is None:
            return {
                "state": "blocked",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "profile_id": profile.profile_id,
                "profile_version": profile.version,
                "reason": context_error,
            }
        attempt_id = self._attempt_id(backend, operation_id, "restore")
        job_spec = JobSpec(owner_id=f"backend:{backend.backend_id}", profile_id=profile.execution_profile_id)
        try:
            handle = self.execution.start(
                job_spec,
                attempt_id=attempt_id,
                execution_root=Path(self._execution_root).expanduser()
                / f"backend-{hashlib.sha256(backend.backend_id.encode('utf-8')).hexdigest()[:24]}",
                execution_context=context,
            )
        except Exception as exc:
            # The core intent and deterministic attempt identity let recovery
            # inspect this operation instead of blindly launching a second one.
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "profile_id": profile.profile_id,
                "profile_version": profile.version,
                "reason": f"backend start outcome requires reconciliation: {type(exc).__name__}",
            }
        readiness = self._wait_ready(profile, handle, timeout_seconds=profile.readiness_timeout_seconds)
        result = {
            "state": "ready" if readiness.get("ready") else readiness.get("state", "unknown"),
            "backend_id": backend.backend_id,
            "operation_id": operation_id,
            "attempt_id": attempt_id,
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "execution_handle": handle.to_dict() if isinstance(handle, ExecutionHandle) else dict(handle),
            "readiness": readiness,
            "evidence": {
                "adapter": self.adapter_id,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        endpoint = self._endpoint(profile, readiness)
        if endpoint is not None:
            result["endpoint"] = endpoint
            result["readiness"] = {**readiness, "endpoint": endpoint}
        return result

    def _wait_ready(
        self,
        profile: BackendProfile,
        handle: ExecutionHandle | Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = self._monotonic() + max(0.0, timeout_seconds)
        last: dict[str, Any] = {"ready": False, "state": "unknown", "reason": "no health observation"}
        while True:
            try:
                execution = self.execution.inspect(handle)
            except (OSError, TimeoutError) as exc:
                return {
                    "ready": False,
                    "state": "unknown",
                    "reason": f"execution inspection failed: {type(exc).__name__}",
                    "identity_verified": False,
                }
            execution_state = _member(execution, "state", "unknown")
            if execution_state != "running":
                last = {
                    "ready": False,
                    "state": "unknown" if execution_state == "unknown" else "failed",
                    "reason": f"backend execution is {execution_state}",
                    "execution": execution,
                    "identity_verified": False,
                }
            else:
                last = self._health_once(profile, timeout_seconds=max(0.1, timeout_seconds))
                last["execution_state"] = execution_state
                if last.get("ready"):
                    return last
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return last
            self._sleep(min(profile.readiness_interval_seconds, remaining))

    def ready(
        self,
        backend: BackendSpec,
        *,
        timeout_seconds: float,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Wait a bounded time for the current invocation and pinned model identity."""
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            return {
                "ready": False,
                "state": "unknown",
                "reason": "timeout_seconds must be finite and non-negative",
                "operation_id": operation_id,
            }
        try:
            profile = self._profile(backend)
        except BackendAdapterError as exc:
            return {"ready": False, "state": "unknown", "reason": str(exc), "operation_id": operation_id}
        handle, error = self._resolve_handle(backend)
        if handle is None:
            return {
                "ready": False,
                "state": "unknown",
                "reason": error,
                "operation_id": operation_id,
                "identity_verified": False,
            }
        result = self._wait_ready(profile, handle, timeout_seconds=timeout_seconds)
        response = {**result, "operation_id": operation_id, "backend_id": backend.backend_id}
        endpoint = self._endpoint(profile, result)
        if endpoint is not None:
            response["endpoint"] = endpoint
            response["readiness"] = {**result, "endpoint": endpoint}
        return response

    def reopen(self, backend: BackendSpec, *, operation_id: str) -> Mapping[str, Any]:
        """Re-enable only the exact admission epoch fenced by this operation."""
        try:
            profile = self._profile(backend)
            control = self._control_for(profile)
        except BackendAdapterError as exc:
            return {
                "state": "blocked",
                "operation_id": operation_id,
                "backend_id": backend.backend_id,
                "admission_open": False,
                "reason": str(exc),
            }
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("reopen requires a non-empty operation_id")
        try:
            current = control.backend_admission_state(backend.backend_id)
            if current.get("backend_id") != backend.backend_id:
                raise BackendAdapterError("FrameShift admission state belongs to a different backend")
            if current.get("operation_id") != operation_id:
                raise BackendAdapterError("FrameShift admission epoch belongs to a different handoff")
            epoch = current.get("epoch")
            valid_epoch = not isinstance(epoch, bool) and (
                (isinstance(epoch, str) and bool(epoch.strip()))
                or (isinstance(epoch, int) and epoch >= 0)
            )
            if not valid_epoch:
                raise BackendAdapterError("FrameShift admission epoch is not verifiable")
            if current.get("closed") is False:
                return {
                    "state": "ready",
                    "backend_id": backend.backend_id,
                    "operation_id": operation_id,
                    "admission_open": True,
                    "admission_epoch": epoch,
                    "open_receipt": {
                        **dict(current),
                        "opened": True,
                        "reconciled": True,
                    },
                }
            if current.get("closed") is not True:
                raise BackendAdapterError("FrameShift admission is not closed at a verifiable epoch")
            receipt = control.open_backend_admission(backend.backend_id, operation_id, epoch)
        except (BackendAdapterError, OSError, TimeoutError, LookupError) as exc:
            return {
                "state": "unknown",
                "backend_id": backend.backend_id,
                "operation_id": operation_id,
                "admission_open": False,
                "reason": str(exc) if isinstance(exc, BackendAdapterError) else f"admission reopen failed: {type(exc).__name__}",
            }
        opened = (
            isinstance(receipt, Mapping)
            and receipt.get("opened") is True
            and receipt.get("backend_id") == backend.backend_id
            and receipt.get("operation_id") == operation_id
            and receipt.get("epoch") == epoch
        )
        return {
            "state": "ready" if opened else "unknown",
            "operation_id": operation_id,
            "backend_id": backend.backend_id,
            "admission_open": opened,
            "admission_epoch": epoch,
            "open_receipt": dict(receipt) if isinstance(receipt, Mapping) else None,
        }


def profile_stop_timeout(profile: BackendProfile | Mapping[str, Any]) -> float:
    value = _member(profile, "stop_timeout_seconds", 30.0)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 30.0
    return min(300.0, max(0.1, timeout))


