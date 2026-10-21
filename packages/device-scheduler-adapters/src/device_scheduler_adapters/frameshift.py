"""Authenticated HTTP adapter for trusted FrameShift continuation references.

References are resolved only from operator configuration. Handoff requests
never supply URLs, tokens, or arbitrary HTTP destinations.
"""

from __future__ import annotations

import ipaddress
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit

from device_scheduler_core.models import AdapterCapability, EnforcementLevel


class FrameShiftAdapterError(RuntimeError):
    """FrameShift operation failed or returned an invalid response."""


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _http_json(method: str, url: str, token: str, body: Mapping[str, Any] | None, timeout: float) -> Mapping[str, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1_048_577)
    except (urllib.error.URLError, OSError) as exc:
        raise FrameShiftAdapterError(f"FrameShift request failed: {type(exc).__name__}") from exc
    if len(raw) > 1_048_576:
        raise FrameShiftAdapterError("FrameShift response exceeds 1 MiB")
    try:
        result = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameShiftAdapterError("FrameShift returned invalid JSON") from exc
    if not isinstance(result, Mapping):
        raise FrameShiftAdapterError("FrameShift response must be a JSON object")
    return result


class FrameShiftContinuationAdapter:
    """Call FrameShift's durable handoff barrier API over authenticated HTTP."""

    api_version = "1"
    adapter_id = "frameshift-http"

    def __init__(
        self,
        continuations: Mapping[str, Mapping[str, Any]],
        *,
        timeout_seconds: float = 10.0,
        request_json: Any = _http_json,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(continuations, Mapping):
            raise ValueError("continuations must be an operator-configured mapping")
        checked: dict[str, dict[str, str]] = {}
        targets: set[tuple[str, str]] = set()
        for ref, config in continuations.items():
            if not isinstance(ref, str) or not ref or not isinstance(config, Mapping):
                raise ValueError("each continuation reference must map to endpoint configuration")
            base_url = config.get("base_url")
            token = config.get("token")
            frame_id = config.get("frame_id")
            if not all(isinstance(item, str) and item for item in (base_url, token, frame_id)):
                raise ValueError(f"continuation {ref!r} requires base_url, token and frame_id")
            parsed = urlsplit(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or not _is_loopback(parsed.hostname):
                raise ValueError("FrameShift endpoint must be an operator-configured loopback HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("FrameShift endpoint URL cannot contain credentials, query or fragment")
            normalized_url = base_url.rstrip("/")
            target = (normalized_url, frame_id)
            if target in targets:
                raise ValueError("each FrameShift frame must have one configured continuation reference")
            targets.add(target)
            checked[ref] = {"base_url": normalized_url, "token": token, "frame_id": frame_id}
        self._continuations = checked
        self._timeout = float(timeout_seconds)
        self._request_json = request_json

    def capabilities(self) -> Sequence[AdapterCapability]:
        return (
            AdapterCapability("durable_turn_boundary_barrier", True, EnforcementLevel.ENFORCED),
            AdapterCapability("idempotent_handoff_delivery", True, EnforcementLevel.ENFORCED),
            AdapterCapability("ordinary_suspend_preserves_backend_residency", True, EnforcementLevel.ENFORCED),
        )

    def _endpoint(self, ref: str) -> tuple[dict[str, str], str]:
        try:
            config = self._continuations[ref]
        except KeyError as exc:
            raise FrameShiftAdapterError(f"continuation reference {ref!r} is not configured") from exc
        frame = quote(config["frame_id"], safe="")
        return config, f"{config['base_url']}/v1/frames/{frame}"

    def validate_ref(self, continuation_ref: str) -> Mapping[str, Any]:
        """Preflight a continuation reference without making a network call."""
        self._endpoint(continuation_ref)
        return {"configured": True, "continuation_ref": continuation_ref}

    def _request(
        self,
        ref: str,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        config, root = self._endpoint(ref)
        try:
            response = self._request_json(method, root + path, config["token"], body, self._timeout)
        except FrameShiftAdapterError:
            raise
        except Exception as exc:
            raise FrameShiftAdapterError(f"FrameShift operation failed: {type(exc).__name__}") from exc
        if not isinstance(response, Mapping):
            raise FrameShiftAdapterError("FrameShift response must be an object")
        if response.get("error"):
            raise FrameShiftAdapterError(str(response["error"]))
        return response

    @staticmethod
    def _barrier_path(handoff_id: str, action: str | None = None) -> str:
        path = f"/handoff-barriers/{quote(handoff_id, safe='')}"
        return path if action is None else f"{path}/{action}"

    def register_consumer(self, continuation_ref: str, *, backend_id: str, contract_id: str) -> Mapping[str, Any]:
        return self._request(
            continuation_ref, "POST", "/handoff-consumers",
            {"backend_id": backend_id, "contract_id": contract_id},
        )

    def revoke_consumer(self, continuation_ref: str, *, backend_id: str, contract_id: str) -> Mapping[str, Any]:
        return self._request(
            continuation_ref, "POST", "/handoff-consumers/revoke",
            {"backend_id": backend_id, "contract_id": contract_id},
        )

    def suspend(self, continuation_ref: str, *, handoff_id: str, deadline: Any) -> Mapping[str, Any]:
        requested = self._request(
            continuation_ref, "POST", self._barrier_path(handoff_id, "suspend"),
            {"deadline": deadline.isoformat() if hasattr(deadline, "isoformat") else str(deadline)},
        )
        if requested.get("safe_boundary") is True and requested.get("durable") is True:
            return requested
        if not hasattr(deadline, "timestamp"):
            raise FrameShiftAdapterError("suspension deadline must be an absolute timestamp")
        end = deadline.timestamp()
        while time.time() < end:
            time.sleep(min(0.1, max(0.0, end - time.time())))
            observed = self.inspect(continuation_ref, handoff_id=handoff_id)
            if observed.get("state") == "unknown":
                raise FrameShiftAdapterError("FrameShift could not verify provider release at the requested boundary")
            if observed.get("safe_boundary") is True and observed.get("durable") is True:
                return observed
        raise FrameShiftAdapterError("FrameShift did not reach a verified turn boundary before the deadline")

    def inspect(self, continuation_ref: str, *, handoff_id: str) -> Mapping[str, Any]:
        return self._request(continuation_ref, "GET", self._barrier_path(handoff_id))

    def deliver(self, continuation_ref: str, *, delivery_id: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        handoff_id = result.get("handoff_id")
        if not isinstance(handoff_id, str) or not handoff_id:
            raise FrameShiftAdapterError("delivery result must identify its handoff")
        return self._request(
            continuation_ref, "POST", self._barrier_path(handoff_id, "deliver"),
            {"delivery_id": delivery_id, "result": dict(result)},
        )

    def resume(self, continuation_ref: str, *, delivery_id: str) -> Mapping[str, Any]:
        # Delivery IDs encode the handoff ID so recovery works after this
        # adapter process restarts; the frame service still validates it
        # against its durable barrier record.
        try:
            prefix, encoded, digest = delivery_id.split(".", 2)
            if prefix != "ds":
                raise ValueError
            handoff_id = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise FrameShiftAdapterError("delivery ID has an invalid stable format") from exc
        if delivery_id != self._delivery_id(handoff_id, continuation_ref) or digest != hashlib.sha256(continuation_ref.encode("utf-8")).hexdigest()[:16]:
            raise FrameShiftAdapterError("delivery ID does not match this continuation")
        return self._request(
            continuation_ref, "POST", self._barrier_path(handoff_id, "resume"),
            {"delivery_id": delivery_id},
        )

    @staticmethod
    def _delivery_id(handoff_id: str, continuation_ref: str) -> str:
        encoded = base64.urlsafe_b64encode(handoff_id.encode("utf-8")).decode("ascii").rstrip("=")
        digest = hashlib.sha256(continuation_ref.encode("utf-8")).hexdigest()[:16]
        return f"ds.{encoded}.{digest}"


__all__ = ["FrameShiftAdapterError", "FrameShiftContinuationAdapter"]
