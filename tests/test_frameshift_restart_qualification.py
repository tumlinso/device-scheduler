"""Integration qualification against FrameShift's real SQLite-backed engine.

Run with FrameShift's source tree installed or on PYTHONPATH, for example:
    PYTHONPATH=/home/tumlinson/FrameShift/src \
      /home/tumlinson/device-scheduler/.venv/bin/python -m pytest -q \
      tests/test_frameshift_restart_qualification.py

No model server is contacted: the continuity test uses a scripted provider, and
the admission-gate test stops at the provider's closed-gate preflight.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest

from device_scheduler_adapters.frameshift import (
    FrameShiftAdapterError,
    FrameShiftContinuationAdapter,
)

pytest.importorskip(
    "frameshift",
    reason="FrameShift qualification requires FrameShift to be installed or on PYTHONPATH",
)

from frameshift.backends.llama import LlamaCppHTTPProvider
from frameshift.backends.scripted import ScriptedProvider
from frameshift.engine import Engine
from frameshift.models import (
    ActionKind,
    AgentType,
    CapabilityGrant,
    GenerationResult,
    JoinPredicate,
    Lifecycle,
    ModelBinding,
    ProposedAction,
    TypedPayload,
    WaitSpec,
)
from frameshift.registry import Registry
from frameshift.service import FrameShiftService, _handler_type


def _registry() -> Registry:
    registry = Registry()
    registry.register_model_binding(ModelBinding(
        "fixture", "1", "fixture-model", (("json", "1"),), (("json", "1"),),
        model_version="fixture-v1",
    ))
    registry.register_agent_type(AgentType(
        "worker", "1", Lifecycle.PERSISTENT, ("json", "1"), ("json", "1"),
        ("json", "1"), ("fixture",),
    ))
    registry.register_application("app", CapabilityGrant(
        "grant", "app", frozenset({"worker"}), max_frames=8, max_children_per_frame=4,
        max_turns=8, max_output_tokens_per_turn=64, max_context_tokens=1024,
    ))
    return registry


class _EngineTransport:
    """Route adapter calls into an actual Engine backed by its SQLite file."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, token, body, timeout):
        assert token == "scheduler-token"
        assert timeout > 0
        parsed = urlsplit(url)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        self.calls.append((method, parsed.path))
        assert parts[:2] == ["v1", "frames"]
        frame_id = parts[2]
        if parts[3] == "handoff-barriers":
            handoff_id = parts[4]
            action = parts[5] if len(parts) > 5 else "inspect"
            if method == "GET" and action == "inspect":
                return self.engine.inspect_handoff_barrier("app", frame_id, handoff_id)
            if method == "POST" and action == "suspend":
                return self.engine.suspend_handoff_continuation(
                    "app", frame_id, handoff_id=handoff_id,
                    deadline=body.get("deadline") if body else None,
                )
            if method == "POST" and action == "deliver":
                return self.engine.deliver_handoff_result(
                    "app", frame_id, handoff_id=handoff_id,
                    delivery_id=body["delivery_id"], result=body["result"],
                )
            if method == "POST" and action == "resume":
                return self.engine.resume_handoff_continuation(
                    "app", frame_id, handoff_id=handoff_id,
                    delivery_id=body["delivery_id"],
                )
        raise AssertionError((method, parsed.path, body))


def _adapter(engine: Engine, transport: _EngineTransport, *, frame_id: str = "frame/1") -> FrameShiftContinuationAdapter:
    return FrameShiftContinuationAdapter({
        "app/frame": {
            "base_url": "http://127.0.0.1:8765",
            "token": "scheduler-token",
            "frame_id": frame_id,
        },
    }, request_json=transport)


def test_real_engine_barrier_survives_restart_blocks_wakes_and_delivers_once(tmp_path: Path):
    database = tmp_path / "frames.sqlite3"
    handoff_id = "handoff/restart-1"
    holder: dict[str, Engine] = {}

    def generate(_request, _lease):
        # The generation is executing when the handoff barrier is requested.
        barrier = holder["engine"].suspend_handoff_continuation(
            "app", "frame/1", handoff_id=handoff_id,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=5),
        )
        assert barrier["state"] == "requested"
        # A later unrelated message must remain a pending wake, not bypass the barrier.
        holder["engine"].send(
            "app", "frame/1", TypedPayload("json", "1", {"unrelated": True}),
            message_id="unrelated-wake-1",
        )
        return GenerationResult(ProposedAction(ActionKind.PARTIAL), partial=True)

    provider = ScriptedProvider(
        model_id="fixture-model", model_version="fixture-v1",
        input_codecs=(("json", "1"),), output_codecs=(("json", "1"),),
        fallback=generate,
    )
    engine = Engine(database, _registry(), (provider,))
    holder["engine"] = engine
    frame = engine.spawn("app", "worker", {"task": "yield at tool boundary"},
                         grant_id="grant", frame_id="frame/1")
    report = engine.run_once()
    assert report is not None and report.released and report.state.value == "suspended"
    assert engine.inspect("app", frame.id).state.value == "suspended"
    barrier = engine.inspect_handoff_barrier("app", frame.id, handoff_id)
    assert barrier["state"] == "acknowledged"
    assert barrier["ack"]["provider_release_verified"] is True

    # Reconstruct the service against the same SQLite authority.
    engine.close()
    restarted = Engine(database, _registry())
    transport = _EngineTransport(restarted)
    adapter = _adapter(restarted, transport)
    assert adapter.inspect("app/frame", handoff_id=handoff_id)["state"] == "acknowledged"

    # An unrelated post-restart wake remains held until this handoff is resumed.
    restarted.send("app", frame.id, TypedPayload("json", "1", {"later": True}),
                   message_id="unrelated-wake-2")
    assert restarted.inspect("app", frame.id).state.value == "suspended"
    assert restarted.inspect_handoff_barrier("app", frame.id, handoff_id)["state"] == "acknowledged"

    delivery_id = adapter._delivery_id(handoff_id, "app/frame")
    result = {"handoff_id": handoff_id, "jobs": [{"state": "succeeded", "exit_code": 0}]}
    first = adapter.deliver("app/frame", delivery_id=delivery_id, result=result)
    second = adapter.deliver("app/frame", delivery_id=delivery_id, result=result)
    assert first == second
    handoff_messages = [m for m in restarted.store.messages_after("app", frame.id, 0)
                        if m.id.startswith(f"handoff:{handoff_id}:")]
    assert len(handoff_messages) == 1
    with pytest.raises(FrameShiftAdapterError, match="FrameShift operation failed"):
        adapter.deliver("app/frame", delivery_id=delivery_id,
                        result={"handoff_id": handoff_id, "jobs": [{"state": "different"}]})
    assert restarted.inspect("app", frame.id).state.value == "suspended"

    assert adapter.resume("app/frame", delivery_id=delivery_id)["resumed"] is True
    resume_events = [event for event in restarted.events("app", frame.id)
                     if event.kind == "handoff_continuation_resumed"]
    assert len(resume_events) == 1
    assert restarted.inspect("app", frame.id).state.value == "runnable"
    # Repeating the same durable resume is idempotent and creates no second transition.
    assert adapter.resume("app/frame", delivery_id=delivery_id)["resumed"] is True
    assert len([event for event in restarted.events("app", frame.id)
                if event.kind == "handoff_continuation_resumed"]) == 1
    restarted.close()


def test_unverified_provider_release_never_acknowledges_safe_handoff(tmp_path: Path):
    holder: dict[str, Engine] = {}

    class ReleaseFailureProvider(ScriptedProvider):
        def release(self, lease):
            raise OSError("provider release could not be confirmed")

    def generate(_request, _lease):
        holder["engine"].suspend_handoff_continuation(
            "app", "frame-failure", handoff_id="handoff-release-failure",
            deadline=datetime.now(timezone.utc) + timedelta(seconds=2),
        )
        return GenerationResult(ProposedAction(ActionKind.PARTIAL), partial=True)

    provider = ReleaseFailureProvider(
        model_id="fixture-model", model_version="fixture-v1",
        input_codecs=(("json", "1"),), output_codecs=(("json", "1"),),
        fallback=generate,
    )
    engine = Engine(tmp_path / "release-failure.sqlite3", _registry(), (provider,))
    holder["engine"] = engine
    frame = engine.spawn("app", "worker", {"task": "release failure"}, grant_id="grant",
                         frame_id="frame-failure")
    report = engine.run_once()
    assert report is not None and report.released is False
    barrier = engine.inspect_handoff_barrier("app", frame.id, "handoff-release-failure")
    assert barrier["state"] == "unknown"
    assert barrier["safe_boundary"] is False
    assert barrier["ack"]["provider_release_verified"] is False
    engine.close()


def test_dependency_wake_is_held_until_handoff_delivery_and_resume(tmp_path: Path):
    holder: dict[str, Engine] = {}
    child_id = "child-frame"

    def generate(request, _lease):
        if request.frame_id == "parent-frame":
            holder["engine"].suspend_handoff_continuation(
                "app", "parent-frame", handoff_id="handoff-dependency",
                deadline=datetime.now(timezone.utc) + timedelta(seconds=5),
            )
            return GenerationResult(ProposedAction(
                ActionKind.WAIT,
                wait=WaitSpec((child_id,), predicate=JoinPredicate.ALL),
            ))
        return GenerationResult(ProposedAction(ActionKind.COMPLETE,
                                                payload=TypedPayload("json", "1", {"done": True})))

    provider = ScriptedProvider(
        model_id="fixture-model", model_version="fixture-v1",
        input_codecs=(("json", "1"),), output_codecs=(("json", "1"),),
        fallback=generate,
    )
    engine = Engine(tmp_path / "dependency.sqlite3", _registry(), (provider,))
    holder["engine"] = engine
    child = engine.spawn("app", "worker", {"task": "dependency"}, grant_id="grant",
                         frame_id=child_id, priority_class="speculative")
    parent = engine.spawn("app", "worker", {"task": "wait for child"}, grant_id="grant",
                          frame_id="parent-frame", priority_class="critical")

    first = engine.run_once()
    assert first is not None and first.frame_id == parent.id
    assert engine.inspect("app", parent.id).state.value == "waiting"
    assert engine.inspect_handoff_barrier("app", parent.id, "handoff-dependency")["state"] == "acknowledged"

    child_report = engine.run_once()
    assert child_report is not None and child_report.frame_id == child.id
    assert child_report.state.value == "completed"
    assert engine.inspect("app", parent.id).state.value == "suspended"
    barrier = engine.inspect_handoff_barrier("app", parent.id, "handoff-dependency")
    assert barrier["state"] == "acknowledged"
    assert barrier["resume_state"] == "runnable"

    transport = _EngineTransport(engine)
    adapter = _adapter(engine, transport, frame_id=parent.id)
    delivery_id = adapter._delivery_id("handoff-dependency", "app/frame")
    adapter.deliver("app/frame", delivery_id=delivery_id,
                    result={"handoff_id": "handoff-dependency", "jobs": []})
    assert engine.inspect("app", parent.id).state.value == "suspended"
    adapter.resume("app/frame", delivery_id=delivery_id)
    assert engine.inspect("app", parent.id).state.value == "runnable"
    engine.close()


def test_closed_backend_gate_rejects_stale_offers_and_active_generation(tmp_path: Path):
    provider = LlamaCppHTTPProvider(
        "http://127.0.0.1:8080", backend_id="backend-1", model_id="llama",
        model_version="pinned-v1", input_codecs=(("text/plain", "1"),),
        output_codecs=(("text/plain", "1"),), max_concurrency=2,
    )
    engine = Engine(tmp_path / "backend-gate.sqlite3", _registry(), (provider,))
    opened = engine.close_backend_admission("backend-1", operation_id="handoff-open-1")
    epoch = opened["epoch"]
    engine.open_backend_admission("backend-1", operation_id="handoff-open-1", epoch=epoch)

    request = SimpleNamespace(
        model_id="llama", input=TypedPayload("text/plain", "1", "no network call"),
        output_codec=("text/plain", "1"), frame_id="frame-gate", frame_generation=1,
        deadline=None, max_output_tokens=8,
    )
    offers = provider.offers(request)
    assert len(offers) == 2
    active = provider.admit(offers[0], request)
    assert active is not None
    assert active.capacity_epoch == offers[0].capacity_epoch

    closed = engine.close_backend_admission("backend-1", operation_id="handoff-close-1")
    assert closed["closed"] is True and closed["active_generations"] == 1
    assert closed["active_generations_state"] == "observed"
    assert provider.offers(request) == ()
    assert provider.admit(offers[1], request) is None
    refused = provider.generate(request, active)
    assert refused.error and "stale" in refused.error

    receipt = provider.release(active)
    assert receipt.released and receipt.verified
    reopened = engine.open_backend_admission(
        "backend-1", operation_id="handoff-close-1", epoch=closed["epoch"],
    )
    assert reopened["open"] is True
    assert provider.admit(offers[1], request) is None
    fresh = provider.offers(request)
    assert fresh and fresh[0].capacity_epoch != offers[0].capacity_epoch
    new_lease = provider.admit(fresh[0], request)
    assert new_lease is not None and new_lease.capacity_epoch == fresh[0].capacity_epoch
    provider.release(new_lease)
    engine.close()


def test_scheduler_gate_http_routes_require_scheduler_identity(tmp_path: Path):
    provider = LlamaCppHTTPProvider(
        "http://127.0.0.1:8080", backend_id="backend-auth", model_id="llama",
        model_version="pinned-v1",
    )
    engine = Engine(tmp_path / "backend-route.sqlite3", _registry(), (provider,))
    service = FrameShiftService(engine, {"app": "application-token"},
                                scheduler_tokens={"device-scheduler": "scheduler-token"})

    def request(method: str, path: str, token: str, payload=None):
        handler_type = _handler_type()
        handler = object.__new__(handler_type)
        handler.server = SimpleNamespace(service=service)
        handler.path = path
        handler.headers = Message()
        handler.headers["Authorization"] = f"Bearer {token}"
        encoded = json.dumps(payload or {}).encode()
        handler.headers["Content-Length"] = str(len(encoded))
        handler.rfile = BytesIO(encoded)
        handler.wfile = BytesIO()
        statuses = []
        handler.send_response = lambda status, *args: statuses.append(status)
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda: None
        getattr(handler, f"do_{method}")()
        body = json.loads(handler.wfile.getvalue()) if handler.wfile.getvalue() else {}
        return statuses[-1], body

    path = "/v1/backends/backend-auth/admission"
    assert request("GET", path, "application-token")[0] == 401
    status, initial = request("GET", path, "scheduler-token")
    assert status == 200 and initial["state"] == "unconfigured"
    close_path = f"{path}/close"
    assert request("POST", close_path, "application-token", {"operation_id": "h1"})[0] == 401
    status, closed = request("POST", close_path, "scheduler-token", {"operation_id": "h1"})
    assert status == 200 and closed["closed"] is True
    assert request("GET", path, "scheduler-token")[1]["open"] is False

    open_path = f"{path}/open"
    assert request("POST", open_path, "application-token",
                   {"operation_id": "h1", "epoch": closed["epoch"]})[0] == 401
    status, opened = request("POST", open_path, "scheduler-token",
                             {"operation_id": "h1", "epoch": closed["epoch"]})
    assert status == 200 and opened["open"] is True
    status, _ = request("POST", open_path, "scheduler-token",
                        {"operation_id": "h1", "epoch": closed["epoch"] + 1})
    assert status == 409
    assert request("GET", path, "scheduler-token")[1]["open"] is True
    engine.close()
