from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import pytest

from device_scheduler_adapters.frameshift import (
    FrameShiftAdapterError,
    FrameShiftContinuationAdapter,
)


class DurableBarrierService:
    """Transport fake whose state persists independently of adapter instances."""

    def __init__(self):
        self.barriers = {}
        self.deliveries = {}
        self.calls = []
        self.poll_count = 0

    def request(self, method, url, token, body, timeout):
        assert url.startswith("http://127.0.0.1:8765/")
        assert token == "trusted-token"
        self.calls.append((method, url, dict(body or {})))
        if url.endswith("/handoff-consumers"):
            return {"registered": True, "backend_id": body["backend_id"], "contract_id": body["contract_id"]}
        if url.endswith("/handoff-consumers/revoke"):
            return {"revoked": True, "backend_id": body["backend_id"]}
        marker = "/handoff-barriers/"
        if marker not in url:
            raise AssertionError(url)
        tail = url.split(marker, 1)[1].split("/", 1)
        handoff_id = unquote(tail[0])
        action = tail[1] if len(tail) == 2 else "inspect"
        key = ("ref", handoff_id)
        if action == "suspend":
            self.barriers.setdefault(key, {"state": "requested", "safe_boundary": False, "durable": False})
            return dict(self.barriers[key])
        if action == "inspect":
            self.poll_count += 1
            if self.poll_count >= 2:
                self.barriers[key] = {"state": "acknowledged", "safe_boundary": True, "durable": True}
            return dict(self.barriers[key])
        if action == "deliver":
            if self.barriers[key]["state"] not in {"acknowledged", "delivered"}:
                raise AssertionError("result delivered before the safe boundary")
            delivery_id = body["delivery_id"]
            result = body["result"]
            existing = self.deliveries.get(key)
            if existing and existing != (delivery_id, result):
                return {"error": "delivery identity conflict"}
            self.deliveries[key] = (delivery_id, result)
            self.barriers[key] = {"state": "delivered", "safe_boundary": True, "durable": True}
            return {"state": "delivered", "delivered": True, "durable": True}
        if action == "resume":
            if self.barriers[key]["state"] != "delivered" or self.deliveries[key][0] != body["delivery_id"]:
                raise AssertionError("resume requires its durable matching delivery")
            self.barriers[key] = {"state": "resumed", "safe_boundary": True, "durable": True}
            return {"state": "resumed", "resumed": True}
        raise AssertionError(action)


def test_trusted_framshift_endpoint_supports_restart_safe_barrier_and_delivery():
    remote = DurableBarrierService()
    config = {"ref": {"base_url": "http://127.0.0.1:8765", "token": "trusted-token", "frame_id": "frame/a"}}
    adapter = FrameShiftContinuationAdapter(config, request_json=remote.request)
    registration = adapter.register_consumer("ref", backend_id="backend", contract_id="contract-v1")
    assert registration["registered"]
    ack = adapter.suspend("ref", handoff_id="handoff/1",
                          deadline=datetime.now(timezone.utc) + timedelta(seconds=3))
    assert ack["safe_boundary"] and ack["durable"]
    delivery_id = adapter._delivery_id("handoff/1", "ref")
    payload = {"handoff_id": "handoff/1", "jobs": [{"state": "succeeded"}]}
    delivered = adapter.deliver("ref", delivery_id=delivery_id, result=payload)
    assert delivered["durable"]

    # A new adapter process can resume using only the stable delivery ID.
    restarted = FrameShiftContinuationAdapter(config, request_json=remote.request)
    resumed = restarted.resume("ref", delivery_id=delivery_id)
    assert resumed["resumed"]
    assert remote.barriers[("ref", "handoff/1")]["state"] == "resumed"
    assert any(path.endswith("/frames/frame%2Fa/handoff-barriers/handoff%2F1/deliver")
               for method, path, _ in remote.calls if method == "POST")


def test_adapter_refuses_unconfigured_or_non_loopback_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        FrameShiftContinuationAdapter({
            "ref": {"base_url": "https://example.invalid", "token": "x", "frame_id": "f"},
        })
    adapter = FrameShiftContinuationAdapter({})
    with pytest.raises(FrameShiftAdapterError, match="not configured"):
        adapter.validate_ref("request-controlled-ref")
    with pytest.raises(FrameShiftAdapterError, match="not configured"):
        adapter.inspect("request-controlled-url", handoff_id="h")


def test_adapter_rejects_ambiguous_duplicate_reference_for_one_frame():
    with pytest.raises(ValueError, match="one configured continuation reference"):
        FrameShiftContinuationAdapter({
            "first": {"base_url": "http://localhost:8765", "token": "one", "frame_id": "same"},
            "second": {"base_url": "http://localhost:8765/", "token": "two", "frame_id": "same"},
        })


def test_delivery_id_cannot_be_replayed_for_another_continuation():
    remote = DurableBarrierService()
    adapter = FrameShiftContinuationAdapter({
        "one": {"base_url": "http://localhost:8765", "token": "trusted-token", "frame_id": "f1"},
        "two": {"base_url": "http://localhost:8765", "token": "trusted-token", "frame_id": "f2"},
    }, request_json=remote.request)
    delivery_id = adapter._delivery_id("handoff", "one")
    with pytest.raises(FrameShiftAdapterError, match="does not match"):
        adapter.resume("two", delivery_id=delivery_id)
