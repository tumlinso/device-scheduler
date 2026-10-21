from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qualify_gpu_allocations import (
    QualificationError,
    exact_gpu_snapshot,
    exclusive_request_waiting,
    _listed_resource_statuses,
    main,
)
from device_scheduler_adapters import Ledger
from device_scheduler_core import ConflictError, Resource, ResourceAllocator, ResourceRequirement, ResourceRequest
from datetime import datetime, timezone
import ctypes
import uuid as uuidlib

from scripts.qualification_gpu_probe import runtime_visible_device_uuid


GPU_A = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU_B = "GPU-fedcba98-7654-3210-fedc-ba9876543210"


class FakeSymbol:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


def test_cuda_uuid_uses_runtime_context_and_driver_v2_abi_without_gpu():
    calls = []
    physical_uuid = uuidlib.UUID("21131915-1488-23af-38dd-1743ae1f5cc8")

    class Runtime:
        cudaSetDevice = FakeSymbol(lambda ordinal: calls.append(("set", ordinal)) or 0)
        cudaFree = FakeSymbol(lambda pointer: calls.append(("free", pointer)) or 0)

    class Driver:
        cuInit = FakeSymbol(lambda flags: calls.append(("init", flags)) or 0)

        @staticmethod
        def _current_device(output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int)).contents.value = 7
            calls.append(("context-device", 7))
            return 0

        @staticmethod
        def _uuid(output, device):
            calls.append(("uuid-v2", device))
            ctypes.memmove(output, physical_uuid.bytes, 16)
            return 0

        cuCtxGetDevice = FakeSymbol(_current_device.__func__)
        cuDeviceGetUuid_v2 = FakeSymbol(_uuid.__func__)

    actual = runtime_visible_device_uuid(Runtime(), driver=Driver())

    assert actual == "GPU-" + str(physical_uuid)
    assert calls == [
        ("set", 0), ("free", None), ("init", 0),
        ("context-device", 7), ("uuid-v2", 7),
    ]


def test_exact_snapshot_filters_unrelated_gpu_but_keeps_matching_isolation_domain():
    gpu_a = {"resource_id": "gpu:host:" + GPU_A, "kind": "gpu", "attributes": {"uuid": GPU_A}}
    gpu_b = {"resource_id": "gpu:host:" + GPU_B, "kind": "gpu", "attributes": {"uuid": GPU_B}}
    domain_a = {
        "resource_id": "isolation:domain-a", "kind": "isolation",
        "attributes": {"member_resource_ids": [gpu_a["resource_id"], gpu_b["resource_id"]]},
    }
    snapshot = {
        "resources": [domain_a, gpu_b, gpu_a],
        "gpu_inventory": {"state": "observed", "devices": [{"uuid": GPU_A}, {"uuid": GPU_B}]},
        "process_telemetry": {"state": "observed", "processes": [
            {"gpu_uuid": GPU_A, "pid": 101}, {"gpu_uuid": GPU_B, "pid": 202},
            {"gpu_uuid": None, "pid": None, "identity_unknown": True},
        ]},
    }

    narrowed = exact_gpu_snapshot(snapshot, GPU_A)

    assert {item["resource_id"] for item in narrowed["resources"]} == {
        gpu_a["resource_id"], domain_a["resource_id"],
    }
    assert narrowed["gpu_inventory"]["devices"] == [{"uuid": GPU_A}]
    assert narrowed["process_telemetry"]["processes"] == [
        {"gpu_uuid": GPU_A, "pid": 101},
        {"gpu_uuid": None, "pid": None, "identity_unknown": True},
    ]
    assert len(snapshot["resources"]) == 3


def test_exact_snapshot_fails_closed_if_grant_uuid_is_absent_or_repeated():
    with pytest.raises(QualificationError, match="exactly one"):
        exact_gpu_snapshot({"resources": []}, GPU_A)
    with pytest.raises(QualificationError, match="exactly one"):
        exact_gpu_snapshot({"resources": [
            {"resource_id": "a", "kind": "gpu", "attributes": {"uuid": GPU_A}},
            {"resource_id": "b", "kind": "gpu", "attributes": {"uuid": GPU_A}},
        ]}, GPU_A)


def test_exclusive_wait_requires_no_attempt_or_lease():
    waiting = {"job_id": "exclusive-1", "state": "blocked", "owner": "qualification", "attempt_id": None, "lease_id": None}
    assert exclusive_request_waiting(waiting)
    assert not exclusive_request_waiting({**waiting, "attempt_id": "attempt-1"})
    assert not exclusive_request_waiting({**waiting, "lease_id": "lease-1"})
    assert not exclusive_request_waiting({**waiting, "state": "running"})


def test_public_resource_status_envelopes_are_unwrapped_and_keep_status():
    resource = {"resource_id": "gpu:host:" + GPU_A, "kind": "gpu", "attributes": {"uuid": GPU_A}}
    rows = _listed_resource_statuses({"resources": [{"resource": resource, "available": True}]})
    assert rows == [(resource, {"resource": resource, "available": True})]


def test_gpu_qualification_selector_uses_actual_allocator_attribute_contract(tmp_path):
    now = datetime.now(timezone.utc)
    resource = Resource(
        "gpu:host:" + GPU_A, "host-a", "gpu", {"memory_bytes": 1024**3},
        attributes={
            "uuid": GPU_A,
            "inventory_observed_at": now.isoformat(),
            "process_observed_at": now.isoformat(),
            "memory_free_bytes_observation": 1024**3,
            "process_telemetry": "compute-apps-visible",
            "process_telemetry_state": "observed",
            "process_count_observation": 0,
            "process_identity_unknown_count": 0,
            "unmanaged_process_count": 0,
            "managed_process_ids": [],
        },
    )
    allocator = ResourceAllocator(Ledger(tmp_path / "ledger.sqlite3"), clock=lambda: now)
    allocator.register_resource(resource)

    correct = ResourceRequirement(kind="gpu", quantities={"memory_bytes": 4 * 1024**2}, selectors={"uuid": GPU_A})
    lease = allocator.request_lease(ResourceRequest("correct-selector", "test", (correct,), 60))
    assert lease.allocations[0].resource_id == resource.resource_id

    incorrect = ResourceRequirement(kind="gpu", quantities={"memory_bytes": 4 * 1024**2}, selectors={"attributes.uuid": GPU_A})
    with pytest.raises(ConflictError, match="no resource matches"):
        allocator.request_lease(ResourceRequest("incorrect-selector", "test", (incorrect,), 60))


def test_live_run_requires_explicit_parent_grant(capsys):
    assert main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "plan-only"
    assert output["gpu_started"] is False
    with pytest.raises(SystemExit):
        main(["--run"])


def test_parent_grant_cannot_enable_live_side_effect_without_run():
    with pytest.raises(SystemExit):
        main(["--parent-grant", str(Path("grant.json"))])
    with pytest.raises(SystemExit):
        main(["--worker", "--gpu-uuid", GPU_A, "--memory-bytes", "4194304", "--receipt", "r.json", "--release-file", "release"])
