from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.qualify_gpu_allocations import (
    QualificationError,
    exact_gpu_snapshot,
    exclusive_request_waiting,
    main,
)


GPU_A = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU_B = "GPU-fedcba98-7654-3210-fedc-ba9876543210"


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
