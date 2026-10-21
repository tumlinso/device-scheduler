"""Focused contracts for the KDA compatibility bridge."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from device_scheduler_adapters import Ledger
from device_scheduler_core import Resource, ResourceAllocator
from device_scheduler_mcp.kda_bridge import (
    BridgeRefusal,
    _controller_request,
    _configured_requirements,
    _normalize_job,
    _persist_job_logs,
    run_request,
)


class FakeClient:
    def __init__(self, states: list[dict]):
        self.states = list(states)
        self.submitted = []
        self.cancelled = []

    async def submit_job(self, spec):
        self.submitted.append(spec)
        return {"job_id": "job:1", "owner": spec.owner_id, "state": "queued"}

    async def wait_job(self, job_id, *, timeout_seconds):
        if self.states:
            return {"job": self.states.pop(0), "timed_out": False}
        return {"job": {"job_id": job_id, "owner": "kda:pilot", "state": "unknown"}, "timed_out": False}

    async def control_job(self, job_id, action, **kwargs):
        self.cancelled.append((job_id, action))
        return {"job": {"job_id": job_id, "owner": "kda:pilot", "state": "cancelled", "cleanup_verified": True}}

    async def get_job(self, job_id):
        return {"job_id": job_id, "owner": "kda:pilot", "state": "unknown"}

    async def job_logs(self, job_id, **kwargs):
        return {"job_id": job_id, "stdout": {"text": "ok"}, "stderr": {"text": ""}, "cleanup_verified": True}


def _config(root: Path, **changes):
    config = {
        "base_url": "http://127.0.0.1:8770",
        "token_file": "/run/user/1000/device-scheduler/kda.token",
        "owner_id": "kda:pilot",
        "execution_roots": [str(root)],
        "max_priority": 70,
        "priority": 60,
        "priority_class": "interactive",
        "resource_requirements": [{
            "kind": "gpu",
            "quantities": {"gpu_memory_bytes": 3_000_000_000},
            "selectors": {"uuid": "GPU-11111111-1111-1111-1111-111111111111"},
            "mode": "exclusive",
        }],
        "quiescence": {"timeout_seconds": 20, "consecutive_idle_samples": 4},
        "environment": {"OMP_NUM_THREADS": "2"},
    }
    config.update(changes)
    return config


def _request(root: Path):
    return {
        "schema_version": 1,
        "request_id": "run-123",
        "argv": ["python", "worker.py", "--case", "one"],
        "cwd": str(root),
        "worker_result_path": str(root / "result.json"),
        "timeout_seconds": 30,
    }


def test_bridge_submits_one_idempotent_job_and_requires_verified_cleanup(tmp_path: Path) -> None:
    client = FakeClient([{
        "job_id": "job:1", "owner": "kda:pilot", "state": "succeeded",
        "cleanup_verified": True, "attempt_id": "attempt:1", "result": {"exit_code": 0},
    }])

    result = asyncio.run(run_request(_config(tmp_path), _request(tmp_path), client=client, poll_interval_seconds=0.1))

    assert result["status"] == "ok"
    assert result["device_scheduler"]["job_id"] == "job:1"
    assert result["device_scheduler"]["attempt_id"] == "attempt:1"
    assert result["device_scheduler"]["cleanup_verified"] is True
    assert result["device_scheduler"]["scope_binding"]["execution_roots"] == [str(tmp_path)]
    assert len(client.submitted) == 1
    spec = client.submitted[0]
    assert spec.idempotency_key == "kda:run-123"
    assert "gpu.exclusive" in spec.tags
    assert spec.preconditions[0].name == "gpu.quiescence"
    assert spec.preconditions[0].parameters["consecutive_samples"] == 4


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"resource_requirements": [{"kind": "gpu", "quantities": {"gpu_memory_bytes": 1}}]}, "UUID"),
        ({"resource_requirements": [{"kind": "gpu", "selectors": {"uuid": "GPU-0"}, "quantities": {}}]}, "non-empty"),
        ({"quiescence": {"unknown_precondition": True}}, "unsupported legacy quiescence fields"),
        ({"environment": {"ACCESS_TOKEN": "not-a-secret"}}, "secret-like"),
        ({"priority": 71}, "priority exceeds"),
    ],
)
def test_bridge_refuses_unscoped_or_unrepresentable_requests(tmp_path: Path, changes, message: str) -> None:
    with pytest.raises(BridgeRefusal, match=message):
        asyncio.run(run_request(_config(tmp_path, **changes), _request(tmp_path), client=FakeClient([])))


def test_bridge_cancels_when_caller_requests_stop_and_reports_failure(tmp_path: Path) -> None:
    client = FakeClient([])
    stop = asyncio.Event()
    stop.set()

    result = asyncio.run(run_request(_config(tmp_path), _request(tmp_path), client=client, stop_event=stop))

    assert result["status"] == "failed"
    assert result["device_scheduler"]["state"] == "cancelled"
    assert result["device_scheduler"]["cleanup_verified"] is True
    assert client.cancelled == [("job:1", "cancel")]


def test_bridge_rejects_a_working_directory_outside_operator_scope(tmp_path: Path) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    client = FakeClient([])
    with pytest.raises(BridgeRefusal, match="outside configured execution_roots"):
        asyncio.run(run_request(_config(root), _request(outside), client=client))
    assert not client.submitted


def test_legacy_controller_translation_keeps_uuid_and_quiescence_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spec = {
        "schema_version": 1,
        "project_root": str(tmp_path),
        "command_cwd": str(tmp_path),
        "resources": {
            "gpus": 1,
            "gpu_uuids": ["GPU-11111111-1111-1111-1111-111111111111"],
        },
        "quiescence": {"timeout_seconds": 20, "consecutive_idle_samples": 4},
        "timeout": 900,
        "argv": ["python", "validate.py"],
        "campaign_id": "pilot-run",
        "recipe": "compute-sanitizer",
    }

    request = _controller_request(tmp_path / "controller-spec.json", spec, config)
    job, envelope = _normalize_job(config, request)

    assert request["request_id"].startswith("controller-spec:")
    assert job.argv == ("python", "validate.py")
    assert job.timeout_seconds == 880
    assert job.preconditions[0].parameters["consecutive_samples"] == 4
    assert envelope["scope_binding"]["caller_contract_digest"]


def test_legacy_controller_translation_refuses_resource_scope_mismatch(tmp_path: Path) -> None:
    spec = {
        "schema_version": 1,
        "command_cwd": str(tmp_path),
        "resources": {"gpus": 1, "gpu_uuids": ["GPU-different"]},
        "timeout": 900,
        "argv": ["python", "validate.py"],
    }
    with pytest.raises(BridgeRefusal, match="UUID selectors differ"):
        _controller_request(tmp_path / "spec.json", spec, _config(tmp_path))


def test_scheduler_logs_are_persisted_with_hashes_and_truncation_evidence(tmp_path: Path) -> None:
    result = {
        "device_scheduler": {
            "logs": {
                "stdout": {"text": "sanitizer output\n", "truncated": False},
                "stderr": {"text": "", "truncated": True},
            },
        },
    }

    _persist_job_logs(result, tmp_path / "receipt.json")

    evidence = result["device_scheduler"]["execution"]
    assert Path(evidence["stdout_path"]).read_text() == "sanitizer output\n"
    assert evidence["stdout_sha256"]
    assert evidence["stdout_truncated"] is False
    assert evidence["stderr_truncated"] is True


def test_bridge_gpu_alias_matches_real_nvidia_capacity_and_uuid_selector(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    config = _config(tmp_path)
    gpu = _configured_requirements(config["resource_requirements"])[0]

    # The bridge accepts the public gpu_memory_bytes spelling, then binds it
    # to the NVIDIA adapter's actual memory_bytes capacity field. The stable
    # UUID selector resolves against Resource.attributes["uuid"].
    assert gpu.quantities == {"memory_bytes": 3_000_000_000}
    assert gpu.selectors == {"uuid": "GPU-11111111-1111-1111-1111-111111111111"}
    ledger = Ledger(tmp_path / "scheduler.sqlite3")
    allocator = ResourceAllocator(ledger, clock=lambda: now)
    resource = Resource(
        "gpu:host-a:GPU-11111111-1111-1111-1111-111111111111",
        "host-a",
        "gpu",
        {"memory_bytes": 16 * 1024**3},
        attributes={
            "uuid": "GPU-11111111-1111-1111-1111-111111111111",
            "inventory_observed_at": now.isoformat(),
            "process_observed_at": now.isoformat(),
            "memory_free_bytes_observation": 16 * 1024**3,
            "process_telemetry": "compute-apps-visible",
            "process_telemetry_state": "observed",
            "process_count_observation": 0,
            "process_identity_unknown_count": 0,
            "unmanaged_process_count": 0,
            "managed_process_ids": [],
        },
        conflict_domains=frozenset({"gpu-exclusive:host-a:GPU-11111111-1111-1111-1111-111111111111"}),
    )
    allocator.register_resource(resource, source_adapter_id="nvidia-test", observed_at=now)

    placement = allocator.preview_allocation((gpu,))

    assert len(placement) == 1
    assert placement[0].resource_id == resource.resource_id
    assert placement[0].quantities == {"memory_bytes": 3_000_000_000}
