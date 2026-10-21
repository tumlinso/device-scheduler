from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core.allocator import ResourceAllocator
from device_scheduler_core.application import Scheduler, SchedulerConfig
from device_scheduler_core.errors import SchedulerError
from device_scheduler_core.models import Principal, Resource

from device_scheduler_adapters.project_control import (
    DeviceSchedulerProjectControlAdapter,
    ReclamationPending,
    UnsupportedLegacyOperation,
)


def _resources() -> list[dict[str, Any]]:
    observation_time = datetime.now(timezone.utc).isoformat()
    return [
        {
            "resource_id": "gpu:host-a:GPU-1",
            "host_id": "host-a",
            "kind": "gpu",
            "capacities": {"memory_bytes": 16 * 1024**3},
            "attributes": {"uuid": "GPU-1", "index_observation": 0, "pci_bus_id": "0000:01:00.0", "memory_free_bytes_observation": 15 * 1024**3, "utilization_percent_observation": 3, "inventory_observed_at": observation_time, "process_observed_at": observation_time, "process_telemetry": "compute-apps-visible", "process_telemetry_state": "observed", "process_count_observation": 0, "process_identity_unknown_count": 0, "unmanaged_process_count": 0, "managed_process_ids": []},
            "topology": {},
            "conflict_domains": ["gpu-exclusive:host-a:GPU-1"],
            "capabilities": ["whole-device-exclusive-admission"],
        },
        {
            "resource_id": "gpu:host-a:GPU-2",
            "host_id": "host-a",
            "kind": "gpu",
            "capacities": {"memory_bytes": 16 * 1024**3},
            "attributes": {"uuid": "GPU-2", "index_observation": 1, "pci_bus_id": "0000:02:00.0", "memory_free_bytes_observation": 14 * 1024**3, "utilization_percent_observation": 8, "inventory_observed_at": observation_time, "process_observed_at": observation_time, "process_telemetry": "compute-apps-visible", "process_telemetry_state": "observed", "process_count_observation": 0, "process_identity_unknown_count": 0, "unmanaged_process_count": 0, "managed_process_ids": []},
            "topology": {},
            "conflict_domains": ["gpu-exclusive:host-a:GPU-2"],
            "capabilities": ["whole-device-exclusive-admission"],
        },
        {
            "resource_id": "host:host-a:domain:gpu-topology:abc",
            "host_id": "host-a",
            "kind": "isolation",
            "capacities": {"slots": 1},
            "attributes": {"domain_id": "host:host-a:domain:gpu-topology:abc", "member_resource_ids": ["gpu:host-a:GPU-1", "gpu:host-a:GPU-2"], "closure": "complete-observed-gpu-inventory"},
            "topology": {"member_resource_ids": ["gpu:host-a:GPU-1", "gpu:host-a:GPU-2"]},
            "conflict_domains": [],
            "capabilities": ["exclusive-scheduler-admission"],
        },
        {
            "resource_id": "host:host-a:cpu",
            "host_id": "host-a",
            "kind": "cpu",
            "capacities": {"cores": 16},
            "attributes": {},
            "topology": {},
            "conflict_domains": [],
            "capabilities": [],
        },
        {
            "resource_id": "host:host-a:memory",
            "host_id": "host-a",
            "kind": "memory",
            "capacities": {"bytes": 64 * 1024**3},
            "attributes": {},
            "topology": {},
            "conflict_domains": [],
            "capabilities": [],
        },
    ]


class FakePublicClient:
    """In-memory public MCP contract fake; no SQLite or process adapters."""

    def __init__(self) -> None:
        self.resources = _resources()
        self.leases: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.release_state = "released"

    async def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((operation, dict(arguments)))
        if operation == "resources.list":
            return {"resources": self.resources}
        if operation == "leases.request":
            request = arguments["request"]
            allocations = []
            for requirement in request["requirements"]:
                selected = requirement["selectors"]["resource_id"]
                if isinstance(selected, dict):
                    available = selected["in"]
                    resource_id = next((item for item in available if item not in {a["resource_id"] for a in allocations}), available[0])
                else:
                    resource_id = selected
                allocations.append({
                    "resource_id": resource_id,
                    "quantities": requirement["quantities"],
                    "mode": requirement["mode"],
                })
            lease = {
                "lease_id": f"lease-{len(self.leases) + 1}",
                "owner_id": request["owner_id"],
                "owner_context": request["owner_context"],
                "allocations": allocations,
                "fencing_token": 1,
                "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
                "state": "active",
            }
            self.leases.append(lease)
            return {"lease": lease}
        if operation == "leases.list":
            return {"leases": self.leases}
        if operation == "leases.renew":
            lease = next(item for item in self.leases if item["lease_id"] == arguments["lease_id"])
            lease["fencing_token"] += 1
            return {"lease": lease}
        if operation == "leases.release":
            lease = next(item for item in self.leases if item["lease_id"] == arguments["lease_id"])
            lease["state"] = self.release_state
            return {"lease": lease}
        raise AssertionError(f"unexpected public operation: {operation}")


class _PublicClientError(RuntimeError):
    def __init__(self, error: SchedulerError) -> None:
        super().__init__(error.message)
        self.code = error.code
        self.details = error.details


class InProcessPublicClient:
    """Exercise the actual public operation boundary without a live service."""

    def __init__(self, scheduler: Scheduler, principal: Principal) -> None:
        self.scheduler = scheduler
        self.principal = principal
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((operation, dict(arguments)))
        try:
            return self.scheduler.dispatch(operation, arguments, self.principal)
        except SchedulerError as error:
            raise _PublicClientError(error) from error


class VerifiedReclaimer:
    """Test-only declared owner of the synthetic inventory and reclamation proof."""

    api_version = "1"
    adapter_id = "test.verified-reclaimer"

    def discover(self) -> list[dict[str, Any]]:
        # The scheduler binds these records to this adapter during public
        # inventory reconciliation; direct allocator registration would omit
        # provenance and must not be treated as trusted release coverage.
        return _resources()

    def reclaim_lease(self, lease: dict[str, Any], *, operation_id: str) -> dict[str, Any]:
        return {
            "verified": True,
            "lease_id": lease["lease_id"],
            "fencing_token": lease["fencing_token"],
            "resource_ids": sorted(item["resource_id"] for item in lease["allocations"]),
            "operation_id": operation_id,
            "evidence": {"operation_id": operation_id, "test_adapter": True,
                         "synthetic_fixture_only": True},
        }


def _in_process_scheduler(tmp_path: Path) -> tuple[Scheduler, Principal]:
    ledger = Ledger(tmp_path / "scheduler.sqlite3")
    allocator = ResourceAllocator(ledger)
    scheduler = Scheduler(
        ledger,
        allocator,
        resource_adapters=(VerifiedReclaimer(),),
        config=SchedulerConfig(state_root=tmp_path / "state"),
    )
    principal = Principal(
        principal_id="project-control-bridge",
        actions=frozenset({
            "resources.list",
            "leases.request", "leases.request:any_owner",
            "leases.list", "leases.read:any_owner",
            "leases.renew", "leases.manage:any_owner",
            "leases.release",
        }),
    )
    return scheduler, principal


def test_inventory_keeps_legacy_ids_and_does_not_mutate_device_scheduler() -> None:
    client = FakePublicClient()
    bridge = DeviceSchedulerProjectControlAdapter(client)

    rows = bridge.discover_gpus()

    assert [item["id"] for item in rows] == ["accelerator:GPU-1", "accelerator:GPU-2"]
    assert rows[0]["tags"]["uuid"] == "GPU-1"
    assert rows[0]["tags"]["nvlink_domain"] == "host:host-a:domain:gpu-topology:abc"
    assert rows[0]["enabled"] is True
    assert [name for name, _ in client.calls] == ["resources.list"]


def test_reserve_background_forwards_atomic_gpu_and_topology_claim_with_owner_context(tmp_path: Path) -> None:
    client = FakePublicClient()
    bridge = DeviceSchedulerProjectControlAdapter(client)

    reservation = bridge.reserve_background(
        project_root=tmp_path,
        job_id="job-3",
        attempt_id="attempt-9",
        resource_request={
            "schema_version": 1,
            "kind": "accelerator",
            "ids": ["accelerator:GPU-1", "accelerator:GPU-2"],
            "exclusive_resources": ["interference:nvlink:runtime:0-1"],
        },
        pid=12345,
    )

    assert reservation is not None
    assert reservation["owner_id"].startswith("pc:background:")
    assert reservation["resource_ids"] == ["accelerator:GPU-1", "accelerator:GPU-2"]
    request = next(args["request"] for name, args in client.calls if name == "leases.request")
    assert request["owner_id"] == reservation["owner_id"]
    assert request["owner_context"] == client.leases[0]["owner_context"]
    assert request["owner_context"]["issuer"] == "project-control"
    assert request["owner_context"]["attempt_id"] == "attempt-9"
    assert request["owner_context"]["pid"] == "12345"
    assert [item["kind"] for item in request["requirements"]] == ["gpu", "gpu", "isolation"]
    assert all(item["mode"] == "exclusive" for item in request["requirements"])
    assert request["requirements"][-1]["selectors"] == {"resource_id": "host:host-a:domain:gpu-topology:abc"}


def test_owner_query_and_heartbeat_use_authoritative_lease_and_rotated_fence(tmp_path: Path) -> None:
    client = FakePublicClient()
    bridge = DeviceSchedulerProjectControlAdapter(client)
    reservation = bridge.reserve_background(
        project_root=tmp_path, job_id="job-3", attempt_id="attempt-9",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]}, pid=12345,
    )
    assert reservation is not None

    owner = bridge.owner(reservation["owner_id"])
    assert owner is not None
    assert owner["attempt_id"] == "attempt-9"
    assert owner["resources"] == ["accelerator:GPU-1"]
    assert bridge.owners_for_attempt(tmp_path, "attempt-9")[0]["owner_id"] == reservation["owner_id"]

    bridge.heartbeat(reservation["owner_id"], pid=12345)

    renew = next(args for name, args in client.calls if name == "leases.renew")
    assert renew == {"lease_id": "lease-1", "fencing_token": 1, "ttl_seconds": 60.0}
    assert bridge.owner(reservation["owner_id"])["fencing_token"] == 2


def test_release_never_reports_success_while_server_state_is_unknown(tmp_path: Path) -> None:
    client = FakePublicClient()
    bridge = DeviceSchedulerProjectControlAdapter(client)
    reservation = bridge.reserve_background(
        project_root=tmp_path, job_id="job-3", attempt_id="attempt-9",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]},
    )
    assert reservation is not None
    client.release_state = "unknown"

    with pytest.raises(ReclamationPending, match="has not verified release"):
        bridge.release(reservation["owner_id"])

    assert bridge.owner(reservation["owner_id"])["state"] == "unknown"


def test_legacy_preemption_residency_and_local_inventory_writes_fail_closed() -> None:
    bridge = DeviceSchedulerProjectControlAdapter(FakePublicClient())

    for operation in (
        lambda: bridge.preempt_requested("pc:service:owner"),
        lambda: bridge.request_preemption("pc:service:owner"),
        lambda: bridge.set_priority("pc:service:owner", "interactive"),
        lambda: bridge.reserve_service(
            project_root="/project",
            service_id="model",
            resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]},
        ),
        lambda: bridge.protect_residency("pc:service:owner", memory_baseline={}),
        lambda: bridge.replace("accelerator", []),
    ):
        with pytest.raises(UnsupportedLegacyOperation):
            operation()


def test_bridge_can_call_async_sdk_from_an_existing_event_loop() -> None:
    client = FakePublicClient()
    bridge = DeviceSchedulerProjectControlAdapter(client)

    async def run() -> list[dict[str, Any]]:
        return bridge.list(kind="accelerator")

    rows = asyncio.run(run())
    assert len(rows) == 2
    assert client.calls[0][0] == "resources.list"


def test_cpu_and_memory_legacy_estimates_become_shared_requirements() -> None:
    bridge = DeviceSchedulerProjectControlAdapter(FakePublicClient())

    requirements, _ = bridge._resource_requirements({
        "kind": "accelerator",
        "ids": ["accelerator:GPU-1"],
        "cpu_threads": 3,
        "ram_bytes": 4096,
    })

    assert [(item["kind"], item["mode"]) for item in requirements] == [
        ("gpu", "exclusive"), ("cpu", "shared"), ("memory", "shared")
    ]
    assert requirements[1]["quantities"] == {"cores": 3.0}
    assert requirements[2]["quantities"] == {"bytes": 4096.0}


def test_unqualified_pcie_isolation_is_not_silently_downgraded() -> None:
    bridge = DeviceSchedulerProjectControlAdapter(FakePublicClient())

    with pytest.raises(UnsupportedLegacyOperation, match="PCIe-root isolation"):
        bridge._resource_requirements({
            "kind": "accelerator", "ids": ["accelerator:GPU-1"], "isolate_pcie_root": True,
        })


def test_bridge_round_trips_legacy_owner_contract_through_real_public_application(tmp_path: Path) -> None:
    project_root = tmp_path / "project with spaces"
    project_root.mkdir()
    scheduler, principal = _in_process_scheduler(tmp_path)
    client = InProcessPublicClient(scheduler, principal)
    bridge = DeviceSchedulerProjectControlAdapter(client)

    reservation = bridge.reserve_background(
        project_root=project_root,
        job_id="model-job-8",
        attempt_id="attempt-2",
        resource_request={
            "kind": "accelerator",
            "ids": ["accelerator:GPU-1"],
            "isolate_nvlink_domain": True,
            "cpu_threads": 2,
            "ram_bytes": 4096,
        },
        pid=os.getpid(),
    )

    assert reservation is not None
    assert reservation["resource_ids"] == ["accelerator:GPU-1"]
    owner = bridge.owner(reservation["owner_id"])
    assert owner is not None
    assert owner["project_root"] == str(project_root.resolve())
    assert owner["job_id"] == "model-job-8"
    assert owner["attempt_id"] == "attempt-2"
    assert owner["resources"] == [
        "accelerator:GPU-1", "host:host-a:domain:gpu-topology:abc",
    ]
    assert owner["priority_class"] is None and owner["preemptible"] is False
    assert owner["cpu_threads"] == 2 and owner["ram_bytes"] == 4096
    assert owner["residency_metadata_json"] == "{}"
    assert owner["acquired_at"] is None and owner["heartbeat_at"] is None
    assert bridge.owners_for_attempt(project_root, "attempt-2") == [owner]

    bridge.heartbeat(reservation["owner_id"], pid=os.getpid())
    bridge.release(reservation["owner_id"])

    assert scheduler.allocator.list_leases(owner_id=reservation["owner_id"])[0].state.value == "released"
    assert bridge.owner_activity([reservation["owner_id"]]) == (0, 0)
    assert bridge.release_observation_counts() == (0, 0, 0)
    assert {name for name, _ in client.calls} >= {
        "resources.list", "leases.request", "leases.list", "leases.renew", "leases.release",
    }


def test_public_api_conflict_maps_to_no_grant_without_preempting_existing_owner(tmp_path: Path) -> None:
    scheduler, principal = _in_process_scheduler(tmp_path)
    client = InProcessPublicClient(scheduler, principal)
    first = DeviceSchedulerProjectControlAdapter(client)
    second = DeviceSchedulerProjectControlAdapter(client)

    reservation = first.reserve_background(
        project_root=tmp_path, job_id="job-1", attempt_id="attempt-1",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]},
    )
    denied = second.reserve_background(
        project_root=tmp_path, job_id="job-2", attempt_id="attempt-1",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]},
    )

    assert reservation is not None
    assert denied is None
    assert scheduler.allocator.list_leases(owner_id=reservation["owner_id"])[0].state.value == "active"
    bridge_owner = second.owner(reservation["owner_id"])
    assert bridge_owner is not None
    assert bridge_owner["state"] == "active"


def test_background_owner_retry_cannot_create_a_second_lease_for_changed_process_identity(tmp_path: Path) -> None:
    scheduler, principal = _in_process_scheduler(tmp_path)
    client = InProcessPublicClient(scheduler, principal)
    bridge = DeviceSchedulerProjectControlAdapter(client)

    first = bridge.reserve_background(
        project_root=tmp_path, job_id="job-stable", attempt_id="attempt-stable",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]}, pid=12001,
    )
    retry = bridge.reserve_background(
        project_root=tmp_path, job_id="job-stable", attempt_id="attempt-stable",
        resource_request={"kind": "accelerator", "ids": ["accelerator:GPU-1"]}, pid=12002,
    )

    assert first is not None
    assert retry is None
    leases = scheduler.allocator.list_leases(owner_id=first["owner_id"])
    assert len(leases) == 1
    assert leases[0].state.value == "active"


def test_foreground_activation_confirms_the_exact_gpu_and_topology_lease(tmp_path: Path) -> None:
    scheduler, principal = _in_process_scheduler(tmp_path)
    client = InProcessPublicClient(scheduler, principal)
    bridge = DeviceSchedulerProjectControlAdapter(client)

    reservation = bridge.begin_foreground(
        project_root=tmp_path,
        resource_request={
            "kind": "accelerator",
            "ids": ["accelerator:GPU-1"],
            "isolate_nvlink_domain": True,
        },
        pid=os.getpid(),
    )

    assert reservation["resource_ids"] == [
        "accelerator:GPU-1", "host:host-a:domain:gpu-topology:abc",
    ]
    assert bridge.activate_foreground(reservation) is True
