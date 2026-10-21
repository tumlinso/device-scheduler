from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from device_scheduler_adapters import Ledger
from device_scheduler_core import (
    AllocationMode,
    ConflictError,
    IdempotencyConflictError,
    LeaseState,
    Resource,
    ResourceAllocator,
    ResourceRequirement,
    ResourceRequest,
    ValidationError,
)


class AllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.now = datetime(2026, 10, 21, tzinfo=timezone.utc)
        self.ledger = Ledger(self.path)
        self.allocator = ResourceAllocator(self.ledger, clock=lambda: self.now)
        observed_gpu = Resource(
            "gpu-0", "host-a", "gpu", {"memory_mib": 100.0, "compute_slots": 8.0},
            attributes={
                "model": "A",
                "inventory_observed_at": self.now.isoformat(),
                "process_observed_at": self.now.isoformat(),
                "memory_free_bytes_observation": 100 * 1024 * 1024,
                "process_telemetry": "compute-apps-visible",
                "process_telemetry_state": "observed",
                "process_count_observation": 0,
                "process_identity_unknown_count": 0,
                "unmanaged_process_count": 0,
                "managed_process_ids": [],
            },
        )
        self.allocator.register_resource(
            observed_gpu
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def gpu_requirement(memory: float, **kwargs) -> ResourceRequirement:
        return ResourceRequirement(kind="gpu", quantities={"memory_mib": memory}, **kwargs)

    def request(self, request_id: str, owner: str, memory: float, *, key: str | None = None) -> ResourceRequest:
        return ResourceRequest(
            request_id=request_id,
            owner_id=owner,
            requirements=(self.gpu_requirement(memory),),
            lease_ttl_seconds=60,
            idempotency_key=key,
        )

    def test_concurrent_connections_cannot_overallocate(self) -> None:
        allocators = [
            ResourceAllocator(Ledger(self.path), clock=lambda: self.now),
            ResourceAllocator(Ledger(self.path), clock=lambda: self.now),
        ]
        barrier = threading.Barrier(2)

        def request(index: int) -> bool:
            barrier.wait(timeout=5)
            try:
                allocators[index].request_lease(self.request(f"req-{index}", f"owner-{index}", 60))
                return True
            except ConflictError:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            granted = list(pool.map(request, range(2)))
        self.assertEqual(sum(granted), 1)
        leases = self.allocator.list_leases(state=LeaseState.ACTIVE)
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0].allocations[0].quantities["memory_mib"], 60)

    def test_idempotent_lease_replay_and_changed_payload_rejection(self) -> None:
        first = self.allocator.request_lease(self.request("req-1", "owner", 20, key="same-key"))
        replay = self.allocator.request_lease(self.request("req-1", "owner", 20, key="same-key"))
        self.assertEqual(first.lease_id, replay.lease_id)
        with self.assertRaises(IdempotencyConflictError):
            self.allocator.request_lease(self.request("req-1", "owner", 21, key="same-key"))

    def test_preview_is_side_effect_free_and_still_uses_current_claims(self) -> None:
        requirement = self.gpu_requirement(20)
        before_events = self.ledger.events()
        placement = self.allocator.preview_allocation((requirement,))
        self.assertEqual(placement[0].resource_id, "gpu-0")
        self.assertTrue(self.allocator.can_allocate((requirement,)))
        self.assertEqual(self.ledger.list("lease"), [])
        self.assertEqual(self.ledger.events(), before_events)
        self.allocator.request_lease(self.request("preview-fill", "owner", 80))
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(20),)))
        self.assertTrue(self.allocator.can_allocate((self.gpu_requirement(20),), ignore_lease_ids=(
            self.allocator.list_leases(state=LeaseState.ACTIVE)[0].lease_id,
        )))
        with self.assertRaises(ConflictError):
            self.allocator.preview_allocation((self.gpu_requirement(20),))

    def test_expiration_remains_held_until_verified_reclamation(self) -> None:
        request = ResourceRequest(
            "short", "owner", (self.gpu_requirement(70),), lease_ttl_seconds=5
        )
        lease = self.allocator.request_lease(request)
        self.now += timedelta(seconds=6)
        self.assertEqual(self.allocator.expire_due(), [lease.lease_id])
        expired_lease = self.allocator.get_lease(lease.lease_id)
        self.assertEqual(expired_lease.state, LeaseState.RECLAIMING)
        self.assertGreater(expired_lease.fencing_token, lease.fencing_token)
        self.assertEqual(self.allocator.expire_due(), [])
        self.assertEqual(self.allocator.get_lease(lease.lease_id).fencing_token, expired_lease.fencing_token)
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(self.request("blocked", "other", 21))
        with self.assertRaises(ConflictError):
            self.allocator.verify_release(
                lease.lease_id, lease.fencing_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": lease.fencing_token, "resource_ids": ["gpu-0"]},
            )
        with self.assertRaises(ValidationError):
            self.allocator.verify_release(
                lease.lease_id, expired_lease.fencing_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": expired_lease.fencing_token},
            )
        self.allocator.verify_release(
            lease.lease_id,
            expired_lease.fencing_token,
            {
                "verified": True,
                "lease_id": lease.lease_id,
                "fencing_token": expired_lease.fencing_token,
                "resource_ids": ["gpu-0"],
                "adapter": "test",
            },
        )
        granted = self.allocator.request_lease(self.request("after-release", "other", 70))
        self.assertEqual(granted.state, LeaseState.ACTIVE)

    def test_release_fences_stale_owner_and_requires_adapter_receipt(self) -> None:
        lease = self.allocator.request_lease(self.request("release", "owner", 20))
        release = self.allocator.request_release(lease.lease_id, lease.fencing_token)
        new_token = release["fencing_token"]
        self.assertNotEqual(lease.fencing_token, new_token)
        with self.assertRaises(ConflictError):
            self.allocator.begin_reclamation(lease.lease_id, lease.fencing_token)
        self.allocator.begin_reclamation(lease.lease_id, new_token)
        self.assertEqual(
            self.allocator.mark_unknown(lease.lease_id, new_token, "adapter disconnected")["state"],
            LeaseState.UNKNOWN.value,
        )
        with self.assertRaises(ValidationError):
            self.allocator.verify_release(lease.lease_id, new_token, {"verified": False})
        with self.assertRaises(ValidationError):
            self.allocator.verify_release(
                lease.lease_id, new_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": new_token - 1, "resource_ids": ["gpu-0"]},
            )
        with self.assertRaises(ValidationError):
            self.allocator.verify_release(
                lease.lease_id, new_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": new_token, "resource_ids": ["gpu-0", "gpu-other"]},
            )
        released = self.allocator.verify_release(
            lease.lease_id, new_token,
            {"verified": True, "lease_id": lease.lease_id, "fencing_token": new_token, "resource_ids": ["gpu-0"]},
        )
        self.assertEqual(released["state"], LeaseState.RELEASED.value)
        self.assertEqual(
            self.allocator.verify_release(
                lease.lease_id, new_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": new_token, "resource_ids": ["gpu-0"]},
            )["state"],
            LeaseState.RELEASED.value,
        )
        with self.assertRaises(ConflictError):
            self.allocator.verify_release(
                lease.lease_id, new_token,
                {"verified": True, "lease_id": lease.lease_id, "fencing_token": new_token, "resource_ids": ["gpu-0", "gpu-other"]},
            )

    def test_renewal_rotates_fence_and_expired_lease_cannot_be_revived(self) -> None:
        lease = self.allocator.request_lease(self.request("renew", "owner", 20))
        renewed = self.allocator.renew_lease(lease.lease_id, lease.fencing_token, 90)
        self.assertGreater(renewed.fencing_token, lease.fencing_token)
        with self.assertRaises(ConflictError):
            self.allocator.request_release(lease.lease_id, lease.fencing_token)
        self.now += timedelta(seconds=91)
        with self.assertRaises(ConflictError):
            self.allocator.renew_lease(lease.lease_id, renewed.fencing_token, 90)
        expired = self.allocator.get_lease(lease.lease_id)
        self.assertEqual(expired.state, LeaseState.RECLAIMING)
        self.assertGreater(expired.fencing_token, renewed.fencing_token)

    def test_reservation_expiry_rotates_fence_only_once(self) -> None:
        reservation = self.allocator.reserve_envelope(
            "expiring-envelope", "handoff", (self.gpu_requirement(20),), 5,
            idempotency_key="expiring-envelope",
        )
        self.now += timedelta(seconds=6)
        self.assertEqual(self.allocator.expire_due(), ["expiring-envelope"])
        expired = self.ledger.get("reservation", "expiring-envelope")
        self.assertEqual(expired["state"], "reclaiming")
        self.assertGreater(expired["fencing_token"], reservation["fencing_token"])
        self.assertEqual(self.allocator.expire_due(), [])
        self.assertEqual(
            self.ledger.get("reservation", "expiring-envelope")["fencing_token"],
            expired["fencing_token"],
        )

    def test_selectors_and_conflict_domains_are_enforced(self) -> None:
        self.allocator.register_resource(
            Resource(
                "gpu-1", "host-a", "gpu", {"memory_mib": 100.0},
                attributes={"model": "B"}, conflict_domains=frozenset({"nvlink-pair-0"}),
            )
        )
        self.allocator.register_resource(
            Resource(
                "gpu-0", "host-a", "gpu", {"memory_mib": 100.0, "compute_slots": 8.0},
                attributes={**self.allocator.get_resource("gpu-0").attributes, "model": "A"},
                conflict_domains=frozenset({"nvlink-pair-0"}),
            )
        )
        first = ResourceRequirement(
            kind="gpu", quantities={"memory_mib": 15}, selectors={"model": "A"},
            topology_constraints={"exclusive_conflict_domains": ["nvlink-pair-0"]},
        )
        lease = self.allocator.request_lease(ResourceRequest("topo-1", "one", (first,), 60))
        self.assertEqual(lease.allocations[0].resource_id, "gpu-0")
        second = ResourceRequirement(
            kind="gpu", quantities={"memory_mib": 15}, selectors={"resource_id": "gpu-1"}
        )
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(ResourceRequest("topo-2", "two", (second,), 60))

    def test_exclusive_device_blocks_shared_allocation(self) -> None:
        exclusive = ResourceRequirement(
            kind="gpu", quantities={"memory_mib": 1}, mode=AllocationMode.EXCLUSIVE
        )
        self.allocator.request_lease(ResourceRequest("exclusive", "one", (exclusive,), 60))
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(ResourceRequest("shared", "two", (self.gpu_requirement(10),), 60))

    def test_atomic_bundle_failure_does_not_leave_partial_claims(self) -> None:
        self.allocator.register_resource(Resource("cpu-0", "host-a", "cpu", {"cores": 1}))
        self.allocator.request_lease(self.request("fill", "one", 80))
        bundle = ResourceRequest(
            "bundle", "two",
            (
                ResourceRequirement(kind="cpu", quantities={"cores": 1}),
                self.gpu_requirement(20),
            ),
            60,
        )
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(bundle)
        self.assertEqual(self.allocator.list_leases(state=LeaseState.ACTIVE).__len__(), 1)
        self.assertEqual(self.ledger.get("resource", "cpu-0")["resource_id"], "cpu-0")

    def test_transferred_envelope_settles_to_fenced_backend_lease_once(self) -> None:
        source = self.allocator.request_lease(self.request("residency", "backend-owner", 40))
        self.ledger.put("backend", "backend-1", {"backend_id": "backend-1", "state": "ready", "lease_id": source.lease_id})
        reservation = self.allocator.transfer_leases_to_envelope(
            [source.lease_id], "handoff-1", "handoff-owner", 120,
            idempotency_key="transfer-1", requirements=(self.gpu_requirement(60),),
            backend_lease_ids={"backend-1": source.lease_id},
        )
        self.assertEqual(reservation["allocations"][0]["quantities"]["memory_mib"], 60)
        backend = self.ledger.get("backend", "backend-1")
        self.assertIsNone(backend["lease_id"])
        self.assertEqual(backend["state"], "draining")
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(self.request("during-handoff", "other", 40))
        phase = self.allocator.phase_lease("handoff-1", "run-phase", (self.gpu_requirement(55),))
        self.assertEqual(phase["state"], "active")
        self.allocator.release_phase("handoff-1", "run-phase")
        receipt = {"verified": True, "backend_ready": True, "reservation_id": "handoff-1"}
        result = self.allocator.settle_envelope_to_leases("handoff-1", ttl_seconds=120, receipt=receipt)
        repeated = self.allocator.settle_envelope_to_leases("handoff-1", ttl_seconds=120, receipt=receipt)
        self.assertEqual(result, repeated)
        self.assertEqual(result[0]["source_lease_id"], source.lease_id)
        restored = result[0]["lease"]
        self.assertNotEqual(restored["lease_id"], source.lease_id)
        self.assertGreater(restored["fencing_token"], source.fencing_token)
        backend = self.ledger.get("backend", "backend-1")
        self.assertEqual(backend["lease_id"], restored["lease_id"])
        self.assertEqual(backend["state"], "ready")
        self.assertEqual(
            len([item for item in self.allocator.list_leases(state=LeaseState.ACTIVE) if item.owner_id == "backend-owner"]),
            1,
        )

    def test_unmanaged_compute_process_blocks_whole_device_exclusivity(self) -> None:
        resource = Resource(
            "gpu-0", "host-a", "gpu", {"memory_mib": 100.0},
            attributes={
                **self.allocator.get_resource("gpu-0").attributes,
                "process_count_observation": 1,
                "unmanaged_process_count": 1,
                "managed_process_ids": [],
            },
        )
        self.allocator.register_resource(resource)
        exclusive = ResourceRequirement(kind="gpu", quantities={"memory_mib": 1}, mode=AllocationMode.EXCLUSIVE)
        self.assertFalse(self.allocator.can_allocate((exclusive,)))
        with self.assertRaises(ConflictError):
            self.allocator.request_lease(ResourceRequest("exclusive-with-unmanaged", "one", (exclusive,), 60))

    def test_unknown_or_stale_gpu_observation_blocks_admission(self) -> None:
        current = self.allocator.get_resource("gpu-0")
        attrs = dict(current.attributes)
        attrs["process_telemetry_state"] = "unknown"
        self.allocator.register_resource(Resource(
            current.resource_id, current.host_id, current.kind, current.capacities, attributes=attrs,
            topology=current.topology, conflict_domains=current.conflict_domains, capabilities=current.capabilities,
        ))
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(10),)))
        attrs["process_telemetry_state"] = "observed"
        attrs["inventory_observed_at"] = (self.now - timedelta(seconds=31)).isoformat()
        self.allocator.register_resource(Resource(
            current.resource_id, current.host_id, current.kind, current.capacities, attributes=attrs,
            topology=current.topology, conflict_domains=current.conflict_domains, capabilities=current.capabilities,
        ))
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(10),)))

    def test_shared_gpu_budget_is_bounded_by_observed_free_memory(self) -> None:
        current = self.allocator.get_resource("gpu-0")
        attrs = dict(current.attributes)
        attrs["memory_free_bytes_observation"] = 5 * 1024 * 1024
        self.allocator.register_resource(Resource(
            current.resource_id, current.host_id, current.kind, current.capacities, attributes=attrs,
            topology=current.topology, conflict_domains=current.conflict_domains, capabilities=current.capabilities,
        ))
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(10),)))

    def test_claims_created_after_gpu_observation_reduce_observed_free_budget(self) -> None:
        current = self.allocator.get_resource("gpu-0")
        attrs = dict(current.attributes)
        attrs["inventory_observed_at"] = (self.now - timedelta(seconds=1)).isoformat()
        attrs["process_observed_at"] = (self.now - timedelta(seconds=1)).isoformat()
        attrs["memory_free_bytes_observation"] = 60 * 1024 * 1024
        self.allocator.register_resource(Resource(
            current.resource_id, current.host_id, current.kind, current.capacities, attributes=attrs,
            topology=current.topology, conflict_domains=current.conflict_domains, capabilities=current.capabilities,
        ))
        self.allocator.request_lease(self.request("recent-claim", "one", 40))
        # Ledger total has room (40 + 20 < 90 MiB), but the current observed
        # free-memory budget is only 54 MiB after headroom.
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(20),)))

    def test_successful_empty_inventory_marks_resource_unavailable_but_keeps_lease_claim(self) -> None:
        resource = self.allocator.get_resource("gpu-0")
        self.allocator.reconcile_resource_inventory("nvidia-smi", [resource], observed_at=self.now)
        stored_resource = self.ledger.get("resource", "gpu-0")
        self.assertNotIn("available", stored_resource)
        self.assertNotIn("source_adapter_id", stored_resource)
        lease = self.allocator.request_lease(self.request("inventory-held", "owner", 20))
        self.allocator.reconcile_resource_inventory("nvidia-smi", [], observed_at=self.now + timedelta(seconds=1))
        self.assertEqual(self.allocator.list_resources(), [])
        self.assertEqual(len(self.allocator.list_resources(available_only=False)), 1)
        status = self.allocator.list_resource_status()[0]
        self.assertEqual(status["resource"]["resource_id"], "gpu-0")
        self.assertFalse(status["available"])
        self.assertEqual(status["source_adapter_id"], "nvidia-smi")
        self.assertEqual(status["availability_reason"], "resource_absent_from_successful_inventory")
        self.assertEqual(self.allocator.get_lease(lease.lease_id).state, LeaseState.ACTIVE)
        self.assertFalse(self.allocator.can_allocate((self.gpu_requirement(10),)))

    def test_owner_context_survives_lease_transfer_and_settlement(self) -> None:
        request = ResourceRequest(
            "context-residency", "backend-owner", (self.gpu_requirement(40),), 60,
            owner_context={"issuer": "external-bridge", "backend_id": "backend-1"},
        )
        source = self.allocator.request_lease(request)
        self.ledger.put("backend", "backend-1", {"backend_id": "backend-1", "state": "ready", "lease_id": source.lease_id})
        reservation = self.allocator.transfer_leases_to_envelope(
            [source.lease_id], "context-handoff", "handoff-owner", 120,
            idempotency_key="context-transfer", requirements=(self.gpu_requirement(60),),
            backend_lease_ids={"backend-1": source.lease_id},
        )
        self.assertEqual(reservation["owner_context"], dict(source.owner_context))
        self.allocator.phase_lease("context-handoff", "context-run", (self.gpu_requirement(55),))
        self.allocator.release_phase("context-handoff", "context-run")
        unknown = self.allocator.mark_envelope_unknown("context-handoff", "restore receipt was delayed")
        renewed = self.allocator.renew_envelope(
            "context-handoff", int(unknown["fencing_token"]), 240
        )
        self.assertEqual(renewed["state"], "unknown")
        result = self.allocator.settle_envelope_to_leases(
            "context-handoff", ttl_seconds=120,
            receipt={"verified": True, "backend_ready": True, "reservation_id": "context-handoff"},
        )
        self.assertEqual(result[0]["lease"]["owner_context"], dict(source.owner_context))

    def test_phase_lease_enforces_same_host_and_same_topology_bundle_constraints(self) -> None:
        resources = [
            Resource("gpu-a", "host-a", "gpu", {"memory_mib": 100}, attributes={
                "inventory_observed_at": self.now.isoformat(), "process_observed_at": self.now.isoformat(),
                "memory_free_bytes_observation": 100 * 1024 * 1024, "process_telemetry": "compute-apps-visible",
                "process_telemetry_state": "observed", "process_count_observation": 0,
                "process_identity_unknown_count": 0, "unmanaged_process_count": 0, "managed_process_ids": [],
            }, topology={"node": "numa-a"}),
            Resource("gpu-b", "host-b", "gpu", {"memory_mib": 100}, attributes={
                "inventory_observed_at": self.now.isoformat(), "process_observed_at": self.now.isoformat(),
                "memory_free_bytes_observation": 100 * 1024 * 1024, "process_telemetry": "compute-apps-visible",
                "process_telemetry_state": "observed", "process_count_observation": 0,
                "process_identity_unknown_count": 0, "unmanaged_process_count": 0, "managed_process_ids": [],
            }, topology={"node": "numa-b"}),
        ]
        for item in resources:
            self.allocator.register_resource(item)
        envelope_reqs = (
            ResourceRequirement(kind="gpu", quantities={"memory_mib": 50}, selectors={"resource_id": "gpu-a"}),
            ResourceRequirement(kind="gpu", quantities={"memory_mib": 50}, selectors={"resource_id": "gpu-b"}),
        )
        # Envelopes may span hosts when every requirement explicitly opts out.
        envelope_reqs = tuple(ResourceRequirement(
            kind=req.kind, quantities=req.quantities, selectors=req.selectors, same_host=False
        ) for req in envelope_reqs)
        self.allocator.reserve_envelope("multi-host-envelope", "handoff", envelope_reqs, 120, idempotency_key="mh-env")
        phase_reqs = (
            ResourceRequirement(kind="gpu", quantities={"memory_mib": 10}, selectors={"resource_id": "gpu-a"}, same_host=True),
            ResourceRequirement(kind="gpu", quantities={"memory_mib": 10}, selectors={"resource_id": "gpu-b"}, same_host=True),
        )
        with self.assertRaises(ConflictError):
            self.allocator.phase_lease("multi-host-envelope", "bad-host-phase", phase_reqs)

        self.allocator.register_resource(Resource("cpu-a", "host-c", "cpu", {"cores": 8}, topology={"node": "numa-a"}))
        self.allocator.register_resource(Resource("cpu-b", "host-c", "cpu", {"cores": 8}, topology={"node": "numa-b"}))
        cpu_envelope = (
            ResourceRequirement(kind="cpu", quantities={"cores": 4}, selectors={"resource_id": "cpu-a"}, same_host=False),
            ResourceRequirement(kind="cpu", quantities={"cores": 4}, selectors={"resource_id": "cpu-b"}, same_host=False),
        )
        self.allocator.reserve_envelope("topology-envelope", "handoff", cpu_envelope, 120, idempotency_key="topo-env")
        cpu_phase = (
            ResourceRequirement(
                kind="cpu", quantities={"cores": 1}, selectors={"resource_id": "cpu-a"}, same_host=False,
                topology_constraints={"same_topology": ["node"]},
            ),
            ResourceRequirement(
                kind="cpu", quantities={"cores": 1}, selectors={"resource_id": "cpu-b"}, same_host=False,
                topology_constraints={"same_topology": ["node"]},
            ),
        )
        with self.assertRaises(ConflictError):
            self.allocator.phase_lease("topology-envelope", "bad-topology-phase", cpu_phase)

    def test_envelope_release_requires_reclamation_state_and_complete_evidence(self) -> None:
        reservation = self.allocator.reserve_envelope(
            "release-envelope", "handoff", (self.gpu_requirement(20),), 120, idempotency_key="release-envelope"
        )
        receipt = {"verified": True, "reservation_id": "release-envelope", "resource_ids": ["gpu-0"]}
        with self.assertRaises(ConflictError):
            self.allocator.verify_envelope_release("release-envelope", receipt)
        self.allocator.release_envelope("release-envelope")
        with self.assertRaises(ValidationError):
            self.allocator.verify_envelope_release(
                "release-envelope", {"verified": True, "reservation_id": "release-envelope", "resource_ids": []}
            )
        released = self.allocator.verify_envelope_release("release-envelope", receipt)
        self.assertEqual(released["state"], "released")
        self.assertEqual(self.allocator.verify_envelope_release("release-envelope", receipt)["state"], "released")
        with self.assertRaises(ConflictError):
            self.allocator.verify_envelope_release(
                "release-envelope", {**receipt, "resource_ids": ["gpu-0", "gpu-1"]}
            )

    def test_owner_context_rejects_secret_shaped_fields_at_allocator_boundary(self) -> None:
        with self.assertRaises(ValidationError):
            self.allocator.reserve_envelope(
                "bad-context", "handoff", (self.gpu_requirement(1),), 120,
                idempotency_key="bad-context", owner_context={"api_key": "do-not-store"},
            )


if __name__ == "__main__":
    unittest.main()
