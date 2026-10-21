from __future__ import annotations

from datetime import datetime, timedelta, timezone

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core import (
    BackendSpec,
    ExecutionContext,
    ExecutionPurpose,
    EvictionGrant,
    HandoffCoordinator,
    HandoffSpec,
    JobSpec,
    Resource,
    ResourceAllocator,
    ResourceRequest,
    ResourceRequirement,
)


class MutableClock:
    def __init__(self):
        self.value = datetime(2030, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeBackend:
    def __init__(self, clock):
        self.clock = clock
        self.active_generations = 0
        self.admission_closed = False
        self.admission_operation_id = None
        self.evicted = False
        self.restore_failure = False
        self.ready_failure = False
        self.calls = []
        self.restore_contexts = []
        self.baseline_calls = 0

    def inspect(self, backend):
        return {"active_generations": self.active_generations,
                "state": "observed", "admission_closed": self.admission_closed,
                "admission_operation_id": self.admission_operation_id,
                "admission_epoch": 1 if self.admission_operation_id else 0,
                "admission_state": "closed" if self.admission_closed else "open",
                "evicted": self.evicted}

    def drain(self, backend, *, operation_id, timeout_seconds):
        self.calls.append("drain")
        self.admission_closed = True
        self.admission_operation_id = operation_id
        return {"state": "admission_closed", "admission_closed": True,
                "operation_id": operation_id, "evidence": "fake-close"}

    def capture_reclamation_baseline(self, backend, *, operation_id):
        self.baseline_calls += 1
        sampled_at = self.clock().isoformat()
        return {
            "state": "observed",
            "backend_id": backend.backend_id,
            "operation_id": operation_id,
            "profile_id": backend.profile_id,
            "profile_version": backend.pinned_configuration["profile_version"],
            "observed_at": sampled_at,
            "source": "test-fixture",
            "execution_identity": {
                "unit_name": "test-backend.service",
                "invocation_id": "invocation-1",
                "control_group": "/user.slice/test-backend.service",
            },
            "process_mapping_state": "observed",
            "gpus": [{
                "resource_id": "host0/gpu0",
                "gpu_uuid": "GPU-test-0",
                "memory_free_bytes": 8192,
                "used_gpu_memory_bytes": 1024,
                "backend_process_ids": [4321],
                "process_ids": [4321],
                "inventory_observed_at": sampled_at,
                "process_observed_at": sampled_at,
            }],
            "evidence": [{"kind": "fixture", "id": "baseline"}],
            "limitations": ["test fixture only"],
        }

    def evict(self, backend, *, operation_id, reclamation_baseline=None):
        self.calls.append("evict")
        assert reclamation_baseline is not None
        assert reclamation_baseline["backend_id"] == backend.backend_id
        self.evicted = True
        return {"state": "evicted", "cleanup_verified": True,
                "operation_id": operation_id,
                "resource_reclamation": {"state": "reclaimed", "evidence": "fake-reclaim"}}

    def restore(self, backend, *, operation_id, execution_context=None):
        self.calls.append("restore")
        self.restore_contexts.append(execution_context)
        if not isinstance(execution_context, ExecutionContext):
            return {"state": "failed", "retry_safe": False, "operation_id": operation_id}
        if self.restore_failure:
            return {"state": "failed", "retry_safe": True,
                    "operation_id": operation_id, "evidence": "no unit started"}
        self.evicted = False
        return {"state": "restored", "operation_id": operation_id,
                "profile_version": "v1", "execution_handle": {"adapter_id": "fake", "external_id": "unit-2"}}

    def ready(self, backend, *, timeout_seconds, operation_id=None):
        self.calls.append("ready")
        if self.ready_failure:
            return {"ready": False, "state": "unknown", "operation_id": operation_id,
                    "evidence": "temporary health observer failure"}
        return {"ready": not self.evicted, "state": "ready" if not self.evicted else "starting",
                "operation_id": operation_id, "evidence": "fake-ready"}

    def reopen(self, backend, *, operation_id):
        self.calls.append("reopen")
        self.admission_closed = False
        self.admission_operation_id = operation_id
        return {"state": "open", "admission_open": True, "operation_id": operation_id}


class FakeContinuations:
    def __init__(self):
        self.calls = []
        self.deliveries = {}
        self.barriers = {}

    def suspend(self, ref, *, handoff_id, deadline):
        self.calls.append(("suspend", ref))
        result = {"safe_boundary": True, "durable": True, "state": "acknowledged",
                  "handoff_id": handoff_id}
        self.barriers[(ref, handoff_id)] = result
        return result

    def inspect(self, ref, *, handoff_id):
        return self.barriers.get((ref, handoff_id), {
            "state": "absent", "safe_boundary": False, "durable": False,
            "delivery_id": None,
        })

    def deliver(self, ref, *, delivery_id, result):
        old = self.deliveries.get((ref, delivery_id))
        if old is not None and old != dict(result):
            raise AssertionError("delivery key was reused with different content")
        self.deliveries[(ref, delivery_id)] = dict(result)
        self.calls.append(("deliver", ref))
        handoff_id = result["handoff_id"]
        self.barriers[(ref, handoff_id)] = {
            "state": "delivered", "safe_boundary": True, "durable": True,
            "delivery_id": delivery_id,
        }
        return {"delivered": True, "durable": True, "delivery_id": delivery_id}

    def resume(self, ref, *, delivery_id):
        assert (ref, delivery_id) in self.deliveries
        self.calls.append(("resume", ref))
        handoff_id = self.deliveries[(ref, delivery_id)]["handoff_id"]
        self.barriers[(ref, handoff_id)] = {
            "state": "resumed", "safe_boundary": True, "durable": True,
            "delivery_id": delivery_id,
        }
        return {"resumed": True, "delivery_id": delivery_id}


class FakeJobs:
    def __init__(self):
        self.by_key = {}
        self.jobs = {}
        self.submit_calls = 0
        self.cancel_calls = []
        self.pending = False

    def submit(self, spec, owner, key, *, reservation_id):
        if key in self.by_key:
            return self.jobs[self.by_key[key]]
        self.submit_calls += 1
        job_id = f"job-{len(self.jobs) + 1}"
        record = {"id": job_id, "owner": owner, "spec": spec,
                  "state": "running" if self.pending else "succeeded",
                  "result": None if self.pending else {"exit_code": 0},
                  "cleanup_verified": not self.pending,
                  "reservation_id": reservation_id}
        self.by_key[key] = job_id
        self.jobs[job_id] = record
        return record

    def get(self, job_id):
        return self.jobs[job_id]

    def cancel(self, job_id, reason):
        self.cancel_calls.append((job_id, reason))
        self.jobs[job_id]["state"] = "cancelled"
        self.jobs[job_id]["cleanup_verified"] = True
        return self.jobs[job_id]


def make_fixture(tmp_path, *, attached_consumer="app/frame"):
    clock = MutableClock()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    allocator = ResourceAllocator(ledger, clock=clock)
    allocator.register_resource(Resource(
        resource_id="cpu0", host_id="host0", kind="cpu", capacities={"cores": 8},
    ))
    residency = ResourceRequirement("cpu", {"cores": 2})
    lease = allocator.request_lease(ResourceRequest(
        request_id="backend-residency", owner_id="backend-owner", requirements=(residency,),
        lease_ttl_seconds=3600, idempotency_key="backend-residency",
    ))
    backend_spec = BackendSpec(
        backend_id="model-backend", host_id="host0", adapter_id="fake", profile_id="llama",
        configuration_ref="model-config", residency_requirements=(residency,),
        attached_consumers=(attached_consumer,), pinned_configuration={"profile_version": "v1"},
    )
    ledger.put("backend", backend_spec.backend_id, {
        "id": backend_spec.backend_id, "owner": "operator", "spec": backend_spec.to_dict(),
        "state": "ready", "lease_id": lease.lease_id, "execution_handle": {"external_id": "unit"},
    })
    clockwork = FakeBackend(clock)
    continuations = FakeContinuations()
    jobs = FakeJobs()
    run_envelope = ResourceRequirement("cpu", {"cores": 4})
    spec = HandoffSpec(
        handoff_id="test-handoff", owner_id="owner",
        jobs=(JobSpec(owner_id="owner", argv=("true",), working_directory="/tmp"),),
        backend_ids=(backend_spec.backend_id,), continuation_refs=("app/frame",),
        eviction_grants=(EvictionGrant(
            grant_id="grant-1", backend_id=backend_spec.backend_id, granted_by="owner",
            scope=f"backend:{backend_spec.backend_id}", expires_at=clock() + timedelta(hours=1),
        ),),
        resource_envelope=(run_envelope,), drain_timeout_seconds=10,
        run_timeout_seconds=100, restore_timeout_seconds=10,
    )
    coordinator = HandoffCoordinator(
        ledger, allocator, clockwork, continuations, jobs.submit, jobs.get, clock,
        cancel_job=jobs.cancel,
    )
    return clock, ledger, allocator, clockwork, continuations, jobs, spec, coordinator


def finish(coordinator, handoff_id, *, restart_each_tick=False, factories=None, max_ticks=40):
    for _ in range(max_ticks):
        record = coordinator.get(handoff_id)
        if record["state"] in {"complete", "blocked", "unknown", "cancelled"}:
            return coordinator, record
        coordinator.tick()
        if restart_each_tick and factories:
            coordinator = factories()
    return coordinator, coordinator.get(handoff_id)


def test_handoff_drains_evicts_runs_restores_then_delivers_and_resumes(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    requested = coordinator.submit(spec, "owner", "submit-1")
    assert requested["state"] == "requested"
    assert coordinator.submit(spec, "owner", "submit-1")["handoff_id"] == spec.handoff_id

    coordinator, completed = finish(coordinator, spec.handoff_id)
    assert completed["state"] == "complete"
    assert completed["job_results"][completed["job_ids"][0]]["result"] == {"exit_code": 0}
    assert backend.calls.index("reopen") < len(backend.calls)
    context = backend.restore_contexts[0]
    assert context.purpose == ExecutionPurpose.BACKEND
    assert context.claim_id == completed["reservation_id"]
    assert context.fencing_token > 0
    assert [allocation.resource_id for allocation in context.allocations] == ["cpu0"]
    assert [resource.resource_id for resource in context.resources] == ["cpu0"]
    assert context.allocations[0].quantities["cores"] == 2
    assert continuations.calls.index(("deliver", "app/frame")) < continuations.calls.index(("resume", "app/frame"))
    assert jobs.submit_calls == 1
    backend_record = ledger.get("backend", "model-backend")
    assert backend_record["state"] == "ready"
    assert backend_record["lease_id"]
    reservation = ledger.get("reservation", completed["reservation_id"])
    assert reservation["state"] == "released"


def test_restore_refuses_source_allocation_outside_protected_envelope(tmp_path):
    _clock, ledger, _allocator, backend, _continuations, _jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(30):
        coordinator.tick()
        if coordinator.get(spec.handoff_id)["phase"] == "restore":
            break
    assert coordinator.get(spec.handoff_id)["phase"] == "restore"

    reservation = ledger.get("reservation", coordinator.get(spec.handoff_id)["reservation_id"])
    corrupted = {key: value for key, value in reservation.items() if key != "_revision"}
    corrupted["allocations"] = []
    with ledger.transaction() as tx:
        tx.put("reservation", reservation["reservation_id"], corrupted, reservation["_revision"])

    blocked = coordinator.tick()["handoffs"][0]
    assert blocked["state"] == "blocked"
    assert "does not cover backend model-backend source allocation cpu0" in blocked["error"]
    assert "restore" not in backend.calls


def test_each_persisted_phase_survives_coordinator_reconstruction(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, _ = make_fixture(tmp_path)
    coordinator = HandoffCoordinator(ledger, allocator, backend, continuations, jobs.submit,
                                     jobs.get, clock, cancel_job=jobs.cancel)
    coordinator.submit(spec, "owner", "submit-1")

    def factory():
        return HandoffCoordinator(ledger, allocator, backend, continuations, jobs.submit,
                                  jobs.get, clock, cancel_job=jobs.cancel)

    coordinator, record = finish(coordinator, spec.handoff_id, restart_each_tick=True, factories=factory)
    assert record["state"] == "complete"
    assert jobs.submit_calls == 1
    assert backend.calls.count("evict") == 1
    assert backend.calls.count("restore") == 1
    assert backend.baseline_calls == 1


def test_drain_timeout_blocks_eviction_and_explicit_recovery_can_continue(tmp_path):
    clock, _ledger, _allocator, backend, _continuations, _jobs, spec, coordinator = make_fixture(tmp_path)
    backend.active_generations = 1
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(5):
        coordinator.tick()
    assert "evict" not in backend.calls
    clock.advance(11)
    coordinator.tick()
    blocked = coordinator.get(spec.handoff_id)
    assert blocked["state"] == "blocked"
    assert "turn boundary" in blocked["error"]
    assert "evict" not in backend.calls
    backend.active_generations = 0
    coordinator.recover(spec.handoff_id)
    coordinator, completed = finish(coordinator, spec.handoff_id)
    assert completed["state"] == "complete"


def test_restore_failure_preserves_completed_results_and_holds_reservation(tmp_path):
    clock, ledger, _allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    backend.restore_failure = True
    coordinator.submit(spec, "owner", "submit-1")
    coordinator, blocked = finish(coordinator, spec.handoff_id)
    assert blocked["state"] == "blocked"
    assert blocked["job_results"]
    assert ledger.get("reservation", blocked["reservation_id"])["state"] == "active"
    assert not any(action == "deliver" for action, _ in continuations.calls)

    backend.restore_failure = False
    coordinator.recover(spec.handoff_id)
    coordinator, completed = finish(coordinator, spec.handoff_id)
    assert completed["state"] == "complete"
    assert completed["job_results"] == blocked["job_results"]
    assert jobs.submit_calls == 1


def test_readiness_timeout_is_blocked_and_recoverable_without_restarting_jobs(tmp_path):
    clock, ledger, _allocator, backend, _continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    backend.ready_failure = True
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(20):
        coordinator.tick()
        if "ready" in backend.calls:
            clock.advance(11)
        if coordinator.get(spec.handoff_id)["state"] == "blocked":
            break
    blocked = coordinator.get(spec.handoff_id)
    assert blocked["state"] == "blocked"
    assert ledger.get("reservation", blocked["reservation_id"])["state"] == "active"
    assert jobs.submit_calls == 1

    backend.ready_failure = False
    coordinator.recover(spec.handoff_id)
    coordinator, completed = finish(coordinator, spec.handoff_id)
    assert completed["state"] == "complete"
    assert jobs.submit_calls == 1


def test_unknown_job_outcome_is_not_replayed_or_delivered(tmp_path):
    clock, ledger, _allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)

    def uncertain_submit(*args, **kwargs):
        jobs.submit(*args, **kwargs)
        raise OSError("response lost after durable acceptance")

    coordinator = HandoffCoordinator(ledger, _allocator, backend, continuations,
                                     uncertain_submit, jobs.get, clock, cancel_job=jobs.cancel)
    coordinator.submit(spec, "owner", "submit-1")
    coordinator, unknown = finish(coordinator, spec.handoff_id)
    assert unknown["state"] == "unknown"
    assert jobs.submit_calls == 1
    assert not any(action == "resume" for action, _ in continuations.calls)
    assert ledger.get("reservation", unknown["reservation_id"])["state"] == "active"


def test_cancel_before_eviction_reopens_admission_without_unloading_backend(tmp_path):
    _clock, ledger, _allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    coordinator.tick()  # transfer backend lease to the protected envelope
    coordinator.tick()  # close admission
    coordinator.cancel(spec.handoff_id, "caller cancelled before tests")

    coordinator, cancelled = finish(coordinator, spec.handoff_id)
    assert cancelled["state"] == "cancelled"
    assert "evict" not in backend.calls
    assert "restore" not in backend.calls
    assert jobs.submit_calls == 0
    assert not backend.admission_closed
    assert ledger.get("reservation", cancelled["reservation_id"])["state"] == "released"
    assert not continuations.calls


def test_cancel_running_batch_waits_for_verified_job_cleanup_then_restores(tmp_path):
    _clock, ledger, _allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    jobs.pending = True
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(12):
        coordinator.tick()
        if coordinator.get(spec.handoff_id)["phase"] == "wait_jobs":
            break
    running = coordinator.get(spec.handoff_id)
    assert running["phase"] == "wait_jobs"
    coordinator.cancel(spec.handoff_id, "caller cancelled running tests")

    coordinator, cancelled = finish(coordinator, spec.handoff_id)
    assert cancelled["state"] == "cancelled"
    assert jobs.cancel_calls == [(running["job_ids"][0], "caller cancelled running tests")]
    assert cancelled["job_results"][running["job_ids"][0]]["state"] == "cancelled"
    assert backend.calls.count("restore") == 1
    assert ledger.get("reservation", cancelled["reservation_id"])["state"] == "released"
    assert continuations.calls.index(("deliver", "app/frame")) < continuations.calls.index(("resume", "app/frame"))


def test_unresolved_operation_intent_is_not_replayed_after_process_reconstruction(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(5):
        coordinator.tick()
    assert coordinator.get(spec.handoff_id)["phase"] == "evict"

    operation_id = coordinator._operation_id(spec.handoff_id, "evict", "model-backend", 0)
    original = ledger.operation_intent
    calls = {"count": 0}

    def persist_then_crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("operation_id") == operation_id:
            calls["count"] += 1
            raise RuntimeError("simulated process loss after durable intent")
        return result

    ledger.operation_intent = persist_then_crash
    try:
        coordinator.tick()
    except RuntimeError:
        pass
    ledger.operation_intent = original

    restarted = HandoffCoordinator(ledger, allocator, backend, continuations,
                                   jobs.submit, jobs.get, clock, cancel_job=jobs.cancel)
    result = restarted.tick()["handoffs"][0]
    assert result["state"] == "unknown"
    assert backend.calls.count("evict") == 0
    assert ledger.get("reservation", result["reservation_id"])["state"] == "active"


def test_missing_reclamation_baseline_blocks_before_evict_intent(tmp_path):
    _clock, ledger, _allocator, backend, _continuations, _jobs, spec, coordinator = make_fixture(tmp_path)
    backend.capture_reclamation_baseline = None
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(5):
        coordinator.tick()
    blocked = coordinator.tick()["handoffs"][0]
    assert blocked["state"] == "blocked"
    assert "pre-eviction resource reclamation baseline" in blocked["error"]
    assert "evict" not in backend.calls
    operation_id = coordinator._operation_id(spec.handoff_id, "evict", "model-backend", 0)
    assert ledger.operation(operation_id) is None


def test_baseline_is_persisted_before_any_evict_intent(tmp_path):
    _clock, ledger, _allocator, backend, _continuations, _jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(5):
        coordinator.tick()
    record = coordinator.get(spec.handoff_id)
    assert record["phase"] == "evict"
    baseline = record["reclamation_baselines"]["model-backend"]
    assert baseline["execution_identity"]["invocation_id"] == "invocation-1"
    assert backend.calls.count("evict") == 0
    operation_id = coordinator._operation_id(spec.handoff_id, "evict", "model-backend", 0)
    assert ledger.operation(operation_id) is None


def test_stale_baseline_is_recaptured_before_eviction_and_never_after_intent(tmp_path):
    clock, _ledger, _allocator, backend, _continuations, _jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(5):
        coordinator.tick()
    assert coordinator.get(spec.handoff_id)["phase"] == "evict"
    clock.advance(31)
    coordinator.tick()
    assert coordinator.get(spec.handoff_id)["phase"] == "capture_baseline"
    coordinator.tick()
    baseline = coordinator.get(spec.handoff_id)["reclamation_baselines"]["model-backend"]
    assert baseline["operation_id"] == f"{spec.handoff_id}:baseline:1"
    assert backend.baseline_calls == 2
    coordinator, completed = finish(coordinator, spec.handoff_id)
    assert completed["state"] == "complete"
    assert backend.calls.count("evict") == 1
    assert backend.baseline_calls == 2


def test_unresolved_suspend_intent_retries_only_after_barrier_inspection_proves_absent(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    coordinator.tick()  # transfer residency into the protected envelope
    coordinator.tick()  # close backend admission
    assert coordinator.get(spec.handoff_id)["phase"] == "suspend_consumers"

    original = ledger.operation_intent

    def persist_then_crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if args and args[0] == "handoff.suspend":
            raise RuntimeError("simulated process loss after durable intent")
        return result

    ledger.operation_intent = persist_then_crash
    try:
        coordinator.tick()
    except RuntimeError:
        pass
    finally:
        ledger.operation_intent = original
    assert not continuations.calls

    restarted = HandoffCoordinator(ledger, allocator, backend, continuations,
                                   jobs.submit, jobs.get, clock, cancel_job=jobs.cancel)
    restarted.tick()
    assert continuations.calls.count(("suspend", "app/frame")) == 1
    restarted, completed = finish(restarted, spec.handoff_id)
    assert completed["state"] == "complete"


def test_unresolved_admission_close_retries_after_inspection_proves_gate_open(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    coordinator.tick()  # transfer residency into the protected envelope
    original = ledger.operation_intent

    def persist_then_crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if args and args[0] == "handoff.drain":
            raise RuntimeError("simulated process loss after durable intent")
        return result

    ledger.operation_intent = persist_then_crash
    try:
        coordinator.tick()
    except RuntimeError:
        pass
    finally:
        ledger.operation_intent = original
    assert not backend.calls

    restarted = HandoffCoordinator(ledger, allocator, backend, continuations,
                                   jobs.submit, jobs.get, clock, cancel_job=jobs.cancel)
    restarted.tick()
    assert backend.calls.count("drain") == 1
    restarted, completed = finish(restarted, spec.handoff_id)
    assert completed["state"] == "complete"


def test_unresolved_delivery_intent_recovers_from_durable_frame_shift_receipt(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(20):
        coordinator.tick()
        if coordinator.get(spec.handoff_id)["phase"] == "deliver":
            break
    assert coordinator.get(spec.handoff_id)["phase"] == "deliver"

    operation_id = coordinator._operation_id(spec.handoff_id, "deliver", "app/frame", 0)
    original_transaction = ledger.transaction

    class TransactionProxy:
        def __init__(self):
            self.context = original_transaction()
            self.tx = None

        def __enter__(self):
            self.tx = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.tx, name)

        def record_receipt(self, candidate_id, receipt, *, status="complete"):
            if candidate_id == operation_id:
                raise RuntimeError("simulated process loss before recording delivery receipt")
            return self.tx.record_receipt(candidate_id, receipt, status=status)

    ledger.transaction = TransactionProxy
    try:
        try:
            coordinator.tick()
        except RuntimeError:
            pass
    finally:
        ledger.transaction = original_transaction
    assert continuations.calls.count(("deliver", "app/frame")) == 1
    assert ledger.operation(operation_id)["status"] == "intent"

    restarted = HandoffCoordinator(ledger, allocator, backend, continuations,
                                   jobs.submit, jobs.get, clock, cancel_job=jobs.cancel)
    restarted.tick()
    assert continuations.calls.count(("deliver", "app/frame")) == 1
    restarted, completed = finish(restarted, spec.handoff_id)
    assert completed["state"] == "complete"


def test_unresolved_resume_intent_recovers_from_durable_frame_shift_receipt(tmp_path):
    clock, ledger, allocator, backend, continuations, jobs, spec, coordinator = make_fixture(tmp_path)
    coordinator.submit(spec, "owner", "submit-1")
    for _ in range(24):
        coordinator.tick()
        if coordinator.get(spec.handoff_id)["phase"] == "resume":
            break
    assert coordinator.get(spec.handoff_id)["phase"] == "resume"

    operation_id = coordinator._operation_id(spec.handoff_id, "resume", "app/frame", 0)
    original_transaction = ledger.transaction

    class TransactionProxy:
        def __init__(self):
            self.context = original_transaction()
            self.tx = None

        def __enter__(self):
            self.tx = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.tx, name)

        def record_receipt(self, candidate_id, receipt, *, status="complete"):
            if candidate_id == operation_id:
                raise RuntimeError("simulated process loss before recording resume receipt")
            return self.tx.record_receipt(candidate_id, receipt, status=status)

    ledger.transaction = TransactionProxy
    try:
        try:
            coordinator.tick()
        except RuntimeError:
            pass
    finally:
        ledger.transaction = original_transaction
    assert continuations.calls.count(("resume", "app/frame")) == 1
    assert ledger.operation(operation_id)["status"] == "intent"

    restarted = HandoffCoordinator(ledger, allocator, backend, continuations,
                                   jobs.submit, jobs.get, clock, cancel_job=jobs.cancel)
    restarted.tick()
    assert continuations.calls.count(("resume", "app/frame")) == 1
    restarted, completed = finish(restarted, spec.handoff_id)
    assert completed["state"] == "complete"
