from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone

import pytest

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core.errors import IdempotencyConflictError, NotFoundError, PermissionDeniedError, ValidationError
from device_scheduler_core.models import EventTrigger, JobSpec, MisfirePolicy, OverlapPolicy, ScheduleSpec
from device_scheduler_core.scheduling import (
    ScheduleCoordinator,
    dependencies_ready,
    dependency_status,
    event_ready,
    scheduled_job_ready,
    validate_dependency_graph,
)

UTC = timezone.utc


class FakeClock:
    def __init__(self, at: datetime):
        self.at = at

    def now(self) -> datetime:
        return self.at

    def advance(self, **delta: float) -> None:
        self.at += timedelta(**delta)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _job_spec(owner: str = "alice") -> JobSpec:
    return JobSpec(owner_id=owner, argv=("/bin/true",), working_directory="/tmp")


def _setup(tmp_path, at: datetime):
    ledger = Ledger(tmp_path / "scheduler.sqlite3")
    clock = FakeClock(at)

    def submit(spec, owner, key):
        payload = dict(spec)
        job_id = "job-" + hashlib.sha256(f"{owner}\0{key}".encode()).hexdigest()[:20]

        def create(tx):
            return tx.put(
                "jobs",
                job_id,
                {
                    "id": job_id,
                    "owner": owner,
                    "spec": payload,
                    "state": "queued",
                    "created_at": _iso(clock.now()),
                    "attempt_id": None,
                },
                None,
            )

        return ledger.idempotent(owner, key, "job.submit", payload, create)

    return ledger, clock, ScheduleCoordinator(ledger, submit, clock)


def _schedule(
    schedule_id: str,
    owner: str,
    *,
    clock: FakeClock,
    interval: float | None = 60,
    calendar: str | None = None,
    timezone_name: str | None = None,
    event_trigger: EventTrigger | None = None,
    overlap: OverlapPolicy = OverlapPolicy.QUEUE,
    misfire: MisfirePolicy = MisfirePolicy.COALESCE,
    max_catch_up: int = 1,
    max_instances: int = 1,
) -> ScheduleSpec:
    return ScheduleSpec(
        schedule_id=schedule_id,
        owner_id=owner,
        job=_job_spec(owner),
        interval_seconds=interval,
        calendar_expression=calendar,
        timezone=timezone_name,
        event_trigger=event_trigger,
        overlap=overlap,
        misfire_policy=misfire,
        max_catch_up=max_catch_up,
        max_instances=max_instances,
    )


def _job_at(record):
    return datetime.fromisoformat(record["spec"]["not_before"].replace("Z", "+00:00"))


def test_interval_coalesces_missed_occurrences_and_tick_is_idempotent(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("timer", "alice", clock=clock), "alice", "schedule-create")
    clock.advance(minutes=5, seconds=30)

    first = coordinator.tick()
    second = coordinator.tick()

    assert len(first) == 1
    assert second == []
    assert _job_at(first[0]) == start + timedelta(minutes=5)
    assert first[0]["spec"]["idempotency_key"].startswith("schedule-occurrence:")
    assert scheduled_job_ready(ledger, first[0])
    assert len(ledger.list("jobs")) == 1
    misfires = [event for event in ledger.events(kind="schedule.misfire")]
    assert misfires and misfires[-1]["payload"]["policy"] == "coalesce"


def test_catch_up_materializes_only_the_configured_recent_backlog(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule(
        "catch-up",
        "alice",
        clock=clock,
        misfire=MisfirePolicy.CATCH_UP,
        max_catch_up=2,
        overlap=OverlapPolicy.ALLOW,
        max_instances=5,
    )
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(minutes=5, seconds=30)

    created = coordinator.tick()

    assert [_job_at(job) for job in created] == [start + timedelta(minutes=4), start + timedelta(minutes=5)]
    assert len(ledger.list("jobs")) == 2


def test_skip_misfire_advances_cursor_without_creating_jobs(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule("skip", "alice", clock=clock, misfire=MisfirePolicy.SKIP)
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(minutes=5)

    assert coordinator.tick() == []
    assert ledger.list("jobs") == []
    assert ledger.get("schedules", "skip")["last_nominal_at"] == _iso(start + timedelta(minutes=5))
    clock.advance(minutes=1)
    created = coordinator.tick()
    assert len(created) == 1
    assert _job_at(created[0]) == start + timedelta(minutes=6)


def test_forbid_overlap_skips_occurrence_then_continues_after_completion(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(
        _schedule("no-overlap", "alice", clock=clock, overlap=OverlapPolicy.FORBID),
        "alice",
        "schedule-create",
    )
    clock.advance(minutes=1)
    first = coordinator.tick()
    clock.advance(minutes=1)

    assert coordinator.tick() == []
    assert any(
        item["state"] == "skipped" and item["reason"] == "overlap_forbidden"
        for item in ledger.list("schedule_occurrences")
    )
    completed = dict(first[0])
    completed["state"] = "succeeded"
    ledger.put("jobs", completed["id"], {k: v for k, v in completed.items() if k != "_revision"}, completed["_revision"])
    clock.advance(minutes=1)

    next_job = coordinator.tick()
    assert len(next_job) == 1
    assert len(ledger.list("jobs")) == 2


def test_queue_overlap_keeps_one_pending_occurrence_until_active_job_finishes(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule("queue", "alice", clock=clock, overlap=OverlapPolicy.QUEUE)
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(minutes=1)
    first = coordinator.tick()
    clock.advance(minutes=1)

    assert coordinator.tick() == []
    queued = ledger.get("schedules", "queue")["pending_occurrences"]
    assert len(queued) == 1
    active = ledger.get("jobs", first[0]["id"])
    ledger.put("jobs", active["id"], {**{k: v for k, v in active.items() if k != "_revision"}, "state": "failed"}, active["_revision"])

    second = coordinator.tick()
    assert len(second) == 1
    assert len(ledger.list("jobs")) == 2


def test_default_overlap_coalesces_long_running_missed_occurrences(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("default-overlap", "alice", clock=clock), "alice", "schedule-create")

    clock.advance(minutes=1)
    first = coordinator.tick()
    assert len(first) == 1

    for _ in range(5):
        clock.advance(minutes=1)
        assert coordinator.tick() == []
        pending = ledger.get("schedules", "default-overlap")["pending_occurrences"]
        assert len(pending) == 1

    latest_missed = start + timedelta(minutes=6)
    assert datetime.fromisoformat(pending[0]["nominal_at"].replace("Z", "+00:00")) == latest_missed
    active = ledger.get("jobs", first[0]["id"])
    ledger.put(
        "jobs",
        active["id"],
        {**{key: value for key, value in active.items() if key != "_revision"}, "state": "succeeded"},
        active["_revision"],
    )

    next_job = coordinator.tick()
    assert len(next_job) == 1
    assert _job_at(next_job[0]) == latest_missed
    assert len(ledger.list("jobs")) == 2


def test_allow_overlap_respects_max_instances(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule(
        "allow",
        "alice",
        clock=clock,
        overlap=OverlapPolicy.ALLOW,
        max_instances=2,
    )
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(minutes=1)
    assert len(coordinator.tick()) == 1
    clock.advance(minutes=1)
    assert len(coordinator.tick()) == 1
    clock.advance(minutes=1)
    assert coordinator.tick() == []
    assert len(ledger.list("jobs")) == 2


def test_event_publication_recovers_after_job_submission_failure_and_restart(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    trigger = EventTrigger(source="worker", event_type="finished")
    coordinator.put(
        _schedule("event-recovery", "alice", clock=clock, interval=None, event_trigger=trigger),
        "alice",
        "schedule-create",
    )
    submit_job = coordinator.submit_job

    def fail_submit(*args):
        raise RuntimeError("simulated dispatcher interruption")

    coordinator.submit_job = fail_submit
    with pytest.raises(RuntimeError, match="interruption"):
        coordinator.publish_event("worker", "evt-1", {"type": "finished"}, "alice", "event-key")
    assert len(ledger.list("events")) == 1
    occurrences = ledger.list("schedule_occurrences")
    assert len(occurrences) == 1
    assert occurrences[0]["state"] == "submitting"
    assert occurrences[0]["job_spec"]

    recovered = ScheduleCoordinator(ledger, submit_job, clock)
    assert len(recovered.tick()) == 1
    assert recovered.tick() == []
    assert len(ledger.list("jobs")) == 1
    assert ledger.list("schedule_occurrences")[0]["state"] == "submitted"


def test_submission_intent_recovers_lost_ack_without_duplicate_job(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("ambiguous-submit", "alice", clock=clock), "alice", "schedule-create")
    clock.advance(minutes=1)
    submit_job = coordinator.submit_job
    calls = 0

    def lose_ack_after_commit(spec, owner, key):
        nonlocal calls
        calls += 1
        record = submit_job(spec, owner, key)
        if calls == 1:
            raise RuntimeError("lost submit acknowledgement")
        return record

    coordinator.submit_job = lose_ack_after_commit
    with pytest.raises(RuntimeError, match="lost submit acknowledgement"):
        coordinator.tick()
    assert len(ledger.list("jobs")) == 1
    assert ledger.list("schedule_occurrences")[0]["state"] == "submitting"

    recovered = ScheduleCoordinator(ledger, submit_job, clock)
    created = recovered.tick()
    assert len(created) == 1
    assert len(ledger.list("jobs")) == 1
    occurrence = ledger.list("schedule_occurrences")[0]
    assert occurrence["state"] == "submitted"
    assert occurrence["job_id"] == created[0]["id"]


def test_cancel_racing_job_submission_cancels_and_never_opens_dispatch_gate(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("cancel-race", "alice", clock=clock), "alice", "schedule-create")
    clock.advance(minutes=1)
    submit_job = coordinator.submit_job
    callback_created = threading.Event()
    allow_ack = threading.Event()
    tick_errors = []

    def blocked_ack(spec, owner, key):
        record = submit_job(spec, owner, key)
        callback_created.set()
        if not allow_ack.wait(timeout=5):
            raise RuntimeError("test timed out waiting for submit acknowledgement")
        return record

    def cancel_job(job_id, owner, key):
        payload = {"job_id": job_id}

        def apply(tx):
            job = tx.get("jobs", job_id)
            if job is None:
                raise NotFoundError("submitted job disappeared")
            if job["state"] not in {"succeeded", "failed", "cancelled"}:
                updated = {name: value for name, value in job.items() if name != "_revision"}
                updated["state"] = "cancelled"
                job = tx.put("jobs", job_id, updated, job["_revision"])
            return job

        return ledger.idempotent(owner, key, "job.cancel", payload, apply)

    coordinator.submit_job = blocked_ack
    coordinator.cancel_job = cancel_job

    def run_tick():
        try:
            coordinator.tick()
        except Exception as exc:  # asserted below so thread failures stay visible
            tick_errors.append(exc)

    worker = threading.Thread(target=run_tick)
    worker.start()
    try:
        assert callback_created.wait(timeout=5)
        job = ledger.list("jobs")[0]
        assert job["state"] == "queued"
        assert not scheduled_job_ready(ledger, job)
        coordinator.control("cancel-race", "cancel", "alice", "schedule-cancel")
        occurrence = ledger.list("schedule_occurrences")[0]
        assert occurrence["state"] == "cancel_requested"
        assert not scheduled_job_ready(ledger, job)
    finally:
        allow_ack.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert tick_errors == []
    schedule = ledger.get("schedules", "cancel-race")
    occurrence = ledger.list("schedule_occurrences")[0]
    job = ledger.get("jobs", occurrence["job_id"])
    assert schedule["state"] == "cancelled"
    assert occurrence["state"] == "cancelled"
    assert job["state"] == "cancelled"
    assert occurrence["job_id"] == job["id"]
    assert job["id"] not in schedule["active_job_ids"]
    assert not scheduled_job_ready(ledger, job)


def test_lost_cancel_ack_keeps_intent_blocked_until_retry_reconciles(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("cancel-recovery", "alice", clock=clock), "alice", "schedule-create")
    clock.advance(minutes=1)
    submit_job = coordinator.submit_job
    submit_calls = 0
    cancel_calls = 0

    def submit_then_cancel_schedule(spec, owner, key):
        nonlocal submit_calls
        submit_calls += 1
        job = submit_job(spec, owner, key)
        if submit_calls == 1:
            coordinator.control("cancel-recovery", "cancel", "alice", "schedule-cancel")
        return job

    def cancel_and_lose_ack(job_id, owner, key):
        nonlocal cancel_calls
        cancel_calls += 1
        payload = {"job_id": job_id}

        def apply(tx):
            job = tx.get("jobs", job_id)
            if job is None:
                raise NotFoundError("submitted job disappeared")
            if job["state"] not in {"succeeded", "failed", "cancelled"}:
                updated = {name: value for name, value in job.items() if name != "_revision"}
                updated["state"] = "cancelled"
                job = tx.put("jobs", job_id, updated, job["_revision"])
            return job

        result = ledger.idempotent(owner, key, "job.cancel", payload, apply)
        if cancel_calls == 1:
            raise RuntimeError("lost cancel acknowledgement")
        return result

    coordinator.submit_job = submit_then_cancel_schedule
    coordinator.cancel_job = cancel_and_lose_ack
    with pytest.raises(RuntimeError, match="lost cancel acknowledgement"):
        coordinator.tick()

    occurrence = ledger.list("schedule_occurrences")[0]
    job = ledger.get("jobs", occurrence["job_id"])
    assert occurrence["state"] == "cancel_requested"
    assert job["state"] == "cancelled"
    assert not scheduled_job_ready(ledger, job)

    recovered = ScheduleCoordinator(ledger, submit_job, clock, cancel_job=cancel_and_lose_ack)
    assert recovered.tick() == []
    occurrence = ledger.list("schedule_occurrences")[0]
    job = ledger.get("jobs", occurrence["job_id"])
    assert occurrence["state"] == "cancelled"
    assert job["state"] == "cancelled"
    assert len(ledger.list("jobs")) == 1
    assert cancel_calls == 2


def test_unknown_cancel_receipt_keeps_dispatch_gate_closed(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("unknown-cancel", "alice", clock=clock), "alice", "schedule-create")
    clock.advance(minutes=1)
    submit_job = coordinator.submit_job

    def submit_then_cancel_schedule(spec, owner, key):
        record = submit_job(spec, owner, key)
        coordinator.control("unknown-cancel", "cancel", "alice", "schedule-cancel")
        return record

    coordinator.submit_job = submit_then_cancel_schedule
    coordinator.cancel_job = lambda _job_id, _owner, _key: {"state": "unknown"}
    assert coordinator.tick() == []

    schedule = ledger.get("schedules", "unknown-cancel")
    occurrence = ledger.list("schedule_occurrences")[0]
    job = ledger.get("jobs", occurrence["job_id"])
    assert schedule["state"] == "cancelled"
    assert len(schedule["submission_intents"]) == 1
    assert occurrence["state"] == "cancel_requested"
    assert job["state"] == "queued"
    assert not scheduled_job_ready(ledger, job)


def test_schedule_pause_resume_and_cancel_control_generation(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _, clock, coordinator = _setup(tmp_path, start)
    coordinator.put(_schedule("controlled", "alice", clock=clock), "alice", "schedule-create")
    coordinator.control("controlled", "pause", "alice", "pause-key")
    clock.advance(minutes=2)
    assert coordinator.tick() == []
    coordinator.control("controlled", "resume", "alice", "resume-key")
    assert len(coordinator.tick()) == 1  # default coalesces the paused interval
    coordinator.control("controlled", "cancel", "alice", "cancel-key")
    clock.advance(minutes=1)
    assert coordinator.tick() == []


def test_calendar_spring_gap_is_skipped_and_later_local_time_runs(tmp_path):
    # 2026-03-29 02:30 does not exist in Europe/Berlin.
    start = datetime(2026, 3, 29, 0, 0, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule(
        "spring",
        "alice",
        clock=clock,
        interval=None,
        calendar="30 2 * * *",
        timezone_name="Europe/Berlin",
    )
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(hours=3)

    assert coordinator.tick() == []
    clock.advance(hours=22)
    created = coordinator.tick()
    assert len(created) == 1
    assert _job_at(created[0]) == datetime(2026, 3, 30, 0, 30, tzinfo=UTC)


def test_calendar_fall_fold_uses_one_nominal_wall_time(tmp_path):
    start = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    spec = _schedule(
        "fall",
        "alice",
        clock=clock,
        interval=None,
        calendar="30 2 * * *",
        timezone_name="Europe/Berlin",
    )
    coordinator.put(spec, "alice", "schedule-create")
    clock.advance(hours=2)

    created = coordinator.tick()
    assert len(created) == 1
    assert _job_at(created[0]) == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)


def test_event_publication_deduplicates_identity_and_materializes_matching_schedule(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    trigger = EventTrigger(source="build-system", event_type="build.complete", filters={"branch": "main"})
    spec = _schedule(
        "event-schedule",
        "alice",
        clock=clock,
        interval=None,
        event_trigger=trigger,
    )
    coordinator.put(spec, "alice", "schedule-create")

    first = coordinator.publish_event(
        "build-system", "evt-1", {"event_type": "build.complete", "branch": "main"}, "alice", "event-1"
    )
    repeated = coordinator.publish_event(
        "build-system", "evt-1", {"event_type": "build.complete", "branch": "main"}, "alice", "event-1-retry"
    )

    assert len(first["jobs"]) == 1
    assert repeated["jobs"] == []
    assert len(ledger.list("jobs")) == 1
    with pytest.raises(IdempotencyConflictError):
        coordinator.publish_event(
            "build-system", "evt-1", {"event_type": "build.complete", "branch": "release"}, "alice", "event-1-changed"
        )


def test_event_job_gate_requires_matching_event_after_submission(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ledger, clock, coordinator = _setup(tmp_path, start)
    trigger = EventTrigger(source="ci", event_type="test.finished", event_key="run-7", filters={"passed": True})
    assert not event_ready(ledger, trigger, created_at=start)
    coordinator.publish_event("ci", "run-7", {"type": "test.finished", "passed": True}, "alice", "e1")
    assert event_ready(ledger, trigger, created_at=start)
    assert not event_ready(ledger, trigger, created_at=start + timedelta(seconds=1))


def test_dependency_cycle_missing_and_job_result_conditions(tmp_path):
    ledger = Ledger(tmp_path / "jobs.sqlite3")
    ledger.put("jobs", "a", {"id": "a", "owner": "alice", "spec": {"dependencies": []}, "state": "succeeded"}, None)
    ledger.put("jobs", "b", {"id": "b", "owner": "alice", "spec": {"dependencies": ["a"]}, "state": "failed"}, None)
    validate_dependency_graph(ledger, "c", ["b"])
    with pytest.raises(NotFoundError):
        validate_dependency_graph(ledger, "d", ["missing"])
    with pytest.raises(ValidationError, match="itself"):
        validate_dependency_graph(ledger, "d", ["d"])

    # Simulate pre-existing/imported corrupt state; the validator must refuse to
    # extend a cyclic job-result graph rather than relying on task state.
    a = ledger.get("jobs", "a")
    ledger.put("jobs", "a", {**{k: v for k, v in a.items() if k != "_revision"}, "spec": {"dependencies": ["b"]}}, a["_revision"])
    with pytest.raises(ValidationError, match="cycle"):
        validate_dependency_graph(ledger, "d", ["a"])

    assert dependency_status(ledger, [{"job_id": "a", "condition": "completed"}])["ready"]
    assert not dependencies_ready(ledger, ["b"])


def test_schedule_owner_and_calendar_expression_validation(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _, clock, coordinator = _setup(tmp_path, start)
    with pytest.raises(ValidationError, match="five numeric cron"):
        coordinator.put(
            _schedule(
                "bad-cron",
                "alice",
                clock=clock,
                interval=None,
                calendar="@daily",
                timezone_name="UTC",
            ),
            "alice",
            "bad-cron-key",
        )
    spec = _schedule("owned", "alice", clock=clock)
    with pytest.raises(PermissionDeniedError):
        coordinator.put(spec, "bob", "wrong-owner")
