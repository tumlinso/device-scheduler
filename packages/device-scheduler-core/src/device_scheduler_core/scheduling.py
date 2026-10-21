"""Durable schedule, dependency, and event coordination.

The coordinator stores schedule progress before materializing work. Job creation
is delegated to the application dispatcher and uses a stable idempotency key, so
a crash between job creation and schedule progress update cannot create duplicate
jobs. Calendar rules intentionally implement the five-field numeric cron subset
using only the standard library.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import ConflictError, IdempotencyConflictError, NotFoundError, PermissionDeniedError, ValidationError
from .models import EventTrigger, JobSpec, MisfirePolicy, OverlapPolicy, ScheduleSpec, ScheduleState

UTC = timezone.utc
SCHEDULE_KIND = "schedules"
JOB_KIND = "jobs"
EVENT_KIND = "events"
OCCURRENCE_KIND = "schedule_occurrences"
_TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled"})
_ACTIVE_JOB_STATES = frozenset({"queued", "waiting", "blocked", "admitted", "running", "suspend_requested", "suspended", "cancelling", "unknown"})
_MAX_EVENT_BACKLOG = 100
_MAX_CALENDAR_DAYS = 36600


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _datetime(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"{name} must be an ISO-8601 datetime") from exc
    else:
        raise ValidationError(f"{name} must be a timezone-aware datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _now(clock: Any) -> datetime:
    value = clock.now() if hasattr(clock, "now") else clock()
    return _datetime(value, "clock.now()")


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _without_revision(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_revision"}


def _job_id(job: Mapping[str, Any]) -> str | None:
    value = job.get("job_id", job.get("id"))
    return value if isinstance(value, str) else None


def _job_state(job: Mapping[str, Any]) -> str:
    state = job.get("state", "unknown")
    return str(getattr(state, "value", state)).lower()


def _dependency_items(dependencies: Sequence[Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in dependencies:
        if isinstance(item, str):
            result.append((item, "success"))
            continue
        if not isinstance(item, Mapping):
            raise ValidationError("each dependency must be a job ID or an object")
        job_id = item.get("job_id", item.get("id"))
        condition = item.get("condition", "success")
        if not isinstance(job_id, str) or not job_id:
            raise ValidationError("dependency job_id must be a non-empty string")
        if condition not in {"success", "completed", "failure"}:
            raise ValidationError(f"unsupported dependency condition: {condition!r}")
        result.append((job_id, condition))
    if len({job_id for job_id, _ in result}) != len(result):
        raise ValidationError("dependencies must not repeat a job")
    return result


def validate_dependency_graph(
    ledger: Any,
    candidate_id: str,
    dependencies: Sequence[Any],
) -> None:
    """Reject missing dependencies, self-dependencies, and cycles.

    Dependencies point to job results. External workflow acceptance is not a
    job result and is deliberately not consulted here.
    """

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValidationError("candidate_id must be a non-empty string")
    candidate_dependencies = _dependency_items(dependencies)
    graph: dict[str, list[str]] = {}
    for record in ledger.list(JOB_KIND):
        job_id = _job_id(record)
        if not job_id:
            continue
        spec = record.get("spec") or {}
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        raw = spec.get("dependencies", ()) if isinstance(spec, Mapping) else ()
        graph[job_id] = [dependency for dependency, _ in _dependency_items(raw)]

    if candidate_id in graph:
        raise ConflictError(f"job {candidate_id!r} already exists")
    graph[candidate_id] = [dependency for dependency, _ in candidate_dependencies]
    for dependency, _ in candidate_dependencies:
        if dependency == candidate_id:
            raise ValidationError("a job cannot depend on itself")
        if dependency not in graph:
            raise NotFoundError(f"dependency job {dependency!r} does not exist")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValidationError("job dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, ()):
            if child not in graph:
                raise NotFoundError(f"dependency job {child!r} does not exist")
            visit(child)
        visiting.remove(node)
        visited.add(node)

    visit(candidate_id)


def dependency_status(ledger: Any, dependencies: Sequence[Any]) -> dict[str, Any]:
    """Return job-result dependency states without consulting external task state."""

    items = _dependency_items(dependencies)
    results: list[dict[str, Any]] = []
    for job_id, condition in items:
        job = ledger.get(JOB_KIND, job_id)
        state = _job_state(job) if job is not None else "missing"
        if condition == "success":
            satisfied = state == "succeeded"
        elif condition == "failure":
            satisfied = state == "failed"
        else:
            satisfied = state in _TERMINAL_JOB_STATES
        results.append({"job_id": job_id, "condition": condition, "state": state, "satisfied": satisfied})
    return {
        "ready": all(item["satisfied"] for item in results),
        "dependencies": results,
    }


def dependencies_ready(ledger: Any, dependencies: Sequence[Any]) -> bool:
    """Boolean convenience for dispatchers enforcing dependency gates."""

    return bool(dependency_status(ledger, dependencies)["ready"])


def _trigger(value: EventTrigger | Mapping[str, Any] | None) -> EventTrigger | None:
    if value is None or isinstance(value, EventTrigger):
        return value
    return EventTrigger.from_dict(value)


def _event_matches(event: Mapping[str, Any], trigger: EventTrigger) -> bool:
    if event.get("source") != trigger.source or event.get("event_type") != trigger.event_type:
        return False
    if trigger.event_key is not None:
        payload = event.get("payload", {})
        event_key = event.get("event_id")
        if isinstance(payload, Mapping):
            event_key = payload.get("event_key", payload.get("key", event_key))
        if trigger.event_key != event.get("event_id") and trigger.event_key != event_key:
            return False
    payload = event.get("payload", {})
    if not isinstance(payload, Mapping):
        return not trigger.filters
    return all(payload.get(key) == expected for key, expected in trigger.filters.items())


def matching_event(
    ledger: Any,
    trigger: EventTrigger | Mapping[str, Any],
    *,
    created_at: datetime | str | None = None,
) -> dict[str, Any] | None:
    """Find the newest stored authenticated event that opens a one-off job gate."""

    normalized = _trigger(trigger)
    assert normalized is not None
    lower_bound = _datetime(created_at, "created_at") if created_at is not None else None
    matches = []
    for event in ledger.list(EVENT_KIND):
        if not _event_matches(event, normalized):
            continue
        try:
            occurred_at = _datetime(event.get("created_at"), "event.created_at")
        except ValidationError:
            continue
        if lower_bound is not None and occurred_at < lower_bound:
            continue
        matches.append((occurred_at, event))
    return dict(max(matches, key=lambda item: item[0])[1]) if matches else None


def event_ready(
    ledger: Any,
    trigger: EventTrigger | Mapping[str, Any] | None,
    *,
    created_at: datetime | str | None = None,
) -> bool:
    """Return whether a queued one-off job's authenticated event has arrived."""

    if trigger is None:
        return True
    return matching_event(ledger, trigger, created_at=created_at) is not None


def scheduled_job_ready(ledger: Any, job_or_spec: Mapping[str, Any]) -> bool:
    """Fail closed until a scheduled job is bound to its durable occurrence.

    The application dispatcher must call this before admitting a job. Ordinary
    jobs pass through unchanged. A scheduled job is ready only after its
    occurrence is `submitted` and names that canonical job ID; this keeps a
    newly created but not-yet-bound job out of execution during cancel races.
    """

    if not isinstance(job_or_spec, Mapping):
        raise ValidationError("job_or_spec must be an object")
    job = job_or_spec
    raw_spec = job.get("spec")
    if isinstance(raw_spec, Mapping):
        spec = raw_spec
    elif hasattr(raw_spec, "to_dict"):
        spec = raw_spec.to_dict()
    else:
        spec = job
    key = spec.get("idempotency_key") if isinstance(spec, Mapping) else None
    prefix = "schedule-occurrence:"
    if not isinstance(key, str) or not key.startswith(prefix):
        return True
    job_id = _job_id(job)
    if job_id is None:
        return False
    entity_id = key[len(prefix) :]
    if not entity_id:
        return False
    occurrence = ledger.get(OCCURRENCE_KIND, entity_id)
    return bool(
        occurrence is not None
        and occurrence.get("state") == "submitted"
        and occurrence.get("job_id") == job_id
    )


def _parse_cron_field(source: str, minimum: int, maximum: int, *, sunday: bool = False) -> tuple[frozenset[int], bool]:
    if not source:
        raise ValidationError("calendar expression contains an empty field")
    wildcard = source == "*"
    values: set[int] = set()
    for component in source.split(","):
        if not component:
            raise ValidationError("calendar expression contains an empty list item")
        base, slash, step_text = component.partition("/")
        if slash:
            if not step_text.isdigit() or int(step_text) < 1:
                raise ValidationError(f"invalid calendar step {component!r}")
            step = int(step_text)
        else:
            step = 1
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            left, right = base.split("-", 1)
            if not left.isdigit() or not right.isdigit():
                raise ValidationError(f"invalid calendar range {component!r}")
            start, end = int(left), int(right)
            if start > end:
                raise ValidationError(f"descending calendar range {component!r}")
        else:
            if not base.isdigit():
                raise ValidationError(f"invalid calendar value {component!r}")
            start = end = int(base)
        allowed_maximum = 7 if sunday else maximum
        if start < minimum or end > allowed_maximum:
            raise ValidationError(f"calendar value in {component!r} is outside {minimum}..{allowed_maximum}")
        values.update(range(start, end + 1, step))
    if sunday and 7 in values:
        values.discard(7)
        values.add(0)
    return frozenset(values), wildcard


class _Cron:
    def __init__(self, expression: str):
        if not isinstance(expression, str):
            raise ValidationError("calendar_expression must be a string")
        parts = expression.split()
        if len(parts) != 5:
            raise ValidationError("calendar_expression must use five numeric cron fields")
        self.minutes, _ = _parse_cron_field(parts[0], 0, 59)
        self.hours, _ = _parse_cron_field(parts[1], 0, 23)
        self.days, day_wildcard = _parse_cron_field(parts[2], 1, 31)
        self.months, _ = _parse_cron_field(parts[3], 1, 12)
        self.weekdays, weekday_wildcard = _parse_cron_field(parts[4], 0, 6, sunday=True)
        self.day_wildcard = day_wildcard
        self.weekday_wildcard = weekday_wildcard
        self.expression = expression

    def matches_day(self, day: date) -> bool:
        if day.month not in self.months:
            return False
        day_match = day.day in self.days
        weekday_match = (day.weekday() + 1) % 7 in self.weekdays
        if self.day_wildcard:
            return weekday_match
        if self.weekday_wildcard:
            return day_match
        return day_match or weekday_match


def _valid_local_occurrence(local_naive: datetime, tz: ZoneInfo) -> datetime | None:
    """Map a wall time to one UTC instant; skip gaps and dedupe repeated folds."""

    candidate = local_naive.replace(tzinfo=tz, fold=0)
    round_trip = candidate.astimezone(UTC).astimezone(tz).replace(tzinfo=None)
    if round_trip != local_naive:
        return None  # nonexistent wall time during the spring-forward gap
    return candidate.astimezone(UTC)  # fold=0 gives one nominal occurrence at fall-back


def _calendar_occurrences(
    expression: str,
    timezone_name: str,
    start_exclusive: datetime,
    end_inclusive: datetime,
    *,
    limit: int,
) -> tuple[list[datetime], bool]:
    """Return the newest ``limit`` cron instants and whether older ones exist."""

    if limit < 1 or end_inclusive <= start_exclusive:
        return [], False
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, TypeError) as exc:
        raise ValidationError(f"timezone is not a recognized IANA timezone: {timezone_name!r}") from exc
    cron = _Cron(expression)
    local_start = start_exclusive.astimezone(tz).date()
    local_end = end_inclusive.astimezone(tz).date()
    span = (local_end - local_start).days + 1
    if span > _MAX_CALENDAR_DAYS:
        raise ValidationError("calendar schedule lookback exceeds 100 years; reconcile the schedule before ticking")
    times = tuple(time(hour, minute) for hour, minute in itertools.product(sorted(cron.hours), sorted(cron.minutes)))
    found: list[datetime] = []
    older = False
    current = local_end
    while current >= local_start:
        if cron.matches_day(current):
            for wall_time in reversed(times):
                nominal = _valid_local_occurrence(datetime.combine(current, wall_time), tz)
                if nominal is None or nominal <= start_exclusive or nominal > end_inclusive:
                    continue
                if len(found) == limit:
                    older = True
                    break
                found.append(nominal)
            if older:
                break
        current -= timedelta(days=1)
    found.reverse()
    return found, older


def _interval_occurrences(
    anchor: datetime,
    last_nominal: datetime,
    now: datetime,
    interval_seconds: float,
    *,
    limit: int,
) -> tuple[list[datetime], bool]:
    elapsed = (now - anchor).total_seconds()
    if elapsed <= 0:
        return [], False
    latest_index = math.floor(elapsed / interval_seconds)
    previous_index = max(0, math.floor((last_nominal - anchor).total_seconds() / interval_seconds))
    first_index = previous_index + 1
    if latest_index < first_index:
        return [], False
    count = latest_index - first_index + 1
    chosen_first = max(first_index, latest_index - limit + 1)
    values = [anchor + timedelta(seconds=interval_seconds * index) for index in range(chosen_first, latest_index + 1)]
    return values, count > limit


def _occurrence_key(schedule_id: str, version: int, nominal: str) -> str:
    digest = hashlib.sha256(f"{schedule_id}\0{version}\0{nominal}".encode()).hexdigest()
    return digest


def _occurrence_entity_id(schedule_id: str, nominal_key: str) -> str:
    digest = hashlib.sha256(f"{schedule_id}\0{nominal_key}".encode()).hexdigest()
    return f"{schedule_id}:{digest}"


class ScheduleCoordinator:
    """Durable schedule/event coordinator using a dispatcher callback for jobs.

    ``submit_job`` is called as ``submit_job(spec_dict, owner, idempotency_key)``.
    It must implement normal job idempotency and return the canonical job record.
    """

    def __init__(
        self,
        ledger: Any,
        submit_job: Callable[[Mapping[str, Any], str, str], Mapping[str, Any]],
        clock: Any,
        *,
        cancel_job: Callable[[str, str, str], Any] | None = None,
    ):
        self.ledger = ledger
        self.submit_job = submit_job
        self.clock = clock
        self.cancel_job = cancel_job

    def put(self, spec: ScheduleSpec | Mapping[str, Any], owner: str, key: str) -> dict[str, Any]:
        schedule = spec if isinstance(spec, ScheduleSpec) else ScheduleSpec.from_dict(spec)
        if schedule.owner_id != owner or schedule.job.owner_id != owner:
            raise PermissionDeniedError("schedule owner and nested job owner must match the authenticated owner")
        if schedule.calendar_expression is not None:
            _Cron(schedule.calendar_expression)
        now = _now(self.clock)
        payload = schedule.to_dict()

        def create_or_update(tx: Any) -> dict[str, Any]:
            current = tx.get(SCHEDULE_KIND, schedule.schedule_id)
            if current is not None and current.get("owner_id") != owner:
                raise PermissionDeniedError("schedule belongs to another owner")
            version = int(current.get("version", 0)) + 1 if current is not None else 1
            active_jobs = list(current.get("active_job_ids", [])) if current is not None else []
            discarded = list(current.get("pending_occurrences", [])) if current is not None else []
            record = {
                "schedule_id": schedule.schedule_id,
                "owner_id": owner,
                "spec": payload,
                "state": schedule.state.value,
                "version": version,
                "created_at": current.get("created_at", _iso(now)) if current else _iso(now),
                "version_started_at": _iso(now),
                "updated_at": _iso(now),
                "anchor_at": _iso(now),
                "last_nominal_at": _iso(now),
                "active_job_ids": active_jobs,
                "submission_intents": list(current.get("submission_intents", [])) if current is not None else [],
                "pending_occurrences": [],
                "last_error": None,
            }
            saved = tx.put(SCHEDULE_KIND, schedule.schedule_id, record, current.get("_revision") if current else None)
            for occurrence in discarded:
                self._record_occurrence(tx, schedule.schedule_id, occurrence, "skipped", reason="schedule_revised")
            tx.event("schedule.revised" if current else "schedule.created", schedule.schedule_id, {"owner": owner, "version": version})
            return saved

        return self.ledger.idempotent(owner, key, "schedule.put", payload, create_or_update)

    def control(self, schedule_id: str, action: str, owner: str, key: str) -> dict[str, Any]:
        if action not in {"pause", "resume", "cancel"}:
            raise ValidationError("schedule action must be pause, resume, or cancel")
        payload = {"schedule_id": schedule_id, "action": action}

        def apply(tx: Any) -> dict[str, Any]:
            current = tx.get(SCHEDULE_KIND, schedule_id)
            if current is None:
                raise NotFoundError(f"schedule {schedule_id!r} does not exist")
            if current.get("owner_id") != owner:
                raise PermissionDeniedError("schedule belongs to another owner")
            state = current.get("state")
            if action == "pause":
                if state == ScheduleState.CANCELLED.value:
                    raise ConflictError("cancelled schedules cannot be paused")
                new_state = ScheduleState.PAUSED.value
            elif action == "resume":
                if state == ScheduleState.CANCELLED.value:
                    raise ConflictError("cancelled schedules cannot be resumed")
                new_state = ScheduleState.ACTIVE.value
            else:
                new_state = ScheduleState.CANCELLED.value
            updated = _without_revision(current)
            updated["state"] = new_state
            updated["updated_at"] = _iso(_now(self.clock))
            if action == "cancel":
                pending = list(updated.get("pending_occurrences", []))
                updated["pending_occurrences"] = []
                for occurrence in pending:
                    self._record_occurrence(tx, schedule_id, occurrence, "skipped", reason="schedule_cancelled")
                for entity_id in updated.get("submission_intents", []):
                    occurrence_record = tx.get(OCCURRENCE_KIND, entity_id)
                    if occurrence_record is not None and occurrence_record.get("state") == "submitting":
                        occurrence = {
                            "occurrence_key": occurrence_record["occurrence_key"],
                            "nominal_at": occurrence_record["nominal_at"],
                            "source": occurrence_record.get("source"),
                            "event_source": occurrence_record.get("event_source"),
                            "event_id": occurrence_record.get("event_id"),
                            "version": occurrence_record.get("version"),
                        }
                        self._record_occurrence(
                            tx,
                            schedule_id,
                            occurrence,
                            "cancel_requested",
                            reason="schedule_cancelled_before_submission_bound",
                        )
            schedule_data = dict(updated["spec"])
            schedule_data["state"] = new_state
            updated["spec"] = schedule_data
            saved = tx.put(SCHEDULE_KIND, schedule_id, updated, current["_revision"])
            tx.event(f"schedule.{action}", schedule_id, {"owner": owner, "state": new_state})
            return saved

        return self.ledger.idempotent(owner, key, f"schedule.{action}", payload, apply)

    def publish_event(
        self,
        source: str,
        event_id: str,
        payload: Mapping[str, Any],
        owner: str,
        key: str,
    ) -> dict[str, Any]:
        if not source or not event_id or not isinstance(payload, Mapping):
            raise ValidationError("event source, event_id, and object payload are required")
        event_type = payload.get("event_type", payload.get("type"))
        if not isinstance(event_type, str) or not event_type:
            raise ValidationError("event payload must contain a non-empty event_type or type")
        event_key = f"{source}:{event_id}"
        now = _now(self.clock)
        request = {"source": source, "event_id": event_id, "event_type": event_type, "payload": _plain(payload)}

        def persist(tx: Any) -> dict[str, Any]:
            existing = tx.get(EVENT_KIND, event_key)
            if existing is not None:
                same = (
                    existing.get("source") == source
                    and existing.get("event_id") == event_id
                    and existing.get("event_type") == event_type
                    and existing.get("payload") == _plain(payload)
                )
                if not same:
                    raise IdempotencyConflictError("event identity was already used with different content")
                return existing
            event = {
                **request,
                "owner": owner,
                "created_at": _iso(now),
            }
            saved = tx.put(EVENT_KIND, event_key, event, None)
            tx.event("event.published", event_key, {"owner": owner, "source": source, "event_type": event_type})
            return saved

        event = self.ledger.idempotent(owner, key, "event.publish", request, persist)
        jobs: list[dict[str, Any]] = []
        for schedule in self.ledger.list(SCHEDULE_KIND):
            if schedule.get("state") != ScheduleState.ACTIVE.value:
                continue
            spec = ScheduleSpec.from_dict(schedule["spec"])
            trigger = spec.event_trigger
            if trigger is None or not _event_matches(event, trigger):
                continue
            try:
                self._enqueue_event_occurrence(schedule, event)
                jobs.extend(self._process_pending(schedule["schedule_id"]))
            except ConflictError:
                # A concurrent tick won the schedule revision. The durable event
                # remains available and the next tick will reconcile it.
                continue
        return {"event": event, "jobs": jobs}

    def tick(self) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        now = _now(self.clock)
        for snapshot in self.ledger.list(SCHEDULE_KIND):
            if snapshot.get("state") == ScheduleState.ACTIVE.value:
                spec = ScheduleSpec.from_dict(snapshot["spec"])
                if spec.event_trigger is not None:
                    # Events are independently durable. This also recovers the window
                    # where a daemon stopped after storing an event but before queueing it.
                    events = sorted(self.ledger.list(EVENT_KIND), key=lambda item: item.get("created_at", ""))
                    for event in events:
                        version_started = snapshot.get("version_started_at", snapshot["created_at"])
                        if _event_matches(event, spec.event_trigger) and _datetime(event["created_at"], "event.created_at") >= _datetime(version_started, "schedule.version_started_at"):
                            try:
                                self._enqueue_event_occurrence(snapshot, event)
                            except ConflictError:
                                continue
                else:
                    try:
                        self._collect_due(snapshot, now)
                    except ConflictError:
                        # Another scheduler tick advanced this durable cursor.
                        # Its committed pending work is still processed below.
                        pass
            # In-flight submission intents are reconciled even after pause or
            # cancellation. They were claimed before the control transition.
            created.extend(self._process_pending(snapshot["schedule_id"]))
        return created

    def _collect_due(self, snapshot: Mapping[str, Any], now: datetime) -> None:
        schedule_id = str(snapshot["schedule_id"])
        spec = ScheduleSpec.from_dict(snapshot["spec"])
        cursor = _datetime(snapshot["last_nominal_at"], "schedule.last_nominal_at")
        anchor = _datetime(snapshot["anchor_at"], "schedule.anchor_at")
        if spec.misfire_policy is MisfirePolicy.COALESCE:
            limit = 1
        elif spec.misfire_policy is MisfirePolicy.SKIP:
            # Two candidates distinguish a single current occurrence from a
            # missed backlog without enumerating unbounded history.
            limit = 2
        else:
            limit = spec.max_catch_up
        if spec.interval_seconds is not None:
            values, older = _interval_occurrences(anchor, cursor, now, spec.interval_seconds, limit=limit)
        else:
            assert spec.calendar_expression is not None and spec.timezone is not None
            values, older = _calendar_occurrences(
                spec.calendar_expression,
                spec.timezone,
                cursor,
                now,
                limit=limit,
            )
        if not values:
            return
        latest = values[-1]
        version = int(snapshot["version"])
        occurrences = [
            {
                "occurrence_key": _occurrence_key(schedule_id, version, _iso(value)),
                "nominal_at": _iso(value),
                "version": version,
                "source": "calendar" if spec.calendar_expression is not None else "interval",
            }
            for value in values
        ]

        def update(record: dict[str, Any], tx: Any) -> dict[str, Any]:
            current_cursor = _datetime(record["last_nominal_at"], "schedule.last_nominal_at")
            if current_cursor != cursor:
                raise ConflictError("schedule tick cursor changed concurrently")
            pending = list(record.get("pending_occurrences", []))
            skipped = 0
            if spec.misfire_policy is MisfirePolicy.SKIP:
                if older or len(occurrences) > 1:
                    skipped = len(occurrences) + (1 if older else 0)
                else:
                    pending.extend(occurrences)
            elif spec.misfire_policy is MisfirePolicy.COALESCE:
                # Prefer the newest missed occurrence; an older backlog is
                # intentionally collapsed into this one durable run.
                for prior in pending:
                    self._record_occurrence(tx, schedule_id, prior, "skipped", reason="coalesced")
                pending = [occurrences[-1]]
                skipped = max(0, len(occurrences) - 1) + (1 if older else 0)
            else:
                capacity = max(0, spec.max_catch_up - len(pending))
                selected = occurrences[:capacity]
                pending.extend(selected)
                skipped = len(occurrences) - len(selected) + (1 if older else 0)
            for occurrence in pending:
                self._record_occurrence(tx, schedule_id, occurrence, "pending")
            updated = _without_revision(record)
            updated["last_nominal_at"] = _iso(latest)
            updated["pending_occurrences"] = pending
            updated["updated_at"] = _iso(now)
            saved = tx.put(SCHEDULE_KIND, schedule_id, updated, record["_revision"])
            if skipped:
                tx.event(
                    "schedule.misfire",
                    schedule_id,
                    {"policy": spec.misfire_policy.value, "skipped_count_at_least": skipped, "through": _iso(latest)},
                )
            return saved

        self._mutate_schedule(schedule_id, update)

    def _enqueue_event_occurrence(self, snapshot: Mapping[str, Any], event: Mapping[str, Any]) -> None:
        schedule_id = str(snapshot["schedule_id"])
        spec = ScheduleSpec.from_dict(snapshot["spec"])
        if spec.event_trigger is None or not _event_matches(event, spec.event_trigger):
            return
        version_started = snapshot.get("version_started_at", snapshot["created_at"])
        if _datetime(event["created_at"], "event.created_at") < _datetime(version_started, "schedule.version_started_at"):
            return
        nominal = f"{event['source']}:{event['event_id']}"
        occurrence = {
            "occurrence_key": hashlib.sha256(f"{schedule_id}\0{nominal}".encode()).hexdigest(),
            "nominal_at": event["created_at"],
            "version": int(snapshot["version"]),
            "source": "event",
            "event_source": event["source"],
            "event_id": event["event_id"],
        }
        occurrence_id = _occurrence_entity_id(schedule_id, occurrence["occurrence_key"])

        def update(record: dict[str, Any], tx: Any) -> dict[str, Any]:
            if tx.get(OCCURRENCE_KIND, occurrence_id) is not None:
                return dict(record)
            pending = list(record.get("pending_occurrences", []))
            if len(pending) >= _MAX_EVENT_BACKLOG:
                self._record_occurrence(tx, schedule_id, occurrence, "skipped", reason="event_backlog_full")
                return dict(record)
            pending.append(occurrence)
            self._record_occurrence(tx, schedule_id, occurrence, "pending")
            updated = _without_revision(record)
            updated["pending_occurrences"] = pending
            updated["updated_at"] = _iso(_now(self.clock))
            return tx.put(SCHEDULE_KIND, schedule_id, updated, record["_revision"])

        self._mutate_schedule(schedule_id, update)

    def _process_pending(self, schedule_id: str) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        while True:
            intent = self._next_submission_intent(schedule_id)
            if intent is not None:
                result, job_record = self._reconcile_submission(intent)
                if result == "submitted" and job_record is not None:
                    created.append(job_record)
                    schedule = self.ledger.get(SCHEDULE_KIND, schedule_id)
                    spec = ScheduleSpec.from_dict(schedule["spec"]) if schedule else None
                    if spec is not None and spec.overlap is OverlapPolicy.QUEUE:
                        return created
                    continue
                if result in {"cancel_pending", "unresolved"}:
                    return created
                continue

            outcome = self._claim_pending_submission(schedule_id)
            if outcome in {"claimed", "skipped"}:
                continue
            return created

    def _next_submission_intent(self, schedule_id: str) -> dict[str, Any] | None:
        schedule = self.ledger.get(SCHEDULE_KIND, schedule_id)
        if schedule is None:
            return None
        candidates = []
        for entity_id in schedule.get("submission_intents", []):
            record = self.ledger.get(OCCURRENCE_KIND, entity_id)
            if record is not None and record.get("state") in {"submitting", "cancel_requested"}:
                candidates.append(record)
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item.get("nominal_at", ""), item.get("occurrence_key", "")))

    def _claim_pending_submission(self, schedule_id: str) -> str:
        """Atomically claim an occurrence before any job submission callback runs."""

        with self.ledger.transaction() as tx:
            schedule = tx.get(SCHEDULE_KIND, schedule_id)
            if schedule is None or schedule.get("state") != ScheduleState.ACTIVE.value:
                return "none"
            pending = list(schedule.get("pending_occurrences", []))
            if not pending:
                return "none"
            occurrence = pending[0]
            occurrence_key = str(occurrence.get("occurrence_key", ""))
            entity_id = _occurrence_entity_id(schedule_id, occurrence_key)
            prior = tx.get(OCCURRENCE_KIND, entity_id)
            if prior is not None and prior.get("state") != "pending":
                return "none"

            spec = ScheduleSpec.from_dict(schedule["spec"])
            active_ids = list(dict.fromkeys(schedule.get("active_job_ids", [])))
            active_count = sum(
                job is None or _job_state(job) not in _TERMINAL_JOB_STATES
                for job in (tx.get(JOB_KIND, job_id) for job_id in active_ids)
            )
            in_flight = len(schedule.get("submission_intents", []))
            active_count += in_flight
            if spec.overlap is OverlapPolicy.FORBID and active_count:
                pending.pop(0)
                updated = _without_revision(schedule)
                updated["pending_occurrences"] = pending
                updated["updated_at"] = _iso(_now(self.clock))
                tx.put(SCHEDULE_KIND, schedule_id, updated, schedule["_revision"])
                self._record_occurrence(tx, schedule_id, occurrence, "skipped", reason="overlap_forbidden")
                tx.event("schedule.occurrence.skipped", entity_id, {"schedule_id": schedule_id, "reason": "overlap_forbidden"})
                return "skipped"
            if spec.overlap is OverlapPolicy.QUEUE and active_count:
                return "wait"
            if spec.overlap is OverlapPolicy.ALLOW and active_count >= spec.max_instances:
                return "wait"

            job_spec = self._job_for_occurrence(spec.job, schedule_id, occurrence)
            submission_key = str(job_spec["idempotency_key"])
            pending.pop(0)
            updated = _without_revision(schedule)
            updated["pending_occurrences"] = pending
            intents = list(updated.get("submission_intents", []))
            if entity_id not in intents:
                intents.append(entity_id)
            updated["submission_intents"] = intents
            updated["updated_at"] = _iso(_now(self.clock))
            tx.put(SCHEDULE_KIND, schedule_id, updated, schedule["_revision"])
            self._record_occurrence(
                tx,
                schedule_id,
                occurrence,
                "submitting",
                owner_id=schedule["owner_id"],
                submission_key=submission_key,
                job_spec=job_spec,
            )
            tx.event("schedule.occurrence.submitting", entity_id, {"schedule_id": schedule_id, "version": schedule["version"]})
            return "claimed"

    def _reconcile_submission(self, intent: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
        schedule_id = str(intent["schedule_id"])
        entity_id = _occurrence_entity_id(schedule_id, str(intent["occurrence_key"]))
        owner = str(intent.get("owner_id", ""))
        submission_key = str(intent.get("submission_key", ""))
        job_spec = intent.get("job_spec")
        if not owner or not submission_key or not isinstance(job_spec, Mapping):
            raise ValidationError("submission intent is missing its durable owner, key, or job specification")

        # The callback stays outside the ledger transaction. The durable intent
        # and stable key make retries safe without nested SQLite transactions.
        job_record = dict(self.submit_job(job_spec, owner, submission_key))
        job_id = _job_id(job_record)
        if not job_id:
            raise ValidationError("submit_job callback must return a job record with id or job_id")

        cancellation_required = False
        with self.ledger.transaction() as tx:
            schedule = tx.get(SCHEDULE_KIND, schedule_id)
            current = tx.get(OCCURRENCE_KIND, entity_id)
            if current is None:
                raise ConflictError("submission intent disappeared before job binding")
            if current.get("state") in {"submitted", "cancelled", "skipped"}:
                return current["state"], job_record
            changed_version = schedule is None or int(schedule.get("version", -1)) != int(current.get("version", -2))
            schedule_cancelled = schedule is None or schedule.get("state") == ScheduleState.CANCELLED.value
            if current.get("state") == "cancel_requested" or changed_version or schedule_cancelled:
                self._record_occurrence(
                    tx,
                    schedule_id,
                    current,
                    "cancel_requested",
                    reason=current.get("reason") or "schedule_changed_before_submission_bound",
                    job_id=job_id,
                )
                cancellation_required = True
            else:
                active = list(schedule.get("active_job_ids", []))
                if job_id not in active:
                    active.append(job_id)
                updated = _without_revision(schedule)
                updated["active_job_ids"] = active
                updated["submission_intents"] = [
                    value for value in updated.get("submission_intents", []) if value != entity_id
                ]
                updated["updated_at"] = _iso(_now(self.clock))
                tx.put(SCHEDULE_KIND, schedule_id, updated, schedule["_revision"])
                self._record_occurrence(tx, schedule_id, current, "submitted", job_id=job_id)
                tx.event("schedule.occurrence.submitted", entity_id, {"schedule_id": schedule_id, "job_id": job_id})

        if not cancellation_required:
            return "submitted", job_record
        if self.cancel_job is None:
            return "cancel_pending", job_record

        cancel_receipt = self.cancel_job(job_id, owner, f"schedule-occurrence-cancel:{entity_id}")
        cancel_state = (
            str(getattr(cancel_receipt.get("state"), "value", cancel_receipt.get("state", ""))).lower()
            if isinstance(cancel_receipt, Mapping)
            else ""
        )
        if cancel_state != "cancelled":
            # Keep the dispatch gate closed until the application has a
            # terminal cancellation receipt. Unknown cleanup is not success.
            return "cancel_pending", job_record
        with self.ledger.transaction() as tx:
            current = tx.get(OCCURRENCE_KIND, entity_id)
            if current is not None and current.get("state") == "cancel_requested":
                self._record_occurrence(
                    tx,
                    schedule_id,
                    current,
                    "cancelled",
                    reason=current.get("reason") or "schedule_cancelled_before_submission_bound",
                    job_id=job_id,
                )
                tx.event("schedule.occurrence.cancelled", entity_id, {"schedule_id": schedule_id, "job_id": job_id})
            schedule = tx.get(SCHEDULE_KIND, schedule_id)
            if schedule is not None:
                updated = _without_revision(schedule)
                updated["submission_intents"] = [
                    value for value in updated.get("submission_intents", []) if value != entity_id
                ]
                updated["updated_at"] = _iso(_now(self.clock))
                tx.put(SCHEDULE_KIND, schedule_id, updated, schedule["_revision"])
        return "cancelled", job_record

    def _job_for_occurrence(
        self,
        template: JobSpec,
        schedule_id: str,
        occurrence: Mapping[str, Any],
    ) -> dict[str, Any]:
        entity_id = _occurrence_entity_id(schedule_id, str(occurrence["occurrence_key"]))
        submission_key = f"schedule-occurrence:{entity_id}"
        spec = replace(
            template,
            not_before=_datetime(occurrence["nominal_at"], "occurrence.nominal_at"),
            event_trigger=None,
            idempotency_key=submission_key,
        )
        return spec.to_dict()

    def _record_occurrence(
        self,
        tx: Any,
        schedule_id: str,
        occurrence: Mapping[str, Any],
        state: str,
        *,
        reason: str | None = None,
        job_id: str | None = None,
        owner_id: str | None = None,
        submission_key: str | None = None,
        job_spec: Mapping[str, Any] | None = None,
    ) -> None:
        entity_id = _occurrence_entity_id(schedule_id, str(occurrence["occurrence_key"]))
        current = tx.get(OCCURRENCE_KIND, entity_id)
        record = _without_revision(current) if current is not None else {}
        record.update({
            "schedule_id": schedule_id,
            "occurrence_key": occurrence["occurrence_key"],
            "nominal_at": occurrence["nominal_at"],
            "source": occurrence.get("source"),
            "event_source": occurrence.get("event_source"),
            "event_id": occurrence.get("event_id"),
            "version": occurrence.get("version"),
            "state": state,
            "reason": reason,
            "updated_at": _iso(_now(self.clock)),
        })
        if job_id is not None:
            record["job_id"] = job_id
        if owner_id is not None:
            record["owner_id"] = owner_id
        if submission_key is not None:
            record["submission_key"] = submission_key
        if job_spec is not None:
            record["job_spec"] = dict(job_spec)
        if current is not None:
            if current.get("state") in {"submitted", "skipped", "cancelled"}:
                return
            if current.get("state") == state == "pending":
                return
            record["created_at"] = current.get("created_at", record["updated_at"])
            tx.put(OCCURRENCE_KIND, entity_id, record, current["_revision"])
        else:
            record["created_at"] = record["updated_at"]
            tx.put(OCCURRENCE_KIND, entity_id, record, None)

    def _mutate_schedule(self, schedule_id: str, update: Callable[[dict[str, Any], Any], dict[str, Any]]) -> dict[str, Any] | None:
        with self.ledger.transaction() as tx:
            current = tx.get(SCHEDULE_KIND, schedule_id)
            if current is None:
                return None
            return update(current, tx)


__all__ = [
    "ScheduleCoordinator",
    "dependencies_ready",
    "dependency_status",
    "event_ready",
    "matching_event",
    "scheduled_job_ready",
    "validate_dependency_graph",
]
