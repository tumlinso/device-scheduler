"""Built-in priority, fairness, aging, and safe backfill policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    request_id: str
    owner_id: str
    priority_class: str
    score: float
    reason_code: str
    explanation: str


@dataclass(frozen=True, slots=True)
class PreemptionProposal:
    """Advisory request to cancel and restart a lower-ranked background job.

    This result grants no authority. The application must re-read both jobs,
    validate interruption contracts and ownership, verify cleanup, and only
    then admit the requester.
    """

    requester_job_id: str
    target_job_id: str
    target_attempt_id: str
    target_lease_id: str
    reason_code: str
    requester_score: float
    target_score: float
    policy_id: str
    policy_version: str
    action: str = "cancel_and_restart"


class WeightedFairPolicy:
    """Rank queued requests and choose compatible work without mutating state.

    Applications may pass a ``can_fit`` predicate backed by the current
    allocator snapshot. When a long-waiting request is protected, lower-ranked
    work is considered only if ``safe_backfill(candidate, protected)`` confirms
    that it will preserve the protected request's feasible placement.
    """

    def __init__(
        self,
        *,
        class_weights: Mapping[str, float] | None = None,
        owner_weights: Mapping[str, float] | None = None,
        age_quantum_seconds: float = 180.0,
        max_age_boost: float = 3.0,
        large_request_protection_seconds: float = 300.0,
        fairness_penalty: float = 0.01,
        policy_id: str = "builtin.weighted-fair",
        version: str = "1",
    ) -> None:
        weights = dict(class_weights or {"interactive": 3.0, "normal": 2.0, "background": 1.0})
        if set(weights) != {"interactive", "normal", "background"}:
            raise ValueError("class_weights must define interactive, normal, and background")
        weights = {key: self._positive_weight(value, "class weights") for key, value in weights.items()}
        numeric_values = {
            "age_quantum_seconds": age_quantum_seconds,
            "max_age_boost": max_age_boost,
            "large_request_protection_seconds": large_request_protection_seconds,
            "fairness_penalty": fairness_penalty,
        }
        parsed_values: dict[str, float] = {}
        for name, value in numeric_values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite")
            try:
                parsed_values[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite") from exc
        if any(not math.isfinite(value) for value in parsed_values.values()):
            raise ValueError("aging and fairness policy values must be finite")
        age_quantum_seconds = parsed_values["age_quantum_seconds"]
        max_age_boost = parsed_values["max_age_boost"]
        large_request_protection_seconds = parsed_values["large_request_protection_seconds"]
        fairness_penalty = parsed_values["fairness_penalty"]
        if age_quantum_seconds <= 0 or max_age_boost < 0 or large_request_protection_seconds < 0 or fairness_penalty < 0:
            raise ValueError("aging and fairness policy values must be non-negative and age quantum positive")
        self.class_weights = weights
        self.owner_weights = self._validated_owner_weights(owner_weights or {})
        self.age_quantum_seconds = float(age_quantum_seconds)
        self.max_age_boost = float(max_age_boost)
        self.large_request_protection_seconds = float(large_request_protection_seconds)
        self.fairness_penalty = float(fairness_penalty)
        self.policy_id = policy_id
        self.version = version
        self.api_version = "1"

    def rank_requests(
        self,
        requests: Sequence[Any],
        *,
        now: datetime | None = None,
        service_by_owner: Mapping[str, float] | None = None,
    ) -> list[PolicyDecision]:
        current = self._aware_now(now)
        service = service_by_owner or {}
        ranked = [self._decision(request, current, service) for request in requests]
        return sorted(ranked, key=lambda item: (-item.score, item.request_id))

    def choose_next(
        self,
        requests: Sequence[Any],
        *,
        can_fit: Callable[[Any], bool],
        safe_backfill: Callable[[Any, Any], bool] | None = None,
        now: datetime | None = None,
        service_by_owner: Mapping[str, float] | None = None,
    ) -> PolicyDecision | None:
        """Choose the top feasible request, respecting aged request protection."""

        current = self._aware_now(now)
        ranked = self.rank_requests(requests, now=current, service_by_owner=service_by_owner)
        by_id = {self._request_id(request): request for request in requests}
        if not ranked:
            return None

        protected = self._protected_request(requests, current)
        if protected is not None and can_fit(protected):
            base = next(item for item in ranked if item.request_id == self._request_id(protected))
            return PolicyDecision(
                base.request_id,
                base.owner_id,
                base.priority_class,
                base.score,
                "aged_request_protected",
                "Selected the oldest request that crossed the protection threshold.",
            )

        for decision in ranked:
            candidate = by_id[decision.request_id]
            if not can_fit(candidate):
                continue
            if protected is not None and candidate is not protected:
                if safe_backfill is not None and not safe_backfill(candidate, protected):
                    continue
                if safe_backfill is None:
                    return PolicyDecision(
                        decision.request_id,
                        decision.owner_id,
                        decision.priority_class,
                        decision.score,
                        "backfill_unchecked",
                        "Selected feasible work because no placement-preservation proof was supplied.",
                    )
                return PolicyDecision(
                    decision.request_id,
                    decision.owner_id,
                    decision.priority_class,
                    decision.score,
                    "safe_backfill",
                    "Selected compatible work while preserving the aged request's feasible placement.",
                )
            return decision
        return None

    def explain(self, request: Any, *, now: datetime | None = None, service_by_owner: Mapping[str, float] | None = None) -> PolicyDecision:
        return self._decision(request, self._aware_now(now), service_by_owner or {})

    def propose_preemption(
        self,
        request: Mapping[str, Any],
        running: Sequence[Mapping[str, Any]],
        observations: Mapping[str, Any] | None = None,
        *,
        would_unblock: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> PreemptionProposal | None:
        """Suggest a safe-contract background victim without causing effects."""

        observations = observations or {}
        current = self._aware_now(observations.get("now"))
        service = observations.get("service_by_owner", {})
        if not isinstance(service, Mapping):
            service = {}
        requester_state = str(self._read(request, "state", "queued"))
        if requester_state not in {"queued", "blocked"}:
            return None
        requester_score = self._decision(request, current, service)
        eligible: list[tuple[float, datetime, str, Mapping[str, Any]]] = []
        for target in running:
            if str(self._read(target, "state", "running")) != "running":
                continue
            if self._request_id(target) == requester_score.request_id:
                continue
            spec = self._read(target, "spec", {})
            interruption = self._read(spec, "interruption", {})
            if not self._read(interruption, "restart_safe", False):
                continue
            if not self._read(interruption, "cancellation_supported", False):
                continue
            if self._priority_class(target) != "background":
                continue
            if self._read(target, "reservation_id"):
                continue
            attempt_id = self._read(target, "attempt_id")
            lease_id = self._read(target, "lease_id")
            if not attempt_id or not lease_id:
                continue
            max_attempts = self._read(spec, "max_attempts", 1)
            attempt_count = self._read(target, "attempt_count", 1)
            if (
                isinstance(max_attempts, bool)
                or not isinstance(max_attempts, int)
                or isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count >= max_attempts
            ):
                continue
            if would_unblock is not None:
                try:
                    if not would_unblock(target):
                        continue
                except Exception:
                    # Feasibility is supplied by the application. An uncertain
                    # check must never produce an interruption proposal.
                    continue
            target_for_score = dict(target)
            target_for_score["created_at"] = current
            target_score = self._decision(target_for_score, current, service)
            if requester_score.score <= target_score.score:
                continue
            started = self._time_field(target, "started_at") or current
            eligible.append((target_score.score, started, self._request_id(target), target))
        if not eligible:
            return None
        target_score_value, _, target_id, target = min(eligible, key=lambda item: (item[0], item[1], item[2]))
        return PreemptionProposal(
            requester_job_id=requester_score.request_id,
            target_job_id=target_id,
            target_attempt_id=str(self._read(target, "attempt_id")),
            target_lease_id=str(self._read(target, "lease_id")),
            reason_code="higher_ranked_request_needs_capacity",
            requester_score=requester_score.score,
            target_score=target_score_value,
            policy_id=self.policy_id,
            policy_version=self.version,
        )

    def _decision(self, request: Any, now: datetime, service: Mapping[str, float]) -> PolicyDecision:
        request_id = self._request_id(request)
        owner_id = str(
            self._read(
                request,
                "owner_id",
                self._read(request, "owner", self._read(self._read(request, "spec", {}), "owner_id", "unknown")),
            )
        )
        priority_class = self._priority_class(request)
        priority = self._numeric(self._read(request, "priority", self._read(self._read(request, "spec", {}), "priority", 50)))
        enqueued = self._time_field(request, "created_at") or self._time_field(request, "enqueued_at") or self._time_field(request, "submitted_at")
        age_seconds = max(0.0, (now - enqueued).total_seconds()) if enqueued else 0.0
        aging = min(self.max_age_boost, age_seconds / self.age_quantum_seconds)
        class_weight = self.class_weights[priority_class]
        owner_weight = self.owner_weights.get(owner_id, 1.0)
        served = max(0.0, self._numeric(service.get(owner_id, 0.0)))
        score = class_weight + priority / 100.0 + aging - self.fairness_penalty * served / (class_weight * owner_weight)
        if aging >= self.max_age_boost:
            reason = "queue_aging"
            explanation = "Queue age has reached the configured aging boost."
        elif served or any(self._numeric(value) > 0 for value in service.values()):
            reason = "weighted_fairness"
            explanation = "Queue ordering includes each owner's recent service relative to its class weight."
        else:
            reason = "priority_class"
            explanation = f"Request uses the server-granted {priority_class} priority class."
        return PolicyDecision(request_id, owner_id, priority_class, score, reason, explanation)

    @staticmethod
    def _validated_owner_weights(weights: Mapping[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for owner, value in weights.items():
            if not isinstance(owner, str) or not owner:
                raise ValueError("owner weights require non-empty owner IDs")
            result[owner] = WeightedFairPolicy._positive_weight(value, "owner weights")
        return result

    @staticmethod
    def _positive_weight(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be finite positive numbers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be finite positive numbers") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{label} must be finite positive numbers")
        return numeric

    def _protected_request(self, requests: Sequence[Any], now: datetime) -> Any | None:
        aged: list[tuple[datetime, str, Any]] = []
        for request in requests:
            enqueued = self._time_field(request, "created_at") or self._time_field(request, "enqueued_at") or self._time_field(request, "submitted_at")
            if enqueued is None:
                continue
            if (now - enqueued).total_seconds() >= self.large_request_protection_seconds:
                aged.append((enqueued, self._request_id(request), request))
        if not aged:
            return None
        return min(aged, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _request_id(request: Any) -> str:
        value = WeightedFairPolicy._read(request, "request_id", None)
        if value is None:
            value = WeightedFairPolicy._read(request, "job_id", None)
        if value is None:
            value = WeightedFairPolicy._read(request, "id", None)
        if value is None:
            raise ValueError("queued request must have request_id, job_id, or id")
        return str(value)

    @classmethod
    def _priority_class(cls, request: Any) -> str:
        spec = cls._read(request, "spec", {})
        explicit = cls._read(request, "priority_class", cls._read(spec, "priority_class", None))
        if isinstance(explicit, str) and explicit in {"interactive", "normal", "background"}:
            return str(explicit)
        priority = int(cls._numeric(cls._read(request, "priority", cls._read(spec, "priority", 50))))
        if priority >= 80:
            return "interactive"
        if priority < 30:
            return "background"
        return "normal"

    @staticmethod
    def _read(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _numeric(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0

    @staticmethod
    def _time_field(value: Any, name: str) -> datetime | None:
        field = WeightedFairPolicy._read(value, name)
        if field is None:
            return None
        if isinstance(field, str):
            try:
                field = datetime.fromisoformat(field.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(field, datetime):
            return None
        if field.tzinfo is None or field.utcoffset() is None:
            return field.replace(tzinfo=timezone.utc)
        return field.astimezone(timezone.utc)

    @staticmethod
    def _aware_now(value: datetime | None) -> datetime:
        result = value or datetime.now(timezone.utc)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return result.astimezone(timezone.utc)


__all__ = ["PolicyDecision", "PreemptionProposal", "WeightedFairPolicy"]
