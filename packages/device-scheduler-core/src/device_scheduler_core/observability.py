"""Pure helpers for decision explanations, execution evidence, and policy replay.

These functions consume ledger snapshots supplied by the application layer. They
do not open artifacts, mutate state, allocate resources, or invoke execution and
backend adapters. Artifact path authorization and destructive retention belong
to the adapter/application boundary that owns those effects.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any

from .errors import UnsupportedCapabilityError, ValidationError
from .models import PolicyDecision

UTC = timezone.utc
_MAX_POLICY_REPLAY_REQUESTS = 1000
_MAX_POLICY_REPLAY_BYTES = 1_048_576


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(child) for child in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _aware(value: datetime | str, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(f"{name} must be an ISO-8601 datetime") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def decision_snapshot(
    decision: PolicyDecision | Mapping[str, Any],
    *,
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any] = (),
    eligible_counts: Mapping[str, int] | None = None,
    conflicts: Sequence[Mapping[str, Any] | str] = (),
    ownership: Sequence[Mapping[str, Any] | str] = (),
) -> dict[str, Any]:
    """Build a versioned, replayable record of decision inputs and explanation."""

    record = _plain(decision)
    if not isinstance(record, Mapping):
        raise ValidationError("decision must be a PolicyDecision or object")
    counts = dict(eligible_counts or record.get("facts", {}).get("eligible_counts", {}))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValidationError("eligible counts must be non-negative integers")
    facts = record.get("facts", {})
    if observations == () and isinstance(facts, Mapping):
        observations = facts.get("observations", ())
    inputs = _plain(observations)
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    if not isinstance(inputs, (list, dict)):
        raise ValidationError("observations must be a sequence or object")
    return {
        "schema_version": 1,
        "decision_id": record.get("decision_id"),
        "policy_id": record.get("policy_id"),
        "policy_version": record.get("policy_version"),
        "subject_id": record.get("subject_id"),
        "action": record.get("action"),
        "accepted": record.get("accepted"),
        "reason": record.get("reason"),
        "selected_resource_ids": _plain(record.get("selected_resource_ids", ())),
        "observed_at": record.get("observed_at"),
        "observations": inputs,
        "eligible_counts": counts,
        "conflicts": _plain(conflicts or (facts.get("conflicts", ()) if isinstance(facts, Mapping) else ())),
        "ownership": _plain(ownership or (facts.get("ownership", ()) if isinstance(facts, Mapping) else ())),
        "facts": _plain(facts),
    }


def explain_decision(
    snapshot: PolicyDecision | Mapping[str, Any],
    *,
    now: datetime | str | None = None,
    max_observation_age_seconds: float = 60.0,
) -> dict[str, Any]:
    """Return an explanation with explicit policy identity and observation age."""

    if not isinstance(max_observation_age_seconds, (int, float)) or not math.isfinite(max_observation_age_seconds) or max_observation_age_seconds < 0:
        raise ValidationError("max_observation_age_seconds must be a finite non-negative number")
    record = _plain(snapshot)
    if not isinstance(record, Mapping):
        raise ValidationError("decision snapshot must be an object")
    observed = record.get("observed_at")
    current = _aware(now, "now") if now is not None else datetime.now(UTC)
    if observed is None:
        age = None
        freshness = "unknown"
    else:
        age = (current - _aware(observed, "observed_at")).total_seconds()
        freshness = "future_clock_skew" if age < 0 else ("fresh" if age <= max_observation_age_seconds else "stale")
    return {
        "decision_id": record.get("decision_id"),
        "policy_id": record.get("policy_id"),
        "policy_version": record.get("policy_version"),
        "subject_id": record.get("subject_id"),
        "action": record.get("action"),
        "accepted": record.get("accepted"),
        "reason": record.get("reason"),
        "selected_resource_ids": _plain(record.get("selected_resource_ids", ())),
        "observed_at": observed,
        "observation_age_seconds": age,
        "observation_freshness": freshness,
        "eligible_counts": _plain(record.get("eligible_counts", {})),
        "conflicts": _plain(record.get("conflicts", ())),
        "ownership": _plain(record.get("ownership", ())),
        "observations": _plain(record.get("observations", ())),
        "facts": _plain(record.get("facts", {})),
    }


def execution_evidence(
    *,
    ledger_state: str,
    adapter_observation: Mapping[str, Any] | None,
    capability: Mapping[str, Any] | Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Keep canonical state, last adapter observation, and enforcement claims distinct."""

    if not isinstance(ledger_state, str) or not ledger_state:
        raise ValidationError("ledger_state must be a non-empty string")
    observed = _plain(adapter_observation) if adapter_observation is not None else None
    capabilities = _plain(capability)
    if isinstance(capabilities, Mapping):
        capabilities = [capabilities]
    if not isinstance(capabilities, list):
        raise ValidationError("capability must be an object or sequence of objects")
    return {
        "ledger": {"state": ledger_state, "authority": "canonical_ledger"},
        "adapter": {
            "observation": observed,
            "verified": bool(observed and observed.get("verified", False)),
            "observed_at": observed.get("observed_at") if isinstance(observed, Mapping) else None,
        },
        "capabilities": capabilities,
        "claim_limit": "Adapter capabilities state whether behavior is enforced, advisory, or observational.",
    }


def bound_output(
    output: str | bytes,
    *,
    max_bytes: int,
    max_lines: int | None = None,
) -> dict[str, Any]:
    """Return bounded UTF-8 output plus digest and explicit truncation metadata."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValidationError("max_bytes must be a non-negative integer")
    if max_lines is not None and (
        isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1
    ):
        raise ValidationError("max_lines must be a positive integer when supplied")
    raw = output.encode("utf-8") if isinstance(output, str) else output
    if not isinstance(raw, bytes):
        raise ValidationError("output must be text or bytes")
    original_size = len(raw)
    digest = hashlib.sha256(raw).hexdigest()
    bounded = raw[:max_bytes]
    truncated = len(bounded) != original_size
    # Ignore a trailing partial codepoint rather than expanding a replacement
    # character beyond the caller's byte cap.
    text = bounded.decode("utf-8", errors="ignore")
    omitted_lines = 0
    if max_lines is not None:
        lines = text.splitlines(keepends=True)
        if len(lines) > max_lines:
            omitted_lines = len(lines) - max_lines
            text = "".join(lines[:max_lines])
            truncated = True
    returned_size = len(text.encode("utf-8"))
    return {
        "text": text,
        "sha256": digest,
        "original_bytes": original_size,
        "returned_bytes": returned_size,
        "truncated": truncated,
        "omitted_lines": omitted_lines,
    }


def health_report(
    *,
    leases: Sequence[Mapping[str, Any]] = (),
    jobs: Sequence[Mapping[str, Any]] = (),
    operations: Sequence[Mapping[str, Any]] = (),
    backends: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize unresolved ownership and unavailable capacity without guessing."""

    unresolved_leases = []
    unavailable_resources: set[str] = set()
    held_leases = 0
    held_resources: set[str] = set()
    for lease in leases:
        state = str(getattr(lease.get("state", "unknown"), "value", lease.get("state", "unknown"))).lower()
        if state == "active":
            held_leases += 1
            for allocation in lease.get("allocations", ()):
                if isinstance(allocation, Mapping) and isinstance(allocation.get("resource_id"), str):
                    held_resources.add(allocation["resource_id"])
        if state in {"release_requested", "reclaiming", "unknown", "expired"}:
            lease_id = lease.get("lease_id", lease.get("id"))
            unresolved_leases.append({"lease_id": lease_id, "state": state})
            for allocation in lease.get("allocations", ()):
                if isinstance(allocation, Mapping) and isinstance(allocation.get("resource_id"), str):
                    unavailable_resources.add(allocation["resource_id"])
    unresolved_jobs = []
    for job in jobs:
        state = str(getattr(job.get("state", "unknown"), "value", job.get("state", "unknown"))).lower()
        if state == "unknown":
            unresolved_jobs.append({"job_id": job.get("job_id", job.get("id")), "state": state})
    unresolved_operations = [
        {"operation_id": operation.get("operation_id", operation.get("id")), "status": operation.get("status")}
        for operation in operations
        if operation.get("status") in {"intent", "unknown"}
    ]
    unresolved_backends = [
        {"backend_id": backend.get("backend_id", backend.get("id")), "state": backend.get("state", "unknown")}
        for backend in backends
        if str(getattr(backend.get("state", "unknown"), "value", backend.get("state", "unknown"))).lower()
        in {"failed", "unknown", "evicting", "restoring", "draining"}
    ]
    unresolved = bool(unresolved_leases or unresolved_jobs or unresolved_operations or unresolved_backends)
    return {
        "status": "degraded" if unresolved else "healthy",
        "held_lease_count": held_leases,
        "held_resource_ids": sorted(held_resources),
        "unresolved_lease_count": len(unresolved_leases),
        "unresolved_job_count": len(unresolved_jobs),
        "unresolved_operation_count": len(unresolved_operations),
        "unresolved_backend_count": len(unresolved_backends),
        "unavailable_resource_ids": sorted(unavailable_resources),
        "unresolved_leases": unresolved_leases,
        "unresolved_jobs": unresolved_jobs,
        "unresolved_operations": unresolved_operations,
        "unresolved_backends": unresolved_backends,
    }


def replay_policy(
    policy: Any,
    observations: Sequence[Mapping[str, Any]],
    *,
    operation: str = "admission",
    now: datetime | str | None = None,
    service_by_owner: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Replay bounded admission observations through the production policy port.

    Every request snapshot must include an explicit boolean ``can_fit`` fact.
    Optional ``safe_backfill_for`` lists protected request IDs for which the
    recorded snapshot proves backfill is safe. The policy receives only copied
    JSON requests and pure predicates over those captured facts; it gets no
    ledger or adapter references.
    """

    if operation != "admission":
        raise UnsupportedCapabilityError(f"policy replay operation {operation!r} is not supported")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValidationError("observations must be a sequence of objects")
    if len(observations) > _MAX_POLICY_REPLAY_REQUESTS:
        raise ValidationError(f"policy replay is limited to {_MAX_POLICY_REPLAY_REQUESTS} requests")
    copied = _plain(observations)
    if not all(isinstance(item, Mapping) for item in copied):
        raise ValidationError("each observation snapshot must be an object")
    # Round-trip through JSON to reject non-JSON inputs and detach nested data.
    try:
        encoded = json.dumps(copied, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("policy replay observations must be finite JSON data") from exc
    if len(encoded.encode("utf-8")) > _MAX_POLICY_REPLAY_BYTES:
        raise ValidationError(f"policy replay observations exceed {_MAX_POLICY_REPLAY_BYTES} bytes")
    detached = json.loads(encoded)
    choose_next = getattr(policy, "choose_next", None)
    if not callable(choose_next):
        raise UnsupportedCapabilityError("configured policy does not implement choose_next admission")
    try:
        parameters = inspect.signature(choose_next).parameters
    except (TypeError, ValueError) as exc:
        raise UnsupportedCapabilityError("configured policy choose_next signature cannot be inspected") from exc
    accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    required_keywords = {"can_fit", "safe_backfill", "now", "service_by_owner"}
    if not accepts_kwargs and not required_keywords.issubset(parameters):
        raise UnsupportedCapabilityError("configured policy choose_next does not implement the scheduler admission contract")

    can_fit_by_id: dict[str, bool] = {}
    safe_backfill_pairs: set[tuple[str, str]] = set()
    for request in detached:
        request_id = request.get("request_id", request.get("job_id", request.get("id")))
        if not isinstance(request_id, str) or not request_id:
            raise ValidationError("each policy request needs request_id, job_id, or id")
        if request_id in can_fit_by_id:
            raise ValidationError(f"duplicate policy request ID {request_id!r}")
        fit = request.get("can_fit")
        if not isinstance(fit, bool):
            raise ValidationError(f"request {request_id!r} must include a boolean can_fit snapshot")
        can_fit_by_id[request_id] = fit
        compatible = request.get("safe_backfill_for", [])
        if not isinstance(compatible, list) or any(not isinstance(item, str) for item in compatible):
            raise ValidationError(f"request {request_id!r} safe_backfill_for must be an array of request IDs")
        safe_backfill_pairs.update((request_id, protected_id) for protected_id in compatible)

    service = dict(service_by_owner or {})
    for owner, amount in service.items():
        if not isinstance(owner, str) or not owner or isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(float(amount)) or amount < 0:
            raise ValidationError("service_by_owner must map owner IDs to finite non-negative numbers")
    current = _aware(now, "now") if now is not None else datetime.now(UTC)

    def request_id(value: Any) -> str | None:
        if isinstance(value, Mapping):
            identifier = value.get("request_id", value.get("job_id", value.get("id")))
        else:
            identifier = getattr(value, "request_id", getattr(value, "job_id", getattr(value, "id", None)))
        return identifier if isinstance(identifier, str) else None

    def can_fit(value: Any) -> bool:
        identifier = request_id(value)
        return bool(identifier is not None and can_fit_by_id.get(identifier, False))

    def safe_backfill(candidate: Any, protected: Any) -> bool:
        candidate_id, protected_id = request_id(candidate), request_id(protected)
        return bool(candidate_id and protected_id and (candidate_id, protected_id) in safe_backfill_pairs)

    result = choose_next(
        detached,
        can_fit=can_fit,
        safe_backfill=safe_backfill,
        now=current,
        service_by_owner=service,
    )
    selected = _plain(result) if result is not None else None
    if selected is not None and not isinstance(selected, Mapping):
        raise ValidationError("choose_next must return a PolicySelection object or None")
    if selected is not None:
        required = {"request_id", "owner_id", "priority_class", "score", "reason_code", "explanation"}
        if required - set(selected):
            raise ValidationError("choose_next result is missing PolicySelection fields")
        score = selected["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValidationError("PolicySelection.score must be finite")
        selected_id = selected["request_id"]
        valid = isinstance(selected_id, str) and selected_id in can_fit_by_id and can_fit_by_id[selected_id]
        validation = {"accepted_by_core": bool(valid), "reason": None if valid else "unknown_or_infeasible_request"}
    else:
        validation = {"accepted_by_core": True, "reason": "no_feasible_request"}
    proposal = dict(selected) if selected is not None else {
        "request_id": None,
        "owner_id": None,
        "priority_class": None,
        "score": None,
        "reason_code": "no_feasible_request",
        "explanation": "The policy found no request that fits the recorded snapshot.",
    }
    try:
        proposal = json.loads(json.dumps(proposal, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("policy replay proposal must be finite JSON data") from exc
    identity = {
        "policy_id": getattr(policy, "policy_id", None),
        "policy_version": getattr(policy, "version", getattr(policy, "policy_version", None)),
    }
    return {
        **identity,
        "operation": operation,
        "proposal": dict(proposal),
        "proposal_validation": validation,
        "request_count": len(detached),
        "observed_at": current.isoformat().replace("+00:00", "Z"),
        "evidence_type": "policy_simulation",
        "physical_outcome": None,
        "resource_allocation_performed": False,
        "execution_started": False,
        "backend_operation_performed": False,
    }


__all__ = [
    "bound_output",
    "decision_snapshot",
    "execution_evidence",
    "explain_decision",
    "health_report",
    "replay_policy",
]
