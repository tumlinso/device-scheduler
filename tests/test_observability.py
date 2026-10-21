from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from device_scheduler_adapters.sqlite import Ledger
from device_scheduler_core.allocator import ResourceAllocator
from device_scheduler_core.application import Scheduler, SchedulerConfig
from device_scheduler_core.errors import UnsupportedCapabilityError, ValidationError
from device_scheduler_core.models import PolicyDecision, Principal
from device_scheduler_core.observability import (
    bound_output,
    decision_snapshot,
    execution_evidence,
    explain_decision,
    health_report,
    replay_policy,
)
from device_scheduler_core.policy import WeightedFairPolicy

UTC = timezone.utc


class FrozenClock:
    def now(self) -> datetime:
        return datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def test_decision_explanation_preserves_policy_version_ownership_and_freshness():
    now = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    decision = PolicyDecision(
        decision_id="decision-1",
        policy_id="default-policy",
        policy_version="4.2",
        subject_id="job-1",
        action="admit",
        accepted=False,
        reason="insufficient_headroom",
        observed_at=now - timedelta(seconds=10),
        facts={"eligible_counts": {"gpu": 2}},
    )
    snapshot = decision_snapshot(
        decision,
        observations=[{"resource_id": "gpu0", "fresh": True}],
        eligible_counts={"gpu": 2},
        conflicts=[{"resource_id": "gpu1", "owner": "alice"}],
        ownership=[{"resource_id": "gpu0", "owner": "bob"}],
    )

    explanation = explain_decision(snapshot, now=now, max_observation_age_seconds=30)

    assert explanation["policy_version"] == "4.2"
    assert explanation["reason"] == "insufficient_headroom"
    assert explanation["observation_freshness"] == "fresh"
    assert explanation["observation_age_seconds"] == 10
    assert explanation["eligible_counts"] == {"gpu": 2}
    assert explanation["ownership"][0]["owner"] == "bob"


def test_stale_and_unknown_observation_freshness_are_explicit():
    now = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    decision = {
        "decision_id": "d",
        "policy_id": "p",
        "policy_version": "1",
        "subject_id": "j",
        "action": "place",
        "accepted": True,
        "reason": "available",
        "observed_at": (now - timedelta(minutes=5)).isoformat(),
    }
    assert explain_decision(decision, now=now, max_observation_age_seconds=5)["observation_freshness"] == "stale"
    decision.pop("observed_at")
    assert explain_decision(decision, now=now)["observation_freshness"] == "unknown"


def test_execution_evidence_keeps_ledger_adapter_and_enforcement_claims_separate():
    evidence = execution_evidence(
        ledger_state="reclaiming",
        adapter_observation={"state": "stopped", "verified": False, "observed_at": "2026-10-21T11:59:00Z"},
        capability={"name": "gpu_memory_limit", "level": "advisory"},
    )
    assert evidence["ledger"]["state"] == "reclaiming"
    assert evidence["adapter"]["observation"]["state"] == "stopped"
    assert evidence["adapter"]["verified"] is False
    assert evidence["capabilities"][0]["level"] == "advisory"


def test_bounded_output_enforces_byte_and_line_limits_with_full_digest():
    output = "alpha\nbeta\ngamma\n"
    bounded = bound_output(output, max_bytes=100, max_lines=2)
    assert bounded["text"] == "alpha\nbeta\n"
    assert bounded["truncated"] is True
    assert bounded["omitted_lines"] == 1
    assert bounded["original_bytes"] == len(output.encode())
    assert len(bounded["sha256"]) == 64

    byte_limited = bound_output("ééé", max_bytes=3)
    assert byte_limited["truncated"] is True
    assert byte_limited["returned_bytes"] <= 3


def test_health_reports_unknown_capacity_and_ambiguous_operations_as_degraded():
    report = health_report(
        leases=[
            {"lease_id": "lease-1", "state": "active", "allocations": [{"resource_id": "gpu0"}]},
            {"lease_id": "lease-2", "state": "unknown", "allocations": [{"resource_id": "gpu1"}]},
        ],
        jobs=[{"job_id": "job-1", "state": "unknown"}],
        operations=[{"operation_id": "op-1", "status": "intent"}],
        backends=[{"backend_id": "llama", "state": "restoring"}],
    )
    assert report["status"] == "degraded"
    assert report["held_lease_count"] == 1
    assert report["held_resource_ids"] == ["gpu0"]
    assert report["unavailable_resource_ids"] == ["gpu1"]
    assert report["unresolved_job_count"] == 1
    assert report["unresolved_operation_count"] == 1
    assert report["unresolved_backend_count"] == 1


def test_policy_replay_calls_the_production_choose_next_contract_without_effects():
    original = [
        {
            "id": "blocked-high-priority",
            "owner_id": "alice",
            "priority_class": "interactive",
            "priority": 100,
            "created_at": "2026-01-01T00:00:00Z",
            "can_fit": False,
        },
        {
            "id": "fits-background",
            "owner_id": "bob",
            "priority_class": "background",
            "priority": 10,
            "created_at": "2026-01-01T00:00:00Z",
            "can_fit": True,
        },
    ]
    result = replay_policy(
        WeightedFairPolicy(),
        original,
        now="2026-01-01T00:01:00Z",
        service_by_owner={"alice": 0, "bob": 0},
    )

    assert len(original) == 2
    assert result["evidence_type"] == "policy_simulation"
    assert result["resource_allocation_performed"] is False
    assert result["execution_started"] is False
    assert result["backend_operation_performed"] is False
    assert result["proposal"]["request_id"] == "fits-background"
    assert result["proposal_validation"] == {"accepted_by_core": True, "reason": None}
    assert result["policy_id"] == "builtin.weighted-fair"
    assert result["policy_version"] == "1"


def test_policy_replay_rejects_policies_without_admission_capability():
    class ExplainOnly:
        def explain(self, request, **kwargs):
            return {}

    with pytest.raises(UnsupportedCapabilityError):
        replay_policy(
            ExplainOnly(),
            [{"id": "job-1", "can_fit": True}],
            now="2026-01-01T00:00:00Z",
        )


def test_public_policy_simulation_uses_builtin_admission_selection_without_ledger_effects(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    scheduler = Scheduler(
        ledger,
        ResourceAllocator(ledger),
        clock=FrozenClock(),
        config=SchedulerConfig(state_root=Path(tmp_path) / "state"),
    )
    principal = Principal("policy-reader", actions=frozenset({"policies.simulate"}))
    observations = [
        {
            "id": "blocked-interactive",
            "owner_id": "alice",
            "priority_class": "interactive",
            "priority": 100,
            "created_at": "2026-10-21T11:00:00Z",
            "can_fit": False,
        },
        {
            "id": "feasible-background",
            "owner_id": "bob",
            "priority_class": "background",
            "priority": 10,
            "created_at": "2026-10-21T11:59:00Z",
            "can_fit": True,
            "safe_backfill_for": ["blocked-interactive"],
        },
    ]
    result = scheduler.dispatch("policies.simulate", {"observations": observations}, principal)

    assert result["proposal"]["request_id"] == "feasible-background"
    assert result["proposal_validation"]["accepted_by_core"] is True
    assert result["resource_allocation_performed"] is False
    assert ledger.list("jobs") == []
    assert ledger.list("lease") == []
    assert ledger.events() == []


def test_invalid_output_bounds_and_non_json_replay_inputs_are_rejected():
    with pytest.raises(ValidationError):
        bound_output("hello", max_bytes=-1)
    with pytest.raises(ValidationError):
        replay_policy(WeightedFairPolicy(), [{"id": "job-1", "can_fit": float("nan")}])
