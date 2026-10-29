from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from device_scheduler_core import QueuePriorityDecision, WeightedFairPolicy


class WeightedFairPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 10, 21, tzinfo=timezone.utc)
        self.policy = WeightedFairPolicy(
            age_quantum_seconds=100,
            max_age_boost=3,
            large_request_protection_seconds=300,
            fairness_penalty=0.1,
        )

    def request(self, request_id: str, owner: str, priority_class: str, *, age: float = 0) -> dict:
        return {
            "request_id": request_id,
            "owner_id": owner,
            "priority_class": priority_class,
            "priority": 50,
            "created_at": (self.now - timedelta(seconds=age)).isoformat(),
        }

    def test_interactive_class_ranks_above_normal_and_background(self) -> None:
        requests = [
            self.request("background", "b", "background"),
            self.request("normal", "n", "normal"),
            self.request("interactive", "i", "interactive"),
        ]
        decisions = self.policy.rank_requests(requests, now=self.now)
        self.assertEqual([item.request_id for item in decisions], ["interactive", "normal", "background"])
        self.assertTrue(all(isinstance(item, QueuePriorityDecision) for item in decisions))

    def test_fresh_interactive_stays_above_aged_background(self) -> None:
        old_background = self.request("old", "b", "background", age=1000)
        new_interactive = self.request("new", "i", "interactive")
        ranked = self.policy.rank_requests([new_interactive, old_background], now=self.now)
        self.assertEqual(ranked[0].request_id, "new")

    def test_aging_can_promote_background_within_its_tier(self) -> None:
        old_background = self.request("old", "b", "background", age=1000)
        new_background = self.request("new", "c", "background")
        ranked = self.policy.rank_requests([new_background, old_background], now=self.now)
        self.assertEqual(ranked[0].request_id, "old")
        self.assertEqual(ranked[0].reason_code, "queue_aging")

    def test_expected_duration_breaks_only_equal_policy_score_ties(self) -> None:
        short = {"job_id": "short", "owner_id": "a", "priority_class": "background", "priority": 50,
                 "created_at": self.now, "spec": {"expected_duration_seconds": 10}}
        long = {"job_id": "long", "owner_id": "a", "priority_class": "background", "priority": 50,
                "created_at": self.now, "spec": {"expected_duration_seconds": 60}}
        ranked = self.policy.rank_requests([long, short], now=self.now)
        assert [item.request_id for item in ranked] == ["short", "long"]

    def test_weighted_fairness_accounts_for_recent_service(self) -> None:
        a = self.request("a", "well-served", "normal")
        b = self.request("b", "underserved", "normal")
        ranked = self.policy.rank_requests(
            [a, b], now=self.now, service_by_owner={"well-served": 100, "underserved": 0}
        )
        self.assertEqual(ranked[0].request_id, "b")
        self.assertEqual(ranked[0].reason_code, "weighted_fairness")

    def test_aged_request_protection_and_safe_backfill(self) -> None:
        large = self.request("large", "old-owner", "normal", age=400)
        small = self.request("small", "other", "normal")
        chosen = self.policy.choose_next(
            [small, large], can_fit=lambda _: True, now=self.now,
        )
        self.assertEqual(chosen.request_id, "large")
        self.assertEqual(chosen.reason_code, "aged_request_protected")

        backfilled = self.policy.choose_next(
            [small, large],
            can_fit=lambda item: item["request_id"] == "small",
            safe_backfill=lambda candidate, protected: candidate["request_id"] == "small",
            now=self.now,
        )
        self.assertEqual(backfilled.request_id, "small")
        self.assertEqual(backfilled.reason_code, "safe_backfill")

        blocked = self.policy.choose_next(
            [small, large],
            can_fit=lambda item: item["request_id"] == "small",
            safe_backfill=lambda _candidate, _protected: False,
            now=self.now,
        )
        self.assertIsNone(blocked)

        unchecked = self.policy.choose_next(
            [small, large],
            can_fit=lambda item: item["request_id"] == "small",
            now=self.now,
        )
        self.assertEqual(unchecked.request_id, "small")
        self.assertEqual(unchecked.reason_code, "backfill_unchecked")

    def test_configured_owner_weights_adjust_weighted_fairness(self) -> None:
        policy = WeightedFairPolicy(
            owner_weights={"favored": 2.0, "default": 1.0},
            fairness_penalty=0.1,
        )
        favored = self.request("favored-job", "favored", "normal")
        default = self.request("default-job", "default", "normal")
        decisions = policy.rank_requests(
            [favored, default],
            now=self.now,
            service_by_owner={"favored": 10, "default": 10},
        )
        self.assertEqual(decisions[0].request_id, "favored-job")

    def test_preemption_is_advisory_and_respects_restart_budget(self) -> None:
        request = {
            "id": "interactive-request",
            "owner": "interactive-owner",
            "state": "queued",
            "priority_class": "interactive",
            "priority": 95,
            "created_at": self.now.isoformat(),
            "spec": {"priority_class": "interactive", "priority": 95},
        }
        target = {
            "id": "background-job",
            "owner": "background-owner",
            "state": "running",
            "attempt_id": "attempt-1",
            "attempt_count": 1,
            "lease_id": "lease-1",
            "reservation_id": None,
            "started_at": self.now.isoformat(),
            "spec": {
                "priority_class": "background",
                "priority": 10,
                "max_attempts": 2,
                "interruption": {"restart_safe": True, "cancellation_supported": True},
            },
        }
        proposal = self.policy.propose_preemption(
            request,
            [target],
            {"now": self.now},
            would_unblock=lambda candidate: candidate["id"] == "background-job",
        )
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.action, "cancel_and_restart")
        self.assertEqual(proposal.target_job_id, "background-job")
        self.assertEqual(proposal.target_attempt_id, "attempt-1")
        self.assertEqual(proposal.target_lease_id, "lease-1")
        self.assertGreater(proposal.requester_score, proposal.target_score)

        exhausted = {**target, "attempt_count": 2}
        self.assertIsNone(self.policy.propose_preemption(request, [exhausted], {"now": self.now}))
        not_feasible = self.policy.propose_preemption(
            request, [target], {"now": self.now}, would_unblock=lambda _candidate: False
        )
        self.assertIsNone(not_feasible)

    def test_policy_does_not_authorize_when_nothing_fits(self) -> None:
        requests = [self.request("one", "one", "interactive")]
        self.assertIsNone(self.policy.choose_next(requests, can_fit=lambda _: False, now=self.now))


if __name__ == "__main__":
    unittest.main()
