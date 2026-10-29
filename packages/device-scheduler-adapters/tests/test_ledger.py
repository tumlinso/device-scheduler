from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from device_scheduler_core.errors import ConflictError, IdempotencyConflictError
from device_scheduler_adapters import Ledger


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / "ledger.sqlite3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_revision_compare_and_swap_and_json_round_trip(self) -> None:
        created = self.ledger.put("job", "job-1", {"name": "test", "nested": {"n": 2}})
        self.assertEqual(created["_revision"], 1)
        self.assertEqual(self.ledger.get("job", "job-1")["nested"], {"n": 2})
        updated = self.ledger.put("job", "job-1", {"name": "updated"}, expected_revision=1)
        self.assertEqual(updated["_revision"], 2)
        with self.assertRaises(ConflictError):
            self.ledger.put("job", "job-1", {"name": "stale"}, expected_revision=1)
        with self.assertRaises(ConflictError):
            self.ledger.put("job", "job-1", {"name": "implicit-overwrite"})

    def test_idempotency_replays_result_and_rejects_changed_payload(self) -> None:
        calls: list[str] = []

        def create(tx):
            calls.append("called")
            tx.put("job", "job-2", {"state": "queued"})
            return {"job_id": "job-2"}

        result = self.ledger.idempotent("alice", "submit-1", "submit_job", {"argv": ["true"]}, create)
        replay = self.ledger.idempotent(
            "alice", "submit-1", "submit_job", {"argv": ["true"]},
            lambda tx: self.fail("same idempotency key must not run callback twice"),
        )
        self.assertEqual(result, replay)
        self.assertEqual(calls, ["called"])
        with self.assertRaises(IdempotencyConflictError):
            self.ledger.idempotent("alice", "submit-1", "submit_job", {"argv": ["false"]}, create)

    def test_callback_failure_rolls_back_intent_and_entity(self) -> None:
        def broken(tx):
            tx.put("job", "partial", {"state": "queued"})
            tx.event("job_created", "partial", {})
            raise RuntimeError("injected failure")

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.ledger.idempotent("alice", "rollback-1", "submit_job", {"id": "partial"}, broken)
        self.assertIsNone(self.ledger.get("job", "partial"))
        self.assertEqual(self.ledger.events(kind="job_created"), [])
        self.assertEqual(
            self.ledger.idempotent("alice", "rollback-1", "submit_job", {"id": "partial"}, lambda tx: {"ok": True}),
            {"ok": True},
        )

    def test_append_only_events_receipts_and_outbox(self) -> None:
        self.ledger.event("job_state", "job-3", {"state": "queued"})
        self.assertEqual(self.ledger.events(kind="job_state")[0]["payload"], {"state": "queued"})
        with self.ledger._connect() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE events SET payload_json='{}'")

        operation_id = self.ledger.operation_intent(
            "launch", {"attempt_id": "attempt-1"}, owner="system", idempotency_key="launch-1"
        )
        same_id = self.ledger.operation_intent(
            "launch", {"attempt_id": "attempt-1"}, owner="system", idempotency_key="launch-1"
        )
        self.assertEqual(operation_id, same_id)
        with self.assertRaises(IdempotencyConflictError):
            self.ledger.operation_intent(
                "launch", {"attempt_id": "attempt-2"}, owner="system", idempotency_key="launch-1"
            )
        receipt_id = self.ledger.record_receipt(operation_id, {"exit_code": 0})
        self.assertEqual(self.ledger.record_receipt(operation_id, {"exit_code": 0}), receipt_id)
        with self.assertRaises(ConflictError):
            self.ledger.record_receipt(operation_id, {"exit_code": 1}, status="failed")
        self.assertEqual(self.ledger.operation(operation_id)["status"], "complete")

        outbox_id = self.ledger.enqueue_outbox("job.result", {"job_id": "job-3"})
        self.assertEqual(self.ledger.pending_outbox()[0]["outbox_id"], outbox_id)
        self.ledger.mark_outbox_delivered(outbox_id)
        self.ledger.mark_outbox_delivered(outbox_id)
        self.assertEqual(self.ledger.pending_outbox(), [])

    def test_integrity_check_and_online_backup(self) -> None:
        self.ledger.put("schedule", "sched-1", {"state": "active"})
        backup = self.ledger.backup(self.root / "backup.sqlite3")
        restored = Ledger(backup)
        self.assertEqual(restored.get("schedule", "sched-1")["state"], "active")
        restored.integrity_check()

    def test_wal_full_sync_migration_and_truthful_capabilities(self) -> None:
        with self.ledger._connect() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            migrations = connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        self.assertEqual(str(journal_mode).lower(), "wal")
        self.assertEqual(synchronous, 2)
        self.assertEqual(schema_version, 1)
        self.assertEqual([row[0] for row in migrations], [1])
        capabilities = {item.name: item for item in self.ledger.capabilities()}
        self.assertIn("ledger.transactions", capabilities)
        self.assertTrue(capabilities["ledger.transactions"].supported)
        self.assertEqual(capabilities["ledger.transactions"].enforcement.value, "enforced")

    def test_autobench_retention_compacts_only_explicit_old_terminal_owner_records(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        new = datetime.now(timezone.utc).isoformat()
        expired_epoch = int(datetime.now(timezone.utc).timestamp()) - 100

        def add_job(job_id: str, *, owner: str = "alice", state: str = "succeeded", opt_in: bool = True, **extra):
            spec = {"owner_id": owner, "retention_policy": "autobench" if opt_in else None, "schema_version": 1,
                    "idempotency_key": f"autobench-v1:{expired_epoch}:{job_id}"}
            record = {
                "id": job_id,
                "job_id": job_id,
                "owner": owner,
                "owner_id": owner,
                "spec": spec,
                "state": state,
                "created_at": old,
                "updated_at": old,
                "attempt_id": f"attempt-{job_id}",
                "attempt_count": 2,
                "cleanup_verified": True,
                "result": {"state": "ok", "large": "x" * 1000},
                **extra,
            }
            self.ledger.put("jobs", job_id, record)
            self.ledger.put("attempts", f"attempt-{job_id}", {
                "attempt_id": f"attempt-{job_id}", "job_id": job_id, "state": state,
                "created_at": old, "finished_at": old, "cleanup_verified": True,
                "result": {"large": "y" * 1000}, "execution_handle": {"metadata": {"large": "z" * 1000}},
            })

        add_job("eligible", lease_id="lease-eligible", decision_id="decision-eligible")
        self.ledger.put("lease", "lease-eligible", {
            "lease_id": "lease-eligible", "owner_id": "alice", "state": "released",
            "request_id": "job:eligible", "idempotency_key": "job-lease:eligible",
        })
        self.ledger.put("decisions", "decision-eligible", {"decision_id": "decision-eligible", "subject_id": "eligible"})
        add_job("not-opted", opt_in=False)
        add_job("active", state="running")
        add_job("unknown", state="unknown")
        add_job("unclean", cleanup_verified=False)
        add_job("protected-lease", lease_id="lease-1")
        add_job("protected-reservation", reservation_id="reservation-1")
        add_job("other-owner", owner="bob")
        add_job("too-recent", state="failed")
        add_job("unexpired", state="succeeded")
        current_unexpired = self.ledger.get("jobs", "unexpired")
        current_unexpired.pop("_revision")
        current_unexpired["spec"]["idempotency_key"] = f"autobench-v1:{int(datetime.now(timezone.utc).timestamp()) + 86400}:unexpired"
        self.ledger.put("jobs", "unexpired", current_unexpired, expected_revision=1)
        current = self.ledger.get("jobs", "too-recent")
        current.pop("_revision")
        current["updated_at"] = new
        self.ledger.put("jobs", "too-recent", current, expected_revision=1)

        self.ledger.event("job.finished", "eligible", {"detail": "e" * 1000})
        submit = self.ledger.idempotent(
            "alice", "submit-eligible", "jobs.submit", {"spec": {"id": "eligible"}},
            lambda tx: {"job": tx.get("jobs", "eligible")},
        )
        operation_id = self.ledger.operation_intent(
            "job.finish", {"job_id": "eligible", "result": "r" * 1000}, owner="alice", idempotency_key="finish-eligible"
        )
        self.ledger.record_receipt(operation_id, {"job_id": "eligible", "receipt": "q" * 1000})
        attempt_operation_id = self.ledger.operation_intent(
            "job.preconditions", {"attempt_id": "attempt-eligible", "detail": "p" * 1000},
            owner="alice", operation_id="preconditions:attempt-eligible",
        )
        self.ledger.record_receipt(attempt_operation_id, {"attempt_id": "attempt-eligible", "receipt": "v" * 1000})
        self.ledger.enqueue_outbox("job.result", {"job_id": "eligible", "attempt_id": "attempt-eligible"}, outbox_id="delivered-eligible")
        self.ledger.mark_outbox_delivered("delivered-eligible")
        self.ledger.event("unrelated", "keep-me", {"small": True})
        unrelated_op = self.ledger.operation_intent("unrelated", {"detail": True}, owner="alice")
        self.ledger.record_receipt(unrelated_op, {"detail": True})
        before = self.ledger.storage_usage()["logical_payload_bytes"]

        result = self.ledger.compact_autobench_history(
            "alice", before=datetime.now(timezone.utc) - timedelta(days=1), max_records=100
        )

        self.assertEqual(result["jobs_purged"], 1)
        self.assertEqual(result["attempts_purged"], 1)
        self.assertEqual(result["events_purged"], 1)
        self.assertEqual(result["operations_purged"], 3)
        self.assertEqual(result["receipts_purged"], 2)
        self.assertEqual(result["leases_purged"], 1)
        self.assertEqual(result["decisions_purged"], 1)
        self.assertEqual(result["outbox_purged"], 1)
        self.assertLess(result["logical_payload_bytes_after"], before)
        self.assertIsNone(self.ledger.get("jobs", "eligible"))
        self.assertIsNone(self.ledger.get("attempts", "attempt-eligible"))
        self.assertIsNone(self.ledger.get("lease", "lease-eligible"))
        self.assertIsNone(self.ledger.get("decisions", "decision-eligible"))
        self.assertEqual(self.ledger.events(kind="job.finished"), [])
        self.assertIsNone(self.ledger.operation(operation_id))
        self.assertGreaterEqual(result["floor_expiry_unix_seconds"], expired_epoch)
        with self.assertRaises(ConflictError):
            self.ledger.validate_autobench_idempotency_key(
                "alice", f"autobench-v1:{expired_epoch}:eligible", now_unix_seconds=expired_epoch - 1
            )
        with self.assertRaises(ConflictError):
            # Scheduler validates before entering the generic idempotency API;
            # without the compact floor a deleted operation key could be reused.
            self.ledger.validate_autobench_idempotency_key(
                "alice", f"autobench-v1:{expired_epoch}:eligible", now_unix_seconds=expired_epoch - 1
            )
        for job_id in ("not-opted", "active", "unknown", "unclean", "protected-lease", "protected-reservation", "other-owner", "too-recent", "unexpired"):
            self.assertFalse(self.ledger.get("jobs", job_id).get("retention_compacted", False), job_id)
        with self.ledger._connect() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE events SET payload_json='{}' WHERE entity_id='keep-me'")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE operation_receipts SET receipt_json='{}' WHERE operation_id=?", (unrelated_op,))
        self.ledger.integrity_check()

    def test_autobench_retention_limits_and_empty_run_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            self.ledger.compact_autobench_history("alice", before="2026-10-01T00:00:00Z", max_records=0)
        with self.assertRaises(ValueError):
            self.ledger.compact_autobench_history("alice", before="2026-10-01T00:00:00", max_records=1)
        result = self.ledger.compact_autobench_history("alice", before="2026-10-01T00:00:00Z", max_records=1)
        self.assertEqual(result["jobs_purged"], 0)
        self.assertGreaterEqual(self.ledger.storage_usage()["allocated_page_bytes"], 0)

    def test_autobench_owner_budget_is_atomic_and_fails_closed_by_bytes_and_rows(self) -> None:
        self.ledger.put("jobs", "budget-job", {
            "job_id": "budget-job", "owner": "alice", "state": "queued",
            "spec": {"retention_policy": "autobench", "idempotency_key": "autobench-v1:1999999999:budget"},
        })
        usage = self.ledger.autobench_owner_usage("alice")
        self.assertEqual(usage["job_count"], 1)
        self.assertEqual(usage["row_count"], 1)
        with self.ledger.transaction() as tx:
            checked = tx.check_autobench_owner_budget(
                "alice", additional_bytes=1, additional_rows=1, limit_bytes=1024, limit_rows=3
            )
            self.assertEqual(checked["projected_rows"], 2)
            with self.assertRaises(ConflictError):
                tx.check_autobench_owner_budget(
                    "alice", additional_bytes=1, additional_rows=1, limit_bytes=1024, limit_rows=1
                )
            with self.assertRaises(ConflictError):
                tx.check_autobench_owner_budget(
                    "alice", additional_bytes=1024, additional_rows=1, limit_bytes=1024, limit_rows=5
                )

    def test_autobench_repeated_expiration_epochs_keep_database_bounded(self) -> None:
        start_bytes = self.ledger.storage_usage()["database_bytes"]
        for epoch in range(8):
            expiry = int(datetime.now(timezone.utc).timestamp()) - 10
            for index in range(20):
                job_id = f"epoch-{epoch}-{index}"
                self.ledger.put("jobs", job_id, {
                    "job_id": job_id, "owner": "alice", "owner_id": "alice", "state": "succeeded",
                    "cleanup_verified": True, "created_at": "2020-01-01T00:00:00+00:00",
                    "updated_at": "2020-01-01T00:00:00+00:00", "result": {"payload": "x" * 2048},
                    "spec": {"owner_id": "alice", "retention_policy": "autobench",
                             "idempotency_key": f"autobench-v1:{expiry}:{job_id}"},
                })
                self.ledger.put("attempts", f"attempt-{job_id}", {
                    "job_id": job_id, "attempt_id": f"attempt-{job_id}", "state": "succeeded",
                    "result": {"payload": "y" * 2048},
                })
                self.ledger.event("job.finished", job_id, {"payload": "z" * 2048})
            result = self.ledger.compact_autobench_history("alice", max_records=100)
            self.assertEqual(result["jobs_purged"], 20)
            self.assertTrue(result["physical_reclaim"])
        final_size = self.ledger.storage_usage()["database_bytes"]
        self.assertLess(final_size, start_bytes + 128 * 1024)
        summary_rows = [row for row in self.ledger.list("autobench_retention") if row["owner_id"] == "alice"]
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0]["jobs_purged_total"], 160)


if __name__ == "__main__":
    unittest.main()
