from __future__ import annotations

import sqlite3
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
