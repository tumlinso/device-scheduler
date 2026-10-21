"""Transactional SQLite ledger used by the single-host scheduler.

The ledger is deliberately environment neutral. It stores versioned JSON entity
records, append-only events and receipts, operation intents, and a durable
outbox. External effects are never run inside a ledger transaction: callers
record intent, perform the effect, then record its receipt.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar

from device_scheduler_core.errors import (
    ConflictError,
    IdempotencyConflictError,
    ReconciliationRequiredError,
)
from device_scheduler_core.models import AdapterCapability, EnforcementLevel


T = TypeVar("T")
_REVISION_KEY = "_revision"


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as ledger JSON")


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding suitable for request digests."""

    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def request_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = json.loads(row["data_json"])
    result[_REVISION_KEY] = row["revision"]
    return result


class LedgerTransaction:
    """Operations scoped to one ``BEGIN IMMEDIATE`` SQLite transaction."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def get(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT revision, data_json FROM entities WHERE kind=? AND entity_id=?",
            (kind, entity_id),
        ).fetchone()
        return _decode_row(row)

    def list(self, kind: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT entity_id, revision, data_json FROM entities WHERE kind=? ORDER BY entity_id",
            (kind,),
        ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = json.loads(row["data_json"])
            value[_REVISION_KEY] = row["revision"]
            values.append(value)
        return values

    def put(
        self,
        kind: str,
        entity_id: str,
        data: Mapping[str, Any],
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Create/update an entity with revision compare-and-swap.

        ``expected_revision=None`` means create-only. For an update, pass the
        revision returned by ``get``/``put``. The returned mapping carries the
        current revision in the reserved ``_revision`` field.
        """

        if not kind or not entity_id:
            raise ValueError("kind and entity_id must be non-empty")
        if _REVISION_KEY in data:
            raise ValueError(f"{_REVISION_KEY} is ledger-owned metadata")
        row = self._connection.execute(
            "SELECT revision FROM entities WHERE kind=? AND entity_id=?",
            (kind, entity_id),
        ).fetchone()
        now = _now()
        value = dict(data)
        encoded = canonical_json(value)
        if row is None:
            if expected_revision is not None:
                raise ConflictError(
                    f"Entity {kind}/{entity_id} does not exist at revision {expected_revision}",
                    details={"kind": kind, "entity_id": entity_id, "expected_revision": expected_revision},
                )
            revision = 1
            self._connection.execute(
                "INSERT INTO entities(kind, entity_id, revision, data_json, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (kind, entity_id, revision, encoded, now, now),
            )
        else:
            current = int(row["revision"])
            if expected_revision is None or current != expected_revision:
                raise ConflictError(
                    f"Entity {kind}/{entity_id} revision mismatch",
                    details={
                        "kind": kind,
                        "entity_id": entity_id,
                        "expected_revision": expected_revision,
                        "actual_revision": current,
                    },
                )
            revision = current + 1
            self._connection.execute(
                "UPDATE entities SET revision=?, data_json=?, updated_at=? "
                "WHERE kind=? AND entity_id=? AND revision=?",
                (revision, encoded, now, kind, entity_id, current),
            )
        value[_REVISION_KEY] = revision
        return value

    def event(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        self._connection.execute(
            "INSERT INTO events(event_id, kind, entity_id, payload_json, created_at) VALUES(?,?,?,?,?)",
            (event_id, kind, entity_id, canonical_json(payload), _now()),
        )
        return event_id

    def events(
        self,
        *,
        kind: str | None = None,
        entity_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        clauses = ["sequence > ?"]
        values: list[Any] = [after_sequence]
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            values.append(entity_id)
        values.append(limit)
        rows = self._connection.execute(
            "SELECT sequence, event_id, kind, entity_id, payload_json, created_at FROM events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY sequence LIMIT ?",
            values,
        ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "kind": row["kind"],
                "entity_id": row["entity_id"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT operation_id, owner, idempotency_key, operation, request_digest, request_json, "
            "status, response_json, created_at, updated_at FROM operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        receipts = self._connection.execute(
            "SELECT receipt_id, status, receipt_json, created_at FROM operation_receipts "
            "WHERE operation_id=? ORDER BY receipt_sequence",
            (operation_id,),
        ).fetchall()
        return {
            "operation_id": row["operation_id"],
            "owner": row["owner"],
            "idempotency_key": row["idempotency_key"],
            "operation": row["operation"],
            "request_digest": row["request_digest"],
            "request": json.loads(row["request_json"]),
            "status": row["status"],
            "response": json.loads(row["response_json"]) if row["response_json"] else None,
            "receipts": [
                {
                    "receipt_id": receipt["receipt_id"],
                    "status": receipt["status"],
                    "receipt": json.loads(receipt["receipt_json"]),
                    "created_at": receipt["created_at"],
                }
                for receipt in receipts
            ],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def operation_intent(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        owner: str = "system",
        idempotency_key: str | None = None,
    ) -> str:
        """Persist an external-operation intent before performing its effect."""

        op_id = operation_id or str(uuid.uuid4())
        digest = request_digest({"operation": operation, "payload": payload})
        if idempotency_key is not None:
            prior = self._connection.execute(
                "SELECT operation_id, operation, request_digest FROM operations "
                "WHERE owner=? AND idempotency_key=?",
                (owner, idempotency_key),
            ).fetchone()
            if prior is not None:
                if prior["operation"] != operation or prior["request_digest"] != digest:
                    raise IdempotencyConflictError(
                        f"Idempotency key {idempotency_key!r} was already used with a different operation",
                        details={"owner": owner, "key": idempotency_key},
                    )
                return str(prior["operation_id"])
        self._connection.execute(
            "INSERT INTO operations(operation_id, owner, idempotency_key, operation, request_digest, "
            "request_json, status, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (op_id, owner, idempotency_key, operation, digest, canonical_json(payload), "intent", _now(), _now()),
        )
        return op_id

    def record_receipt(
        self,
        operation_id: str,
        receipt: Mapping[str, Any],
        *,
        status: str = "complete",
    ) -> str:
        if status not in {"complete", "failed", "unknown"}:
            raise ValueError("receipt status must be complete, failed, or unknown")
        row = self._connection.execute(
            "SELECT status FROM operations WHERE operation_id=?", (operation_id,)
        ).fetchone()
        if row is None:
            raise ConflictError(f"Unknown operation intent {operation_id}")
        current_status = row["status"]
        prior_receipt = self._connection.execute(
            "SELECT receipt_id, status, receipt_json FROM operation_receipts "
            "WHERE operation_id=? ORDER BY receipt_sequence DESC LIMIT 1",
            (operation_id,),
        ).fetchone()
        encoded = canonical_json(receipt)
        if current_status in {"complete", "failed"}:
            if prior_receipt is not None and prior_receipt["status"] == status and prior_receipt["receipt_json"] == encoded:
                return str(prior_receipt["receipt_id"])
            raise ConflictError("terminal operation receipt cannot be replaced")
        if current_status == "unknown" and status == "unknown" and prior_receipt is not None:
            if prior_receipt["receipt_json"] == encoded:
                return str(prior_receipt["receipt_id"])
        receipt_id = str(uuid.uuid4())
        self._connection.execute(
            "INSERT INTO operation_receipts(receipt_id, operation_id, status, receipt_json, created_at) "
            "VALUES(?,?,?,?,?)",
            (receipt_id, operation_id, status, encoded, _now()),
        )
        self._connection.execute(
            "UPDATE operations SET status=?, updated_at=? WHERE operation_id=?",
            (status, _now(), operation_id),
        )
        return receipt_id

    def enqueue_outbox(self, topic: str, payload: Mapping[str, Any], *, outbox_id: str | None = None) -> str:
        message_id = outbox_id or str(uuid.uuid4())
        self._connection.execute(
            "INSERT INTO outbox(outbox_id, topic, payload_json, created_at) VALUES(?,?,?,?)",
            (message_id, topic, canonical_json(payload), _now()),
        )
        return message_id

    def idempotent(
        self,
        owner: str,
        key: str,
        operation: str,
        payload: Mapping[str, Any],
        callback: Callable[["LedgerTransaction"], T],
    ) -> T:
        if not owner or not key:
            raise ValueError("owner and idempotency key must be non-empty")
        digest = request_digest({"operation": operation, "payload": payload})
        existing = self._connection.execute(
            "SELECT operation, request_digest, status, response_json, operation_id FROM operations "
            "WHERE owner=? AND idempotency_key=?",
            (owner, key),
        ).fetchone()
        if existing is not None:
            if existing["operation"] != operation or existing["request_digest"] != digest:
                raise IdempotencyConflictError(
                    f"Idempotency key {key!r} was already used with a different request",
                    details={"owner": owner, "key": key},
                )
            if existing["status"] != "complete":
                raise ReconciliationRequiredError(
                    f"Operation {existing['operation_id']} has not completed",
                    details={"operation_id": existing["operation_id"], "status": existing["status"]},
                )
            if existing["response_json"] is None:
                raise ReconciliationRequiredError(
                    f"Operation {existing['operation_id']} completed through the receipt path without a callback response",
                    details={"operation_id": existing["operation_id"], "status": existing["status"]},
                )
            return json.loads(existing["response_json"])

        operation_id = str(uuid.uuid4())
        now = _now()
        self._connection.execute(
            "INSERT INTO operations(operation_id, owner, idempotency_key, operation, request_digest, "
            "request_json, status, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (operation_id, owner, key, operation, digest, canonical_json(payload), "intent", now, now),
        )
        result = callback(self)
        response_json = canonical_json(result)
        self._connection.execute(
            "UPDATE operations SET status='complete', response_json=?, updated_at=? WHERE operation_id=?",
            (response_json, _now(), operation_id),
        )
        return json.loads(response_json)

    def next_fencing_token(self) -> int:
        self._connection.execute(
            "UPDATE fencing_epoch SET epoch=epoch+1 WHERE singleton=1"
        )
        row = self._connection.execute(
            "SELECT epoch FROM fencing_epoch WHERE singleton=1"
        ).fetchone()
        return int(row["epoch"])


class Ledger:
    """A process-safe, revisioned entity ledger backed by one SQLite database."""

    api_version = "1"

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_seconds: float = 30.0):
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout_seconds = busy_timeout_seconds
        self._schema_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def capabilities(self) -> tuple[AdapterCapability, ...]:
        """Describe the persistence guarantees this adapter actually provides."""

        names = (
            "transactions",
            "revision_cas",
            "idempotency",
            "append_only_events",
            "operation_journal",
            "delivery_outbox",
            "online_backup",
            "integrity_check",
        )
        return tuple(
            AdapterCapability(
                name=f"ledger.{name}",
                supported=True,
                enforcement=EnforcementLevel.ENFORCED,
                details={"engine": "sqlite", "journal_mode": "wal", "synchronous": "full"},
            )
            for name in names
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        with self._schema_lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("BEGIN EXCLUSIVE")
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version > 1:
                    raise RuntimeError(f"Ledger schema version {version} is newer than this software supports")
                if version < 1:
                    self._migration_1(conn)
                    conn.execute("PRAGMA user_version=1")
                    conn.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES(1,?)", (_now(),)
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        self.integrity_check()

    @staticmethod
    def _migration_1(conn: sqlite3.Connection) -> None:
        statements = [
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
            "CREATE TABLE entities(kind TEXT NOT NULL, entity_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0), data_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(kind, entity_id))",
            "CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, entity_id TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)",
            "CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END",
            "CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END",
            "CREATE TABLE operations(operation_id TEXT PRIMARY KEY, owner TEXT NOT NULL, idempotency_key TEXT, operation TEXT NOT NULL, request_digest TEXT NOT NULL, request_json TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('intent','complete','failed','unknown')), response_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(owner, idempotency_key))",
            "CREATE TABLE operation_receipts(receipt_sequence INTEGER PRIMARY KEY AUTOINCREMENT, receipt_id TEXT NOT NULL UNIQUE, operation_id TEXT NOT NULL REFERENCES operations(operation_id), status TEXT NOT NULL, receipt_json TEXT NOT NULL, created_at TEXT NOT NULL)",
            "CREATE TRIGGER operation_receipts_no_update BEFORE UPDATE ON operation_receipts BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END",
            "CREATE TRIGGER operation_receipts_no_delete BEFORE DELETE ON operation_receipts BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END",
            "CREATE TABLE outbox(outbox_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, delivered_at TEXT, attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT)",
            "CREATE INDEX outbox_pending_idx ON outbox(delivered_at, created_at)",
            "CREATE TABLE fencing_epoch(singleton INTEGER PRIMARY KEY CHECK(singleton=1), epoch INTEGER NOT NULL CHECK(epoch>=0))",
            "INSERT INTO fencing_epoch(singleton, epoch) VALUES(1,0)",
        ]
        for statement in statements:
            conn.execute(statement)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[LedgerTransaction]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            tx = LedgerTransaction(conn)
            yield tx
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get(self, kind: str, entity_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT revision, data_json FROM entities WHERE kind=? AND entity_id=?",
                (kind, entity_id),
            ).fetchone()
            return _decode_row(row)

    def list(self, kind: str) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT entity_id, revision, data_json FROM entities WHERE kind=? ORDER BY entity_id",
                (kind,),
            ).fetchall()
            values = []
            for row in rows:
                value = json.loads(row["data_json"])
                value[_REVISION_KEY] = row["revision"]
                values.append(value)
            return values

    def put(
        self,
        kind: str,
        entity_id: str,
        data: Mapping[str, Any],
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as tx:
            return tx.put(kind, entity_id, data, expected_revision)

    def event(self, kind: str, entity_id: str, payload: Mapping[str, Any]) -> str:
        with self.transaction() as tx:
            return tx.event(kind, entity_id, payload)

    def events(
        self,
        *,
        kind: str | None = None,
        entity_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as conn:
            return LedgerTransaction(conn).events(
                kind=kind,
                entity_id=entity_id,
                after_sequence=after_sequence,
                limit=limit,
            )

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        with contextlib.closing(self._connect()) as conn:
            return LedgerTransaction(conn).operation(operation_id)

    def operations(self, *, entity_id: str | None = None) -> list[dict[str, Any]]:
        """Return operation intents and their current receipt status.

        The ledger does not infer a canonical entity link from arbitrary JSON.
        When ``entity_id`` is supplied, records are matched against common
        identity fields in the durable request and response objects.
        """

        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute("SELECT operation_id FROM operations ORDER BY created_at, operation_id").fetchall()
            values = [LedgerTransaction(conn).operation(row["operation_id"]) for row in rows]
        result = [value for value in values if value is not None]
        if entity_id is None:
            return result
        fields = {"entity_id", "job_id", "attempt_id", "lease_id", "reservation_id", "backend_id", "handoff_id", "schedule_id"}
        return [
            value for value in result
            if any(
                isinstance(value.get(group), Mapping)
                and any(value[group].get(field) == entity_id for field in fields)
                for group in ("request", "response")
            )
        ]

    def idempotent(
        self,
        owner: str,
        key: str,
        operation: str,
        payload: Mapping[str, Any],
        callback: Callable[[LedgerTransaction], T],
    ) -> T:
        with self.transaction() as tx:
            return tx.idempotent(owner, key, operation, payload, callback)

    def operation_intent(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        owner: str = "system",
        idempotency_key: str | None = None,
    ) -> str:
        with self.transaction() as tx:
            return tx.operation_intent(
                operation,
                payload,
                operation_id=operation_id,
                owner=owner,
                idempotency_key=idempotency_key,
            )

    def record_receipt(self, operation_id: str, receipt: Mapping[str, Any], *, status: str = "complete") -> str:
        with self.transaction() as tx:
            return tx.record_receipt(operation_id, receipt, status=status)

    def enqueue_outbox(self, topic: str, payload: Mapping[str, Any], *, outbox_id: str | None = None) -> str:
        with self.transaction() as tx:
            return tx.enqueue_outbox(topic, payload, outbox_id=outbox_id)

    def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT outbox_id, topic, payload_json, created_at, attempts FROM outbox "
                "WHERE delivered_at IS NULL ORDER BY created_at, outbox_id LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "outbox_id": row["outbox_id"],
                    "topic": row["topic"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                    "attempts": row["attempts"],
                }
                for row in rows
            ]

    def outbox(self, *, consumer: str | None = None) -> list[dict[str, Any]]:
        """Return pending delivery records; this single-host adapter has one consumer cursor."""

        if consumer not in (None, "default"):
            raise ValueError("SQLite ledger outbox supports only the default consumer")
        return self.pending_outbox(limit=100_000)

    def mark_outbox_delivered(self, outbox_id: str) -> None:
        with self.transaction() as tx:
            row = tx._connection.execute(
                "SELECT delivered_at FROM outbox WHERE outbox_id=?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise ConflictError(f"Unknown outbox message {outbox_id}")
            if row["delivered_at"] is not None:
                return
            tx._connection.execute(
                "UPDATE outbox SET delivered_at=?, attempts=attempts+1, last_error=NULL WHERE outbox_id=?",
                (_now(), outbox_id),
            )

    def mark_outbox_failed(self, outbox_id: str, error: str) -> None:
        with self.transaction() as tx:
            cur = tx._connection.execute(
                "UPDATE outbox SET attempts=attempts+1, last_error=? WHERE outbox_id=? AND delivered_at IS NULL",
                (error, outbox_id),
            )
            if cur.rowcount != 1:
                raise ConflictError(f"Unknown or delivered outbox message {outbox_id}")

    def next_fencing_token(self) -> int:
        with self.transaction() as tx:
            return tx.next_fencing_token()

    def integrity_check(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            messages = [str(row[0]) for row in rows]
            if messages != ["ok"]:
                raise sqlite3.DatabaseError("SQLite integrity check failed: " + "; ".join(messages))
            foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise sqlite3.DatabaseError(f"SQLite foreign key check failed: {foreign_key_errors!r}")

    def backup(self, destination: str | os.PathLike[str]) -> Path:
        """Create a transactionally consistent SQLite backup at a new path."""

        target = Path(destination).expanduser().resolve()
        if target == self.path:
            raise ValueError("backup destination must differ from the live ledger")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        source = self._connect()
        dest = sqlite3.connect(target)
        try:
            source.backup(dest)
            dest.commit()
        finally:
            dest.close()
            source.close()
        backup = Ledger(target, busy_timeout_seconds=self.busy_timeout_seconds)
        backup.integrity_check()
        return target


__all__ = ["Ledger", "LedgerTransaction", "canonical_json", "request_digest"]
