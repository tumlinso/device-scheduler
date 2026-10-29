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
import re
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


_AUTOBENCH_KEY = re.compile(r"^autobench-v1:([1-9][0-9]*):([A-Za-z0-9._~-]+)$")


def _autobench_key_expiry(record: Mapping[str, Any]) -> int | None:
    spec = record.get("spec")
    key = spec.get("idempotency_key") if isinstance(spec, Mapping) else None
    if not isinstance(key, str):
        return None
    match = _AUTOBENCH_KEY.fullmatch(key)
    return int(match.group(1)) if match else None


def _autobench_mentions_job(value: Any, job_ids: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(_autobench_mentions_job(child, job_ids) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_autobench_mentions_job(child, job_ids) for child in value)
    return isinstance(value, str) and value in job_ids


def _autobench_owner_usage(connection: sqlite3.Connection, owner_id: str) -> dict[str, Any]:
    """Estimate bytes owned by explicitly opted-in Autobench records."""
    job_rows = connection.execute("SELECT entity_id, data_json FROM entities WHERE kind='jobs'").fetchall()
    job_ids: set[str] = set()
    logical_bytes = 0
    for row in job_rows:
        record = json.loads(row["data_json"])
        spec = record.get("spec")
        if record.get("owner", record.get("owner_id")) != owner_id or not isinstance(spec, Mapping) or spec.get("retention_policy") != "autobench":
            continue
        job_ids.add(str(row["entity_id"]))
        logical_bytes += len(row["data_json"].encode("utf-8"))

    lease_ids: set[str] = set()
    for row in job_rows:
        record = json.loads(row["data_json"])
        spec = record.get("spec")
        if (record.get("owner", record.get("owner_id")) == owner_id
                and isinstance(spec, Mapping) and spec.get("retention_policy") == "autobench"
                and isinstance(record.get("lease_id"), str)):
            lease_ids.add(record["lease_id"])
    lease_count = 0
    for lease_id in lease_ids:
        lease = connection.execute("SELECT data_json FROM entities WHERE kind='lease' AND entity_id=?", (lease_id,)).fetchone()
        if lease is not None:
            lease_count += 1
            logical_bytes += len(lease["data_json"].encode("utf-8"))

    attempt_count = 0
    attempt_ids: set[str] = set()
    for row in connection.execute("SELECT entity_id,data_json FROM entities WHERE kind='attempts'").fetchall():
        value = json.loads(row["data_json"])
        if value.get("job_id") in job_ids:
            attempt_count += 1
            attempt_ids.add(str(value.get("attempt_id", row["entity_id"])))
            logical_bytes += len(row["data_json"].encode("utf-8"))

    event_entity_ids = set(job_ids)
    for job_id in job_ids:
        job = connection.execute("SELECT data_json FROM entities WHERE kind='jobs' AND entity_id=?", (job_id,)).fetchone()
        record = json.loads(job["data_json"]) if job else {}
        lease_id = record.get("lease_id")
        if isinstance(lease_id, str):
            lease = connection.execute("SELECT data_json FROM entities WHERE kind='lease' AND entity_id=?", (lease_id,)).fetchone()
            lease_record = json.loads(lease["data_json"]) if lease else {}
            if (lease_record.get("request_id") == f"job:{job_id}"
                    or lease_record.get("idempotency_key") == f"job-lease:{job_id}"):
                event_entity_ids.add(lease_id)
    event_count = 0
    if event_entity_ids:
        placeholders = ",".join("?" for _ in event_entity_ids)
        rows = connection.execute(f"SELECT payload_json FROM events WHERE entity_id IN ({placeholders})", tuple(sorted(event_entity_ids))).fetchall()
        event_count = len(rows)
        logical_bytes += sum(len(row["payload_json"].encode("utf-8")) for row in rows)

    decision_count = 0
    if job_ids:
        placeholders = ",".join("?" for _ in job_ids)
        rows = connection.execute(
            f"SELECT data_json FROM entities WHERE kind='decisions' AND json_extract(data_json,'$.subject_id') IN ({placeholders})",
            tuple(sorted(job_ids)),
        ).fetchall()
        decision_count = len(rows)
        logical_bytes += sum(len(row["data_json"].encode("utf-8")) for row in rows)

    outbox_count = 0
    if job_ids or attempt_ids:
        for row in connection.execute("SELECT payload_json FROM outbox").fetchall():
            if _autobench_mentions_job(json.loads(row["payload_json"]), job_ids | attempt_ids):
                outbox_count += 1
                logical_bytes += len(row["payload_json"].encode("utf-8"))

    operation_ids: set[str] = set()
    operation_count = 0
    for row in connection.execute("SELECT operation_id, owner, idempotency_key, request_json, response_json FROM operations").fetchall():
        if row["owner"] != owner_id:
            continue
        request = json.loads(row["request_json"])
        response = json.loads(row["response_json"]) if row["response_json"] else None
        key_match = isinstance(row["idempotency_key"], str) and row["idempotency_key"].startswith("autobench-v1:")
        if key_match or _autobench_mentions_job(request, job_ids | attempt_ids) or _autobench_mentions_job(response, job_ids | attempt_ids):
            operation_ids.add(str(row["operation_id"]))
            operation_count += 1
            logical_bytes += len(row["request_json"].encode("utf-8"))
            logical_bytes += len(row["response_json"].encode("utf-8")) if row["response_json"] else 0

    receipt_count = 0
    if operation_ids:
        placeholders = ",".join("?" for _ in operation_ids)
        rows = connection.execute(f"SELECT receipt_json FROM operation_receipts WHERE operation_id IN ({placeholders})", tuple(sorted(operation_ids))).fetchall()
        receipt_count = len(rows)
        logical_bytes += sum(len(row["receipt_json"].encode("utf-8")) for row in rows)
    floor_row = connection.execute(
        "SELECT data_json FROM entities WHERE kind='autobench_retention' AND entity_id=?", (owner_id,)
    ).fetchone()
    floor = int(json.loads(floor_row["data_json"]).get("floor_expiry_unix_seconds", 0)) if floor_row else 0
    return {
        "owner_id": owner_id,
        "logical_payload_bytes": logical_bytes,
        "job_count": len(job_ids),
        "attempt_count": attempt_count,
        "event_count": event_count,
        "decision_count": decision_count,
        "operation_count": operation_count,
        "receipt_count": receipt_count,
        "lease_count": lease_count,
        "outbox_count": outbox_count,
        "row_count": len(job_ids) + lease_count + attempt_count + event_count + decision_count + operation_count + receipt_count + outbox_count,
        "floor_expiry_unix_seconds": floor,
    }


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

    def autobench_owner_usage(self, owner_id: str) -> dict[str, Any]:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        return _autobench_owner_usage(self._connection, owner_id)

    def check_autobench_owner_budget(
        self,
        owner_id: str,
        *,
        additional_bytes: int,
        additional_rows: int = 1,
        limit_bytes: int = 64 * 1024 * 1024,
        limit_rows: int = 10_000,
    ) -> dict[str, Any]:
        """Fail closed when an owner would exceed its Autobench ledger budget.

        Call this on the transaction passed to the job-submit idempotency
        callback, before inserting the new job, to make admission atomic with
        its ledger mutation.
        """
        for name, value in (("additional_bytes", additional_bytes), ("additional_rows", additional_rows), ("limit_bytes", limit_bytes), ("limit_rows", limit_rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if limit_bytes == 0 or limit_rows == 0:
            raise ValueError("limits must be positive")
        usage = self.autobench_owner_usage(owner_id)
        projected = int(usage["logical_payload_bytes"]) + additional_bytes
        projected_rows = int(usage["row_count"]) + additional_rows
        usage.update({"additional_bytes": additional_bytes, "projected_bytes": projected, "limit_bytes": limit_bytes,
                      "additional_rows": additional_rows, "projected_rows": projected_rows, "limit_rows": limit_rows})
        if projected > limit_bytes or projected_rows > limit_rows:
            raise ConflictError(
                "Autobench owner ledger budget would be exceeded",
                details=usage,
            )
        return usage

    def validate_autobench_idempotency_key(
        self,
        owner_id: str,
        key: str,
        *,
        now_unix_seconds: int | None = None,
    ) -> int:
        """Reject expired or already-compacted Autobench submission keys."""
        match = _AUTOBENCH_KEY.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            raise ConflictError("invalid Autobench idempotency key")
        expiry = int(match.group(1))
        now_seconds = int(datetime.now(timezone.utc).timestamp()) if now_unix_seconds is None else now_unix_seconds
        if isinstance(now_seconds, bool) or not isinstance(now_seconds, int):
            raise ValueError("now_unix_seconds must be an integer")
        floor = Ledger._autobench_floor(self._connection, owner_id)
        if expiry <= now_seconds or expiry <= floor:
            raise ConflictError("Autobench idempotency key has expired")
        return expiry


class Ledger:
    """A process-safe, revisioned entity ledger backed by one SQLite database."""

    api_version = "1"

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_seconds: float = 30.0):
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout_seconds = busy_timeout_seconds
        self._schema_lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
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

    def storage_usage(self) -> dict[str, int]:
        """Return physical SQLite size and live logical payload bytes.

        SQLite reuses free pages after compaction; this method deliberately
        reports both logical payload size and on-disk database/WAL size so a
        caller cannot mistake logical compaction for a hard physical quota.
        """
        with contextlib.closing(self._connect()) as conn:
            page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            logical = 0
            for table, columns in {
                "entities": ("data_json",),
                "events": ("payload_json",),
                "operations": ("request_json", "response_json"),
                "operation_receipts": ("receipt_json",),
                "outbox": ("payload_json",),
            }.items():
                expression = " + ".join(f"COALESCE(SUM(length({column})),0)" for column in columns)
                logical += int(conn.execute(f"SELECT {expression} FROM {table}").fetchone()[0])
        wal_path = Path(str(self.path) + "-wal")
        shm_path = Path(str(self.path) + "-shm")
        return {
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            "shm_bytes": shm_path.stat().st_size if shm_path.exists() else 0,
            "logical_payload_bytes": logical,
            "allocated_page_bytes": page_count * page_size,
        }

    def autobench_owner_usage(self, owner_id: str) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as conn:
            return _autobench_owner_usage(conn, owner_id)

    def validate_autobench_idempotency_key(self, owner_id: str, key: str, *, now_unix_seconds: int | None = None) -> int:
        with contextlib.closing(self._connect()) as conn:
            return LedgerTransaction(conn).validate_autobench_idempotency_key(
                owner_id, key, now_unix_seconds=now_unix_seconds
            )

    def _reclaim_storage(self) -> dict[str, Any]:
        """Checkpoint WAL and run SQLite VACUUM under exclusive maintenance."""
        with self._maintenance_lock:
            conn = self._connect()
            try:
                checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is not None and int(checkpoint[0]) != 0:
                    return {"physical_reclaim": False, "checkpoint_busy": int(checkpoint[0])}
                conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return {"physical_reclaim": True}
            except sqlite3.OperationalError as exc:
                return {"physical_reclaim": False, "maintenance_error": type(exc).__name__}
            finally:
                conn.close()

    def compact_autobench_history(
        self,
        owner_id: str,
        *,
        before: datetime | str | None = None,
        max_records: int = 1000,
    ) -> dict[str, Any]:
        """Purge expired opted-in history, retain one owner audit aggregate."""
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= 10_000:
            raise ValueError("max_records must be from 1 through 10000")
        cutoff = None
        if before is not None:
            if isinstance(before, str):
                try:
                    cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("before must be an ISO-8601 datetime") from exc
            elif isinstance(before, datetime):
                cutoff = before
            else:
                raise ValueError("before must be an ISO-8601 datetime")
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError("before must be timezone-aware")
            cutoff = cutoff.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        now_epoch = int(now.timestamp())
        physical_before = self.storage_usage()
        with self.transaction() as tx:
            conn = tx._connection
            owner_usage_before = _autobench_owner_usage(conn, owner_id)
            rows = conn.execute("SELECT entity_id,data_json FROM entities WHERE kind='jobs' ORDER BY updated_at,entity_id").fetchall()
            owner_autobench_ids: set[str] = set()
            for row in rows:
                record = json.loads(row["data_json"])
                spec = record.get("spec")
                if (record.get("owner", record.get("owner_id")) == owner_id
                        and isinstance(spec, Mapping) and spec.get("retention_policy") == "autobench"):
                    owner_autobench_ids.add(str(row["entity_id"]))
            jobs: dict[str, dict[str, Any]] = {}
            expiries: dict[str, int] = {}
            for row in rows:
                job_id = str(row["entity_id"])
                record = json.loads(row["data_json"])
                spec = record.get("spec")
                if not isinstance(spec, Mapping) or spec.get("retention_policy") != "autobench":
                    continue
                if record.get("owner", record.get("owner_id")) != owner_id:
                    continue
                if str(record.get("state", "")).lower() not in {"succeeded", "failed", "cancelled"}:
                    continue
                if record.get("cleanup_verified") is not True:
                    continue
                if record.get("lease_binding") in {"supervised", "protected"} or record.get("protected") is True:
                    continue
                if record.get("reservation_id") is not None:
                    continue
                tags = spec.get("tags", ())
                if isinstance(tags, (list, tuple, set, frozenset)) and {str(x).lower() for x in tags} & {"protected", "scheduler.protected"}:
                    continue
                expiry = _autobench_key_expiry(record)
                # before can restrict deletion, never authorize an unexpired key.
                if expiry is None or expiry > now_epoch:
                    continue
                updated = record.get("updated_at") or record.get("created_at")
                if cutoff is not None:
                    if not isinstance(updated, str):
                        continue
                    try:
                        updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if updated_at.tzinfo is None or updated_at.astimezone(timezone.utc) >= cutoff:
                        continue
                lease_id = record.get("lease_id")
                if lease_id is not None:
                    lease_row = conn.execute("SELECT data_json FROM entities WHERE kind='lease' AND entity_id=?", (str(lease_id),)).fetchone()
                    lease = json.loads(lease_row["data_json"]) if lease_row else None
                    internal = isinstance(lease, Mapping) and (
                        lease.get("internal") is True or lease.get("managed_by") == "device-scheduler" or lease.get("lease_type") == "job_internal"
                        or lease.get("request_id") == f"job:{job_id}"
                        or lease.get("idempotency_key") == f"job-lease:{job_id}"
                    )
                    if (not internal or lease.get("owner_id", lease.get("owner")) != owner_id
                        or str(lease.get("state", "")).lower() not in {"released", "expired"}
                        or lease.get("usage") == "supervised" or lease.get("protected") is True
                        or lease.get("reservation_id") is not None):
                        continue
                jobs[job_id] = record
                expiries[job_id] = expiry
                if len(jobs) >= max_records:
                    break

            job_ids = set(jobs)
            attempt_ids_by_job: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
            owner_attempts_by_job: dict[str, set[str]] = {job_id: set() for job_id in owner_autobench_ids}
            for attempt_row in conn.execute("SELECT entity_id,data_json FROM entities WHERE kind='attempts'").fetchall():
                attempt = json.loads(attempt_row["data_json"])
                job_id = attempt.get("job_id")
                if job_id in owner_autobench_ids:
                    attempt_id = str(attempt.get("attempt_id", attempt_row["entity_id"]))
                    owner_attempts_by_job[job_id].add(attempt_id)
                    if job_id in job_ids:
                        attempt_ids_by_job[job_id].add(attempt_id)
            linked: dict[str, list[sqlite3.Row]] = {job_id: [] for job_id in job_ids}
            operation_matches: dict[str, set[str]] = {}
            for op in conn.execute("SELECT * FROM operations WHERE owner=?", (owner_id,)).fetchall():
                request = json.loads(op["request_json"])
                response = json.loads(op["response_json"]) if op["response_json"] else None
                referenced = {
                    job_id for job_id in owner_autobench_ids
                    if (_autobench_mentions_job(request, {job_id} | owner_attempts_by_job[job_id])
                        or _autobench_mentions_job(response, {job_id} | owner_attempts_by_job[job_id]))
                }
                for job_id in tuple(job_ids):
                    spec = jobs[job_id].get("spec", {})
                    key = spec.get("idempotency_key") if isinstance(spec, Mapping) else None
                    attempt_ids = attempt_ids_by_job[job_id]
                    start_operation = any(str(op["operation_id"]).endswith(attempt_id) for attempt_id in attempt_ids)
                    if op["idempotency_key"] == key or start_operation or job_id in referenced:
                        linked[job_id].append(op)
                        operation_matches.setdefault(str(op["operation_id"]), set()).update(referenced | {job_id})
            pending = set()
            outbox_rows = conn.execute("SELECT outbox_id,payload_json,delivered_at FROM outbox").fetchall()
            delivered_outbox: dict[str, set[str]] = {}
            for row in outbox_rows:
                payload = json.loads(row["payload_json"])
                matched = {job_id for job_id in job_ids if _autobench_mentions_job(payload, {job_id} | attempt_ids_by_job[job_id])}
                if matched and row["delivered_at"] is None:
                    pending.update(matched)
                elif matched:
                    delivered_outbox[str(row["outbox_id"])] = matched
            for job_id in tuple(job_ids):
                if job_id in pending or any(op["status"] not in {"complete", "failed"} for op in linked[job_id]):
                    jobs.pop(job_id, None)
                    expiries.pop(job_id, None)
                    linked.pop(job_id, None)
            job_ids = set(jobs)

            if not job_ids:
                result = {"jobs_purged": 0, "attempts_purged": 0, "events_purged": 0,
                          "operations_purged": 0, "receipts_purged": 0, "leases_purged": 0,
                          "logical_payload_bytes_after": _autobench_owner_usage(conn, owner_id)["logical_payload_bytes"],
                          "floor_expiry_unix_seconds": self._autobench_floor(conn, owner_id)}
            else:
                for trigger in ("events_no_update", "events_no_delete", "operation_receipts_no_update", "operation_receipts_no_delete"):
                    conn.execute(f"DROP TRIGGER {trigger}")
                op_ids = {
                    op_id for op_id, matches in operation_matches.items()
                    if matches and matches <= job_ids
                }
                receipts = 0
                if op_ids:
                    marks = ",".join("?" for _ in op_ids)
                    receipts = int(conn.execute(f"SELECT COUNT(*) FROM operation_receipts WHERE operation_id IN ({marks})", tuple(sorted(op_ids))).fetchone()[0])
                    conn.execute(f"DELETE FROM operation_receipts WHERE operation_id IN ({marks})", tuple(sorted(op_ids)))
                    conn.execute(f"DELETE FROM operations WHERE operation_id IN ({marks})", tuple(sorted(op_ids)))
                outbox_purged = 0
                purge_outbox_ids = {
                    outbox_id for outbox_id, matches in delivered_outbox.items()
                    if matches & job_ids
                }
                if purge_outbox_ids:
                    outbox_marks = ",".join("?" for _ in purge_outbox_ids)
                    outbox_purged = conn.execute(
                        f"DELETE FROM outbox WHERE outbox_id IN ({outbox_marks}) AND delivered_at IS NOT NULL",
                        tuple(sorted(purge_outbox_ids)),
                    ).rowcount
                marks = ",".join("?" for _ in job_ids)
                params = tuple(sorted(job_ids))
                attempts = int(conn.execute(f"SELECT COUNT(*) FROM entities WHERE kind='attempts' AND json_extract(data_json,'$.job_id') IN ({marks})", params).fetchone()[0])
                lease_ids_for_events = {
                    str(record["lease_id"]) for record in jobs.values()
                    if isinstance(record.get("lease_id"), str)
                }
                event_entity_ids = job_ids | lease_ids_for_events
                event_marks = ",".join("?" for _ in event_entity_ids)
                event_params = tuple(sorted(event_entity_ids))
                events = int(conn.execute(f"SELECT COUNT(*) FROM events WHERE entity_id IN ({event_marks})", event_params).fetchone()[0])
                conn.execute(f"DELETE FROM entities WHERE kind='attempts' AND json_extract(data_json,'$.job_id') IN ({marks})", params)
                decisions = int(conn.execute(
                    f"SELECT COUNT(*) FROM entities WHERE kind='decisions' AND json_extract(data_json,'$.subject_id') IN ({marks})", params
                ).fetchone()[0])
                conn.execute(f"DELETE FROM entities WHERE kind='decisions' AND json_extract(data_json,'$.subject_id') IN ({marks})", params)
                conn.execute(f"DELETE FROM events WHERE entity_id IN ({event_marks})", event_params)
                conn.execute(f"DELETE FROM entities WHERE kind='jobs' AND entity_id IN ({marks})", params)
                # Only delivered records are purged. Pending rows were used
                # above to reject their associated jobs from this purge.
                leases = 0
                for lease_id in {str(j["lease_id"]) for j in jobs.values() if j.get("lease_id") is not None}:
                    leases += conn.execute("DELETE FROM entities WHERE kind='lease' AND entity_id=?", (lease_id,)).rowcount

                prior = conn.execute("SELECT revision,data_json FROM entities WHERE kind='autobench_retention' AND entity_id=?", (owner_id,)).fetchone()
                summary = json.loads(prior["data_json"]) if prior else {}
                floor = max([int(summary.get("floor_expiry_unix_seconds", 0)), *expiries.values()])
                key_hashes = sorted(hashlib.sha256(str(j.get("spec", {}).get("idempotency_key", "")).encode()).hexdigest() for j in jobs.values())
                material = canonical_json({"owner": owner_id, "job_ids": sorted(job_ids), "key_hashes": key_hashes, "expired_at": now_iso})
                previous_hash = str(summary.get("audit_chain_sha256", "0" * 64))
                chain = hashlib.sha256((previous_hash + material).encode()).hexdigest()
                audit = {"owner_id": owner_id, "floor_expiry_unix_seconds": floor,
                         "jobs_purged_total": int(summary.get("jobs_purged_total", 0)) + len(job_ids),
                         "last_compacted_at": now_iso, "audit_chain_sha256": chain}
                tx.put("autobench_retention", owner_id, audit, int(prior["revision"]) if prior else None)
                for statement in (
                    "CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END",
                    "CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END",
                    "CREATE TRIGGER operation_receipts_no_update BEFORE UPDATE ON operation_receipts BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END",
                    "CREATE TRIGGER operation_receipts_no_delete BEFORE DELETE ON operation_receipts BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END",
                ):
                    conn.execute(statement)
                usage = _autobench_owner_usage(conn, owner_id)
                result = {"jobs_purged": len(job_ids), "attempts_purged": attempts, "events_purged": events,
                          "operations_purged": len(op_ids), "receipts_purged": receipts, "leases_purged": leases,
                          "decisions_purged": decisions, "outbox_purged": outbox_purged,
                          "logical_payload_bytes_after": usage["logical_payload_bytes"],
                          "floor_expiry_unix_seconds": floor, "audit_chain_sha256": chain}

        physical_after = self.storage_usage()
        reclaim = self._reclaim_storage() if result.get("jobs_purged", 0) else {"physical_reclaim": False, "reason": "no_expired_records"}
        physical_final = self.storage_usage()
        result.update({"logical_payload_bytes_before": owner_usage_before["logical_payload_bytes"],
                       "database_bytes_before": physical_before["database_bytes"],
                       "database_bytes_after_compaction": physical_after["database_bytes"],
                       "database_bytes_after_reclaim": physical_final["database_bytes"],
                       "wal_bytes_after_reclaim": physical_final["wal_bytes"], **reclaim})
        return result

    @staticmethod
    def _autobench_floor(connection: sqlite3.Connection, owner_id: str) -> int:
        row = connection.execute("SELECT data_json FROM entities WHERE kind='autobench_retention' AND entity_id=?", (owner_id,)).fetchone()
        return int(json.loads(row["data_json"]).get("floor_expiry_unix_seconds", 0)) if row else 0

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
