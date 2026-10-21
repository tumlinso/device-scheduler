"""Stable domain errors shared by the scheduler packages."""

from __future__ import annotations

from typing import Any


class SchedulerError(Exception):
    """Base error with a stable machine-readable code and optional details."""

    code = "scheduler_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(SchedulerError, ValueError):
    code = "validation_error"


class ConflictError(SchedulerError):
    code = "conflict"


class NotFoundError(SchedulerError, LookupError):
    code = "not_found"


class PermissionDeniedError(SchedulerError):
    code = "permission_denied"


class UnsupportedCapabilityError(SchedulerError):
    code = "unsupported_capability"


class IdempotencyConflictError(ConflictError):
    code = "idempotency_conflict"


class ReconciliationRequiredError(SchedulerError):
    code = "reconciliation_required"

