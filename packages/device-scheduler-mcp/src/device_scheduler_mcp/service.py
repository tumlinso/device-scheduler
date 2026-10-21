"""Transport-neutral client-facing dispatch boundary."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from device_scheduler_core.errors import SchedulerError
from device_scheduler_core.models import Principal

from .registry import operation_names


class SchedulerDispatch(Protocol):
    def dispatch(self, operation: str, args: dict[str, Any], principal: Principal) -> dict[str, Any]: ...
    def tick(self) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...


def json_value(value: Any) -> Any:
    """Normalize core DTOs and time values for structured MCP output."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_value(value.to_dict())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_value(child) for child in value]
    return value


class SchedulerService:
    """Serialize dispatch and background ticks around one scheduler instance."""

    def __init__(self, scheduler: SchedulerDispatch) -> None:
        if not all(callable(getattr(scheduler, name, None)) for name in ("dispatch", "tick", "status")):
            raise TypeError("scheduler must implement dispatch, tick, and status")
        self.scheduler = scheduler
        # Mutating client calls serialize with each other. Periodic ticks use a
        # separate lock: resource probes can take several seconds, and holding
        # the client lock during one can starve an otherwise fast submission.
        self._dispatch_lock = threading.RLock()
        self._tick_lock = threading.Lock()

    def dispatch(self, operation: str, args: Mapping[str, Any], principal: Principal) -> dict[str, Any]:
        if operation not in operation_names():
            return {"error": {"code": "unknown_operation", "message": "operation is not available", "details": {}}}
        if not isinstance(args, Mapping):
            return {"error": {"code": "validation_error", "message": "arguments must be an object", "details": {}}}
        try:
            # The core deliberately keeps jobs.wait outside its scheduler lock
            # while polling. Let ticks make progress so a wait does not block
            # the completion it is waiting for.
            if operation == "jobs.wait":
                result = self.scheduler.dispatch(operation, dict(args), principal)
            else:
                with self._dispatch_lock:
                    result = self.scheduler.dispatch(operation, dict(args), principal)
        except SchedulerError as exc:
            return {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": json_value(exc.details),
                }
            }
        except Exception:
            # Adapter exception messages may contain command arguments or environment data.
            return {"error": {"code": "internal_error", "message": "scheduler operation failed", "details": {}}}
        try:
            normalized = json_value(result)
            if not isinstance(normalized, dict):
                return {"error": {"code": "invalid_core_response", "message": "scheduler returned a non-object result", "details": {}}}
            # Fail closed if an adapter or policy accidentally returns non-JSON data.
            json.dumps(normalized, allow_nan=False, separators=(",", ":"))
            return normalized
        except (TypeError, ValueError, OverflowError):
            return {"error": {"code": "invalid_core_response", "message": "scheduler response is not valid JSON", "details": {}}}

    def tick(self) -> dict[str, Any]:
        with self._tick_lock:
            result = self.scheduler.tick()
        normalized = json_value(result)
        return normalized if isinstance(normalized, dict) else {"status": "unknown"}

    def status(self) -> dict[str, Any]:
        with self._dispatch_lock:
            result = self.scheduler.status()
        normalized = json_value(result)
        return normalized if isinstance(normalized, dict) else {"status": "unknown"}
