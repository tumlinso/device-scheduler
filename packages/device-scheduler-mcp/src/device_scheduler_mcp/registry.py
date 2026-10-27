"""Shared mapping from core operation names to MCP and CLI spellings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints
import types

from device_scheduler_core.models import BackendSpec, HandoffSpec, JobSpec, ResourceRequest, ScheduleSpec


BOOTSTRAP_OPERATIONS = (
    "resources.list", "resources.capabilities", "access.inspect",
    "leases.request", "leases.get", "leases.list", "leases.renew", "leases.release",
    "jobs.submit", "jobs.get", "jobs.list", "jobs.wait", "jobs.logs", "jobs.control",
    "backends.register", "backends.get", "backends.list", "backends.control", "backends.attach",
    "handoffs.submit", "handoffs.get", "handoffs.list", "handoffs.control",
    "schedules.put", "schedules.get", "schedules.list", "schedules.control",
    "events.publish", "policies.explain", "policies.simulate",
    "admin.status", "admin.reconcile", "admin.backup",
)

READ_ONLY_OPERATIONS = frozenset({
    "resources.list", "resources.capabilities", "access.inspect",
    "leases.get", "leases.list",
    "jobs.get", "jobs.list", "jobs.wait", "jobs.logs",
    "backends.get", "backends.list",
    "handoffs.get", "handoffs.list",
    "schedules.get", "schedules.list",
    "policies.explain", "policies.simulate", "admin.status",
})

DESCRIPTIONS = {
    "resources.list": "List current host resources and their observed capabilities.",
    "resources.capabilities": "Describe the resource and adapter capabilities available to this scheduler.",
    "access.inspect": "Show this caller's effective scheduler access and resource scope.",
    "leases.request": "Request an atomic resource lease using the caller's granted scope.",
    "leases.get": "Inspect one resource lease and its reclamation state.",
    "leases.list": "List leases visible to the authenticated principal.",
    "leases.renew": "Renew an owned lease within its grant and policy limits.",
    "leases.release": "Request release and verified reclamation of an owned lease.",
    "jobs.submit": "Submit a durable execution job.",
    "jobs.get": "Inspect a job and its attempts, results, and cleanup state.",
    "jobs.list": "List jobs visible to the authenticated principal.",
    "jobs.wait": "Wait for a bounded period for a job state change; timing out does not cancel it.",
    "jobs.logs": "Read bounded logs or artifact metadata for a job attempt.",
    "jobs.control": "Request an authorized job lifecycle action.",
    "backends.register": "Register a managed backend profile under administrative authority.",
    "backends.get": "Inspect one managed backend and its verified residency state.",
    "backends.list": "List managed backends visible to the authenticated principal.",
    "backends.control": "Request an authorized managed backend lifecycle action.",
    "backends.attach": "Attach a consumer to a managed backend under its eviction contract.",
    "handoffs.submit": "Submit an explicit finite request-suspend-evict-run-restore-resume handoff.",
    "handoffs.get": "Inspect a handoff, its durable phase, results, and restoration state.",
    "handoffs.list": "List handoffs visible to the authenticated principal.",
    "handoffs.control": "Request cancellation or recovery of an authorized handoff.",
    "schedules.put": "Create or revise a durable delayed or recurring schedule.",
    "schedules.get": "Inspect one durable schedule and its occurrence state.",
    "schedules.list": "List schedules visible to the authenticated principal.",
    "schedules.control": "Pause, resume, or cancel an authorized schedule.",
    "events.publish": "Publish an authenticated, deduplicated scheduler event.",
    "policies.explain": "Explain a recorded or current scheduling decision.",
    "policies.simulate": "Replay a policy against a recorded observation without side effects.",
    "admin.status": "Inspect scheduler health and verified adapter state.",
    "admin.reconcile": "Inspect and reconcile operations within administrative authority.",
    "admin.backup": "Create or inspect a consistent scheduler ledger backup.",
}


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    tool_name: str
    read_only: bool
    description: str
    input_schema: dict[str, Any]


_OBJECT = {"type": "object", "additionalProperties": True}
_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_NUMBER = {"type": "number"}


def _schema(fields: Mapping[str, dict[str, Any]] | None = None, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(fields or {}),
        "required": list(required),
        "additionalProperties": False,
    }


# These are the public operation contracts. Nested core DTOs remain open JSON
# objects because the core validates their authoritative, versioned shapes.
_OPERATION_FIELDS: dict[str, tuple[dict[str, dict[str, Any]], tuple[str, ...]]] = {
    "resources.list": ({}, ()), "resources.capabilities": ({}, ()), "access.inspect": ({}, ()),
    "leases.request": ({"request": _OBJECT, "idempotency_key": _STRING}, ("request",)),
    "leases.get": ({"lease_id": _STRING}, ("lease_id",)),
    "leases.list": ({"state": _STRING}, ()),
    "leases.renew": ({"lease_id": _STRING, "fencing_token": _INTEGER, "ttl_seconds": _NUMBER}, ("lease_id", "fencing_token", "ttl_seconds")),
    "leases.release": ({"lease_id": _STRING, "fencing_token": _INTEGER}, ("lease_id", "fencing_token")),
    "jobs.submit": ({"spec": _OBJECT, "lease_id": _STRING, "fencing_token": _INTEGER, "idempotency_key": _STRING}, ("spec",)),
    "jobs.get": ({"job_id": _STRING}, ("job_id",)),
    "jobs.list": ({}, ()),
    "jobs.wait": ({"job_id": _STRING, "timeout_seconds": _NUMBER, "terminal_states": {"type": "array", "items": _STRING}}, ("job_id",)),
    "jobs.logs": ({"job_id": _STRING, "attempt_id": _STRING, "max_bytes": _INTEGER}, ("job_id",)),
    "jobs.control": ({"job_id": _STRING, "action": _STRING, "reason": _STRING}, ("job_id", "action")),
    "backends.register": ({"spec": _OBJECT}, ("spec",)),
    "backends.get": ({"backend_id": _STRING}, ("backend_id",)), "backends.list": ({}, ()),
    "backends.control": ({"backend_id": _STRING, "action": _STRING}, ("backend_id", "action")),
    "backends.attach": ({"backend_id": _STRING, "continuation_ref": _STRING, "contract_id": _STRING, "idempotency_key": _STRING}, ("backend_id", "continuation_ref", "contract_id", "idempotency_key")),
    "handoffs.submit": ({"spec": _OBJECT}, ("spec",)), "handoffs.get": ({"handoff_id": _STRING}, ("handoff_id",)),
    "handoffs.list": ({}, ()), "handoffs.control": ({"handoff_id": _STRING, "action": _STRING}, ("handoff_id", "action")),
    "schedules.put": ({"spec": _OBJECT}, ("spec",)), "schedules.get": ({"schedule_id": _STRING}, ("schedule_id",)),
    "schedules.list": ({}, ()), "schedules.control": ({"schedule_id": _STRING, "action": _STRING}, ("schedule_id", "action")),
    "events.publish": ({"source": _STRING, "event_id": _STRING, "payload": _OBJECT, "idempotency_key": _STRING}, ("source", "event_id", "payload")),
    "policies.explain": ({"decision_id": _STRING, "job_id": _STRING, "request": _OBJECT, "observations": {"type": "array", "items": _OBJECT}, "max_observation_age_seconds": _NUMBER}, ()),
    "policies.simulate": ({"observations": {"type": "array", "items": _OBJECT}, "operation": _STRING, "service_by_owner": _OBJECT}, ("observations",)),
    "admin.status": ({}, ()), "admin.reconcile": ({}, ()),
    "admin.backup": ({"destination": _STRING}, ()),
}

_NESTED_MODELS: dict[str, dict[str, type[Any]]] = {
    "leases.request": {"request": ResourceRequest},
    "jobs.submit": {"spec": JobSpec},
    "backends.register": {"spec": BackendSpec},
    "handoffs.submit": {"spec": HandoffSpec},
    "schedules.put": {"spec": ScheduleSpec},
}


def _type_schema(annotation: Any) -> dict[str, Any]:
    """Describe the core DTO's Python type contract without restating its fields."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (Union, types.UnionType):
        choices = [_type_schema(item) for item in args]
        return {"anyOf": choices}
    if origin in (list, tuple, set, frozenset):
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            return {"type": "array", "prefixItems": [_type_schema(item) for item in args], "minItems": len(args), "maxItems": len(args)}
        child = args[0] if args else Any
        return {"type": "array", "items": _type_schema(child)}
    if origin is not None and isinstance(origin, type) and issubclass(origin, Mapping):
        value_type = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": _type_schema(value_type)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_schema(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is datetime:
        return {"type": "string", "format": "date-time"}
    if annotation is Path:
        return {"type": "string"}
    return {}


def _default_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, (tuple, set, frozenset)):
        return [_default_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _default_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _dataclass_schema(model: type[Any]) -> dict[str, Any]:
    hints = get_type_hints(model)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in fields(model):
        if not item.init:
            continue
        property_schema = _type_schema(hints.get(item.name, Any))
        if item.default is not MISSING:
            default = _default_value(item.default)
            property_schema["default"] = default
        elif item.default_factory is not MISSING:
            default = _default_value(item.default_factory())
            property_schema["default"] = default
        else:
            required.append(item.name)
        properties[item.name] = property_schema
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def operation_names() -> tuple[str, ...]:
    """Use the core's canonical list, with the bootstrap list for partial imports."""
    try:
        from device_scheduler_core.application import OPERATION_NAMES
    except ModuleNotFoundError as exc:
        if exc.name == "device_scheduler_core.application":
            return BOOTSTRAP_OPERATIONS
        raise
    values = tuple(OPERATION_NAMES.keys()) if isinstance(OPERATION_NAMES, Mapping) else tuple(OPERATION_NAMES)
    if not values or any(not isinstance(value, str) or "." not in value for value in values):
        raise RuntimeError("core OPERATION_NAMES must contain canonical dot-separated names")
    if len(set(values)) != len(values):
        raise RuntimeError("core OPERATION_NAMES contains duplicates")
    return values


def operations() -> tuple[Operation, ...]:
    result: list[Operation] = []
    tools: set[str] = set()
    for name in operation_names():
        tool_name = name.replace(".", "_")
        if tool_name in tools:
            raise RuntimeError("core operations collide after MCP tool-name mapping")
        tools.add(tool_name)
        if name not in _OPERATION_FIELDS:
            raise RuntimeError(f"missing MCP argument schema for canonical operation {name}")
        fields, required = _OPERATION_FIELDS[name]
        field_schemas = dict(fields)
        for field_name, model in _NESTED_MODELS.get(name, {}).items():
            field_schemas[field_name] = _dataclass_schema(model)
        result.append(Operation(
            name,
            tool_name,
            name in READ_ONLY_OPERATIONS,
            DESCRIPTIONS.get(name, f"Dispatch the {name} scheduler operation."),
            _schema(field_schemas, required),
        ))
    return tuple(result)


def operation_for_tool(tool_name: str) -> str:
    for operation in operations():
        if operation.tool_name == tool_name:
            return operation.name
    raise ValueError(f"unknown scheduler MCP operation: {tool_name}")


def tool_for_operation(name: str) -> str:
    if name not in operation_names():
        raise ValueError(f"unknown scheduler operation: {name}")
    return name.replace(".", "_")
