"""Shared mapping from core operation names to MCP and CLI spellings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


BOOTSTRAP_OPERATIONS = (
    "resources.list", "resources.capabilities",
    "leases.request", "leases.get", "leases.list", "leases.renew", "leases.release",
    "jobs.submit", "jobs.get", "jobs.list", "jobs.wait", "jobs.logs", "jobs.control",
    "backends.register", "backends.get", "backends.list", "backends.control", "backends.attach",
    "handoffs.submit", "handoffs.get", "handoffs.list", "handoffs.control",
    "schedules.put", "schedules.get", "schedules.list", "schedules.control",
    "events.publish", "policies.explain", "policies.simulate",
    "admin.status", "admin.reconcile", "admin.backup",
)

READ_ONLY_OPERATIONS = frozenset({
    "resources.list", "resources.capabilities",
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
        result.append(Operation(name, tool_name, name in READ_ONLY_OPERATIONS,
                                DESCRIPTIONS.get(name, f"Dispatch the {name} scheduler operation.")))
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
