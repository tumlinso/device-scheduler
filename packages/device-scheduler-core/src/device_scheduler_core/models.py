"""Environment-neutral, validated and JSON-friendly scheduler contracts."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import ValidationError
from .validation import (
    aware_datetime,
    finite_number,
    identifier,
    json_mapping,
    owner_context_mapping,
    quantity_mapping,
)

T = TypeVar("T", bound="JsonModel")
GPU_EXCLUSIVE_TAG = "gpu.exclusive"


class JsonModel:
    """Mixin that converts typed DTOs to and from ordinary JSON-shaped data."""

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        if not isinstance(data, Mapping):
            raise ValidationError(f"{cls.__name__} input must be an object")
        if not is_dataclass(cls):
            raise TypeError("JsonModel subclasses must be dataclasses")
        known = {item.name for item in fields(cls) if item.init}
        unknown = set(data) - known
        if unknown:
            raise ValidationError(f"unknown {cls.__name__} fields: {', '.join(sorted(unknown))}")
        hints = get_type_hints(cls)
        values: dict[str, Any] = {}
        for item in fields(cls):
            if not item.init or item.name not in data:
                continue
            values[item.name] = _coerce(hints.get(item.name, Any), data[item.name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("JsonModel subclasses must be dataclasses")
        return {item.name: _to_json(getattr(self, item.name)) for item in fields(self) if item.init}


def _coerce(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        failures: list[Exception] = []
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(candidate, value)
            except (TypeError, ValueError, ValidationError) as exc:
                failures.append(exc)
        if failures:
            raise failures[-1]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValidationError("expected an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], child) for child in value)
        if args and len(value) != len(args):
            raise ValidationError("array has incorrect length")
        return tuple(_coerce(kind, child) for kind, child in zip(args, value)) if args else tuple(value)
    if origin is frozenset:
        if not isinstance(value, (list, tuple, set, frozenset)):
            raise ValidationError("expected an array")
        subtype = args[0] if args else Any
        return frozenset(_coerce(subtype, child) for child in value)
    if origin is not None and issubclass(origin, Mapping):
        if not isinstance(value, Mapping):
            raise ValidationError("expected an object")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce(key_type, key): _coerce(value_type, child)
            for key, child in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            raise ValidationError(f"invalid {annotation.__name__} value: {value!r}") from exc
    if isinstance(annotation, type) and issubclass(annotation, JsonModel):
        if isinstance(value, annotation):
            return value
        return annotation.from_dict(value)
    if annotation is datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValidationError("invalid ISO-8601 datetime") from exc
    if annotation is Path:
        return Path(value)
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValidationError("expected a boolean")
        return value
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation in (str, int, float):
        if annotation is int and isinstance(value, bool):
            raise ValidationError("expected an integer")
        if not isinstance(value, annotation):
            raise ValidationError(f"expected {annotation.__name__}")
        return value
    return value


def _to_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, JsonModel):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_json(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_to_json(child) for child in value]
    return value


class AllocationMode(str, Enum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"


class ExecutionPurpose(str, Enum):
    JOB = "job"
    BACKEND = "backend"


class LeaseState(str, Enum):
    ACTIVE = "active"
    RELEASE_REQUESTED = "release_requested"
    RECLAIMING = "reclaiming"
    UNKNOWN = "unknown"
    RELEASED = "released"
    EXPIRED = "expired"


class LeaseUsage(str, Enum):
    """How a lease may be consumed by a client.

    ``external`` preserves the historical reservation contract and requires a
    resource adapter that can independently verify reclamation. ``supervised``
    leases are usable only by jobs launched and cleaned up by Device Scheduler.
    """

    EXTERNAL = "external"
    SUPERVISED = "supervised"


class JobState(str, Enum):
    QUEUED = "queued"
    BLOCKED = "blocked"
    ADMITTED = "admitted"
    RUNNING = "running"
    SUSPEND_REQUESTED = "suspend_requested"
    SUSPENDED = "suspended"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AttemptState(str, Enum):
    PREPARING = "preparing"
    STARTING = "starting"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COLLECTING = "collecting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class BackendState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    EVICTING = "evicting"
    RESTORING = "restoring"
    FAILED = "failed"
    UNKNOWN = "unknown"


class HandoffState(str, Enum):
    REQUESTED = "requested"
    RESERVED = "reserved"
    DRAINING = "draining"
    EVICTING = "evicting"
    RUNNING = "running"
    CLEANING = "cleaning"
    RESTORING = "restoring"
    DELIVERING = "delivering"
    RESUMING = "resuming"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ScheduleState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class OverlapPolicy(str, Enum):
    FORBID = "forbid"
    ALLOW = "allow"
    QUEUE = "queue"


class MisfirePolicy(str, Enum):
    COALESCE = "coalesce"
    SKIP = "skip"
    CATCH_UP = "catch_up"


class PriorityClass(str, Enum):
    INTERACTIVE = "interactive"
    NORMAL = "normal"
    BACKGROUND = "background"


class EnforcementLevel(str, Enum):
    ENFORCED = "enforced"
    ADVISORY = "advisory"
    OBSERVATIONAL = "observational"


@dataclass(frozen=True, slots=True)
class Resource(JsonModel):
    resource_id: str
    host_id: str
    kind: str
    capacities: Mapping[str, float]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    topology: Mapping[str, Any] = field(default_factory=dict)
    conflict_domains: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.resource_id, "resource_id")
        identifier(self.host_id, "host_id")
        identifier(self.kind, "kind")
        # A discovered but currently unavailable resource can truthfully report
        # zero capacity; requested quantities must still be strictly positive.
        object.__setattr__(
            self,
            "capacities",
            quantity_mapping(self.capacities, "capacities", allow_empty=True, allow_zero=True),
        )
        object.__setattr__(self, "attributes", json_mapping(self.attributes, "attributes"))
        object.__setattr__(self, "topology", json_mapping(self.topology, "topology"))
        object.__setattr__(self, "conflict_domains", frozenset(_strings(self.conflict_domains, "conflict_domains", allow_unordered=True)))
        object.__setattr__(self, "capabilities", frozenset(_strings(self.capabilities, "capabilities", allow_unordered=True)))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class AdapterCapability(JsonModel):
    """A truthful declaration of an adapter operation or guarantee."""

    name: str
    supported: bool
    enforcement: EnforcementLevel | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.name, "name")
        if not isinstance(self.supported, bool):
            raise ValidationError("supported must be a boolean")
        if self.enforcement is not None and not isinstance(self.enforcement, EnforcementLevel):
            object.__setattr__(self, "enforcement", _enum(self.enforcement, EnforcementLevel, "enforcement"))
        if self.supported and self.enforcement is None:
            raise ValidationError("supported capabilities must state their enforcement level")
        if not self.supported and self.enforcement is not None:
            raise ValidationError("unsupported capabilities cannot claim an enforcement level")
        object.__setattr__(self, "details", json_mapping(self.details, "details"))


@dataclass(frozen=True, slots=True)
class CapabilityPrecondition(JsonModel):
    """Generic adapter guarantee that must hold immediately before execution.

    Names and parameter schemas belong to adapters. The core only validates
    the versioned JSON shape and requires the adapter to report an explicit
    refusal when a requested condition is unsupported or cannot be observed.
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.name, "name")
        object.__setattr__(self, "parameters", json_mapping(self.parameters, "parameters"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ResourceRequirement(JsonModel):
    kind: str
    quantities: Mapping[str, float]
    selectors: Mapping[str, Any] = field(default_factory=dict)
    mode: AllocationMode = AllocationMode.SHARED
    same_host: bool = True
    topology_constraints: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.kind, "kind")
        object.__setattr__(self, "quantities", quantity_mapping(self.quantities, "quantities"))
        object.__setattr__(self, "selectors", json_mapping(self.selectors, "selectors"))
        object.__setattr__(self, "topology_constraints", json_mapping(self.topology_constraints, "topology_constraints"))
        if not isinstance(self.mode, AllocationMode):
            object.__setattr__(self, "mode", _enum(self.mode, AllocationMode, "mode"))
        if not isinstance(self.same_host, bool):
            raise ValidationError("same_host must be a boolean")
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ResourceRequest(JsonModel):
    """Atomic request plus optional descriptive external-owner identity.

    ``owner_context`` contains flat descriptive strings such as issuer, project,
    attempt, task, owner kind, or a canonical project-root path. It contains no
    secrets and grants no authority; authorization derives exclusively from the
    authenticated owner.
    """

    request_id: str
    owner_id: str
    requirements: tuple[ResourceRequirement, ...]
    lease_ttl_seconds: float
    idempotency_key: str | None = None
    schema_version: int = 1
    owner_context: Mapping[str, str] = field(default_factory=dict)
    usage: LeaseUsage = LeaseUsage.EXTERNAL

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        identifier(self.owner_id, "owner_id")
        requirements = _models(self.requirements, ResourceRequirement, "requirements", nonempty=True)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "lease_ttl_seconds", finite_number(self.lease_ttl_seconds, "lease_ttl_seconds", strict=True))
        if self.idempotency_key is not None:
            identifier(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "owner_context", owner_context_mapping(self.owner_context))
        if not isinstance(self.usage, LeaseUsage):
            object.__setattr__(self, "usage", _enum(self.usage, LeaseUsage, "usage"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ResourceAllocation(JsonModel):
    resource_id: str
    quantities: Mapping[str, float]
    mode: AllocationMode = AllocationMode.SHARED

    def __post_init__(self) -> None:
        identifier(self.resource_id, "resource_id")
        object.__setattr__(self, "quantities", quantity_mapping(self.quantities, "quantities"))
        if not isinstance(self.mode, AllocationMode):
            object.__setattr__(self, "mode", _enum(self.mode, AllocationMode, "mode"))


@dataclass(frozen=True, slots=True)
class ExecutionContext(JsonModel):
    """The physical grant and descriptors bound to one supervised launch.

    Allocation requests describe desired resources; this context records the
    resources actually granted by the scheduler. Empty allocations are valid
    for CPU-only work or a launch without a scheduler lease. The context is
    descriptive input for adapters and never grants authority by itself.
    """

    allocations: tuple[ResourceAllocation, ...] = ()
    resources: tuple[Resource, ...] = ()
    claim_id: str | None = None
    fencing_token: int | None = None
    purpose: ExecutionPurpose = ExecutionPurpose.JOB
    schema_version: int = 1

    def __post_init__(self) -> None:
        allocations = _models(self.allocations, ResourceAllocation, "allocations")
        resources = _models(self.resources, Resource, "resources")
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "resources", resources)
        allocation_ids = [item.resource_id for item in allocations]
        resource_ids = [item.resource_id for item in resources]
        if len(set(allocation_ids)) != len(allocation_ids):
            raise ValidationError("execution allocations must not repeat a resource")
        if len(set(resource_ids)) != len(resource_ids):
            raise ValidationError("execution resource descriptors must not repeat a resource")
        if set(allocation_ids) != set(resource_ids):
            raise ValidationError("execution allocations must match resource descriptors by identity")
        if self.claim_id is not None:
            identifier(self.claim_id, "claim_id")
        if self.fencing_token is not None:
            if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token <= 0:
                raise ValidationError("fencing_token must be a positive integer when supplied")
            if self.claim_id is None:
                raise ValidationError("fencing_token requires a claim_id")
        if not isinstance(self.purpose, ExecutionPurpose):
            object.__setattr__(self, "purpose", _enum(self.purpose, ExecutionPurpose, "purpose"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class Lease(JsonModel):
    """Granted allocation with optional descriptive external-owner identity.

    Context is copied from the authenticated request and is never evidence of
    permission, acceptance, or resource reclamation.
    """

    lease_id: str
    owner_id: str
    allocations: tuple[ResourceAllocation, ...]
    fencing_token: int
    expires_at: datetime
    state: LeaseState = LeaseState.ACTIVE
    request_digest: str | None = None
    schema_version: int = 1
    owner_context: Mapping[str, str] = field(default_factory=dict)
    usage: LeaseUsage = LeaseUsage.EXTERNAL

    def __post_init__(self) -> None:
        identifier(self.lease_id, "lease_id")
        identifier(self.owner_id, "owner_id")
        allocations = _models(self.allocations, ResourceAllocation, "allocations", nonempty=True)
        object.__setattr__(self, "allocations", allocations)
        if len({item.resource_id for item in self.allocations}) != len(self.allocations):
            raise ValidationError("lease allocations must not repeat a resource")
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token <= 0:
            raise ValidationError("fencing_token must be a positive integer")
        object.__setattr__(self, "expires_at", aware_datetime(self.expires_at, "expires_at", required=True))
        if not isinstance(self.state, LeaseState):
            object.__setattr__(self, "state", _enum(self.state, LeaseState, "state"))
        if self.request_digest is not None:
            identifier(self.request_digest, "request_digest")
        object.__setattr__(self, "owner_context", owner_context_mapping(self.owner_context))
        if not isinstance(self.usage, LeaseUsage):
            object.__setattr__(self, "usage", _enum(self.usage, LeaseUsage, "usage"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class InterruptionContract(JsonModel):
    cooperative_yield: bool = False
    checkpoint_supported: bool = False
    restart_safe: bool = False
    cancellation_supported: bool = True

    def __post_init__(self) -> None:
        for name in ("cooperative_yield", "checkpoint_supported", "restart_safe", "cancellation_supported"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class EventTrigger(JsonModel):
    source: str
    event_type: str
    event_key: str | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.source, "source")
        identifier(self.event_type, "event_type")
        if self.event_key is not None:
            identifier(self.event_key, "event_key")
        object.__setattr__(self, "filters", json_mapping(self.filters, "filters"))


@dataclass(frozen=True, slots=True)
class JobSpec(JsonModel):
    owner_id: str
    argv: tuple[str, ...] | None = None
    profile_id: str | None = None
    working_directory: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    resources: tuple[ResourceRequirement, ...] = ()
    tags: frozenset[str] = frozenset()
    priority_class: str = "normal"
    priority: int = 50
    timeout_seconds: float = 3600.0
    max_attempts: int = 1
    not_before: datetime | None = None
    deadline: datetime | None = None
    interruption: InterruptionContract = InterruptionContract()
    dependencies: tuple[str, ...] = ()
    event_trigger: EventTrigger | None = None
    idempotency_key: str | None = None
    secret_refs: tuple[str, ...] = ()
    schema_version: int = 1
    preconditions: tuple[CapabilityPrecondition, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.owner_id, "owner_id")
        if (self.argv is None) == (self.profile_id is None):
            raise ValidationError("exactly one of argv or profile_id is required")
        if self.argv is not None:
            argv = _strings(self.argv, "argv", nonempty=True)
            object.__setattr__(self, "argv", argv)
            if not argv[0].strip() or any("\x00" in arg for arg in argv):
                raise ValidationError("argv must contain non-empty executable and valid string arguments")
            if (
                not isinstance(self.working_directory, str)
                or not self.working_directory.strip()
                or "\x00" in self.working_directory
            ):
                raise ValidationError("working_directory is required for argv jobs")
        if self.profile_id is not None:
            identifier(self.profile_id, "profile_id")
        if self.working_directory is not None and not self.working_directory.strip():
            raise ValidationError("working_directory must not be empty")
        if self.environment is None:
            object.__setattr__(self, "environment", {})
        if not isinstance(self.environment, Mapping):
            raise ValidationError("environment must be an object")
        for key, value in self.environment.items():
            identifier(key, "environment key")
            if not isinstance(value, str) or "\x00" in value:
                raise ValidationError("environment values must be strings without NUL")
        object.__setattr__(self, "environment", json_mapping(self.environment, "environment"))
        resources = _models(self.resources, ResourceRequirement, "resources")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "tags", frozenset(_strings(self.tags, "tags", allow_unordered=True)))
        if not isinstance(self.priority_class, str) or self.priority_class not in {item.value for item in PriorityClass}:
            raise ValidationError("priority_class must be interactive, normal, or background")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or not 0 <= self.priority <= 100:
            raise ValidationError("priority must be an integer from 0 through 100")
        object.__setattr__(self, "timeout_seconds", finite_number(self.timeout_seconds, "timeout_seconds", strict=True))
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValidationError("max_attempts must be a positive integer")
        object.__setattr__(self, "not_before", aware_datetime(self.not_before, "not_before"))
        object.__setattr__(self, "deadline", aware_datetime(self.deadline, "deadline"))
        if self.not_before and self.deadline and self.deadline <= self.not_before:
            raise ValidationError("deadline must be later than not_before")
        if not isinstance(self.interruption, InterruptionContract):
            object.__setattr__(self, "interruption", InterruptionContract.from_dict(self.interruption))
        if self.max_attempts > 1 and not self.interruption.restart_safe:
            raise ValidationError("max_attempts greater than 1 requires restart_safe interruption contract")
        dependencies = _strings(self.dependencies, "dependencies")
        object.__setattr__(self, "dependencies", dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise ValidationError("dependencies must be unique")
        if self.idempotency_key is not None:
            identifier(self.idempotency_key, "idempotency_key")
        if self.event_trigger is not None and not isinstance(self.event_trigger, EventTrigger):
            object.__setattr__(self, "event_trigger", EventTrigger.from_dict(self.event_trigger))
        if self.event_trigger is not None and not isinstance(self.event_trigger, EventTrigger):
            raise ValidationError("event_trigger must be an EventTrigger")
        object.__setattr__(self, "secret_refs", _strings(self.secret_refs, "secret_refs"))
        preconditions = _models(self.preconditions, CapabilityPrecondition, "preconditions")
        names = [item.name for item in preconditions]
        if len(set(names)) != len(names):
            raise ValidationError("preconditions must have unique capability names")
        object.__setattr__(self, "preconditions", preconditions)
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class Job(JsonModel):
    job_id: str
    spec: JobSpec
    state: JobState = JobState.QUEUED
    created_at: datetime | None = None
    updated_at: datetime | None = None
    current_attempt_id: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.job_id, "job_id")
        if not isinstance(self.spec, JobSpec):
            object.__setattr__(self, "spec", JobSpec.from_dict(self.spec))
        for name in ("created_at", "updated_at"):
            object.__setattr__(self, name, aware_datetime(getattr(self, name), name))
        if self.current_attempt_id is not None:
            identifier(self.current_attempt_id, "current_attempt_id")
        if not isinstance(self.state, JobState):
            object.__setattr__(self, "state", _enum(self.state, JobState, "state"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ExecutionHandle(JsonModel):
    adapter_id: str
    external_id: str
    identity: str
    started_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.adapter_id, "adapter_id")
        identifier(self.external_id, "external_id")
        identifier(self.identity, "identity")
        object.__setattr__(self, "started_at", aware_datetime(self.started_at, "started_at"))
        object.__setattr__(self, "metadata", json_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class Attempt(JsonModel):
    attempt_id: str
    job_id: str
    state: AttemptState = AttemptState.PREPARING
    execution_handle: ExecutionHandle | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] = field(default_factory=dict)
    cleanup_verified: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.attempt_id, "attempt_id")
        identifier(self.job_id, "job_id")
        if not isinstance(self.state, AttemptState):
            object.__setattr__(self, "state", _enum(self.state, AttemptState, "state"))
        if self.execution_handle is not None and not isinstance(self.execution_handle, ExecutionHandle):
            object.__setattr__(self, "execution_handle", ExecutionHandle.from_dict(self.execution_handle))
        for name in ("started_at", "finished_at"):
            object.__setattr__(self, name, aware_datetime(getattr(self, name), name))
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValidationError("finished_at cannot precede started_at")
        object.__setattr__(self, "result", json_mapping(self.result, "result"))
        object.__setattr__(self, "error", json_mapping(self.error, "error"))
        if not isinstance(self.cleanup_verified, bool):
            raise ValidationError("cleanup_verified must be a boolean")
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class EvictionContract(JsonModel):
    consumer_ack_required: bool = True
    drain_timeout_seconds: float = 30.0
    model_eviction_requires_explicit_grant: bool = True

    def __post_init__(self) -> None:
        for name in ("consumer_ack_required", "model_eviction_requires_explicit_grant"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be a boolean")
        object.__setattr__(self, "drain_timeout_seconds", finite_number(self.drain_timeout_seconds, "drain_timeout_seconds", strict=True))


@dataclass(frozen=True, slots=True)
class BackendSpec(JsonModel):
    backend_id: str
    host_id: str
    adapter_id: str
    profile_id: str
    configuration_ref: str
    residency_requirements: tuple[ResourceRequirement, ...]
    attached_consumers: tuple[str, ...] = ()
    eviction_contract: EvictionContract = EvictionContract()
    pinned_configuration: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("backend_id", "host_id", "adapter_id", "profile_id", "configuration_ref"):
            identifier(getattr(self, name), name)
        residency = _models(self.residency_requirements, ResourceRequirement, "residency_requirements", nonempty=True)
        object.__setattr__(self, "residency_requirements", residency)
        consumers = _strings(self.attached_consumers, "attached_consumers")
        object.__setattr__(self, "attached_consumers", consumers)
        if len(set(consumers)) != len(consumers):
            raise ValidationError("attached_consumers must be unique")
        if not isinstance(self.eviction_contract, EvictionContract):
            object.__setattr__(self, "eviction_contract", EvictionContract.from_dict(self.eviction_contract))
        object.__setattr__(self, "pinned_configuration", json_mapping(self.pinned_configuration, "pinned_configuration"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class EvictionGrant(JsonModel):
    grant_id: str
    backend_id: str
    granted_by: str
    scope: str
    expires_at: datetime

    def __post_init__(self) -> None:
        for name in ("grant_id", "backend_id", "granted_by", "scope"):
            identifier(getattr(self, name), name)
        object.__setattr__(self, "expires_at", aware_datetime(self.expires_at, "expires_at", required=True))


@dataclass(frozen=True, slots=True)
class HandoffSpec(JsonModel):
    handoff_id: str
    owner_id: str
    jobs: tuple[JobSpec, ...]
    backend_ids: tuple[str, ...]
    continuation_refs: tuple[str, ...]
    eviction_grants: tuple[EvictionGrant, ...]
    resource_envelope: tuple[ResourceRequirement, ...]
    drain_timeout_seconds: float
    run_timeout_seconds: float
    restore_timeout_seconds: float
    deadline: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.handoff_id, "handoff_id")
        identifier(self.owner_id, "owner_id")
        jobs = _models(self.jobs, JobSpec, "jobs", nonempty=True)
        object.__setattr__(self, "jobs", jobs)
        backend_ids = _strings(self.backend_ids, "backend_ids", nonempty=True)
        object.__setattr__(self, "backend_ids", backend_ids)
        if len(set(backend_ids)) != len(backend_ids):
            raise ValidationError("backend_ids must be unique")
        object.__setattr__(self, "continuation_refs", _strings(self.continuation_refs, "continuation_refs"))
        grants = _models(self.eviction_grants, EvictionGrant, "eviction_grants")
        object.__setattr__(self, "eviction_grants", grants)
        grant_backends = {grant.backend_id for grant in grants}
        if len(grant_backends) != len(grants):
            raise ValidationError("each target backend must have exactly one eviction grant")
        extra_grants = grant_backends - set(backend_ids)
        if extra_grants:
            raise ValidationError(f"eviction grants target unrequested backends: {', '.join(sorted(extra_grants))}")
        missing_grants = set(backend_ids) - grant_backends
        if missing_grants:
            raise ValidationError(f"missing explicit eviction grants for backends: {', '.join(sorted(missing_grants))}")
        envelope = _models(self.resource_envelope, ResourceRequirement, "resource_envelope", nonempty=True)
        object.__setattr__(self, "resource_envelope", envelope)
        for name in ("drain_timeout_seconds", "run_timeout_seconds", "restore_timeout_seconds"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name, strict=True))
        object.__setattr__(self, "deadline", aware_datetime(self.deadline, "deadline"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ScheduleSpec(JsonModel):
    schedule_id: str
    owner_id: str
    job: JobSpec
    interval_seconds: float | None = None
    calendar_expression: str | None = None
    timezone: str | None = None
    event_trigger: EventTrigger | None = None
    overlap: OverlapPolicy = OverlapPolicy.QUEUE
    misfire_policy: MisfirePolicy = MisfirePolicy.COALESCE
    max_catch_up: int = 1
    max_instances: int = 1
    state: ScheduleState = ScheduleState.ACTIVE
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.schedule_id, "schedule_id")
        identifier(self.owner_id, "owner_id")
        if not isinstance(self.job, JobSpec):
            object.__setattr__(self, "job", JobSpec.from_dict(self.job))
        trigger_count = sum(
            value is not None for value in (self.interval_seconds, self.calendar_expression, self.event_trigger)
        )
        if trigger_count != 1:
            raise ValidationError("exactly one of interval_seconds, calendar_expression, or event_trigger is required")
        if self.interval_seconds is not None:
            object.__setattr__(self, "interval_seconds", finite_number(self.interval_seconds, "interval_seconds", strict=True))
            if self.timezone is not None:
                raise ValidationError("timezone applies only to calendar schedules")
        elif self.calendar_expression is not None:
            identifier(self.calendar_expression, "calendar_expression")
            identifier(self.timezone, "timezone")
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, TypeError) as exc:
                raise ValidationError(f"timezone is not a recognized IANA timezone: {self.timezone!r}") from exc
        elif self.timezone is not None:
            raise ValidationError("timezone applies only to calendar schedules")
        if self.event_trigger is not None and not isinstance(self.event_trigger, EventTrigger):
            object.__setattr__(self, "event_trigger", EventTrigger.from_dict(self.event_trigger))
        if self.event_trigger is not None and self.job.event_trigger is not None:
            raise ValidationError("event schedule trigger is separate from the nested job's one-off event gate")
        if not isinstance(self.overlap, OverlapPolicy):
            object.__setattr__(self, "overlap", _enum(self.overlap, OverlapPolicy, "overlap"))
        if not isinstance(self.misfire_policy, MisfirePolicy):
            object.__setattr__(self, "misfire_policy", _enum(self.misfire_policy, MisfirePolicy, "misfire_policy"))
        if not isinstance(self.state, ScheduleState):
            object.__setattr__(self, "state", _enum(self.state, ScheduleState, "state"))
        if isinstance(self.max_catch_up, bool) or not isinstance(self.max_catch_up, int) or not 1 <= self.max_catch_up <= 100:
            raise ValidationError("max_catch_up must be an integer from 1 through 100")
        if self.misfire_policy is not MisfirePolicy.CATCH_UP and self.max_catch_up != 1:
            raise ValidationError("max_catch_up must be 1 unless misfire_policy is catch_up")
        if self.event_trigger is not None and self.misfire_policy is not MisfirePolicy.COALESCE:
            raise ValidationError("misfire_policy does not apply to event-triggered schedules")
        if isinstance(self.max_instances, bool) or not isinstance(self.max_instances, int) or not 1 <= self.max_instances <= 1000:
            raise ValidationError("max_instances must be an integer from 1 through 1000")
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class PolicyDecision(JsonModel):
    decision_id: str
    policy_id: str
    policy_version: str
    subject_id: str
    action: str
    accepted: bool
    reason: str
    selected_resource_ids: tuple[str, ...] = ()
    observed_at: datetime | None = None
    facts: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("decision_id", "policy_id", "policy_version", "subject_id", "action", "reason"):
            identifier(getattr(self, name), name)
        if not isinstance(self.accepted, bool):
            raise ValidationError("accepted must be a boolean")
        object.__setattr__(self, "selected_resource_ids", _strings(self.selected_resource_ids, "selected_resource_ids"))
        object.__setattr__(self, "observed_at", aware_datetime(self.observed_at, "observed_at"))
        object.__setattr__(self, "facts", json_mapping(self.facts, "facts"))
        _version(self.schema_version)


@dataclass(frozen=True, slots=True)
class Principal(JsonModel):
    principal_id: str
    actions: frozenset[str] = frozenset()
    execution_roots: tuple[str, ...] = ()
    backend_ids: frozenset[str] = frozenset()
    max_priority: int = 50
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifier(self.principal_id, "principal_id")
        object.__setattr__(self, "actions", frozenset(_strings(self.actions, "actions", allow_unordered=True)))
        object.__setattr__(self, "backend_ids", frozenset(_strings(self.backend_ids, "backend_ids", allow_unordered=True)))
        object.__setattr__(self, "execution_roots", _strings(self.execution_roots, "execution_roots"))
        if isinstance(self.max_priority, bool) or not isinstance(self.max_priority, int) or not 0 <= self.max_priority <= 100:
            raise ValidationError("max_priority must be an integer from 0 through 100")
        _version(self.schema_version)


def _require_nonempty(value: Any, name: str) -> None:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValidationError(f"{name} must be a non-empty array")


def _strings(
    value: Any, name: str, *, nonempty: bool = False, allow_unordered: bool = False
) -> tuple[str, ...]:
    allowed = (tuple, list, set, frozenset) if allow_unordered else (tuple, list)
    if not isinstance(value, allowed):
        raise ValidationError(f"{name} must be an array of identifiers")
    result = tuple(sorted(value)) if isinstance(value, (set, frozenset)) else tuple(value)
    if nonempty and not result:
        raise ValidationError(f"{name} must be a non-empty array")
    _identifiers(result, name)
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    try:
        return enum_type(value)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"invalid {name}: {value!r}") from exc


def _models(value: Any, model: type[T], name: str, *, nonempty: bool = False) -> tuple[T, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValidationError(f"{name} must be an array")
    result: list[T] = []
    for child in value:
        if isinstance(child, model):
            result.append(child)
        elif isinstance(child, Mapping):
            result.append(model.from_dict(child))
        else:
            raise ValidationError(f"{name} items must be {model.__name__} values")
    if nonempty and not result:
        raise ValidationError(f"{name} must be a non-empty array")
    return tuple(result)


def _identifiers(values: Any, name: str) -> None:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise ValidationError(f"{name} must be a collection of identifiers")
    for value in values:
        identifier(value, name)


def _version(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError("schema_version must be a positive integer")

