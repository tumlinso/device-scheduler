"""Focused invariants for the public core contract surface."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from device_scheduler_core import (
    AdapterCapability,
    AllocationMode,
    Attempt,
    AttemptState,
    CapabilityPrecondition,
    EventTrigger,
    EnforcementLevel,
    ExecutionContext,
    ExecutionPurpose,
    GPU_EXCLUSIVE_TAG,
    EvictionGrant,
    HandoffSpec,
    InterruptionContract,
    JobSpec,
    Lease,
    LeaseState,
    MisfirePolicy,
    OverlapPolicy,
    ExplainableSchedulerPolicy,
    PreemptionPolicy,
    PreemptionProposal,
    PreemptionSelection,
    Resource,
    ResourceAllocation,
    ResourceRequest,
    ResourceRequirement,
    RankableSchedulerPolicy,
    ScheduleSpec,
    SchedulerPolicy,
    ValidationError,
    WeightedFairPolicy,
)


NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def test_job_spec_round_trips_as_plain_json_with_enums_and_datetimes() -> None:
    original = JobSpec(
        owner_id="agent:kernel-work",
        argv=("pytest", "-q", "tests/test_kernel.py"),
        working_directory="/work/kernel",
        resources=(
            ResourceRequirement(
                kind="gpu",
                quantities={"gpu_memory_bytes": 2_000_000_000},
                selectors={"vendor": "nvidia"},
                mode=AllocationMode.EXCLUSIVE,
                topology_constraints={"same_pci_root": True},
            ),
        ),
        priority=73,
        not_before=NOW,
        deadline=NOW + timedelta(minutes=15),
        interruption=InterruptionContract(cooperative_yield=True, restart_safe=True),
        dependencies=("job:build-kernel",),
        idempotency_key="test-round-1",
        secret_refs=("cuda/registry-token",),
    )

    encoded = json.dumps(original.to_dict())
    restored = JobSpec.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.resources[0].mode is AllocationMode.EXCLUSIVE
    assert restored.not_before == NOW


def test_job_spec_packet_estimates_limits_and_preemption_round_trip_with_legacy_defaults() -> None:
    legacy = JobSpec.from_dict({
        "owner_id": "agent:legacy", "argv": ["true"], "working_directory": "/tmp",
    })
    assert legacy.timeout_seconds == 3600
    assert legacy.expected_duration_seconds is None
    assert legacy.kill_timeout_seconds is None
    assert legacy.termination_grace_seconds is None
    assert legacy.preemption_action.value == "restart"
    assert legacy.flexible is False

    packet = JobSpec.from_dict({
        "owner_id": "autobench", "argv": ["bench"], "working_directory": "/tmp",
        "expected_duration_seconds": 20, "kill_timeout_seconds": 30,
        "termination_grace_seconds": 5, "preemption_action": "abandon",
        "flexible": True, "retention_policy": "autobench",
    })
    restored = JobSpec.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert restored == packet
    with pytest.raises(ValidationError, match="less than"):
        JobSpec.from_dict({
            "owner_id": "autobench", "argv": ["bench"], "working_directory": "/tmp",
            "timeout_seconds": 20, "expected_duration_seconds": 20,
        })


def test_capability_preconditions_are_generic_versioned_json_contracts() -> None:
    original = JobSpec(
        owner_id="agent:kernel-work",
        argv=("pytest", "-q"),
        working_directory="/work/kernel",
        preconditions=(CapabilityPrecondition(
            "gpu.quiescence",
            {"timeout_seconds": 15, "interval_seconds": 0.25, "consecutive_samples": 4},
        ),),
    )

    restored = JobSpec.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original
    assert restored.preconditions[0].name == "gpu.quiescence"
    assert restored.preconditions[0].parameters["consecutive_samples"] == 4
    assert CapabilityPrecondition.from_dict(original.preconditions[0].to_dict()) == original.preconditions[0]


def test_job_preconditions_require_valid_unique_contract_names() -> None:
    with pytest.raises(ValidationError):
        CapabilityPrecondition("", {})
    with pytest.raises(ValidationError):
        CapabilityPrecondition("gpu.quiescence", {"nan": float("nan")})
    with pytest.raises(ValidationError, match="unique"):
        JobSpec(
            owner_id="agent:kernel-work",
            argv=("pytest",),
            working_directory="/work/kernel",
            preconditions=(CapabilityPrecondition("gpu.quiescence"), CapabilityPrecondition("gpu.quiescence")),
        )


def test_resource_is_host_qualified_and_copies_nested_json_values() -> None:
    attributes = {"vendor": "nvidia", "labels": ["cuda", "test"]}
    resource = Resource(
        resource_id="gpu:0",
        host_id="host:lab",
        kind="gpu",
        capacities={"gpu_memory_bytes": 24_000_000_000, "device_count": 1},
        attributes=attributes,
        topology={"pci_root": "0000:40", "numa_node": 0},
        conflict_domains=frozenset({"profiler:host:lab"}),
        capabilities=frozenset({"exclusive", "concurrent"}),
    )
    attributes["vendor"] = "changed"

    assert resource.attributes["vendor"] == "nvidia"
    assert resource.to_dict()["host_id"] == "host:lab"
    with pytest.raises(TypeError):
        resource.attributes["vendor"] = "changed"  # type: ignore[index]

    unavailable = Resource(
        resource_id="gpu:unavailable",
        host_id="host:lab",
        kind="gpu",
        capacities={"gpu_memory_bytes": 0},
    )
    assert unavailable.capacities["gpu_memory_bytes"] == 0


def test_execution_context_round_trips_the_actual_grant_and_descriptors() -> None:
    gpu = Resource(
        "gpu:uuid-123",
        "host:lab",
        "gpu",
        {"gpu_memory_bytes": 24_000_000_000},
        attributes={"uuid": "uuid-123", "vendor": "nvidia"},
    )
    allocation = ResourceAllocation(
        gpu.resource_id,
        {"gpu_memory_bytes": 3_000_000_000},
        AllocationMode.SHARED,
    )
    context = ExecutionContext(
        allocations=(allocation,),
        resources=(gpu,),
        claim_id="lease:gpu-work",
        fencing_token=17,
        purpose=ExecutionPurpose.JOB,
    )

    restored = ExecutionContext.from_dict(json.loads(json.dumps(context.to_dict())))
    assert restored == context
    assert restored.resources[0].attributes["uuid"] == "uuid-123"
    assert restored.allocations[0].quantities["gpu_memory_bytes"] == 3_000_000_000
    assert ExecutionContext().allocations == ()
    assert ExecutionContext(purpose="backend").purpose is ExecutionPurpose.BACKEND


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allocations": (ResourceAllocation("gpu:0", {"gpu_memory_bytes": 1}),)},
        {"resources": (Resource("gpu:0", "host:1", "gpu", {"gpu_memory_bytes": 10}),)},
        {
            "allocations": (ResourceAllocation("gpu:0", {"gpu_memory_bytes": 1}),),
            "resources": (Resource("gpu:1", "host:1", "gpu", {"gpu_memory_bytes": 10}),),
        },
        {"fencing_token": 3},
        {"claim_id": "lease:1", "fencing_token": 0},
        {"claim_id": "lease:1", "fencing_token": True},
    ],
)
def test_execution_context_rejects_unbound_resources_and_invalid_fencing(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExecutionContext(**kwargs)  # type: ignore[arg-type]


def test_lease_keeps_unknown_reclamation_distinct_from_release() -> None:
    lease = Lease(
        lease_id="lease:1",
        owner_id="job:1",
        allocations=(ResourceAllocation("gpu:0", {"gpu_memory_bytes": 4_000_000_000}),),
        fencing_token=8,
        expires_at=NOW + timedelta(minutes=2),
        state=LeaseState.UNKNOWN,
    )

    assert Lease.from_dict(lease.to_dict()).state is LeaseState.UNKNOWN
    assert lease.state is not LeaseState.RELEASED


def test_owner_context_round_trips_as_immutable_descriptive_strings() -> None:
    context = {
        "issuer": "project-control",
        "project_id": "device-scheduler",
        "attempt_id": "attempt:42",
        "task_id": "DS-13",
        "owner_kind": "agent",
        "project_root": "/home/example/Kernel Worktree",
    }
    requirement = ResourceRequirement("gpu", {"gpu_memory_bytes": 2_000_000})

    # Preserve the pre-context positional order, including schema_version.
    request = ResourceRequest(
        "request:42",
        "project-control:device-scheduler",
        (requirement,),
        30,
        "request-key",
        1,
        context,
    )
    assert ResourceRequest.from_dict(request.to_dict()) == request
    assert request.owner_context["issuer"] == "project-control"
    with pytest.raises(TypeError):
        request.owner_context["task_id"] = "other"  # type: ignore[index]

    lease = Lease(
        "lease:42",
        request.owner_id,
        (ResourceAllocation("gpu:0", requirement.quantities),),
        42,
        NOW + timedelta(minutes=1),
        LeaseState.ACTIVE,
        None,
        1,
        context,
    )
    assert Lease.from_dict(lease.to_dict()) == lease
    with pytest.raises(TypeError):
        lease.owner_context["issuer"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    "owner_context",
    [
        {"issuer": {"name": "project-control"}},
        {"access_token": "opaque-value"},
        {"APIKey": "opaque-value"},
        {"api-key": "opaque-value"},
        {"key": "opaque-value"},
        {"key_material": "opaque-value"},
        {f"field_{index}": "value" for index in range(17)},
        {"issuer": "x\x00y"},
        {"project_root": "/" + "x" * 4096},
    ],
)
def test_owner_context_rejects_nested_sensitive_or_unbounded_values(owner_context: object) -> None:
    requirement = ResourceRequirement("gpu", {"gpu_memory_bytes": 1})
    with pytest.raises(ValidationError, match="owner_context"):
        ResourceRequest(
            request_id="request:invalid-context",
            owner_id="agent:1",
            requirements=(requirement,),
            lease_ttl_seconds=30,
            owner_context=owner_context,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource_id": "gpu:0", "host_id": "host:1", "kind": "gpu", "capacities": {"memory": float("nan")}},
        {"resource_id": "gpu:0", "host_id": "host:1", "kind": "gpu", "capacities": {"memory": -1}},
    ],
)
def test_resource_rejects_invalid_capacity(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Resource(**kwargs)  # type: ignore[arg-type]


def test_requirement_rejects_zero_or_nonfinite_amount() -> None:
    with pytest.raises(ValidationError):
        ResourceRequirement(kind="gpu", quantities={"gpu_memory_bytes": 0})
    with pytest.raises(ValidationError):
        ResourceRequirement(kind="gpu", quantities={"gpu_memory_bytes": float("inf")})


def test_command_requires_explicit_working_directory_and_profile_is_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        JobSpec(owner_id="client:1", argv=("true",))
    with pytest.raises(ValidationError):
        JobSpec(owner_id="client:1", argv=("true",), profile_id="profile:1")
    with pytest.raises(ValidationError, match="working_directory"):
        JobSpec(owner_id="client:1", argv=("true",), working_directory=42)  # type: ignore[arg-type]


def test_job_retries_require_restart_safety_and_exclusive_tag_is_explicit() -> None:
    with pytest.raises(ValidationError, match="restart_safe"):
        JobSpec(owner_id="client:1", profile_id="test", max_attempts=2)

    spec = JobSpec(
        owner_id="client:1",
        profile_id="test",
        tags=frozenset({GPU_EXCLUSIVE_TAG}),
        priority_class="interactive",
        interruption=InterruptionContract(restart_safe=True),
        max_attempts=2,
        timeout_seconds=120,
    )
    assert JobSpec.from_dict(spec.to_dict()) == spec


def test_handoff_requires_an_explicit_grant_for_every_target_backend() -> None:
    job = JobSpec(owner_id="agent:1", profile_id="cuda-test", resources=(
        ResourceRequirement("gpu", {"gpu_memory_bytes": 1_000_000}),
    ))
    with pytest.raises(ValidationError, match="explicit eviction grants"):
        HandoffSpec(
            handoff_id="handoff:1",
            owner_id="agent:1",
            jobs=(job,),
            backend_ids=("llama:main",),
            continuation_refs=("frame:1",),
            eviction_grants=(),
            resource_envelope=job.resources,
            drain_timeout_seconds=15,
            run_timeout_seconds=60,
            restore_timeout_seconds=60,
        )

    spec = HandoffSpec(
        handoff_id="handoff:1",
        owner_id="agent:1",
        jobs=(job,),
        backend_ids=("llama:main",),
        continuation_refs=("frame:1",),
        eviction_grants=(EvictionGrant("grant:1", "llama:main", "operator:1", "evict", NOW + timedelta(minutes=5)),),
        resource_envelope=job.resources,
        drain_timeout_seconds=15,
        run_timeout_seconds=60,
        restore_timeout_seconds=60,
    )
    assert HandoffSpec.from_dict(spec.to_dict()) == spec

    with pytest.raises(ValidationError, match="exactly one eviction grant"):
        HandoffSpec(
            handoff_id="handoff:duplicate-grants",
            owner_id="agent:1",
            jobs=(job,),
            backend_ids=("llama:main",),
            continuation_refs=("frame:1",),
            eviction_grants=(
                EvictionGrant("grant:1", "llama:main", "operator:1", "evict", NOW + timedelta(minutes=5)),
                EvictionGrant("grant:2", "llama:main", "operator:1", "evict", NOW + timedelta(minutes=5)),
            ),
            resource_envelope=job.resources,
            drain_timeout_seconds=15,
            run_timeout_seconds=60,
            restore_timeout_seconds=60,
        )


def test_calendar_schedule_requires_explicit_timezone_and_one_trigger_kind() -> None:
    job = JobSpec(owner_id="client:1", profile_id="daily-check")
    with pytest.raises(ValidationError):
        ScheduleSpec(
            schedule_id="schedule:1",
            owner_id="client:1",
            job=job,
            calendar_expression="0 9 * * *",
        )
    with pytest.raises(ValidationError):
        ScheduleSpec(
            schedule_id="schedule:1",
            owner_id="client:1",
            job=job,
            interval_seconds=30,
            calendar_expression="0 9 * * *",
            timezone="UTC",
        )

    calendar = ScheduleSpec(
        schedule_id="schedule:calendar",
        owner_id="client:1",
        job=job,
        calendar_expression="0 9 * * *",
        timezone="Europe/Berlin",
        misfire_policy=MisfirePolicy.CATCH_UP,
        max_catch_up=3,
    )
    assert ScheduleSpec.from_dict(calendar.to_dict()) == calendar

    event_schedule = ScheduleSpec(
        schedule_id="schedule:event",
        owner_id="client:1",
        job=job,
        event_trigger=EventTrigger("project-control", "gpu-free"),
    )
    assert event_schedule.event_trigger.event_type == "gpu-free"
    assert event_schedule.overlap is OverlapPolicy.QUEUE


def test_attempt_does_not_accept_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Attempt("attempt:1", "job:1", state=AttemptState.RUNNING, started_at=datetime(2026, 10, 21))


def test_adapter_capability_must_state_truthful_enforcement_level() -> None:
    capability = AdapterCapability(
        "gpu.memory_budget",
        supported=True,
        enforcement=EnforcementLevel.ADVISORY,
        details={"reason": "admission accounting only"},
    )
    assert AdapterCapability.from_dict(capability.to_dict()) == capability
    with pytest.raises(ValidationError, match="enforcement level"):
        AdapterCapability("gpu.memory_budget", supported=True)
    with pytest.raises(ValidationError, match="cannot claim"):
        AdapterCapability("gpu.memory_budget", supported=False, enforcement=EnforcementLevel.ENFORCED)


def test_builtin_policy_satisfies_versioned_selection_and_optional_capability_ports() -> None:
    policy = WeightedFairPolicy()
    assert isinstance(policy, SchedulerPolicy)
    assert isinstance(policy, ExplainableSchedulerPolicy)
    assert isinstance(policy, RankableSchedulerPolicy)
    assert isinstance(policy, PreemptionPolicy)

    proposal = policy.propose_preemption(
        {
            "job_id": "job:interactive",
            "owner_id": "agent:interactive",
            "state": "queued",
            "created_at": NOW,
            "spec": {"priority_class": "interactive", "priority": 80},
        },
        [
            {
                "job_id": "job:background",
                "owner_id": "agent:background",
                "state": "running",
                "created_at": NOW,
                "attempt_id": "attempt:background",
                "lease_id": "lease:background",
                "attempt_count": 1,
                "spec": {
                    "priority_class": "background",
                    "priority": 10,
                    "max_attempts": 2,
                    "interruption": {"restart_safe": True, "cancellation_supported": True},
                },
            }
        ],
        {"now": NOW},
        would_unblock=lambda _target: True,
    )
    assert isinstance(proposal, PreemptionProposal)
    assert isinstance(proposal, PreemptionSelection)
    assert proposal is not None and proposal.target_attempt_id == "attempt:background"

