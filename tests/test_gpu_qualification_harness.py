from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

import pytest

from scripts.qualify_gpu_handoff import (
    APP_ID,
    AGENT_TYPE_ID,
    BACKEND_PROFILE_ID,
    CONSUMER_CONTRACT,
    CONTINUATION_REF,
    EXECUTION_PROFILE_ID,
    MODEL_BINDING_ID,
    TOOLKIT_ID,
    QualificationError,
    _action_objects,
    _cancel_handoff_until_terminal,
    _context_from_prompt,
    _encode_qualification_context,
    _frameshift_runtime,
    _reconcile_backend_on_failure,
    _select_scheduler_gpu_resource,
    _wait_backend_ready,
    capture_source_identity,
    _source_identity_matches,
    _write_scheduler_configuration,
    validate_parent_grant,
)
from scripts.qualification_gpu_probe import validate_probe_request


GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def _execution_handle(*, identity: str = "attempt-qualification") -> dict:
    from device_scheduler_core.models import ExecutionHandle

    return ExecutionHandle(
        adapter_id="systemd-user-unit",
        external_id="device-scheduler-attempt-qualification.service",
        identity=identity,
        metadata={"invocation_id": "invocation-qualification"},
    ).to_dict()


def test_backend_start_waits_for_ready_and_preserves_handle_seen_while_starting():
    handle = _execution_handle()

    class FakeClient:
        responses = iter((
            {"backend_id": "backend-a", "state": "starting", "execution_handle": handle},
            {"backend_id": "backend-a", "state": "ready", "execution_handle": handle},
        ))
        calls = []

        async def call(self, operation, arguments):
            self.calls.append((operation, arguments))
            return next(self.responses)

    client = FakeClient()
    observed = []
    started, returned_handle = asyncio.run(_wait_backend_ready(
        client,
        "backend-a",
        {"backend_id": "backend-a", "state": "starting", "execution_handle": handle},
        timeout_seconds=1,
        poll_interval_seconds=0.001,
        on_record=observed.append,
    ))

    assert started["state"] == "ready"
    assert returned_handle == handle
    assert observed[0]["state"] == "starting"
    assert observed[0]["execution_handle"] == handle
    assert client.calls == [
        ("backends.get", {"backend_id": "backend-a"}),
        ("backends.get", {"backend_id": "backend-a"}),
    ]


def test_backend_failure_cleanup_refreshes_handle_before_exact_unit_cancel():
    initial_handle = _execution_handle(identity="stale-attempt")
    current_handle = _execution_handle(identity="current-attempt")

    class FakeClient:
        async def call(self, operation, arguments):
            assert operation == "backends.get"
            assert arguments == {"backend_id": "backend-a"}
            return {
                "backend_id": "backend-a",
                "state": "ready",
                "execution_handle": current_handle,
            }

    cancelled = []

    def cancel(handle, *, operation_id, timeout_seconds):
        cancelled.append((handle.to_dict(), operation_id, timeout_seconds))
        return {"state": "stopped", "cleanup_verified": True}

    report = {}
    returned = asyncio.run(_reconcile_backend_on_failure(
        FakeClient(),
        "backend-a",
        initial_handle,
        cleanup_allowed=True,
        report=report,
        cancel=cancel,
    ))

    assert returned == current_handle
    assert report["backend_record_at_failure"]["execution_handle"] == current_handle
    assert report["early_backend_cleanup"]["state"] == "complete"
    assert cancelled == [(
        current_handle,
        "qualification-early-cleanup:backend-a",
        30,
    )]


def test_backend_failure_cleanup_does_not_stop_after_frame_spawn_may_have_occurred():
    handle = _execution_handle()

    class FakeClient:
        async def call(self, _operation, _arguments):
            return {"backend_id": "backend-a", "state": "ready", "execution_handle": handle}

    def forbidden_cancel(*_args, **_kwargs):
        raise AssertionError("backend must remain untouched after FrameShift spawn may have occurred")

    report = {}
    asyncio.run(_reconcile_backend_on_failure(
        FakeClient(),
        "backend-a",
        None,
        cleanup_allowed=False,
        report=report,
        cancel=forbidden_cancel,
    ))

    assert report["early_backend_cleanup"]["attempted"] is False
    assert report["early_backend_cleanup"]["reason"] == "FrameShift spawn or consumer attachment may have occurred"


def _grant(path: Path, *, project_root: Path, resource_ids: list[str] | None = None) -> dict:
    document = {
        "schema_version": 1,
        "owner_id": "qualification-parent-1",
        "gpu_uuid": GPU_UUID,
        "resource_ids": resource_ids or [f"accelerator:{GPU_UUID}", "host0/cpu"],
        "project_root": str(project_root),
        "pid": 12345,
        "process_start": "987654",
        "state": "active",
        "created_at": "2030-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o600)
    return document


def test_parent_grant_requires_exact_live_public_owner_and_single_gpu(tmp_path):
    path = tmp_path / "parent-grant.json"
    document = _grant(path, project_root=tmp_path)
    owner = {
        "owner_id": document["owner_id"],
        "state": "active",
        "pid": document["pid"],
        "process_start": document["process_start"],
        "project_root": str(tmp_path),
        "resources": document["resource_ids"],
    }

    result = validate_parent_grant(
        path,
        lambda owner_id: owner if owner_id == document["owner_id"] else None,
        project_root=tmp_path,
        current_uid=os.getuid(),
        now=datetime(2030, 1, 1, tzinfo=timezone.utc),
        process_check=lambda pid, start: pid == 12345 and start == "987654",
    )
    assert result["gpu_uuid"] == GPU_UUID
    assert result["resource_ids"] == sorted(document["resource_ids"])

    with pytest.raises(QualificationError, match="exactly its one declared accelerator UUID"):
        _grant(path, project_root=tmp_path, resource_ids=[
            f"accelerator:{GPU_UUID}", "accelerator:GPU-fedcba98-7654-3210-fedc-ba9876543210",
        ])
        validate_parent_grant(
            path, lambda _owner_id: owner, project_root=tmp_path, current_uid=os.getuid(),
            now=datetime(2030, 1, 1, tzinfo=timezone.utc), process_check=lambda *_: True,
        )


def test_parent_grant_rejects_symlink_wrong_process_and_owner_drift(tmp_path):
    real = tmp_path / "grant.json"
    document = _grant(real, project_root=tmp_path)
    link = tmp_path / "grant-link.json"
    link.symlink_to(real)
    with pytest.raises(QualificationError, match="regular non-symlink"):
        validate_parent_grant(link, lambda _owner_id: {}, project_root=tmp_path)

    owner = {
        "state": "active", "pid": document["pid"],
        "process_start": document["process_start"], "project_root": str(tmp_path),
        "resources": document["resource_ids"],
    }
    kwargs = {
        "project_root": tmp_path, "current_uid": os.getuid(),
        "now": datetime(2030, 1, 1, tzinfo=timezone.utc),
    }
    with pytest.raises(QualificationError, match="PID birth identity"):
        validate_parent_grant(real, lambda _owner_id: owner, process_check=lambda *_: False, **kwargs)
    with pytest.raises(QualificationError, match="no longer exactly matches"):
        validate_parent_grant(
            real, lambda _owner_id: {**owner, "resources": ["accelerator:other"]},
            process_check=lambda *_: True, **kwargs,
        )


def test_cuda_probe_request_is_bounded_and_uuid_pinned():
    validate_probe_request(GPU_UUID, 128 * 1024 * 1024, GPU_UUID)
    with pytest.raises(ValueError, match="name exactly"):
        validate_probe_request(GPU_UUID, 128 * 1024 * 1024, "0")
    with pytest.raises(ValueError, match="between 1 MiB and 256 MiB"):
        validate_probe_request(GPU_UUID, 257 * 1024 * 1024, GPU_UUID)
    with pytest.raises(ValueError, match="stable NVIDIA GPU UUID"):
        validate_probe_request("0", 128 * 1024 * 1024, "0")


def test_qualification_source_identity_requires_exact_before_after_match():
    baseline = {
        "production_source_hashes": {"packages/core.py": "a" * 64},
        "external_frameshift": {"git_head": "abc", "source_hashes": {"frameshift/engine.py": "b" * 64}},
        "llama_binary": {"sha256": "c" * 64},
        "model_identity": {"inode": 123, "size_bytes": 456},
    }
    assert _source_identity_matches(baseline, dict(baseline))
    changed = {**baseline, "production_source_hashes": {"packages/core.py": "d" * 64}}
    assert not _source_identity_matches(baseline, changed)


def test_source_identity_capture_returns_production_and_external_hashes(tmp_path):
    frameshift_source = tmp_path / "frameshift-src"
    frameshift_package = frameshift_source / "frameshift"
    frameshift_package.mkdir(parents=True)
    frame_file = frameshift_package / "engine.py"
    frame_file.write_text("ENGINE_SENTINEL = 1\n", encoding="utf-8")
    llama_binary = tmp_path / "llama-server"
    llama_binary.write_bytes(b"small qualification binary\n")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"small model identity sentinel\n")

    identity = capture_source_identity(
        frameshift_source=frameshift_source,
        llama_binary=llama_binary,
        model=model,
    )

    assert identity["production_source_hashes"]
    assert identity["qualification_source_hashes"]["scripts/qualify_gpu_handoff.py"]
    assert identity["external_frameshift"]["source_hashes"] == {
        "frameshift/engine.py": hashlib.sha256(frame_file.read_bytes()).hexdigest(),
    }
    assert identity["llama_binary"]["size_bytes"] == llama_binary.stat().st_size
    assert identity["model_identity"]["content_sha256_status"] == "not-computed-large-local-model"


def test_handoff_signal_cancellation_waits_for_terminal_and_checks_parent_grant():
    class FakeGuard:
        calls = 0

        def verify(self):
            self.calls += 1
            return {"state": "active"}

    class FakeClient:
        states = iter(("running", "cancelled"))
        requests = []

        async def control_handoff(self, handoff_id, action, **options):
            self.requests.append((handoff_id, action, options))
            return {"handoff_id": handoff_id, "cancel_requested": True}

        async def get_handoff(self, handoff_id):
            return {"handoff_id": handoff_id, "state": next(self.states)}

    client = FakeClient()
    guard = FakeGuard()
    result = asyncio.run(_cancel_handoff_until_terminal(
        client, guard, "handoff-test", timeout_seconds=1,
    ))
    assert result["state"] == "cancelled"
    assert client.requests == [(
        "handoff-test", "cancel", {"reason": "qualification received SIGINT or SIGTERM"},
    )]
    assert guard.calls >= 2


def test_real_model_prompt_and_decoder_candidates_are_explicit_json():
    text = _encode_qualification_context({"messages": []}, gpu_uuid=GPU_UUID, handoff_id="handoff-a")
    assert "FRAME_CONTEXT_JSON:" in text
    assert '"memory_bytes":134217728' in text
    context = _context_from_prompt(text)
    assert context["gpu_uuid"] == GPU_UUID
    assert context["expected_handoff_id"] == "handoff-a"
    assert _action_objects('preface {"action":"tool_request","arguments":{"x":1}} tail') == [
        {"action": "tool_request", "arguments": {"x": 1}},
    ]
    assert _action_objects("plain text without a JSON action") == []


def test_actual_frameshift_engine_offer_prompt_contains_granted_handoff_identity(tmp_path):
    frameshift_source = Path("/home/tumlinson/FrameShift/src").resolve()
    if not (frameshift_source / "frameshift" / "engine.py").is_file():
        pytest.skip("FrameShift source checkout is unavailable")
    if str(frameshift_source) not in sys.path:
        sys.path.insert(0, str(frameshift_source))

    from frameshift.backends.scripted import ScriptedProvider
    from frameshift.engine import Engine
    from frameshift.models import (
        ActionKind,
        AgentType,
        CapabilityGrant,
        FrameBudget,
        GenerationResult,
        Lifecycle,
        ModelBinding,
        PayloadCodec,
        ProposedAction,
        Toolkit,
        TypedPayload,
    )
    from frameshift.registry import Registry

    app_id = "scheduler-qualification-test"
    type_id = "qualification-agent"
    toolkit_id = "qualification-tool"
    handoff_id = "qualification-handoff-identity"
    provider_prompt: dict[str, str] = {}

    class CapturingProvider(ScriptedProvider):
        def offers(self, request):
            provider_prompt["text"] = request.input.data
            return super().offers(request)

    def complete(_request, _lease):
        output = TypedPayload("json", "1", {"report": "captured"})
        return GenerationResult(
            action=ProposedAction(ActionKind.COMPLETE, payload=output), output=output,
        )

    registry = Registry()
    registry.register_codec(PayloadCodec(
        "qualification.prompt", "1", "text", count_tokens=lambda value: len(str(value).split()),
        encode=functools.partial(
            _encode_qualification_context, gpu_uuid=GPU_UUID, handoff_id=handoff_id,
        ),
    ))
    registry.register_model_binding(ModelBinding(
        "qualification-model", "1", "qualification-model",
        (("qualification.prompt", "1"),), (("json", "1"),), max_context_tokens=4096,
    ))
    registry.register_toolkit(Toolkit(toolkit_id, "1"))
    registry.register_agent_type(AgentType(
        type_id, "1", Lifecycle.PERSISTENT,
        ("qualification.prompt", "1"), ("json", "1"), ("json", "1"),
        ("qualification-model",), toolkit_ids=(toolkit_id,),
        allowed_actions=frozenset({ActionKind.COMPLETE}), max_context_tokens=4096,
    ))
    registry.register_application(app_id, CapabilityGrant(
        "qualification-grant", app_id, frozenset({type_id}),
        allowed_toolkits=frozenset({toolkit_id}), max_frames=1, max_turns=1,
        max_context_tokens=4096,
    ))
    provider = CapturingProvider(
        provider_id="qualification-provider", model_id="qualification-model",
        model_version="1", input_codecs=(("qualification.prompt", "1"),),
        output_codecs=(("json", "1"),), script=[complete],
    )
    engine = Engine(str(tmp_path / "frames.sqlite3"), registry, (provider,))
    try:
        engine.spawn(
            app_id, type_id, "run one bounded GPU test", grant_id="qualification-grant",
            frame_id="qualification-frame",
            budget=FrameBudget(1, 128, 4096),
        )
        report = engine.run_once()
        assert report is not None and report.state.value == "completed"
        context = _context_from_prompt(provider_prompt["text"])
        assert context["gpu_uuid"] == GPU_UUID
        assert context["expected_handoff_id"] == handoff_id
        assert context["messages"] == []
    finally:
        engine.close()


def test_frameshift_runtime_builds_real_registry_without_starting_a_listener(tmp_path, monkeypatch):
    frameshift_source = Path("/home/tumlinson/FrameShift/src").resolve()
    if not (frameshift_source / "frameshift" / "engine.py").is_file():
        pytest.skip("FrameShift source checkout is unavailable")
    if str(frameshift_source) not in sys.path:
        sys.path.insert(0, str(frameshift_source))

    from frameshift.service import FrameShiftService

    class DisabledListener:
        def __init__(self, service):
            self.service = service
            self.server_address = ("127.0.0.1", 0)

        def serve_forever(self):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr(
        FrameShiftService,
        "create_server",
        lambda self, *_args, **_kwargs: DisabledListener(self),
    )
    server, thread, _client, _config = _frameshift_runtime(
        tmp_path, backend_id="qualification-backend", llama_port=6553,
        gpu_uuid=GPU_UUID, handoff_id="qualification-handoff", frame_id="qualification-frame",
        app_token="app-token", scheduler_token="scheduler-token", timeout_seconds=1,
    )
    engine = server.service.engine
    try:
        registry = engine.registry
        assert registry.application_exists(APP_ID)
        assert registry.agent_type(AGENT_TYPE_ID).toolkit_ids == (TOOLKIT_ID,)
        assert registry.model_binding(MODEL_BINDING_ID).max_context_tokens == 4096
        assert registry.toolkit(TOOLKIT_ID).capabilities == frozenset({"gpu.probe"})
        assert registry.grant(APP_ID, "qualification-grant").allowed_toolkits == frozenset({TOOLKIT_ID})
        assert len(engine.providers) == 1
        assert engine.providers[0].backend_id == "qualification-backend"
        assert not thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
        engine.close()


def test_generated_configuration_is_private_loopback_and_uuid_pinned(tmp_path):
    work = tmp_path / "isolated"
    work.mkdir(mode=0o700)
    scheduler_token_file = work / "frameshift-scheduler.token"
    app_token_file = work / "frameshift-app.token"
    scheduler_token_file.write_text("scheduler-secret-qualification-token-0001\n", encoding="utf-8")
    app_token_file.write_text("app-secret-qualification-token-0001\n", encoding="utf-8")
    scheduler_token_file.chmod(0o600)
    app_token_file.chmod(0o600)

    config_path, token_file, token_dir, state_dir = _write_scheduler_configuration(
        work,
        port=18770,
        token="scheduler-client-token",
        scheduler_token_file=scheduler_token_file,
        fs_url="http://127.0.0.1:18771",
        fs_app_token_file=app_token_file,
        frame_id="qualification-frame",
        llama_binary=Path("/opt/llama-server"),
        model=Path("/opt/model.gguf"),
        llama_port=18772,
        gpu_uuid=GPU_UUID,
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    execution = tomllib.loads((work / "profiles.toml").read_text(encoding="utf-8"))
    backend = tomllib.loads((work / "backend-profiles.toml").read_text(encoding="utf-8"))
    continuations = tomllib.loads((work / "continuations.toml").read_text(encoding="utf-8"))

    assert config["server"]["host"] == "127.0.0.1"
    assert config["paths"]["state_directory"] == str(state_dir)
    assert execution["profiles"][EXECUTION_PROFILE_ID]["environment"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    assert execution["profiles"][EXECUTION_PROFILE_ID]["argv"][execution["profiles"][EXECUTION_PROFILE_ID]["argv"].index("--ctx-size") + 1] == "4096"
    assert execution["profiles"][EXECUTION_PROFILE_ID]["argv"][-1] == "device-scheduler-qualification"
    assert backend["backend_profiles"][BACKEND_PROFILE_ID]["access_mode"] == "frameshift-only"
    assert backend["backend_profiles"][BACKEND_PROFILE_ID]["slots_url"].endswith("/slots")
    assert continuations["continuations"][CONTINUATION_REF]["frame_id"] == "qualification-frame"
    assert all(path.stat().st_mode & 0o077 == 0 for path in (token_file, token_dir, state_dir))
    assert (work / "backend-profiles.toml").stat().st_mode & 0o077 == 0


def test_generated_configuration_loads_and_composes_real_scheduler_without_starting_work(tmp_path):
    from device_scheduler_adapters.systemd import ExecutionAdapterError
    from device_scheduler_core.models import (
        ExecutionContext,
        JobSpec,
        Resource,
        ResourceAllocation,
        ResourceRequirement,
    )
    from device_scheduler_mcp.config import ServiceConfig
    from device_scheduler_mcp.server import build_default_scheduler

    work = tmp_path / "scheduler-composition"
    work.mkdir(mode=0o700)
    scheduler_token_file = work / "frameshift-scheduler.token"
    app_token_file = work / "frameshift-app.token"
    scheduler_token_file.write_text("scheduler-secret-qualification-token-0001\n", encoding="utf-8")
    app_token_file.write_text("app-secret-qualification-token-0001\n", encoding="utf-8")
    scheduler_token_file.chmod(0o600)
    app_token_file.chmod(0o600)
    config_path, token_file, token_dir, state_dir = _write_scheduler_configuration(
        work,
        port=18780,
        token="scheduler-client-token",
        scheduler_token_file=scheduler_token_file,
        fs_url="http://127.0.0.1:18781",
        fs_app_token_file=app_token_file,
        frame_id="qualification-frame",
        llama_binary=Path("/opt/llama-server"),
        model=Path("/opt/model.gguf"),
        llama_port=18782,
        gpu_uuid=GPU_UUID,
    )

    service_config = ServiceConfig.from_file(config_path)
    assert service_config.token_directory == token_dir
    assert service_config.token_file == token_file
    assert service_config.state_directory == state_dir
    assert service_config.profiles_path == work / "profiles.toml"
    assert service_config.backend_profiles_path == work / "backend-profiles.toml"
    assert service_config.continuations_path == work / "continuations.toml"

    scheduler = build_default_scheduler(service_config)
    assert scheduler.backend_adapter is not None
    assert scheduler.continuation_adapter is not None
    assert scheduler.config.profiles[EXECUTION_PROFILE_ID]["environment"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    assert scheduler.config.backend_profiles[BACKEND_PROFILE_ID]["access_mode"] == "frameshift-only"
    resolved = scheduler.executor._profile_resolver(EXECUTION_PROFILE_ID)
    assert resolved["limits"] == {
        "cpu_quota_percent": 800.0,
        "memory_max_bytes": 51539607552,
        "tasks_max": 128,
    }

    job = JobSpec(
        owner_id="qualification",
        profile_id=EXECUTION_PROFILE_ID,
        resources=(ResourceRequirement(
            kind="gpu",
            quantities={"memory_bytes": 128 * 1024 * 1024},
            selectors={"uuid": GPU_UUID},
        ),),
    )
    resource = Resource(
        resource_id="qualification-gpu",
        host_id=socket.gethostname(),
        kind="gpu",
        capacities={"memory_bytes": 16 * 1024 * 1024 * 1024},
        attributes={"uuid": GPU_UUID},
    )
    context = ExecutionContext(
        allocations=(ResourceAllocation(
            resource_id=resource.resource_id,
            quantities={"memory_bytes": 128 * 1024 * 1024},
        ),),
        resources=(resource,),
        claim_id="qualification-parent-grant",
        fencing_token=1,
    )
    bound, _ = scheduler.executor._bind_execution_context(job, resolved, context)
    assert bound["environment"]["CUDA_VISIBLE_DEVICES"] == GPU_UUID
    properties = scheduler.executor._unit_properties(bound)
    assert "CPUQuota=800%" in properties
    assert "MemoryMax=51539607552" in properties
    assert "TasksMax=128" in properties
    with pytest.raises(ExecutionAdapterError, match="conflicts with the scheduler-granted GPU UUIDs"):
        scheduler.executor._bind_execution_context(job, resolved, None)
    wrong_gpu = Resource(
        resource_id="qualification-wrong-gpu",
        host_id=socket.gethostname(),
        kind="gpu",
        capacities={"memory_bytes": 16 * 1024 * 1024 * 1024},
        attributes={"uuid": "GPU-fedcba98-7654-3210-fedc-ba9876543210"},
    )
    wrong_context = ExecutionContext(
        allocations=(ResourceAllocation(
            resource_id=wrong_gpu.resource_id,
            quantities={"memory_bytes": 128 * 1024 * 1024},
        ),),
        resources=(wrong_gpu,),
        claim_id="qualification-wrong-parent-grant",
        fencing_token=2,
    )
    with pytest.raises(ExecutionAdapterError, match="conflicts with the scheduler-granted GPU UUIDs"):
        scheduler.executor._bind_execution_context(job, resolved, wrong_context)


def test_resources_list_dispatch_uses_allocator_status_envelopes(tmp_path):
    from device_scheduler_adapters.sqlite import Ledger
    from device_scheduler_core.allocator import ResourceAllocator
    from device_scheduler_core.application import Scheduler
    from device_scheduler_core.models import Principal, Resource

    ledger = Ledger(tmp_path / "resource-list-ledger.sqlite3")
    allocator = ResourceAllocator(ledger)
    gpu = Resource(
        resource_id="qualification-gpu",
        host_id="qualification-host",
        kind="gpu",
        capacities={"memory_bytes": 16 * 1024 * 1024 * 1024},
        attributes={"uuid": GPU_UUID},
    )
    allocator.register_resource(gpu, source_adapter_id="qualification-nvidia")
    scheduler = Scheduler(ledger=ledger, allocator=allocator, resource_adapters=())
    principal = Principal(principal_id="qualification", actions=frozenset({"resources.list"}))

    response = scheduler.dispatch("resources.list", {}, principal)
    selected, status = _select_scheduler_gpu_resource(response, GPU_UUID)
    assert selected["resource_id"] == gpu.resource_id
    assert selected["attributes"]["uuid"] == GPU_UUID
    assert status["available"] is True
    assert status["source_adapter_id"] == "qualification-nvidia"

    allocator.reconcile_resource_inventory(
        "qualification-nvidia", (), error="qualification telemetry unavailable",
    )
    unavailable_response = scheduler.dispatch("resources.list", {}, principal)
    with pytest.raises(QualificationError, match="reports the grant's GPU unavailable"):
        _select_scheduler_gpu_resource(unavailable_response, GPU_UUID)


def test_harness_uses_distinct_registry_identities():
    assert len({APP_ID, TOOLKIT_ID, MODEL_BINDING_ID, CONSUMER_CONTRACT}) == 4
