from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import sys
import tomllib

import pytest

from scripts.qualify_gpu_handoff import (
    APP_ID,
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
    _source_identity_matches,
    _write_scheduler_configuration,
    validate_parent_grant,
)
from scripts.qualification_gpu_probe import validate_probe_request


GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


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


def test_generated_configuration_is_private_loopback_and_uuid_pinned(tmp_path):
    work = tmp_path / "isolated"
    work.mkdir(mode=0o700)
    scheduler_token_file = work / "frameshift-scheduler.token"
    app_token_file = work / "frameshift-app.token"
    scheduler_token_file.write_text("scheduler-secret\n", encoding="utf-8")
    app_token_file.write_text("app-secret\n", encoding="utf-8")
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


def test_harness_uses_distinct_registry_identities():
    assert len({APP_ID, TOOLKIT_ID, MODEL_BINDING_ID, CONSUMER_CONTRACT}) == 4
