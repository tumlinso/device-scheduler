#!/usr/bin/env python3
"""Source-bound acceptance checks for the native Device Scheduler plan.

The native check runs this standard-library entry point from the claimed task
workspace. DS-00 validates the accepted design package and native plan itself.
Later tasks must add a hash-bound ``planning/evidence/DS-NN/manifest.json``;
the verifier also executes the task's locked, focused pytest suite. DS-12 to
DS-14 have additional live-evidence assertions and cannot pass from unit tests
or placeholder files alone.

Manifest shape::

    {"schema_version": 1, "work_item_id": "DS-01", "verified_at": "...Z",
     "source_commit": "...", "test_python": "/host/configured/python",
     "test_command": ["-m", "pytest", "-q", "..."],
     "execution": {"argv": ["python", "-m", "pytest", "-q", "..."],
       "exit_code": 0, "source_commit": "...", "receipt_id": "...",
       "started_at": "...Z", "finished_at": "...Z"},
     "source_snapshot": {"sha256": "...", "files": {"relative/path.py": "..."}},
     "evidence": {
      "core-contract-api": {"path": "...", "sha256": "...",
        "summary": "...", "assertions": ["..."], "source_files": ["..."],
        "provenance": {"source_commit": "...", "receipt_id": "..."}}, ...}}

Every entry must bind one pre-existing acceptance evidence name to a file in
the task workspace by SHA-256, include a substantive summary, and make a
non-empty assertion list. The external native command receipt separately
records the actual exit status of this verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_PATH = Path("planning/implementation-plan.json")
EVIDENCE_ROOT = Path("planning/evidence")

EXPECTED_TASK_IDS = [f"DS-{index:02d}" for index in range(15)]
EXPECTED_DEPENDENCIES = {
    "DS-01": {"DS-00"},
    "DS-02": {"DS-01"},
    "DS-03": {"DS-02"},
    "DS-04": {"DS-02"},
    "DS-05": {"DS-03", "DS-04"},
    "DS-06": {"DS-04", "DS-05"},
    "DS-07": {"DS-03", "DS-06"},
    "DS-08": {"DS-07"},
    "DS-09": {"DS-01", "DS-02"},
    "DS-10": {"DS-03"},
    "DS-11": {"DS-02", "DS-03"},
    "DS-12": {"DS-04", "DS-05", "DS-06", "DS-07", "DS-08", "DS-09", "DS-10", "DS-11"},
    "DS-13": {"DS-12"},
    "DS-14": {"DS-13"},
}

# These focused suites are deliberately tied to the acceptance workstream.
# DS-12..14 additionally require real evidence below; passing pytest alone is
# never sufficient for GPU qualification, cutover, or final acceptance.
TASK_TESTS: dict[str, tuple[str, ...]] = {
    "DS-01": ("tests/test_contracts.py", "tests/test_package_boundaries.py"),
    "DS-02": ("tests/test_application.py", "packages/device-scheduler-adapters/tests/test_ledger.py"),
    "DS-03": ("tests/test_scheduling.py", "tests/test_handoff.py", "packages/device-scheduler-core/tests"),
    "DS-04": ("tests/test_systemd.py",),
    "DS-05": ("tests/test_nvidia.py",),
    "DS-06": ("tests/test_backends.py",),
    "DS-07": ("tests/test_handoff.py", "tests/test_backends.py"),
    "DS-08": ("tests/test_frameshift_integration.py", "tests/test_frameshift_restart_qualification.py"),
    "DS-09": ("tests/test_api.py", "tests/test_auth.py", "tests/test_client.py", "tests/test_cli.py"),
    "DS-10": ("tests/test_scheduling.py", "tests/test_application.py"),
    "DS-11": ("tests/test_observability.py",),
    "DS-12": ("tests/test_integration_qualification.py", "tests/test_handoff.py", "tests/test_frameshift_restart_qualification.py"),
    "DS-13": ("packages/device-scheduler-adapters/tests/test_project_control_adapter.py", "tests/test_integration_qualification.py"),
    "DS-14": ("tests", "packages"),
}

TASK_PACKAGES: dict[str, tuple[str, ...]] = {
    "DS-01": ("device_scheduler_core",),
    "DS-02": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-03": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-04": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-05": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-06": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-07": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-08": ("device_scheduler_core", "device_scheduler_adapters"),
    "DS-09": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
    "DS-10": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
    "DS-11": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
    "DS-12": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
    "DS-13": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
    "DS-14": ("device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"),
}

TASK_WRITABLE_PACKAGES = {
    "DS-01": {"device_scheduler_core"},
    "DS-02": {"device_scheduler_core", "device_scheduler_adapters"},
    "DS-03": {"device_scheduler_core"},
    "DS-04": {"device_scheduler_adapters"},
    "DS-05": {"device_scheduler_adapters"},
    "DS-06": {"device_scheduler_adapters"},
    "DS-07": {"device_scheduler_core", "device_scheduler_adapters"},
    "DS-08": {"device_scheduler_adapters"},
    "DS-09": {"device_scheduler_core", "device_scheduler_mcp"},
    "DS-10": {"device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"},
    "DS-11": {"device_scheduler_core", "device_scheduler_adapters", "device_scheduler_mcp"},
}

PACKAGE_DIRS = {
    "device_scheduler_core": "packages/device-scheduler-core/src/device_scheduler_core",
    "device_scheduler_adapters": "packages/device-scheduler-adapters/src/device_scheduler_adapters",
    "device_scheduler_mcp": "packages/device-scheduler-mcp/src/device_scheduler_mcp",
}
PACKAGE_ROOTS = {
    "device_scheduler_core": "packages/device-scheduler-core",
    "device_scheduler_adapters": "packages/device-scheduler-adapters",
    "device_scheduler_mcp": "packages/device-scheduler-mcp",
}

SOURCE_IDENTITY_PATHS = (
    "/home/tumlinson/project-control/src/project_control/services/performance_probe.py",
    "/home/tumlinson/project-control/src/project_control/cli.py",
    "/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/controller.py",
    "/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/supervisor.py",
    "/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/service.py",
    "/home/tumlinson/project-control/src/project_control/host_resources/facade.py",
    "/home/tumlinson/project-control/src/project_control/host_resources/coordinator.py",
    "/home/tumlinson/project-control/src/project_control/native_providers.py",
    "/home/tumlinson/project-control/src/project_control/assistance/lab_gpu.py",
    "/home/tumlinson/project-control/src/project_control/as1_jobs.py",
    "/home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py",
)

SPECIAL_EVIDENCE = {
    "DS-12": {
        "gpu-handoff-e2e-receipt": {
            "truthy": (
                "real_gpu", "isolated_pool", "ordinary_suspend_weights_resident",
                "explicit_handoff_evicted", "reclamation_verified",
                "result_persisted_before_restore", "backend_ready_before_resume",
                "continuations_resumed",
            ),
            "required": (
                "host", "resource_pool_id", "resource_uuids", "handoff_id", "attempt_ids",
                "observed_at", "receipt_id", "daemon_source_commit", "production_source_hashes",
                "invocation_ids", "operation_receipt_ids",
            ),
            "equals": {"verdict": "pass", "simulated": False},
        },
        "qualification-report": {
            "required": ("source_commit", "observed_at", "receipt_id", "reviewer", "test_commands", "limitations"),
            "equals": {"verdict": "pass", "work_item_id": "DS-12"},
        },
    },
    "DS-13": {
        "consumer-migration-receipts": {
            "required": ("observed_at", "source_commit", "receipt_id", "consumer_ids", "resource_pool_ids"),
            "truthy": ("resource_allocation_migrated", "ownership_queries_migrated"),
            "equals": {"project_control_agent_lifecycle_migrated": False},
        },
        "allocator-fence-evidence": {
            "required": ("observed_at", "host", "resource_pool_id", "source_fingerprint", "receipt_id"),
            "truthy": ("old_admissions_fenced", "known_ownership_reconciled"),
        },
        "single-authority-audit": {
            "required": ("observed_at", "host", "resource_pool_id", "source_fingerprint", "receipt_id"),
            "truthy": ("device_scheduler_authoritative",),
            "equals": {"active_allocator_count": 1, "old_controller_grants": 0},
        },
        "legacy-controller-disabled-evidence": {
            "required": ("observed_at", "host", "source_fingerprint", "receipt_id"),
            "truthy": ("disabled", "runtime_source_fingerprint_verified"),
        },
    },
    "DS-14": {
        "final-acceptance-record": {
            "required": ("source_commit", "observed_at", "acceptance_gates", "authority_review", "receipt_id"),
            "truthy": ("all_required_gates_passed", "single_authority_verified", "reviewer"),
            "equals": {"work_item_id": "DS-14", "verdict": "accepted"},
        },
        "recovery-drill-record": {
            "required": ("observed_at", "host", "source_commit", "receipt_id"),
            "truthy": ("daemon_restart_passed", "host_reboot_recovery_assessed"),
        },
    },
}


def _read_text(root: Path, relative: str, errors: list[str]) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"missing or unreadable {relative}: {exc}")
        return ""


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"missing or invalid JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"expected JSON object at {path}")
        return None
    return value


def _validate_special_payload(task_id: str, name: str, payload: dict[str, Any], errors: list[str]) -> None:
    spec = SPECIAL_EVIDENCE.get(task_id, {}).get(name)
    if spec is None:
        return
    for key in spec.get("required", ()):
        if key not in payload or payload[key] in (None, "", [], {}):
            errors.append(f"{task_id} {name} must include provenance field {key}")
    for key in spec.get("truthy", ()):
        if not payload.get(key):
            errors.append(f"{task_id} {name} must verify {key}=true")
    for key, value in spec.get("equals", {}).items():
        if payload.get(key) != value:
            errors.append(f"{task_id} {name} must verify {key}={value!r}")
    observed = payload.get("observed_at")
    if observed is not None:
        try:
            parsed_observed = datetime.fromisoformat(observed.replace("Z", "+00:00")) if isinstance(observed, str) else None
            if parsed_observed is None or parsed_observed.tzinfo is None:
                raise ValueError("timezone-qualified timestamp required")
        except ValueError:
            errors.append(f"{task_id} {name} observed_at must be timezone-qualified")
    if name == "gpu-handoff-e2e-receipt":
        uuids = payload.get("resource_uuids")
        if not isinstance(uuids, list) or not uuids or any(not isinstance(value, str) or not value.startswith("GPU-") for value in uuids):
            errors.append("DS-12 GPU receipt must identify real NVIDIA GPU UUIDs")
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(payload.get("daemon_source_commit", ""))):
            errors.append("DS-12 GPU receipt must bind the daemon source commit")


def _validate_production_source_hashes(
    hashes: Any, current_files: dict[str, str], errors: list[str]
) -> None:
    if not isinstance(hashes, dict) or not hashes:
        errors.append("DS-12 GPU receipt must bind production_source_hashes")
        return
    for source_path, digest in hashes.items():
        if source_path not in current_files:
            errors.append(f"DS-12 GPU receipt source is outside its workspace snapshot: {source_path}")
        elif current_files[source_path] != digest:
            errors.append(f"DS-12 GPU receipt source hash differs from current bytes: {source_path}")


def _expected_requirements(item: dict[str, Any]) -> set[str]:
    return set(item["acceptance"]["required_evidence"])


def verify_ds00(root: Path) -> list[str]:
    """Validate design content, dated inventory, source identities and plan graph."""
    errors: list[str] = []
    docs = {
        "planning/product-spec.md": (
            "Project Control remains responsible", "FrameShift retains", "whole-device exclusivity",
            "ordinary suspension", "round-trip reservation", "systemd user service units",
        ),
        "planning/contracts.md": (
            "## DS-00 baseline: current host and consumer inventory", "2026-10-21",
            "V100-SXM2-16GB", "0 MiB", "GPU-21131915", "GPU-cf22c41f",
            "GPU-d745da9c", "GPU-6c1cac7f", "historical", "does not establish",
            "Shared GPU admission is the default", "ordinary FrameShift suspension",
            "feasible round-trip reservation", "TODO_BACKGROUND_HOST_RUNTIME_DIR",
        ),
        "planning/capability-matrix.md": (
            "Current evidence", "Pending", "explicit authorized finite request", "CUDA controller retirement",
        ),
        "planning/acceptance.md": (
            "Request–suspend–evict–run–restart journey", "exactly one active allocator",
            "host-reboot recovery", "unmanaged work",
        ),
        "planning/README.md": (
            "Product specification", "Contracts", "Capability matrix", "Acceptance",
            "Project Control owns", "FrameShift owns",
        ),
    }
    contents: dict[str, str] = {}
    for rel, required in docs.items():
        text = _read_text(root, rel, errors)
        contents[rel] = text
        if len(text.strip()) < 700:
            errors.append(f"{rel} is unexpectedly short for an accepted design document")
        for phrase in required:
            if phrase.casefold() not in text.casefold():
                errors.append(f"{rel} is missing required design content: {phrase!r}")

    contracts = contents.get("planning/contracts.md", "")
    for source in SOURCE_IDENTITY_PATHS:
        if source not in contracts:
            errors.append(f"consumer/source identity inventory omits {source}")
    hashes = re.findall(r"\|\s*`?(/[^|`]+)`?\s*\|\s*`?([0-9a-f]{64})`?\s*\|", contracts)
    hash_map = {path.strip(): digest for path, digest in hashes}
    for source in SOURCE_IDENTITY_PATHS:
        if source not in hash_map:
            errors.append(f"consumer/source identity inventory lacks a SHA-256 for {source}")
    if not re.search(r"Project Control checkout was at Git commit `[0-9a-f]{40}`", contracts):
        errors.append("contracts.md must bind the Project Control source snapshot to a full commit id")
    if not re.search(r"todo-orchestrator` version `[^`]+`.*?fingerprint `[0-9a-f]{64}`", contracts, re.S):
        errors.append("contracts.md must record the Todo runtime package version and source fingerprint")
    if "does not establish the package/source identity selected by any live controller invocation" not in contracts:
        errors.append("contracts.md must retain the explicit uncertainty about live controller binding")

    plan = _load_json(root / PLAN_PATH, errors)
    if plan is None:
        return errors
    if plan.get("format") != "todo-orchestrator.plan" or plan.get("schema_version") != 1:
        errors.append("implementation plan must remain Todo Orchestrator schema version 1")
    project = plan.get("project", {})
    if project != {
        "project_id": "5c878e94-2d97-4902-b594-9adb76c3e7fe",
        "authority_id": "9b19daf8-d7c5-5e56-8b9f-df3630d94961",
    }:
        errors.append("native plan project/authority identity changed")
    items = plan.get("work_items", [])
    if [item.get("id") for item in items] != EXPECTED_TASK_IDS:
        errors.append("native plan must preserve ordered work items DS-00 through DS-14")
    by_id = {item.get("id"): item for item in items}
    if len(by_id) != 15:
        errors.append("native plan must contain exactly fifteen distinct work items")

    actual_dependencies: dict[str, set[str]] = {}
    for dependency in plan.get("dependencies", []):
        source = dependency.get("source", {}).get("entity_id")
        members = dependency.get("members", [])
        targets = {
            member.get("target", {}).get("entity_id")
            for member in members
            if member.get("relation") == "completion"
        }
        if source:
            actual_dependencies[source] = targets
    if actual_dependencies != EXPECTED_DEPENDENCIES:
        errors.append("native plan dependency graph differs from the accepted DS-00..DS-14 sequence")

    requirements = plan.get("requirements", [])
    requirements_by_work: dict[str, list[dict[str, Any]]] = {}
    for requirement in requirements:
        requirements_by_work.setdefault(requirement.get("work_item_id", ""), []).append(requirement)
    if set(requirements_by_work) != set(EXPECTED_TASK_IDS):
        errors.append("every DS work item must have a formal verification requirement")
    if len(requirements) != 15:
        errors.append("native plan must contain one auditable verification requirement per work item")

    for task_id in EXPECTED_TASK_IDS:
        item = by_id.get(task_id)
        task_reqs = requirements_by_work.get(task_id, [])
        if not item or len(task_reqs) != 1:
            continue
        requirement = task_reqs[0]
        expected_id = f"{task_id}-native-evidence"
        if requirement.get("id") != expected_id or requirement.get("check_key") != expected_id:
            errors.append(f"{task_id} verification requirement ID/check_key is not stable")
        if requirement.get("kind") != "check" or requirement.get("work_item_id") != task_id:
            errors.append(f"{task_id} must use a check requirement scoped to itself")
        expected_evidence = _expected_requirements(item)
        actual_evidence = {
            entry.get("name") for entry in requirement.get("required_evidence", [])
            if entry.get("kind") == "command_result" and entry.get("version") == 1
        }
        if actual_evidence != expected_evidence:
            errors.append(f"{task_id} formal evidence requirements do not exactly preserve its accepted names")
        verification = requirement.get("verification", {})
        config = verification.get("config", {})
        argv = ["python3", "-B", "scripts/verify_native_evidence.py", task_id]
        if verification.get("kind") != "command" or verification.get("version") != 1:
            errors.append(f"{task_id} verification must use the version-1 command contract")
        if config.get("argv") != argv or config.get("cwd") != ".":
            errors.append(f"{task_id} verifier must run the task-specific source-bound command")
        if config.get("expected_exit_codes") != [0] or config.get("timeout_seconds") != 60:
            errors.append(f"{task_id} verifier must use bounded timeout and require success")
        group = item.get("requirement_groups", [])
        if group != [{"kind": "check", "mode": "all", "members": [expected_id]}]:
            errors.append(f"{task_id} must require its formal check through an all-check group")

    # Ensure script and test files themselves are included in DS-00's owned
    # scope, then read-only in all later task scopes.
    for task_id in EXPECTED_TASK_IDS:
        item = by_id.get(task_id)
        if not item:
            continue
        paths = {entry.get("path"): entry.get("access") for entry in item.get("scope", {}).get("paths", [])}
        required = {"scripts/verify_native_evidence.py": "write" if task_id == "DS-00" else "read"}
        if task_id == "DS-00":
            required["tests/test_native_evidence.py"] = "write"
        for path, access in required.items():
            if paths.get(path) != access:
                errors.append(f"{task_id} scope must grant {access} access to {path}")
        for package in TASK_PACKAGES.get(task_id, ()):
            package_path = PACKAGE_DIRS[package]
            declared = paths.get(package_path)
            if declared is None:
                declared = next((mode for path, mode in paths.items() if package_path.startswith(path.rstrip("/") + "/")), None)
            broad_write = paths.get("packages") == "write"
            expected_access = "write" if package in TASK_WRITABLE_PACKAGES.get(task_id, set()) or broad_write else "read"
            if declared != expected_access:
                errors.append(f"{task_id} scope must grant {expected_access} access to tested package {package_path}")
        if task_id != "DS-00":
            for config_path in ("pyproject.toml", "uv.lock"):
                if paths.get(config_path) not in {"read", "write"}:
                    errors.append(f"{task_id} scope must grant read access to test configuration {config_path}")
        for test_path in TASK_TESTS.get(task_id, ()):
            declared = paths.get(test_path)
            if declared is None:
                declared = next((mode for path, mode in paths.items() if test_path.startswith(path.rstrip("/") + "/")), None)
            if declared not in {"read", "write"}:
                errors.append(f"{task_id} scope must grant read access to focused test path {test_path}")
        evidence_dir = f"planning/evidence/{task_id}"
        if task_id != "DS-00" and not any(path == "planning" or evidence_dir.startswith(path.rstrip("/") + "/") or path == evidence_dir for path in paths):
            errors.append(f"{task_id} scope must include {evidence_dir} for evidence records")

    return errors


def _manifest(root: Path, task_id: str, expected: set[str], errors: list[str]) -> dict[str, dict[str, Any]]:
    manifest_path = root / EVIDENCE_ROOT / task_id / "manifest.json"
    manifest = _load_json(manifest_path, errors)
    if manifest is None:
        return {}
    if manifest.get("schema_version") != 1 or manifest.get("work_item_id") != task_id:
        errors.append(f"{manifest_path} has an unsupported schema or work-item identity")
    verified_at = manifest.get("verified_at")
    try:
        parsed_at = datetime.fromisoformat(verified_at.replace("Z", "+00:00")) if isinstance(verified_at, str) else None
        if parsed_at is None or parsed_at.tzinfo is None:
            raise ValueError("verified_at must be a timezone-qualified ISO timestamp")
    except ValueError as exc:
        errors.append(f"{manifest_path}: {exc}")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        errors.append(f"{manifest_path}: source_commit must be a full Git object id")
    snapshot = manifest.get("source_snapshot")
    expected_files = _source_file_digests(root, task_id, errors)
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit):
        _validate_source_bundle(root, commit, expected_files, manifest.get("source_bundle"), errors)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("files"), dict):
        errors.append(f"{manifest_path}: source_snapshot.files is required")
    else:
        supplied_files = snapshot["files"]
        if supplied_files != expected_files:
            errors.append(f"{manifest_path}: source_snapshot file list/digests do not match this task workspace")
        canonical = json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        actual_snapshot = hashlib.sha256(canonical).hexdigest()
        if snapshot.get("sha256") != actual_snapshot:
            errors.append(f"{manifest_path}: source_snapshot SHA-256 does not match current source/test bytes")
    expected_test_command = ["-m", "pytest", "-q", *TASK_TESTS.get(task_id, ())]
    if manifest.get("test_command") != expected_test_command:
        errors.append(f"{manifest_path}: test_command does not match this work item's focused pytest contract")
    if manifest.get("test_python") != str(Path(sys.executable).absolute()):
        errors.append(f"{manifest_path}: test_python must match the interpreter running this verifier")
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        errors.append(f"{manifest_path}: execution provenance is required")
    else:
        if execution.get("argv") != ["python", *expected_test_command] or execution.get("exit_code") != 0:
            errors.append(f"{manifest_path}: execution provenance must record the exact successful focused test command")
        if execution.get("source_commit") != commit or not isinstance(execution.get("receipt_id"), str) or not execution.get("receipt_id"):
            errors.append(f"{manifest_path}: execution provenance must bind a receipt to the source commit")
        for timestamp_name in ("started_at", "finished_at"):
            timestamp = execution.get(timestamp_name)
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if isinstance(timestamp, str) else None
                if parsed is None or parsed.tzinfo is None:
                    raise ValueError("timezone-qualified timestamp required")
            except ValueError:
                errors.append(f"{manifest_path}: execution.{timestamp_name} must be timezone-qualified")
        receipt_rel = execution.get("receipt_path")
        receipt_digest = execution.get("receipt_sha256")
        if not isinstance(receipt_rel, str) or Path(receipt_rel).is_absolute() or ".." in Path(receipt_rel).parts:
            errors.append(f"{manifest_path}: execution.receipt_path must be workspace-relative")
        else:
            receipt_path = root / receipt_rel
            try:
                if not receipt_path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("execution receipt resolves outside workspace")
                receipt_bytes = receipt_path.read_bytes()
                receipt = json.loads(receipt_bytes)
                if hashlib.sha256(receipt_bytes).hexdigest() != receipt_digest:
                    errors.append(f"{manifest_path}: execution receipt SHA-256 does not match")
                if not isinstance(receipt, dict) or receipt.get("receipt_id") != execution.get("receipt_id"):
                    errors.append(f"{manifest_path}: execution receipt identity does not match")
                else:
                    unsigned_receipt = {key: value for key, value in receipt.items() if key != "receipt_id"}
                    if hashlib.sha256(_canonical_json(unsigned_receipt)).hexdigest() != receipt.get("receipt_id"):
                        errors.append(f"{manifest_path}: execution receipt id is not its content digest")
                    if receipt.get("exit_code") != 0 or receipt.get("source_commit") != commit:
                        errors.append(f"{manifest_path}: execution receipt does not record successful tests for source_commit")
                    captured_command = receipt.get("command")
                    if (
                        receipt.get("interpreter") != manifest.get("test_python")
                        or not isinstance(captured_command, list)
                        or captured_command[1:] != expected_test_command
                    ):
                        errors.append(f"{manifest_path}: execution receipt command/interpreter do not match the manifest")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{manifest_path}: execution receipt unavailable or invalid: {exc}")
    records = manifest.get("evidence")
    if not isinstance(records, dict) or set(records) != expected:
        errors.append(f"{manifest_path} must bind exactly these evidence names: {', '.join(sorted(expected))}")
        return {}
    root_real = root.resolve()
    for name, record in records.items():
        if not isinstance(record, dict):
            errors.append(f"{manifest_path}: {name} entry must be an object")
            continue
        rel = record.get("path")
        digest = record.get("sha256")
        summary = record.get("summary")
        assertions = record.get("assertions")
        provenance = record.get("provenance")
        source_refs = record.get("source_files")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            errors.append(f"{manifest_path}: {name} path must be a safe workspace-relative path")
            continue
        artifact = root / rel
        try:
            if not artifact.resolve().is_relative_to(root_real):
                raise ValueError("path resolves outside workspace")
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            errors.append(f"{manifest_path}: {name} artifact unavailable or unsafe: {exc}")
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or actual != digest:
            errors.append(f"{manifest_path}: {name} SHA-256 does not match artifact bytes")
        if not isinstance(summary, str) or len(summary.strip()) < 40:
            errors.append(f"{manifest_path}: {name} needs a substantive summary")
        if not isinstance(assertions, list) or not assertions or any(not isinstance(v, str) or len(v.strip()) < 12 for v in assertions):
            errors.append(f"{manifest_path}: {name} needs substantive assertion descriptions")
        if not isinstance(provenance, dict) or provenance.get("source_commit") != commit or not isinstance(provenance.get("receipt_id"), str) or not provenance.get("receipt_id"):
            errors.append(f"{manifest_path}: {name} must bind evidence provenance to the worktree commit and a receipt ID")
        if not isinstance(source_refs, list) or not source_refs or any(path not in expected_files for path in source_refs):
            errors.append(f"{manifest_path}: {name} must reference source files in the hashed workspace snapshot")
        if name in SPECIAL_EVIDENCE.get(task_id, {}):
            payload = _load_json(artifact, errors)
            if payload is None:
                continue
            _validate_special_payload(task_id, name, payload, errors)
            if task_id == "DS-12" and name == "gpu-handoff-e2e-receipt":
                _validate_production_source_hashes(payload.get("production_source_hashes"), expected_files, errors)
            provenance = record.get("provenance", {})
            for payload_key, provenance_key in (
                ("receipt_id", "receipt_id"),
                ("source_commit", "source_commit"),
                ("daemon_source_commit", "source_commit"),
                ("observed_at", "observed_at"),
            ):
                if payload.get(payload_key) is not None and provenance.get(provenance_key) != payload[payload_key]:
                    errors.append(f"{task_id} {name} manifest provenance does not match receipt field {payload_key}")
    return records


def _run_tests(root: Path, task_id: str, errors: list[str]) -> None:
    test_paths = TASK_TESTS.get(task_id)
    if not test_paths:
        errors.append(f"no focused test contract is registered for {task_id}")
        return
    missing = [path for path in test_paths if not (root / path).exists()]
    if missing:
        errors.extend(f"{task_id} test contract missing path: {path}" for path in missing)
        return
    python = Path(sys.executable).absolute()
    if not python.is_absolute() or not python.is_file():
        errors.append("the interpreter running the verifier must be an existing absolute path")
        return
    package_paths = []
    for package in TASK_PACKAGES.get(task_id, ()):
        package_root = root / PACKAGE_DIRS[package]
        if not package_root.is_dir():
            errors.append(f"{task_id} workspace lacks its required tested package source: {PACKAGE_DIRS[package]}")
            return
        package_paths.append(str(package_root.parent))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([*package_paths, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # Confirm Python resolves every package from this claimed task workspace.
    probe = "import importlib.util,json; names=" + repr(list(TASK_PACKAGES[task_id])) + "; print(json.dumps({n: (s.origin if (s:=importlib.util.find_spec(n)) else None) for n in names}))"
    try:
        resolved = subprocess.run([str(python), "-c", probe], cwd=root, env=env, text=True, capture_output=True, timeout=10, check=False)
        origins = json.loads(resolved.stdout) if resolved.returncode == 0 else {}
        root_real = root.resolve()
        origin_mismatch = False
        for package, origin in origins.items():
            if not origin or not Path(origin).resolve().is_relative_to(root_real):
                errors.append(f"{task_id} import {package} resolves outside the task workspace: {origin!r}")
                origin_mismatch = True
        if resolved.returncode != 0:
            errors.append(f"{task_id} trusted interpreter import probe failed: {(resolved.stdout + resolved.stderr)[-2000:]}")
            return
        if origin_mismatch:
            return
        completed = subprocess.run([str(python), *probe_args(test_paths)], cwd=root, env=env, text=True, capture_output=True, timeout=50, check=False)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        errors.append(f"focused pytest invocation failed for {task_id}: {exc}")
        return
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-5000:]
        errors.append(f"focused pytest suite failed for {task_id} (exit {completed.returncode}):\n{detail}")


def probe_args(test_paths: tuple[str, ...]) -> list[str]:
    return ["-m", "pytest", "-q", *test_paths]


def _source_file_digests(root: Path, task_id: str, errors: list[str]) -> dict[str, str]:
    paths: set[Path] = {Path("scripts/verify_native_evidence.py")}
    if task_id == "DS-00":
        paths.add(Path("tests/test_native_evidence.py"))
    else:
        paths.update({Path("pyproject.toml"), Path("uv.lock")})
    for test_path in TASK_TESTS.get(task_id, ()):
        candidate = root / test_path
        if candidate.is_dir():
            paths.update(path.relative_to(root) for path in candidate.rglob("*.py") if "__pycache__" not in path.parts)
        elif candidate.is_file():
            paths.add(Path(test_path))
    for package in TASK_PACKAGES.get(task_id, ()):
        package_root = root / PACKAGE_ROOTS[package]
        pyproject = package_root / "pyproject.toml"
        if pyproject.is_file():
            paths.add(pyproject.relative_to(root))
        candidate = root / PACKAGE_DIRS[package]
        paths.update(path.relative_to(root) for path in candidate.rglob("*.py") if "__pycache__" not in path.parts)
    result: dict[str, str] = {}
    root_real = root.resolve()
    for relative in sorted(paths):
        path = root / relative
        try:
            if not path.resolve().is_relative_to(root_real):
                raise ValueError("source file resolves outside task workspace")
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            errors.append(f"source snapshot file unavailable {relative}: {exc}")
    return result


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, timeout=10, check=True
    ).stdout.strip()


def _validate_source_commit(root: Path, commit: str, source_files: dict[str, str], errors: list[str]) -> None:
    """Require the recorded source commit to be an ancestor containing these bytes.

    A manifest is commonly committed after the code and test sources. Requiring
    equality with HEAD would make that evidence-only commit invalidate itself.
    """
    try:
        head = _git_head(root)
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"cannot verify source commit ancestry: {exc}")
        return
    if ancestry.returncode != 0:
        errors.append("source_commit must be an ancestor of the current task worktree HEAD")
        return
    _validate_committed_files(root, commit, source_files, errors)


def _validate_committed_files(root: Path, commit: str, source_files: dict[str, str], errors: list[str]) -> None:
    for relative, expected_digest in source_files.items():
        try:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=root,
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"source_commit does not contain {relative}: {exc}")
            continue
        if hashlib.sha256(committed).hexdigest() != expected_digest:
            errors.append(f"task source/test drift from source_commit at {relative}")


def _validate_source_bundle(
    root: Path, commit: str, source_files: dict[str, str], record: Any, errors: list[str]
) -> None:
    """Verify the committed source without exposing the host's Git directory.

    Native command isolation mounts only the task worktree, whose .git file
    points outside that mount. A hash-bound bundle supplies the actual commit
    and objects. The host capture checks ancestry; native source qualification
    separately binds this verifier's current worktree. Both paths require the
    exact tested source bytes, including the verifier itself.
    """
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        errors.append("source_bundle must bind a workspace-relative Git bundle and SHA-256")
        return
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        errors.append("source_bundle path is unsafe")
        return
    bundle = root / relative
    try:
        if not bundle.resolve().is_relative_to(root.resolve()) or bundle.is_symlink():
            raise ValueError("bundle resolves outside the task workspace or is a symlink")
        if bundle.stat().st_size > 64 * 1024**2:
            raise ValueError("bundle exceeds 64 MiB verification limit")
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != record.get("sha256"):
            raise ValueError("bundle SHA-256 mismatch")
        with tempfile.TemporaryDirectory(prefix="device-scheduler-source-proof-") as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "--bare", "-q", str(repository)],
                           capture_output=True, check=True, timeout=10)
            subprocess.run(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                            "fetch", "--no-tags", "--no-write-fetch-head", str(bundle.resolve()),
                            "HEAD:refs/heads/source"], capture_output=True, check=True, timeout=15)
            if _git_head_ref(repository, "refs/heads/source") != commit:
                raise ValueError("bundle HEAD differs from source_commit")
            _validate_committed_files(repository, commit, source_files, errors)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f"cannot verify source bundle: {exc}")


def _git_head_ref(root: Path, ref: str) -> str:
    return subprocess.run(["git", "rev-parse", ref], cwd=root, text=True,
                          capture_output=True, check=True, timeout=10).stdout.strip()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def capture_manifest(root: Path, task_id: str) -> list[str]:
    """Run a task's focused tests and build its hash-bound evidence manifest.

    Human-authored evidence summaries and paths live in manifest-input.json;
    command outcomes, source hashes and artifact hashes are measured here.
    This helper does not certify physical claims in evidence payloads.
    """
    errors: list[str] = []
    if task_id not in set(EXPECTED_TASK_IDS) - {"DS-00"}:
        return ["capture supports DS-01 through DS-14"]
    evidence_dir = root / EVIDENCE_ROOT / task_id
    if any((evidence_dir / name).exists() for name in ("manifest.json", "pytest-run.json", "source.bundle")):
        return ["capture outputs already exist; preserve them and use a fresh task evidence capture"]
    input_path = evidence_dir / "manifest-input.json"
    inputs = _load_json(input_path, errors)
    if inputs is None:
        return errors
    if inputs.get("schema_version") != 1 or inputs.get("work_item_id") != task_id:
        errors.append(f"{input_path}: schema/work-item identity mismatch")
    evidence_input = inputs.get("evidence")
    plan = _load_json(root / PLAN_PATH, errors)
    item = next((row for row in (plan or {}).get("work_items", []) if row.get("id") == task_id), None)
    expected = _expected_requirements(item) if item else set()
    if not isinstance(evidence_input, dict) or set(evidence_input) != expected:
        errors.append(f"{input_path}: evidence must define exactly {', '.join(sorted(expected))}")
        return errors
    for name, record in evidence_input.items():
        if not isinstance(record, dict):
            errors.append(f"{input_path}: {name} must be an object")
            continue
        if not isinstance(record.get("path"), str) or not isinstance(record.get("summary"), str):
            errors.append(f"{input_path}: {name} needs path and summary")
        if not isinstance(record.get("assertions"), list) or not record["assertions"]:
            errors.append(f"{input_path}: {name} needs authored assertions")
        if not isinstance(record.get("source_files"), list) or not record["source_files"]:
            errors.append(f"{input_path}: {name} needs source_files")
    if errors:
        return errors

    python = Path(sys.executable).absolute()
    if not python.is_absolute() or not python.is_file():
        return ["the interpreter running this helper must be an existing absolute path"]
    test_paths = TASK_TESTS[task_id]
    missing = [path for path in test_paths if not (root / path).exists()]
    if missing:
        return [f"{task_id} test contract missing path: {path}" for path in missing]
    package_paths = []
    for package in TASK_PACKAGES[task_id]:
        package_root = root / PACKAGE_DIRS[package]
        if not package_root.is_dir():
            return [f"{task_id} workspace lacks required package source: {PACKAGE_DIRS[package]}"]
        package_paths.append(str(package_root.parent))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([*package_paths, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    probe = "import importlib.util,json; names=" + repr(list(TASK_PACKAGES[task_id])) + "; print(json.dumps({n: (s.origin if (s:=importlib.util.find_spec(n)) else None) for n in names}))"
    try:
        resolved = subprocess.run([str(python), "-c", probe], cwd=root, env=env, text=True, capture_output=True, timeout=10, check=False)
        origins = json.loads(resolved.stdout) if resolved.returncode == 0 else {}
        if resolved.returncode != 0 or any(
            not origin or not Path(origin).resolve().is_relative_to(root.resolve()) for origin in origins.values()
        ):
            return [f"{task_id} imports do not resolve exclusively from the task workspace"]
        source_files = _source_file_digests(root, task_id, errors)
        if errors:
            return errors
        commit = _git_head(root)
        _validate_source_commit(root, commit, source_files, errors)
        if errors:
            return errors
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        command = [str(python), *probe_args(test_paths)]
        completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=50, check=False)
        finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return [f"cannot capture focused test result: {exc}"]
    if completed.returncode != 0:
        return [f"focused pytest suite failed for {task_id} (exit {completed.returncode}):\n{(completed.stdout + completed.stderr)[-5000:]}"]

    snapshot_hash = hashlib.sha256(_canonical_json(source_files)).hexdigest()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "work_item_id": task_id,
        "command": command,
        "interpreter": str(python),
        "cwd": ".",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "source_commit": commit,
        "source_snapshot_sha256": snapshot_hash,
        "started_at": started,
        "finished_at": finished,
    }
    receipt["receipt_id"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    receipt_path = evidence_dir / "pytest-run.json"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = evidence_dir / "source.bundle"
    try:
        subprocess.run(["git", "bundle", "create", str(bundle_path.resolve()), "HEAD"],
                       cwd=root, capture_output=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"cannot capture committed source bundle: {exc}"]
    receipt_bytes = _canonical_json(receipt) + b"\n"
    receipt_path.write_bytes(receipt_bytes)

    evidence_records: dict[str, Any] = {}
    for name, record in evidence_input.items():
        artifact = root / record["path"]
        try:
            if not artifact.resolve().is_relative_to(root.resolve()):
                raise ValueError("artifact resolves outside task workspace")
            artifact_bytes = artifact.read_bytes()
            payload = json.loads(artifact_bytes) if artifact.suffix == ".json" else {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return [f"cannot capture {name} artifact: {exc}"]
        provenance = {
            "source_commit": payload.get("source_commit", commit),
            "receipt_id": payload.get("receipt_id", receipt["receipt_id"]),
        }
        if payload.get("observed_at") is not None:
            provenance["observed_at"] = payload["observed_at"]
        evidence_records[name] = {
            "path": record["path"],
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "summary": record["summary"],
            "assertions": record["assertions"],
            "source_files": record["source_files"],
            "provenance": provenance,
        }
    manifest = {
        "schema_version": 1,
        "work_item_id": task_id,
        "verified_at": finished,
        "source_commit": commit,
        "source_bundle": {"path": bundle_path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()},
        "test_python": str(python),
        "test_command": probe_args(test_paths),
        "execution": {
            "argv": ["python", *probe_args(test_paths)],
            "exit_code": completed.returncode,
            "source_commit": commit,
            "receipt_id": receipt["receipt_id"],
            "started_at": started,
            "finished_at": finished,
            "receipt_path": receipt_path.relative_to(root).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        "source_snapshot": {"sha256": snapshot_hash, "files": source_files},
        "evidence": evidence_records,
    }
    (evidence_dir / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
    return []


def verify_task(root: Path, task_id: str) -> list[str]:
    if task_id == "DS-00":
        # Bootstrap verification is standard-library-only. Its regression
        # suite is run during development, not imposed on the native provider.
        return verify_ds00(root)

    errors: list[str] = []
    plan = _load_json(root / PLAN_PATH, errors)
    if plan is None:
        return errors
    item = next((entry for entry in plan.get("work_items", []) if entry.get("id") == task_id), None)
    if item is None:
        return [f"unknown work item {task_id!r}"]
    expected = _expected_requirements(item)
    _manifest(root, task_id, expected, errors)
    _run_tests(root, task_id, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 2 and args[0] == "capture" and args[1] in EXPECTED_TASK_IDS[1:]:
        errors = capture_manifest(Path.cwd(), args[1])
        if errors:
            for error in errors:
                print(f"FAIL: {error}", file=sys.stderr)
            return 1
        print(f"CAPTURED: {args[1]} source-bound test and evidence manifest")
        return 0
    if len(args) != 1 or args[0] not in EXPECTED_TASK_IDS:
        print("usage: verify_native_evidence.py DS-00..DS-14 | capture DS-01..DS-14", file=sys.stderr)
        return 2
    root = Path.cwd()
    errors = verify_task(root, args[0])
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {args[0]} native evidence and focused checks verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
