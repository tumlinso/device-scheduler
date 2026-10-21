from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import scripts.verify_native_evidence as verifier
from scripts.verify_native_evidence import (
    _validate_production_source_hashes,
    _validate_source_commit,
    _validate_special_payload,
    verify_ds00,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_FILES = (
    "product-spec.md",
    "contracts.md",
    "capability-matrix.md",
    "acceptance.md",
    "README.md",
    "implementation-plan.json",
)


def _baseline_copy(target: Path) -> Path:
    planning = target / "planning"
    planning.mkdir(parents=True)
    for name in BASELINE_FILES:
        shutil.copyfile(ROOT / "planning" / name, planning / name)
    return target


def test_current_design_baseline_has_source_bound_native_checks() -> None:
    assert verify_ds00(ROOT) == []


def test_ds00_rejects_missing_consumer_source_digest(tmp_path: Path) -> None:
    root = _baseline_copy(tmp_path)
    contracts = root / "planning/contracts.md"
    text = contracts.read_text(encoding="utf-8")
    text = text.replace("`9fda247e594a20520e974c4eff1cfc5b65715a476b1d55007585e06074614efa`", "`missing-digest`")
    contracts.write_text(text, encoding="utf-8")

    errors = verify_ds00(root)

    assert any("lacks a SHA-256" in error for error in errors)


def test_ds00_rejects_changed_accepted_dependency_graph(tmp_path: Path) -> None:
    root = _baseline_copy(tmp_path)
    plan_path = root / "planning/implementation-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dependency = next(item for item in plan["dependencies"] if item["source"]["entity_id"] == "DS-12")
    dependency["members"] = dependency["members"][:-1]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    errors = verify_ds00(root)

    assert any("dependency graph" in error for error in errors)


def test_ds00_rejects_absent_formal_evidence_requirements(tmp_path: Path) -> None:
    root = _baseline_copy(tmp_path)
    plan_path = root / "planning/implementation-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.pop("requirements", None)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    errors = verify_ds00(root)

    assert any("formal verification requirement" in error for error in errors)
    assert any("one auditable verification requirement" in error for error in errors)


def test_gpu_handoff_gate_rejects_simulation_and_missing_hardware_provenance() -> None:
    errors: list[str] = []
    _validate_special_payload(
        "DS-12",
        "gpu-handoff-e2e-receipt",
        {
            "verdict": "pass",
            "simulated": True,
            "real_gpu": False,
            "isolated_pool": True,
        },
        errors,
    )

    assert any("simulated=False" in error for error in errors)
    assert any("resource_uuids" in error for error in errors)
    assert any("real NVIDIA GPU UUIDs" in error for error in errors)
    assert any("daemon source commit" in error for error in errors)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def _temporary_source_repo(root: Path) -> tuple[str, dict[str, str]]:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Native evidence test")
    _git(root, "config", "user.email", "native-evidence-test@example.invalid")
    source = root / "packages/device-scheduler-core/src/core.py"
    test = root / "packages/device-scheduler-core/tests/test_core.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_text("answer = 42\n", encoding="utf-8")
    test.write_text("def test_answer():\n    assert 42 == 42\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "commit source and tests")
    source_commit = _git(root, "rev-parse", "HEAD")
    snapshot = {
        source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest(),
        test.relative_to(root).as_posix(): hashlib.sha256(test.read_bytes()).hexdigest(),
    }
    return source_commit, snapshot


def test_source_commit_allows_later_evidence_only_commit(tmp_path: Path) -> None:
    commit, snapshot = _temporary_source_repo(tmp_path)
    evidence = tmp_path / "planning/evidence/DS-01/receipt.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"verdict":"pass"}\n', encoding="utf-8")
    _git(tmp_path, "add", "planning/evidence/DS-01/receipt.json")
    _git(tmp_path, "commit", "-qm", "add evidence")

    errors: list[str] = []
    _validate_source_commit(tmp_path, commit, snapshot, errors)

    assert errors == []


def test_source_commit_rejects_source_drift_after_evidence_commit(tmp_path: Path) -> None:
    commit, _snapshot = _temporary_source_repo(tmp_path)
    source = tmp_path / "packages/device-scheduler-core/src/core.py"
    source.write_text("answer = 43\n", encoding="utf-8")
    _git(tmp_path, "add", "packages/device-scheduler-core/src/core.py")
    _git(tmp_path, "commit", "-qm", "change source")

    errors: list[str] = []
    current_snapshot = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in (
            "packages/device-scheduler-core/src/core.py",
            "packages/device-scheduler-core/tests/test_core.py",
        )
    }
    _validate_source_commit(tmp_path, commit, current_snapshot, errors)

    assert any("source/test drift" in error for error in errors)


def test_source_bundle_verifies_without_host_git_and_rejects_drift(tmp_path: Path) -> None:
    commit, snapshot = _temporary_source_repo(tmp_path)
    bundle = tmp_path / "source.bundle"
    _git(tmp_path, "bundle", "create", str(bundle), "HEAD")
    record = {"path": "source.bundle", "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    # Reproduce the native mount boundary: Git's host metadata is inaccessible.
    (tmp_path / ".git").rename(tmp_path / "unmounted-host-metadata")
    errors: list[str] = []
    verifier._validate_source_bundle(tmp_path, commit, snapshot, record, errors)
    assert errors == []

    relative = "packages/device-scheduler-core/src/core.py"
    (tmp_path / relative).write_text("answer = 43\n", encoding="utf-8")
    changed = {**snapshot, relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()}
    verifier._validate_source_bundle(tmp_path, commit, changed, record, errors)
    assert any("source/test drift" in error for error in errors)

    errors.clear()
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    verifier._validate_source_bundle(tmp_path, commit, snapshot, record, errors)
    assert any("SHA-256 mismatch" in error for error in errors)


def test_gpu_receipt_binds_named_production_bytes_without_whole_test_snapshot() -> None:
    production = {"packages/device-scheduler-core/src/device_scheduler_core/allocator.py": "a" * 64}
    errors: list[str] = []
    _validate_production_source_hashes(production, production, errors)
    assert errors == []

    _validate_production_source_hashes(
        {"packages/device-scheduler-core/src/device_scheduler_core/allocator.py": "b" * 64},
        production,
        errors,
    )
    assert any("source hash differs from current bytes" in error for error in errors)


def test_capture_helper_records_actual_pytest_receipt_and_manifest(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    (root / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/verify_native_evidence.py", root / "scripts/verify_native_evidence.py")
    (root / "packages/core/src/device_scheduler_core").mkdir(parents=True)
    package_source = root / "packages/core/src/device_scheduler_core/__init__.py"
    package_source.write_text("VALUE = 42\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    test_file = tests / "test_core.py"
    test_file.write_text("from device_scheduler_core import VALUE\n\ndef test_value():\n    assert VALUE == 42\n", encoding="utf-8")
    (root / "packages/core/pyproject.toml").write_text("[project]\nname='core'\nversion='0.0.1'\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='root'\nversion='0.0.1'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    evidence_dir = root / "planning/evidence/DS-01"
    evidence_dir.mkdir(parents=True)
    artifact = evidence_dir / "api.txt"
    artifact.write_text("Versioned core domain API implementation evidence.\n", encoding="utf-8")
    source_ref = "packages/core/src/device_scheduler_core/__init__.py"
    (root / "planning/implementation-plan.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "planning/implementation-plan.json").write_text(json.dumps({
        "work_items": [{"id": "DS-01", "acceptance": {"required_evidence": ["core-contract-api"]}}]
    }), encoding="utf-8")
    (evidence_dir / "manifest-input.json").write_text(json.dumps({
        "schema_version": 1,
        "work_item_id": "DS-01",
        "evidence": {
            "core-contract-api": {
                "path": "planning/evidence/DS-01/api.txt",
                "summary": "The API file documents and implements the source-bound contract.",
                "assertions": ["The contract source resolves from this task workspace."],
                "source_files": [source_ref],
            }
        },
    }), encoding="utf-8")
    monkeypatch.setitem(verifier.TASK_TESTS, "DS-01", ("tests/test_core.py",))
    monkeypatch.setitem(verifier.TASK_PACKAGES, "DS-01", ("device_scheduler_core",))
    monkeypatch.setitem(verifier.PACKAGE_DIRS, "device_scheduler_core", "packages/core/src/device_scheduler_core")
    monkeypatch.setitem(verifier.PACKAGE_ROOTS, "device_scheduler_core", "packages/core")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Native evidence test")
    _git(root, "config", "user.email", "native-evidence-test@example.invalid")
    _git(root, "add", "scripts/verify_native_evidence.py", "packages", "tests", "pyproject.toml", "uv.lock", "planning/implementation-plan.json")
    _git(root, "commit", "-qm", "commit source")

    errors = verifier.capture_manifest(root, "DS-01")

    assert errors == []
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((evidence_dir / "pytest-run.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["receipt_id"] == receipt["receipt_id"]
    assert receipt["exit_code"] == 0
    assert verifier.verify_task(root, "DS-01") == []
    assert (evidence_dir / "source.bundle").is_file()
    (root / ".git").rename(root / "unmounted-host-metadata")
    assert verifier.verify_task(root, "DS-01") == []
