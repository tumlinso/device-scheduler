#!/usr/bin/env python3
"""Static source-identity audit for the prepared Project Control cutover.

This tool reads the source snapshot and local Git metadata only. It never
connects to Project Control, Device Scheduler, systemd, a CUDA runtime, or a
GPU, and it cannot establish that either allocator is fenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
MAP_PATH = REPOSITORY / "planning" / "cutover-v1" / "consumer-map.json"


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def audit(project_control_root: Path) -> dict[str, Any]:
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for consumer in mapping["consumers"]:
        for source in consumer["files"]:
            expected[source["path"]] = source["sha256"]

    sources: list[dict[str, str | bool | None]] = []
    for relative_path, expected_digest in sorted(expected.items()):
        path = project_control_root / relative_path
        actual = _sha256(path)
        sources.append(
            {
                "path": str(path),
                "exists": actual is not None,
                "expected_sha256": expected_digest,
                "actual_sha256": actual,
                "matches_snapshot": actual == expected_digest,
            }
        )

    head = _git(project_control_root, "rev-parse", "HEAD")
    status = _git(project_control_root, "status", "--porcelain")
    static_ok = all(item["matches_snapshot"] is True for item in sources)
    return {
        "schema_version": 1,
        "scope": "static_source_identity_only",
        "project_control_root": str(project_control_root),
        "project_control_head": head,
        "project_control_worktree_clean": None if status is None else status == "",
        "recorded_preparation_head": mapping["project_control_source"]["preparation_head"],
        "source_snapshot_matches": static_ok,
        "sources": sources,
        "cutover_state": {
            "old_allocator_fenced": "unverified",
            "consumers_switched": "unverified",
            "service_health": "not_queried",
            "single_active_allocator": "unverified",
            "device_scheduler_health": "not_queried",
        },
        "completion_claim": False,
        "limitations": [
            "This audit does not query a live service or inspect active leases.",
            "Matching source hashes do not establish deployed runtime source identity.",
            "A passing static audit is not cutover, fence, recovery, or single-authority evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-control-root",
        type=Path,
        default=Path("/home/tumlinson/project-control"),
        help="Project Control source checkout to fingerprint (default: %(default)s)",
    )
    parser.add_argument("--check", action="store_true", help="exit nonzero when a source is missing or changed")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON (the default)")
    args = parser.parse_args()
    report = audit(args.project_control_root.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check and report["source_snapshot_matches"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
