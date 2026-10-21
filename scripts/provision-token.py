#!/usr/bin/env python3
"""Create one local client token and its hashed daemon grant without printing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path

from device_scheduler_core.models import Principal


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"credential directory is not a real directory: {path}")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError(f"credential directory must be owned by this user with mode 0700: {path}")


def _create_private_file(path: Path, content: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    config_root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "device-scheduler"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--principal-id", default="operator")
    parser.add_argument("--actions", nargs="+", required=True, help="granted operation names; use '*' only for a full operator token")
    parser.add_argument("--execution-root", action="append", default=[])
    parser.add_argument("--backend-id", action="append", default=[])
    parser.add_argument("--max-priority", type=int, default=50)
    parser.add_argument("--token-file", type=Path, default=config_root / "client.token")
    parser.add_argument("--token-directory", type=Path, default=config_root / "tokens")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.principal_id):
        raise ValueError(
            "principal ID must be 1 to 128 letters, digits, dots, underscores, or hyphens "
            "and start with a letter or digit"
        )

    principal = Principal(
        principal_id=args.principal_id,
        actions=frozenset(args.actions),
        execution_roots=tuple(args.execution_root),
        backend_ids=frozenset(args.backend_id),
        max_priority=args.max_priority,
    )
    token_path = args.token_file.expanduser()
    grant_root = args.token_directory.expanduser()
    _private_directory(token_path.parent)
    _private_directory(grant_root)
    grant_path = grant_root / f"{principal.principal_id}.toml"
    token = secrets.token_urlsafe(32)
    grant = (
        "[principal]\n"
        f"token_sha256 = {json.dumps(hashlib.sha256(token.encode('utf-8')).hexdigest())}\n"
        f"principal_id = {json.dumps(principal.principal_id)}\n"
        f"actions = {json.dumps(sorted(principal.actions))}\n"
        f"execution_roots = {json.dumps(list(principal.execution_roots))}\n"
        f"backend_ids = {json.dumps(sorted(principal.backend_ids))}\n"
        f"max_priority = {principal.max_priority}\n"
    )
    token_created = False
    try:
        _create_private_file(token_path, token + "\n")
        token_created = True
        _create_private_file(grant_path, grant)
    except Exception:
        if token_created:
            token_path.unlink(missing_ok=True)
        raise
    print(f"Created token file: {token_path}")
    print(f"Created hashed principal grant: {grant_path}")
    print("The token value was not printed. Keep the client token file private.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"provision-token: {exc}")
