"""Static, file-backed bearer authentication for the local scheduler daemon.

The server stores only SHA-256 token digests and grants. Client bearer tokens
live in separate private files and are never copied into job specifications.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from device_scheduler_core.models import Principal
from mcp.server.auth.provider import AccessToken


class AuthConfigurationError(ValueError):
    """The configured local authentication store is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class TokenGrant:
    token_sha256: str
    principal: Principal


class StaticTokenStore:
    """Resolve bearer tokens to immutable core principals."""

    def __init__(self, grants: tuple[TokenGrant, ...] | list[TokenGrant]) -> None:
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        normalized: list[TokenGrant] = []
        for grant in grants:
            if not isinstance(grant, TokenGrant):
                raise TypeError("grants must contain TokenGrant values")
            digest = grant.token_sha256.lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise AuthConfigurationError("token_sha256 must be a 64-character lowercase hex digest")
            if digest in seen_digests:
                raise AuthConfigurationError("duplicate token digest")
            if grant.principal.principal_id in seen_ids:
                raise AuthConfigurationError("duplicate principal_id")
            seen_digests.add(digest)
            seen_ids.add(grant.principal.principal_id)
            normalized.append(TokenGrant(digest, grant.principal))
        self._grants = tuple(normalized)

    @classmethod
    def from_directory(cls, directory: Path | str) -> "StaticTokenStore":
        root = Path(directory).expanduser()
        _require_private_directory(root)
        grants: list[TokenGrant] = []
        for path in sorted(root.glob("*.toml")):
            _require_private_file(path)
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                if set(data) != {"principal"} or not isinstance(data["principal"], dict):
                    raise AuthConfigurationError("grant file must contain only a [principal] table")
                record: dict[str, Any] = data["principal"]
                expected = {"token_sha256", "principal_id", "actions", "execution_roots", "backend_ids", "max_priority"}
                if set(record) - expected:
                    raise AuthConfigurationError("grant file contains unknown principal fields")
                digest = record.get("token_sha256")
                if not isinstance(digest, str):
                    raise AuthConfigurationError("token_sha256 is required")
                principal_data = {key: value for key, value in record.items() if key != "token_sha256"}
                principal = Principal.from_dict(principal_data)
                grants.append(TokenGrant(digest, principal))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
                if isinstance(exc, AuthConfigurationError):
                    raise
                raise AuthConfigurationError(f"invalid token grant file: {path.name}") from exc
        return cls(grants)

    def authenticate(self, bearer_token: str) -> Principal | None:
        if not isinstance(bearer_token, str) or not bearer_token or len(bearer_token) > 4096:
            return None
        digest = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()
        matched: Principal | None = None
        candidate = digest.encode("ascii")
        for grant in self._grants:
            if hmac.compare_digest(candidate, grant.token_sha256.encode("ascii")):
                matched = grant.principal
        return matched

    def __len__(self) -> int:
        return len(self._grants)


class StaticTokenVerifier:
    """Official MCP SDK verifier that returns the resolved core principal."""

    def __init__(self, token_store: StaticTokenStore) -> None:
        self._token_store = token_store

    async def verify_token(self, token: str) -> AccessToken | None:
        principal = self._token_store.authenticate(token)
        if principal is None:
            return None
        return AccessToken(
            token=token,
            client_id=principal.principal_id,
            subject=principal.principal_id,
            scopes=sorted(principal.actions),
            claims={"device_scheduler_principal": principal.to_dict()},
        )


def principal_from_request(request: Any) -> Principal:
    """Extract the verified principal attached by the MCP SDK auth middleware."""
    user = getattr(request, "user", None)
    access_token = getattr(user, "access_token", None)
    claims = getattr(access_token, "claims", None)
    data = claims.get("device_scheduler_principal") if isinstance(claims, dict) else None
    if not isinstance(data, dict):
        raise AuthConfigurationError("authenticated request is missing its principal")
    return Principal.from_dict(data)


def read_client_token(path: Path | str) -> str:
    """Read one private bearer-token file without following symlinks."""
    token_path = Path(path).expanduser()
    _require_private_file(token_path)
    try:
        token = token_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuthConfigurationError("cannot read client token file") from exc
    if token.endswith("\n"):
        token = token[:-1]
        if token.endswith("\r"):
            token = token[:-1]
    if not token or len(token) < 32 or len(token) > 4096 or any(char.isspace() for char in token):
        raise AuthConfigurationError("client token must be one opaque value of 32 to 4096 characters")
    return token


def _require_private_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AuthConfigurationError(f"token directory is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AuthConfigurationError("token directory must be a real directory")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise AuthConfigurationError("token directory must be owned by the service user with mode 0700 or stricter")


def _require_private_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise AuthConfigurationError(f"private file is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AuthConfigurationError("private credential and grant files must be regular files")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise AuthConfigurationError("private credential and grant files must be owned by the service user with mode 0600")
