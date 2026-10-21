"""Small validation and immutability helpers for core DTOs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
import re
from types import MappingProxyType
from typing import Any

from .errors import ValidationError

_OWNER_CONTEXT_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_OWNER_CONTEXT_MAX_VALUE_LENGTH = 4096
_SENSITIVE_CONTEXT_KEY_PARTS = (
    "secret",
    "password",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "private_key",
    "privatekey",
    "token",
    "bearer",
)


def identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValidationError(f"{name} must be a non-empty identifier")
    return value


def finite_number(value: float, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        comparator = "greater than" if strict else "at least"
        raise ValidationError(f"{name} must be finite and {comparator} {minimum}")
    return result


def aware_datetime(value: datetime | None, name: str, *, required: bool = False) -> datetime | None:
    if value is None:
        if required:
            raise ValidationError(f"{name} is required")
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def validate_json(value: Any, name: str = "value", *, depth: int = 0) -> None:
    if depth > 32:
        raise ValidationError(f"{name} exceeds maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} object keys must be strings")
            validate_json(child, f"{name}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_json(child, f"{name}[{index}]", depth=depth + 1)
        return
    raise ValidationError(f"{name} must contain only JSON-compatible values")


def freeze_json(value: Any) -> Any:
    """Copy JSON-shaped values into recursively immutable containers."""
    validate_json(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(child) for child in value)
    return value


def json_mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    validate_json(value, name)
    return freeze_json(value)


def quantity_mapping(
    value: Mapping[str, float], name: str, *, allow_empty: bool = False, allow_zero: bool = False
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise ValidationError(f"{name} must be a non-empty quantity mapping")
    result: dict[str, float] = {}
    for key, quantity in value.items():
        identifier(key, f"{name} key")
        result[key] = finite_number(quantity, f"{name}.{key}", minimum=0.0, strict=not allow_zero)
    return MappingProxyType(result)


def owner_context_mapping(value: Mapping[str, str] | None, name: str = "owner_context") -> Mapping[str, str]:
    """Freeze bounded, flat identity metadata; never store credentials here.

    Keys identify context fields such as ``issuer``, ``project_id``,
    ``attempt_id``, ``task_id``, ``owner_kind`` or a canonical project-root
    path. Values are bounded descriptive strings, not secret material. Key-name
    checks reject common credential fields but cannot identify every secret;
    callers must keep credentials in the host-managed secret store. Context
    remains descriptive and never grants permission.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a flat string mapping")
    if len(value) > 16:
        raise ValidationError(f"{name} may contain at most 16 entries")
    result: dict[str, str] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not _OWNER_CONTEXT_KEY.fullmatch(key):
            raise ValidationError(f"{name} keys must be identifier names of at most 64 characters")
        # Treat punctuation as a separator so ``api-key``, ``api.key`` and
        # ``api_key`` receive the same credential-field check.
        normalized_key = re.sub(r"[._-]+", "_", key.casefold())
        if "key" in normalized_key.split("_") or any(
            part in normalized_key for part in _SENSITIVE_CONTEXT_KEY_PARTS
        ):
            raise ValidationError(f"{name} cannot contain credential or secret fields")
        if (
            not isinstance(child, str)
            or not child
            or "\x00" in child
            or len(child) > _OWNER_CONTEXT_MAX_VALUE_LENGTH
        ):
            raise ValidationError(
                f"{name} values must be non-empty strings without NUL and at most "
                f"{_OWNER_CONTEXT_MAX_VALUE_LENGTH} characters"
            )
        result[key] = child
    return MappingProxyType(result)

