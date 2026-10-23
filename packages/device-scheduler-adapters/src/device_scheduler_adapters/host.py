"""Read-only Linux host resource discovery using the Python standard library."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from device_scheduler_core.models import AdapterCapability, EnforcementLevel, Resource


def _read_meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.strip().split()
        if not fields:
            continue
        try:
            amount = int(fields[0])
        except ValueError:
            continue
        # Linux reports these values in kB (1024 bytes).
        unit = fields[1].lower() if len(fields) > 1 else "b"
        multiplier = 1024 if unit in {"kb", "kib"} else 1
        values[key] = amount * multiplier
    return values


class LinuxHostAdapter:
    """Observe CPU and memory totals without claiming scheduler enforcement."""

    api_version = "1"
    adapter_id = "linux-host"

    def __init__(
        self,
        *,
        proc_meminfo: str | os.PathLike[str] = "/proc/meminfo",
        cpu_count: Any = os.cpu_count,
        hostname: Any = socket.gethostname,
    ) -> None:
        self._meminfo = Path(proc_meminfo)
        self._cpu_count = cpu_count
        self._hostname = hostname

    def capabilities(self) -> tuple[AdapterCapability, ...]:
        return (
            AdapterCapability(
                "host-cpu-count",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "os.cpu_count", "unit": "cores", "count_semantics": "logical CPUs", "scope": "whole host"},
            ),
            AdapterCapability(
                "host-memory-capacity",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "/proc/meminfo:MemTotal"},
            ),
            AdapterCapability(
                "host-memory-available",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "/proc/meminfo:MemAvailable", "missing_value": "unknown"},
            ),
            AdapterCapability(
                "per-job-cgroup-limits",
                True,
                EnforcementLevel.ADVISORY,
                {"provided_by": "systemd-user-unit-adapter", "host_qualification_required": True},
            ),
            AdapterCapability(
                "cpu-memory-supervised-lease-inputs",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {
                    "resource_kinds": ["cpu", "memory"],
                    "consumption": "scheduler-supervised jobs only",
                    "unmanaged_consumers": "unsupported",
                    "meaning": "ledger admission quantities; not physical CPU reservation or memory preallocation",
                },
            ),
        )

    def _build_snapshot(self) -> dict[str, Any]:
        errors: list[str] = []
        try:
            cpus = self._cpu_count()
            if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus <= 0:
                cpus = None
                errors.append("cpu_count_unavailable")
        except (OSError, RuntimeError):
            cpus = None
            errors.append("cpu_count_unavailable")

        try:
            memory = _read_meminfo(self._meminfo)
        except (OSError, UnicodeError):
            memory = {}
            errors.append("meminfo_unavailable")

        total_bytes = memory.get("MemTotal")
        available_bytes = memory.get("MemAvailable")
        if total_bytes is not None and total_bytes <= 0:
            total_bytes = None
        if available_bytes is not None and (
            available_bytes < 0 or (total_bytes is not None and available_bytes > total_bytes)
        ):
            available_bytes = None
        if total_bytes is None:
            errors.append("memory_total_unknown")
        if available_bytes is None:
            errors.append("memory_available_unknown")
        try:
            host_id = str(self._hostname())
        except OSError:
            host_id = "unknown-host"
            errors.append("hostname_unavailable")

        state = "ready" if not errors else ("partial" if cpus or total_bytes else "unknown")
        resources: list[Resource] = []
        if cpus is not None:
            resources.append(
                Resource(
                    resource_id=f"host:{host_id}:cpu",
                    host_id=host_id,
                    kind="cpu",
                    capacities={"cores": float(cpus)},
                    attributes={"observed_logical_cpus": cpus, "scope": "whole host"},
                    capabilities=frozenset({
                        "systemd-cgroup-request-dependent",
                        "supervised-lease-job-only",
                    }),
                )
            )
        if total_bytes is not None:
            observations: dict[str, Any] = {}
            if available_bytes is not None:
                observations["available_bytes"] = available_bytes
            resources.append(
                Resource(
                    resource_id=f"host:{host_id}:memory",
                    host_id=host_id,
                    kind="memory",
                    capacities={"bytes": float(total_bytes)},
                    attributes=observations,
                    capabilities=frozenset({
                        "systemd-cgroup-request-dependent",
                        "supervised-lease-job-only",
                    }),
                )
            )
        return {
            "adapter": self.adapter_id,
            "state": state,
            "host_id": host_id,
            "resources": resources,
            "capabilities": [item.to_dict() for item in self.capabilities()],
            "errors": errors,
            "evidence": {
                "meminfo_path": str(self._meminfo),
                "memory_total_field": "MemTotal",
                "memory_available_field": "MemAvailable",
            },
        }

    def discover(self) -> list[Resource]:
        """Return only capacities actually observed by the host APIs."""
        return list(self._build_snapshot()["resources"])

    def snapshot(self) -> dict[str, Any]:
        """Return discovery status and diagnostics alongside typed resources."""
        # Callers can inspect partial/unknown telemetry instead of interpreting
        # an absent resource capacity as zero.
        return self._build_snapshot()

    def inspect(self, resource_id: str) -> Resource | None:
        return next((resource for resource in self.discover() if resource.resource_id == resource_id), None)
