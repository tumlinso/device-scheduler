"""Read-only NVIDIA inventory, topology and process-quiescence observations.

This adapter invokes ``nvidia-smi`` only when a method is called.  It never
imports CUDA libraries or launches work.  GPU process telemetry is explicitly
limited to compute processes visible to the installed ``nvidia-smi`` version;
that evidence must not be presented as a hostile-process isolation guarantee.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from device_scheduler_core.models import (
    AdapterCapability,
    AllocationMode,
    CapabilityPrecondition,
    EnforcementLevel,
    ExecutionContext,
    ExecutionPurpose,
    Resource,
    ResourceAllocation,
)


_GPU_LABEL = re.compile(r"^GPU(?P<index>\d+)$")
_SGR_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _number(value: str) -> int | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.upper() in {"N/A", "NA", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return int(float(cleaned))
    except (OverflowError, ValueError):
        return None


def _csv_rows(output: str) -> list[list[str]]:
    return [row for row in csv.reader(io.StringIO(output)) if row and any(cell.strip() for cell in row)]


class NvidiaAdapter:
    """Discover NVIDIA GPUs and conservatively observe compute-process drain."""

    api_version = "1"
    adapter_id = "nvidia-smi"

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        hostname: Callable[[], str] = socket.gethostname,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        command_timeout: float = 8.0,
        external_lease_headroom_bytes: int = 512 * 1024 * 1024,
        external_lease_samples: int = 3,
        external_lease_interval_seconds: float = 0.5,
        external_lease_timeout_seconds: float = 30.0,
    ) -> None:
        if (
            isinstance(command_timeout, bool)
            or not isinstance(command_timeout, (int, float))
            or not math.isfinite(command_timeout)
            or command_timeout <= 0
        ):
            raise ValueError("command_timeout must be a finite positive number")
        if (
            isinstance(external_lease_headroom_bytes, bool)
            or not isinstance(external_lease_headroom_bytes, int)
            or external_lease_headroom_bytes < 0
        ):
            raise ValueError("external_lease_headroom_bytes must be a non-negative integer")
        if (
            isinstance(external_lease_samples, bool)
            or not isinstance(external_lease_samples, int)
            or external_lease_samples < 1
        ):
            raise ValueError("external_lease_samples must be a positive integer")
        if (
            isinstance(external_lease_interval_seconds, bool)
            or not isinstance(external_lease_interval_seconds, (int, float))
            or not math.isfinite(external_lease_interval_seconds)
            or external_lease_interval_seconds < 0
            or isinstance(external_lease_timeout_seconds, bool)
            or not isinstance(external_lease_timeout_seconds, (int, float))
            or not math.isfinite(external_lease_timeout_seconds)
            or external_lease_timeout_seconds < 0
        ):
            raise ValueError("external lease observation interval and timeout must be non-negative")
        self._runner = runner or self._default_runner
        self._hostname = hostname
        self._clock = clock
        self._sleep = sleeper
        self._command_timeout = command_timeout
        self._external_lease_headroom_bytes = external_lease_headroom_bytes
        self._external_lease_samples = external_lease_samples
        self._external_lease_interval_seconds = float(external_lease_interval_seconds)
        self._external_lease_timeout_seconds = float(external_lease_timeout_seconds)

    @staticmethod
    def _default_runner(argv: Sequence[str], *, timeout: float) -> Any:
        return subprocess.run(list(argv), text=True, capture_output=True, timeout=timeout, check=False)

    def capabilities(self) -> tuple[AdapterCapability, ...]:
        return (
            AdapterCapability(
                "nvidia-gpu-inventory",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "nvidia-smi query-gpu", "identity": "UUID"},
            ),
            AdapterCapability(
                "gpu-memory-observation",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"unit": "bytes", "source_unit": "MiB"},
            ),
            AdapterCapability(
                "gpu-memory-budget",
                True,
                EnforcementLevel.ADVISORY,
                {"reason": "inventory telemetry does not enforce per-workload limits"},
            ),
            AdapterCapability(
                "compute-process-telemetry",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"scope": "nvidia-smi-visible-compute-applications"},
            ),
            AdapterCapability(
                "gpu-quiescence-observation",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"requires": "consecutive successful empty process observations"},
            ),
            AdapterCapability(
                "gpu.quiescence",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {
                    "precondition_schema_version": 1,
                    "parameters": {
                        "consecutive_samples": "integer 1..20",
                        "interval_seconds": "number 0..10",
                        "timeout_seconds": "number 0..60",
                        "max_utilization_percent": "integer 0..100, default 0",
                        "minimum_free_bytes": "optional non-negative integer, never below grant plus configured headroom",
                    },
                    "binds_to": "GPU allocations and descriptors in ExecutionContext",
                    "scope": "read-only nvidia-smi-visible compute telemetry",
                },
            ),
            AdapterCapability(
                "device-exclusive-admission",
                True,
                EnforcementLevel.ADVISORY,
                {"enforced_by": "scheduler ledger", "does_not_terminate_unmanaged_processes": True},
            ),
            AdapterCapability(
                "gpu-topology-links",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {"source": "nvidia-smi topo -m", "contention_established": False},
            ),
            AdapterCapability(
                "topology-derived-profiler-isolation-domains",
                True,
                EnforcementLevel.ADVISORY,
                {
                    "resource_kind": "isolation",
                    "requires": "complete symmetric GPU topology matrix",
                    "scope": "scheduler admission only",
                    "does_not_claim_hardware_contention": True,
                },
            ),
            AdapterCapability(
                "external-gpu-lease-observation",
                True,
                EnforcementLevel.OBSERVATIONAL,
                {
                    "requires": "exclusive GPU-only lease and exact discovered resource descriptors",
                    "operations": ["capture_lease_baseline", "inspect_lease_release"],
                    "headroom_bytes": self._external_lease_headroom_bytes,
                    "side_effects": "read-only nvidia-smi observations",
                    "limitations": "does not prove hostile-process isolation",
                },
            ),
        )

    def _run(self, argv: Sequence[str]) -> tuple[str | None, str | None]:
        try:
            result = self._runner(list(argv), timeout=self._command_timeout)
        except FileNotFoundError:
            return None, "nvidia_smi_unavailable"
        except subprocess.TimeoutExpired:
            return None, "nvidia_smi_timeout"
        except OSError as exc:
            return None, f"nvidia_smi_os_error:{type(exc).__name__}"
        returncode = result.get("returncode", 1) if isinstance(result, Mapping) else getattr(result, "returncode", 1)
        stdout = result.get("stdout", "") if isinstance(result, Mapping) else getattr(result, "stdout", "")
        if returncode != 0:
            stderr = result.get("stderr", "") if isinstance(result, Mapping) else getattr(result, "stderr", "")
            detail = str(stderr).strip().replace("\n", " ")[:300]
            return None, f"nvidia_smi_failed:{detail or returncode}"
        return str(stdout or ""), None

    def _gpus(self) -> tuple[list[dict[str, Any]] | None, str | None]:
        fields = "index,uuid,name,pci.bus_id,memory.total,memory.free,utilization.gpu"
        output, error = self._run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        )
        if error:
            return None, error
        assert output is not None
        rows = _csv_rows(output)
        devices: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        seen_uuids: set[str] = set()
        for row in rows:
            if len(row) != 7:
                return None, "gpu_inventory_schema_unknown"
            index = _number(row[0])
            uuid = row[1].strip()
            if index is None or not uuid or uuid.upper() in {"N/A", "NA"}:
                return None, "gpu_identity_unknown"
            if index in seen_indices or uuid in seen_uuids:
                return None, "gpu_identity_duplicate"
            seen_indices.add(index)
            seen_uuids.add(uuid)
            devices.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "name": row[2].strip(),
                    "pci_bus_id": row[3].strip(),
                    "memory_total_bytes": None if _number(row[4]) is None else _number(row[4]) * 1024 * 1024,
                    "memory_free_bytes": None if _number(row[5]) is None else _number(row[5]) * 1024 * 1024,
                    "utilization_percent": _number(row[6]),
                }
            )
        return devices, None

    def _processes(self) -> tuple[list[dict[str, Any]] | None, str | None]:
        fields = "gpu_uuid,pid,process_name,used_gpu_memory"
        output, error = self._run(
            ["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"]
        )
        if error:
            return None, error
        assert output is not None
        if output.strip().lower() in {
            "no running processes found",
            "no running processes",
            "no processes found",
        }:
            return [], None
        rows = _csv_rows(output)
        processes: list[dict[str, Any]] = []
        for row in rows:
            if len(row) != 4:
                return None, "compute_process_telemetry_schema_unknown"
            uuid = row[0].strip()
            pid_text = row[1].strip()
            pid = int(pid_text) if re.fullmatch(r"[0-9]+", pid_text) else None
            if not uuid or uuid.upper() in {"N/A", "NA"} or pid is None or pid <= 0:
                processes.append(
                    {
                        "gpu_uuid": None,
                        "pid": pid,
                        "process_name": row[2].strip(),
                        "used_memory_bytes": None,
                        "identity_unknown": True,
                    }
                )
            else:
                memory_mb = _number(row[3])
                processes.append(
                    {
                        "gpu_uuid": uuid,
                        "pid": pid,
                        "process_name": row[2].strip(),
                        "used_memory_bytes": None if memory_mb is None else memory_mb * 1024 * 1024,
                        "identity_unknown": False,
                    }
                )
        return processes, None

    def _topology(
        self, devices: Sequence[Mapping[str, Any]], host_id: str
    ) -> tuple[dict[str, Any], str | None]:
        output, error = self._run(["nvidia-smi", "topo", "-m"])
        if error:
            return {
                "state": "unknown",
                "links": [],
                "isolation_domains": [],
                "matrix_complete": False,
            }, error
        assert output is not None
        # Some drivers underline the header even on captured stdout. Formatting
        # must not remove a GPU column from the observed topology matrix.
        output = _SGR_ESCAPE.sub("", output)
        lines = [line.split() for line in output.splitlines() if line.strip()]
        header_index = next(
            (i for i, row in enumerate(lines) if any(_GPU_LABEL.fullmatch(cell) for cell in row)),
            None,
        )
        if header_index is None:
            return {
                "state": "unknown",
                "links": [],
                "isolation_domains": [],
                "matrix_complete": False,
            }, "topology_schema_unknown"
        header = lines[header_index]
        columns = [cell for cell in header if _GPU_LABEL.fullmatch(cell)]
        index_to_uuid = {int(device["index"]): str(device["uuid"]) for device in devices}
        label_to_uuid: dict[str, str] = {
            f"GPU{index}": uuid for index, uuid in index_to_uuid.items()
        }
        links: list[dict[str, Any]] = []
        matrix: dict[str, dict[str, str]] = {}
        for row in lines[header_index + 1 :]:
            if not row or not _GPU_LABEL.fullmatch(row[0]):
                continue
            source = row[0]
            cells = row[1 : 1 + len(columns)]
            if source not in label_to_uuid:
                continue
            if len(cells) == len(columns):
                matrix[source] = dict(zip(columns, cells))

        expected_labels = set(label_to_uuid)
        matrix_complete = (
            set(columns) == expected_labels
            and set(matrix) == expected_labels
            and all(
                relation.strip() and relation.strip() not in {"?", "N/A", "-"}
                for source in expected_labels
                for target, relation in matrix.get(source, {}).items()
                if target != source
            )
            and all(matrix[label].get(label) == "X" for label in expected_labels if label in matrix)
        )
        if matrix_complete:
            for left_index, source in enumerate(sorted(expected_labels)):
                for target in sorted(expected_labels)[left_index + 1 :]:
                    relation = matrix[source].get(target, "").strip()
                    reverse = matrix[target].get(source, "").strip()
                    if not relation or relation != reverse:
                        matrix_complete = False
                        break
                if not matrix_complete:
                    break

        graph: dict[str, set[str]] = {label: set() for label in expected_labels}
        relation_by_pair: dict[tuple[str, str], str] = {}
        if matrix_complete:
            for left_index, source in enumerate(sorted(expected_labels)):
                for target in sorted(expected_labels)[left_index + 1 :]:
                    relation = matrix[source][target]
                    if relation == "X":
                        continue
                    normalized = relation.upper()
                    if normalized.startswith("NV"):
                        link_type = "nvlink"
                    elif normalized in {"PIX", "PXB", "PHB"}:
                        link_type = "pcie"
                    elif normalized in {"NODE", "SYS"}:
                        link_type = normalized.lower()
                    else:
                        link_type = "unknown"
                    links.append(
                        {
                            "source_gpu_uuid": label_to_uuid[source],
                            "target_gpu_uuid": label_to_uuid[target],
                            "relation": relation,
                            "type": link_type,
                        }
                    )
                    if normalized.startswith("NV") or normalized in {"PIX", "PXB", "PHB"}:
                        graph[source].add(target)
                        graph[target].add(source)
                        relation_by_pair[(source, target)] = relation

        isolation_domains: list[dict[str, Any]] = []
        if matrix_complete:
            remaining = set(expected_labels)
            while remaining:
                seed = min(remaining)
                component: set[str] = set()
                pending = [seed]
                while pending:
                    current = pending.pop()
                    if current in component:
                        continue
                    component.add(current)
                    pending.extend(graph[current] - component)
                remaining -= component
                if len(component) < 2:
                    continue
                member_uuids = sorted(label_to_uuid[label] for label in component)
                digest = hashlib.sha256("\0".join(member_uuids).encode("utf-8")).hexdigest()[:16]
                member_ids = [f"gpu:{host_id}:{uuid}" for uuid in member_uuids]
                pairs = [
                    {"source": left, "target": right, "relation": relation_by_pair[(left, right)]}
                    for left_index, left in enumerate(sorted(component))
                    for right in sorted(component)[left_index + 1 :]
                    if (left, right) in relation_by_pair
                ]
                domain_id = f"host:{host_id}:domain:gpu-topology:{digest}"
                isolation_domains.append(
                    {
                        "domain_id": domain_id,
                        "resource_id": domain_id,
                        "resource_ids": member_ids,
                        "relations": pairs,
                        "evidence": "nvidia-smi-topo-m",
                        "closure": "complete-observed-gpu-inventory",
                        "enforcement": "scheduler-admission-only",
                        "contention_established": False,
                    }
                )
        state = "observed" if matrix_complete else "partial"
        error = None if matrix_complete else "gpu_topology_matrix_incomplete; topology isolation guarantees unsupported"
        return {
            "state": state,
            "links": links,
            "isolation_domains": isolation_domains,
            "matrix_complete": matrix_complete,
        }, error

    def snapshot(self) -> dict[str, Any]:
        """Return telemetry with independent status for inventory and optional views."""
        try:
            host_id = str(self._hostname())
        except OSError:
            host_id = "unknown-host"
        devices, gpu_error = self._gpus()
        inventory_observed_at = datetime.now(timezone.utc).isoformat()
        if devices is None:
            return {
                "adapter": self.adapter_id,
                "host_id": host_id,
                "state": "unknown",
                "gpu_inventory": {"state": "unknown", "error": gpu_error},
                "process_telemetry": {"state": "unknown", "processes": None},
                "topology": {"state": "unknown", "links": [], "conflict_domains": []},
                "isolation_domain_support": {
                    "state": "unsupported",
                    "domains": [],
                    "reason": "GPU inventory unavailable",
                },
                "resources": [],
                "capabilities": [item.to_dict() for item in self.capabilities()],
                "inventory_observed_at": inventory_observed_at,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            }
        processes, process_error = self._processes()
        process_observed_at = datetime.now(timezone.utc).isoformat()
        process_identity_unknown = processes is not None and any(
            process["identity_unknown"] for process in processes
        )
        topology, topology_error = self._topology(devices, host_id)
        resources: list[Resource] = []
        for device in devices:
            uuid = str(device["uuid"])
            resource_id = f"gpu:{host_id}:{uuid}"
            capacities = {}
            if device["memory_total_bytes"] is not None and device["memory_total_bytes"] > 0:
                capacities["memory_bytes"] = float(device["memory_total_bytes"])
            attrs: dict[str, Any] = {
                "uuid": uuid,
                "index_observation": device["index"],
                "name": device["name"],
                "pci_bus_id": device["pci_bus_id"],
                "memory_free_bytes_observation": device["memory_free_bytes"],
                "utilization_percent_observation": device["utilization_percent"],
                "memory_enforcement": "accounting_only",
                "process_telemetry": "unknown"
                if processes is None or process_identity_unknown
                else "compute-apps-visible",
                "process_count_observation": None
                if processes is None or process_identity_unknown
                else sum(1 for p in processes if p["gpu_uuid"] == uuid),
                "processes_observation": None
                if processes is None
                else [
                    dict(process)
                    for process in processes
                    if process["gpu_uuid"] == uuid or process["identity_unknown"]
                ],
                "inventory_observed_at": inventory_observed_at,
                "process_observed_at": process_observed_at,
            }
            resources.append(
                Resource(
                    resource_id=resource_id,
                    host_id=host_id,
                    kind="gpu",
                    capacities=capacities,
                    attributes=attrs,
                    topology={
                        "pci_bus_id": device["pci_bus_id"],
                        "links": [
                            link
                            for link in topology["links"]
                            if link["source_gpu_uuid"] == uuid or link["target_gpu_uuid"] == uuid
                        ],
                        "topology_state": topology["state"],
                    },
                    conflict_domains=frozenset({f"gpu-exclusive:{host_id}:{uuid}"}
                    ),
                    capabilities=frozenset(
                        {
                            "shared-admission",
                            "whole-device-exclusive-admission",
                            "memory-accounting-only",
                            "nvidia-smi-compute-process-telemetry",
                        }
                    ),
                )
            )
        isolation_resources: list[Resource] = []
        for domain in topology.get("isolation_domains", []):
            isolation_resources.append(
                Resource(
                    resource_id=domain["resource_id"],
                    host_id=host_id,
                    kind="isolation",
                    capacities={"slots": 1.0},
                    attributes={
                        "domain_id": domain["domain_id"],
                        "member_resource_ids": domain["resource_ids"],
                        "relations": domain["relations"],
                        "evidence": domain["evidence"],
                        "closure": domain["closure"],
                        "contention_established": domain["contention_established"],
                        "enforcement": domain["enforcement"],
                    },
                    topology={"member_resource_ids": domain["resource_ids"]},
                    capabilities=frozenset(
                        {"exclusive-scheduler-admission", "topology-derived", "not-hardware-enforced"}
                    ),
                )
            )
        resources.extend(isolation_resources)
        return {
            "adapter": self.adapter_id,
            "host_id": host_id,
            "state": "ready" if not gpu_error else "partial",
            "gpu_inventory": {"state": "observed", "devices": devices},
            "process_telemetry": {
                "state": "unknown"
                if processes is None
                else "partial"
                if process_identity_unknown
                else "observed",
                "processes": processes,
                "error": process_error,
                "scope": "nvidia-smi-visible-compute-applications",
                "observed_at": process_observed_at,
            },
            "topology": {**topology, "error": topology_error},
            "isolation_domain_support": {
                "state": "observed" if topology.get("matrix_complete") else "unsupported",
                "domains": topology.get("isolation_domains", []),
                "reason": None if topology.get("matrix_complete") else topology_error,
            },
            "resources": resources,
            "capabilities": [item.to_dict() for item in self.capabilities()],
            "inventory_observed_at": inventory_observed_at,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }

    def discover(self) -> list[Resource]:
        snapshot = self.snapshot()
        if snapshot["gpu_inventory"]["state"] != "observed":
            return []
        return list(snapshot["resources"])

    def inspect(self, resource_id: str) -> Resource | None:
        return next((resource for resource in self.discover() if resource.resource_id == resource_id), None)

    @staticmethod
    def _lease_value(lease: Any, key: str, default: Any = None) -> Any:
        return lease.get(key, default) if isinstance(lease, Mapping) else getattr(lease, key, default)

    @staticmethod
    def _memory_bytes(quantities: Mapping[str, Any]) -> int:
        factors = {
            "bytes": 1,
            "memory_bytes": 1,
            "gpu_memory_bytes": 1,
            "memory_mib": 1024**2,
            "memory_gib": 1024**3,
        }
        total = 0
        for key, amount in quantities.items():
            factor = factors.get(str(key).casefold())
            if factor is None:
                continue
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)
                or amount <= 0
            ):
                raise ValueError("GPU lease memory quantities must be finite and positive")
            total += math.ceil(amount * factor)
        return total

    def _external_lease_inputs(
        self,
        lease: Any,
        resource_descriptors: Sequence[Resource | Mapping[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None, str]:
        lease_id = self._lease_value(lease, "lease_id")
        token = self._lease_value(lease, "fencing_token")
        allocations_raw = self._lease_value(lease, "allocations", ())
        if not isinstance(lease_id, str) or not lease_id.strip() or "\x00" in lease_id:
            return None, "lease identity is invalid", "unsupported"
        if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
            return None, "lease fencing token is invalid", "blocked"
        if isinstance(allocations_raw, (str, bytes)) or not isinstance(allocations_raw, Sequence) or not allocations_raw:
            return None, "lease allocations are unavailable", "unsupported"
        if isinstance(resource_descriptors, (str, bytes)) or not isinstance(resource_descriptors, Sequence):
            return None, "resource descriptors are unavailable", "unsupported"
        try:
            allocations = tuple(
                value if isinstance(value, ResourceAllocation) else ResourceAllocation.from_dict(value)
                for value in allocations_raw
            )
            resources = tuple(
                value if isinstance(value, Resource) else Resource.from_dict(value)
                for value in resource_descriptors
            )
        except (TypeError, ValueError) as exc:
            return None, f"lease or resource descriptor is invalid: {exc}", "unsupported"
        allocation_ids = [item.resource_id for item in allocations]
        resource_ids = [item.resource_id for item in resources]
        if len(set(allocation_ids)) != len(allocation_ids) or set(allocation_ids) != set(resource_ids):
            return None, "lease allocations and resource descriptors do not match exactly", "blocked"
        if any(item.mode != AllocationMode.EXCLUSIVE for item in allocations):
            return None, "external GPU leases require exclusive allocations", "unsupported"
        if any(item.kind.casefold() != "gpu" for item in resources):
            return None, "external lease inspection supports GPU-only bundles", "unsupported"
        try:
            local_host = str(self._hostname())
        except (OSError, RuntimeError) as exc:
            return None, f"local host identity is unavailable: {type(exc).__name__}", "unknown"
        targets: list[dict[str, Any]] = []
        seen_uuids: set[str] = set()
        by_id = {item.resource_id: item for item in resources}
        for allocation in allocations:
            resource = by_id[allocation.resource_id]
            gpu_uuid = resource.attributes.get("uuid")
            if resource.host_id != local_host:
                return None, "external GPU lease belongs to a different host", "unsupported"
            if not isinstance(gpu_uuid, str) or not gpu_uuid.strip() or "\x00" in gpu_uuid:
                return None, "GPU resource has no stable UUID", "unknown"
            if resource.resource_id != f"gpu:{local_host}:{gpu_uuid}":
                return None, "GPU resource identity does not match its host-qualified UUID", "blocked"
            if gpu_uuid in seen_uuids:
                return None, "external lease repeats a GPU UUID", "unsupported"
            try:
                memory_budget = self._memory_bytes(allocation.quantities)
            except ValueError as exc:
                return None, str(exc), "unsupported"
            seen_uuids.add(gpu_uuid)
            targets.append(
                {
                    "resource_id": allocation.resource_id,
                    "gpu_uuid": gpu_uuid,
                    "memory_budget_bytes": memory_budget,
                }
            )
        return {
            "lease_id": lease_id,
            "fencing_token": token,
            "resource_ids": sorted(allocation_ids),
            "targets": sorted(targets, key=lambda item: item["resource_id"]),
        }, None, ""

    def _external_gpu_sample(self, targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        devices, gpu_error = self._gpus()
        inventory_at = datetime.now(timezone.utc).isoformat()
        processes, process_error = self._processes()
        processes_at = datetime.now(timezone.utc).isoformat()
        if devices is None:
            return {
                "state": "unknown",
                "reason": gpu_error or "GPU inventory is unavailable",
                "inventory_observed_at": inventory_at,
                "process_observed_at": processes_at,
                "gpus": [],
            }
        if processes is None:
            return {
                "state": "unknown",
                "reason": process_error or "compute process telemetry is unavailable",
                "inventory_observed_at": inventory_at,
                "process_observed_at": processes_at,
                "gpus": [],
            }
        if any(item.get("identity_unknown") is True for item in processes):
            return {
                "state": "unknown",
                "reason": "compute process telemetry contains unattributed process identities",
                "inventory_observed_at": inventory_at,
                "process_observed_at": processes_at,
                "gpus": [],
            }
        by_uuid = {str(item.get("uuid")): item for item in devices}
        observations: list[dict[str, Any]] = []
        busy = False
        for target in targets:
            gpu_uuid = str(target["gpu_uuid"])
            device = by_uuid.get(gpu_uuid)
            if device is None:
                return {
                    "state": "unknown",
                    "reason": f"GPU {gpu_uuid} is absent from fresh inventory",
                    "inventory_observed_at": inventory_at,
                    "process_observed_at": processes_at,
                    "gpus": observations,
                }
            free = device.get("memory_free_bytes")
            total = device.get("memory_total_bytes")
            utilization = device.get("utilization_percent")
            if (
                isinstance(free, bool)
                or not isinstance(free, int)
                or free < 0
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total <= 0
                or free > total
                or isinstance(utilization, bool)
                or not isinstance(utilization, int)
                or not 0 <= utilization <= 100
            ):
                return {
                    "state": "unknown",
                    "reason": f"GPU {gpu_uuid} memory or utilization telemetry is unknown",
                    "inventory_observed_at": inventory_at,
                    "process_observed_at": processes_at,
                    "gpus": observations,
                }
            target_processes = [
                item for item in processes if item.get("gpu_uuid") == gpu_uuid
            ]
            required_free = int(target.get("required_free_bytes", 0))
            max_utilization = int(target.get("max_utilization_percent", 0))
            if free < required_free or utilization > max_utilization or target_processes:
                busy = True
            observations.append(
                {
                    "resource_id": target["resource_id"],
                    "gpu_uuid": gpu_uuid,
                    "memory_total_bytes": total,
                    "memory_free_bytes": free,
                    "utilization_percent": utilization,
                    "process_ids": sorted(int(item["pid"]) for item in target_processes),
                    "required_free_bytes": required_free,
                    "inventory_observed_at": inventory_at,
                    "process_observed_at": processes_at,
                }
            )
        return {
            "state": "busy" if busy else "idle",
            "reason": "GPU process, utilization, or free-memory requirement is not satisfied" if busy else None,
            "inventory_observed_at": inventory_at,
            "process_observed_at": processes_at,
            "gpus": observations,
            "process_scope": "nvidia-smi-visible-compute-applications",
        }

    def _wait_external_gpu_samples(
        self,
        targets: Sequence[Mapping[str, Any]],
        *,
        consecutive_samples: int | None = None,
        interval_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        required_samples = self._external_lease_samples if consecutive_samples is None else consecutive_samples
        interval = self._external_lease_interval_seconds if interval_seconds is None else interval_seconds
        timeout = self._external_lease_timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = self._clock() + timeout
        samples: list[dict[str, Any]] = []
        qualifying = 0
        while True:
            sample = self._external_gpu_sample(targets)
            samples.append(sample)
            if sample["state"] == "unknown":
                return {"state": "unknown", "samples": samples, "reason": sample.get("reason")}
            if sample["state"] == "idle":
                qualifying += 1
                if qualifying >= required_samples:
                    return {
                        "state": "observed",
                        "samples": samples,
                        "consecutive_samples": qualifying,
                    }
            else:
                qualifying = 0
            remaining = deadline - self._clock()
            if remaining <= 0:
                return {
                    "state": "blocked",
                    "samples": samples,
                    "reason": "GPU lease quiescence and memory threshold were not observed before the deadline",
                }
            self._sleep(min(interval, remaining))

    def check_execution_preconditions(
        self,
        execution_context: ExecutionContext,
        preconditions: Sequence[CapabilityPrecondition],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Check GPU quiescence preconditions against the actual granted GPUs.

        This operation is read-only. Its receipt is evidence for the scheduler's
        prelaunch gate, not a reservation and not a process-isolation guarantee.
        """
        if not isinstance(operation_id, str) or not operation_id.strip() or "\x00" in operation_id:
            raise ValueError("operation_id must be a non-empty NUL-free string")
        if isinstance(preconditions, (str, bytes)) or not isinstance(preconditions, Sequence) or not preconditions:
            return {
                "state": "unsupported",
                "verified": False,
                "adapter_id": self.adapter_id,
                "operation_id": operation_id,
                "reason": "at least one typed capability precondition is required",
                "preconditions": [],
            }
        try:
            normalized = tuple(
                item if isinstance(item, CapabilityPrecondition) else CapabilityPrecondition.from_dict(item)
                for item in preconditions
            )
        except (TypeError, ValueError) as exc:
            return {
                "state": "unsupported",
                "verified": False,
                "adapter_id": self.adapter_id,
                "operation_id": operation_id,
                "reason": f"capability precondition is invalid: {exc}",
                "preconditions": [],
            }

        results: list[dict[str, Any]] = []

        def refused(state: str, reason: str) -> dict[str, Any]:
            for item in normalized[len(results) :]:
                results.append(
                    {
                        "name": item.name,
                        "schema_version": item.schema_version,
                        "state": state,
                        "evidence": {"reason": reason},
                    }
                )
            response: dict[str, Any] = {
                "state": state,
                "verified": False,
                "adapter_id": self.adapter_id,
                "operation_id": operation_id,
                "preconditions": results,
                "reason": reason,
            }
            if isinstance(execution_context, ExecutionContext):
                response["claim_id"] = execution_context.claim_id
                response["fencing_token"] = execution_context.fencing_token
            return response

        if not isinstance(execution_context, ExecutionContext):
            return refused("unsupported", "a typed granted ExecutionContext is required")
        if execution_context.purpose != ExecutionPurpose.JOB:
            return refused("unsupported", "GPU prelaunch quiescence requires a job-purpose ExecutionContext")
        if execution_context.claim_id is None or execution_context.fencing_token is None:
            return refused("unsupported", "GPU prelaunch quiescence requires a fenced scheduler grant")
        if any(item.name != "gpu.quiescence" or item.schema_version != 1 for item in normalized):
            return refused("unsupported", "NVIDIA adapter supports only gpu.quiescence schema version 1")
        try:
            local_host = str(self._hostname())
        except (OSError, RuntimeError) as exc:
            return refused("unknown", f"local host identity is unavailable: {type(exc).__name__}")
        allocations = {item.resource_id: item for item in execution_context.allocations}
        resources = {item.resource_id: item for item in execution_context.resources}
        gpu_targets: list[dict[str, Any]] = []
        for resource_id, allocation in allocations.items():
            resource = resources.get(resource_id)
            if resource is None:
                return refused("unknown", f"grant descriptor for {resource_id} is absent")
            if resource.kind.casefold() != "gpu":
                continue
            gpu_uuid = resource.attributes.get("uuid")
            if resource.host_id != local_host or not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
                return refused("unsupported", f"GPU grant {resource_id} is not a local stable UUID resource")
            if resource.resource_id != f"gpu:{local_host}:{gpu_uuid}":
                return refused("blocked", f"GPU grant {resource_id} does not match its host-qualified UUID")
            try:
                grant_memory = self._memory_bytes(allocation.quantities)
            except ValueError as exc:
                return refused("unsupported", f"GPU grant quantities are invalid: {exc}")
            gpu_targets.append(
                {
                    "resource_id": resource_id,
                    "gpu_uuid": gpu_uuid,
                    "grant_memory_bytes": grant_memory,
                }
            )
        if not gpu_targets:
            return refused("unsupported", "GPU quiescence precondition has no granted GPU resources to inspect")

        for index, precondition in enumerate(normalized):
            params = dict(precondition.parameters)
            allowed = {
                "consecutive_samples",
                "interval_seconds",
                "timeout_seconds",
                "max_utilization_percent",
                "minimum_free_bytes",
            }
            extras = set(params) - allowed
            if extras:
                return refused("unsupported", f"unsupported gpu.quiescence parameters: {sorted(extras)}")
            samples = params.get("consecutive_samples", 3)
            interval = params.get("interval_seconds", 0.5)
            timeout = params.get("timeout_seconds", 10.0)
            max_utilization = params.get("max_utilization_percent", 0)
            requested_floor = params.get("minimum_free_bytes")
            if (
                isinstance(samples, bool)
                or not isinstance(samples, int)
                or not 1 <= samples <= 20
                or isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(interval)
                or not 0 <= interval <= 10
                or (samples > 1 and interval <= 0)
                or isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or not 0 <= timeout <= 60
                or timeout < interval * (samples - 1)
                or isinstance(max_utilization, bool)
                or not isinstance(max_utilization, int)
                or not 0 <= max_utilization <= 100
                or (
                    requested_floor is not None
                    and (
                        isinstance(requested_floor, bool)
                        or not isinstance(requested_floor, int)
                        or requested_floor < 0
                    )
                )
            ):
                return refused("unsupported", "gpu.quiescence parameters are outside the supported bounded schema")
            targets = []
            for target in gpu_targets:
                grant_floor = target["grant_memory_bytes"] + self._external_lease_headroom_bytes
                if requested_floor is not None and requested_floor < grant_floor:
                    return refused(
                        "blocked",
                        "minimum_free_bytes cannot be lower than the granted GPU budget plus configured headroom",
                    )
                targets.append(
                    {
                        "resource_id": target["resource_id"],
                        "gpu_uuid": target["gpu_uuid"],
                        "required_free_bytes": max(grant_floor, requested_floor or 0),
                        "max_utilization_percent": max_utilization,
                    }
                )
            observation = self._wait_external_gpu_samples(
                targets,
                consecutive_samples=samples,
                interval_seconds=float(interval),
                timeout_seconds=float(timeout),
            )
            result = {
                "name": precondition.name,
                "schema_version": precondition.schema_version,
                "state": "verified" if observation["state"] == "observed" else observation["state"],
                "resource_ids": [item["resource_id"] for item in targets],
                "evidence": {
                    "precondition_index": index,
                    "parameters": {
                        "consecutive_samples": samples,
                        "interval_seconds": float(interval),
                        "timeout_seconds": float(timeout),
                        "max_utilization_percent": max_utilization,
                        "minimum_free_bytes": [item["required_free_bytes"] for item in targets],
                    },
                    "observation": observation,
                },
            }
            results.append(result)
            if observation["state"] != "observed":
                return {
                    "state": observation["state"],
                    "verified": False,
                    "adapter_id": self.adapter_id,
                    "operation_id": operation_id,
                    "claim_id": execution_context.claim_id,
                    "fencing_token": execution_context.fencing_token,
                    "resource_ids": sorted(item["resource_id"] for item in gpu_targets),
                    "preconditions": results + [
                        {
                            "name": item.name,
                            "schema_version": item.schema_version,
                            "state": observation["state"],
                            "evidence": {"reason": observation.get("reason")},
                        }
                        for item in normalized[len(results) :]
                    ],
                    "reason": observation.get("reason"),
                }
        return {
            "state": "verified",
            "verified": True,
            "adapter_id": self.adapter_id,
            "operation_id": operation_id,
            "claim_id": execution_context.claim_id,
            "fencing_token": execution_context.fencing_token,
            "resource_ids": sorted(item["resource_id"] for item in gpu_targets),
            "preconditions": results,
            "limitations": [
                "read-only prelaunch observation is not a physical reservation",
                "compute-process telemetry covers only nvidia-smi-visible applications",
                "GPU memory capacity remains accounting-only",
            ],
        }

    def capture_lease_baseline(
        self,
        lease: Any,
        resource_descriptors: Sequence[Resource | Mapping[str, Any]],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Capture a trusted idle baseline before returning an external lease."""
        if not isinstance(operation_id, str) or not operation_id.strip() or "\x00" in operation_id:
            raise ValueError("operation_id must be a non-empty NUL-free string")
        parsed, error, unsupported_state = self._external_lease_inputs(lease, resource_descriptors)
        if parsed is None:
            return {
                "state": unsupported_state,
                "lease_id": self._lease_value(lease, "lease_id"),
                "fencing_token": self._lease_value(lease, "fencing_token"),
                "resource_ids": [],
                "operation_id": operation_id,
                "verified": False,
                "reason": error,
            }
        targets = [
            {
                **target,
                "required_free_bytes": target["memory_budget_bytes"]
                + self._external_lease_headroom_bytes,
            }
            for target in parsed["targets"]
        ]
        observation = self._wait_external_gpu_samples(targets)
        samples = observation.get("samples", [])
        if observation.get("state") != "observed":
            return {
                "state": observation.get("state", "unknown"),
                "lease_id": parsed["lease_id"],
                "fencing_token": parsed["fencing_token"],
                "resource_ids": parsed["resource_ids"],
                "operation_id": operation_id,
                "observation_id": uuid.uuid4().hex,
                "verified": False,
                "reason": observation.get("reason"),
                "samples": samples,
            }
        last_sample = samples[-1]
        baseline_gpus = []
        for item in last_sample["gpus"]:
            target = next(target for target in parsed["targets"] if target["resource_id"] == item["resource_id"])
            baseline_gpus.append(
                {
                    **item,
                    "memory_budget_bytes": target["memory_budget_bytes"],
                    "headroom_bytes": self._external_lease_headroom_bytes,
                }
            )
        observed_at = last_sample["inventory_observed_at"]
        return {
            "state": "observed",
            "lease_id": parsed["lease_id"],
            "fencing_token": parsed["fencing_token"],
            "resource_ids": parsed["resource_ids"],
            "operation_id": operation_id,
            "observation_id": uuid.uuid4().hex,
            "observed_at": observed_at,
            "verified": True,
            "baseline": {
                "schema_version": 1,
                "adapter_id": self.adapter_id,
                "lease_id": parsed["lease_id"],
                "fencing_token": parsed["fencing_token"],
                "resource_ids": parsed["resource_ids"],
                "captured_at": observed_at,
                "headroom_bytes": self._external_lease_headroom_bytes,
                "gpus": baseline_gpus,
            },
            "evidence": {
                "source": "nvidia-smi",
                "sample_count": observation["consecutive_samples"],
                "samples": samples,
                "process_scope": "nvidia-smi-visible-compute-applications",
                "limitations": [
                    "process telemetry covers only compute applications visible to this nvidia-smi version",
                    "read-only quiescence does not establish hostile-process isolation",
                    "GPU memory remains an accounting observation, not an enforced limit",
                ],
            },
        }

    def inspect_lease_release(
        self,
        lease: Any,
        baseline: Mapping[str, Any],
        resource_descriptors: Sequence[Resource | Mapping[str, Any]],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Observe external GPU release without signalling or unloading work."""
        if not isinstance(operation_id, str) or not operation_id.strip() or "\x00" in operation_id:
            raise ValueError("operation_id must be a non-empty NUL-free string")
        parsed, error, invalid_state = self._external_lease_inputs(lease, resource_descriptors)
        if parsed is None:
            return {
                "verified": False,
                "state": invalid_state,
                "lease_id": self._lease_value(lease, "lease_id"),
                "fencing_token": self._lease_value(lease, "fencing_token"),
                "resource_ids": [],
                "operation_id": operation_id,
                "reason": error,
            }
        if not isinstance(baseline, Mapping):
            return {
                "verified": False,
                "state": "blocked",
                "lease_id": parsed["lease_id"],
                "fencing_token": parsed["fencing_token"],
                "resource_ids": parsed["resource_ids"],
                "operation_id": operation_id,
                "reason": "external lease release requires a persisted baseline mapping",
            }
        nested_baseline = baseline.get("baseline")
        raw_baseline = nested_baseline if isinstance(nested_baseline, Mapping) else baseline
        baseline_token = raw_baseline.get("fencing_token") if isinstance(raw_baseline, Mapping) else None
        current_token = parsed["fencing_token"]
        if (
            not isinstance(raw_baseline, Mapping)
            or baseline.get("state") != "observed"
            or baseline.get("verified") is not True
            or baseline.get("lease_id") != parsed["lease_id"]
            or baseline.get("resource_ids") != parsed["resource_ids"]
            or raw_baseline.get("adapter_id") != self.adapter_id
            or raw_baseline.get("lease_id") != parsed["lease_id"]
            or raw_baseline.get("resource_ids") != parsed["resource_ids"]
            or isinstance(baseline_token, bool)
            or not isinstance(baseline_token, int)
            or isinstance(current_token, bool)
            or not isinstance(current_token, int)
            or current_token <= baseline_token
        ):
            return {
                "verified": False,
                "state": "blocked",
                "lease_id": parsed["lease_id"],
                "fencing_token": current_token,
                "resource_ids": parsed["resource_ids"],
                "operation_id": operation_id,
                "reason": "external lease release requires its exact persisted baseline and a rotated fencing token",
            }
        baseline_gpus = raw_baseline.get("gpus")
        if isinstance(baseline_gpus, (str, bytes)) or not isinstance(baseline_gpus, Sequence):
            baseline_gpus = ()
        baseline_by_id = {
            item.get("resource_id"): item
            for item in baseline_gpus
            if isinstance(item, Mapping)
        }
        if set(baseline_by_id) != set(parsed["resource_ids"]):
            return {
                "verified": False,
                "state": "blocked",
                "lease_id": parsed["lease_id"],
                "fencing_token": current_token,
                "resource_ids": parsed["resource_ids"],
                "operation_id": operation_id,
                "reason": "persisted external lease baseline does not cover the exact allocation",
            }
        capture_headroom = raw_baseline.get("headroom_bytes")
        if isinstance(capture_headroom, bool) or not isinstance(capture_headroom, int) or capture_headroom < 0:
            return {
                "verified": False,
                "state": "blocked",
                "lease_id": parsed["lease_id"],
                "fencing_token": current_token,
                "resource_ids": parsed["resource_ids"],
                "operation_id": operation_id,
                "reason": "persisted external lease baseline has invalid headroom",
            }
        release_slack = min(capture_headroom, self._external_lease_headroom_bytes)
        targets: list[dict[str, Any]] = []
        for target in parsed["targets"]:
            item = baseline_by_id[target["resource_id"]]
            if item.get("gpu_uuid") != target["gpu_uuid"]:
                return {
                    "verified": False,
                    "state": "blocked",
                    "lease_id": parsed["lease_id"],
                    "fencing_token": current_token,
                    "resource_ids": parsed["resource_ids"],
                    "operation_id": operation_id,
                    "reason": "persisted baseline GPU UUID does not match the current resource descriptor",
                }
            free_at_capture = item.get("memory_free_bytes")
            total_at_capture = item.get("memory_total_bytes")
            if (
                isinstance(free_at_capture, bool)
                or not isinstance(free_at_capture, int)
                or free_at_capture < 0
                or isinstance(total_at_capture, bool)
                or not isinstance(total_at_capture, int)
                or total_at_capture <= 0
            ):
                return {
                    "verified": False,
                    "state": "blocked",
                    "lease_id": parsed["lease_id"],
                    "fencing_token": current_token,
                    "resource_ids": parsed["resource_ids"],
                    "operation_id": operation_id,
                    "reason": "persisted external lease memory baseline is invalid",
                }
            targets.append(
                {
                    **target,
                    "required_free_bytes": max(0, free_at_capture - release_slack),
                    "baseline_total_bytes": total_at_capture,
                }
            )
        observation = self._wait_external_gpu_samples(targets)
        samples = observation.get("samples", [])
        verified = observation.get("state") == "observed"
        if verified:
            for target, observed in zip(targets, samples[-1]["gpus"]):
                if observed["memory_total_bytes"] != target["baseline_total_bytes"]:
                    verified = False
                    observation = {
                        **observation,
                        "state": "unknown",
                        "reason": "GPU memory capacity changed since the external lease baseline",
                    }
                    break
        return {
            "verified": verified,
            "state": "reclaimed" if verified else observation.get("state", "unknown"),
            "lease_id": parsed["lease_id"],
            "fencing_token": current_token,
            "baseline_fencing_token": baseline_token,
            "resource_ids": parsed["resource_ids"],
            "operation_id": operation_id,
            "observation_id": uuid.uuid4().hex,
            "observed_at": samples[-1].get("inventory_observed_at") if samples else None,
            "reason": observation.get("reason"),
            "evidence": {
                "source": "nvidia-smi",
                "sample_count": observation.get("consecutive_samples", len(samples)),
                "samples": samples,
                "free_memory_floor": "persisted-pregrant-baseline-minus-configured-headroom",
                "process_scope": "nvidia-smi-visible-compute-applications",
                "limitations": [
                    "read-only observation; no process was signalled, killed, or unloaded",
                    "process telemetry covers only compute applications visible to this nvidia-smi version",
                    "GPU memory remains an accounting observation, not an enforced limit",
                ],
            },
        }

    def observe_processes(self, gpu_uuid: str) -> dict[str, Any]:
        processes, error = self._processes()
        observed_at = datetime.now(timezone.utc).isoformat()
        if processes is None:
            return {
                "state": "unknown",
                "gpu_uuid": gpu_uuid,
                "processes": None,
                "observed_at": observed_at,
                "reason": error,
                "scope": "nvidia-smi-visible-compute-applications",
            }
        if any(process["identity_unknown"] for process in processes):
            return {
                "state": "unknown",
                "gpu_uuid": gpu_uuid,
                "processes": processes,
                "observed_at": observed_at,
                "reason": "one-or-more-reported-compute-processes-have-unknown-device-identity",
                "scope": "nvidia-smi-visible-compute-applications",
            }
        selected = [process for process in processes if process["gpu_uuid"] == gpu_uuid]
        return {
            "state": "busy" if selected else "empty",
            "gpu_uuid": gpu_uuid,
            "processes": selected,
            "observed_at": observed_at,
            "scope": "nvidia-smi-visible-compute-applications",
        }

    def wait_quiescent(
        self,
        gpu_uuid: str,
        *,
        consecutive_samples: int = 3,
        interval_seconds: float = 0.5,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Require repeated successful empty process observations for one UUID."""
        if consecutive_samples < 1:
            raise ValueError("consecutive_samples must be at least one")
        if (
            isinstance(consecutive_samples, bool)
            or not isinstance(consecutive_samples, int)
            or consecutive_samples < 1
        ):
            raise ValueError("consecutive_samples must be a positive integer")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds < 0
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("interval and timeout must be non-negative")
        deadline = self._clock() + timeout_seconds
        samples: list[dict[str, Any]] = []
        empty_count = 0
        while True:
            sample = self.observe_processes(gpu_uuid)
            samples.append(sample)
            if sample["state"] == "unknown":
                return {
                    "state": "unknown",
                    "gpu_uuid": gpu_uuid,
                    "samples": samples,
                    "reason": sample["reason"],
                    "scope": sample["scope"],
                }
            if sample["state"] == "busy":
                empty_count = 0
            else:
                empty_count += 1
                if empty_count >= consecutive_samples:
                    return {
                        "state": "quiescent",
                        "gpu_uuid": gpu_uuid,
                        "samples": samples,
                        "consecutive_empty_samples": empty_count,
                        "scope": sample["scope"],
                        "limitations": [
                            "only compute processes reported by this nvidia-smi version are observed",
                            "process absence does not prove hostile-process isolation",
                        ],
                    }
            remaining = deadline - self._clock()
            if remaining <= 0:
                return {
                    "state": "busy" if sample["state"] == "busy" else "unknown",
                    "gpu_uuid": gpu_uuid,
                    "samples": samples,
                    "reason": "quiescence_sample_threshold_not_met_before_deadline",
                    "scope": sample["scope"],
                }
            self._sleep(min(interval_seconds, remaining))

    def wait_memory_reclaimed(
        self,
        gpu_uuid: str,
        minimum_free_bytes: int,
        *,
        consecutive_samples: int = 3,
        interval_seconds: float = 0.5,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Verify both process quiescence and a requested observed free-memory floor."""
        if (
            isinstance(minimum_free_bytes, bool)
            or not isinstance(minimum_free_bytes, int)
            or minimum_free_bytes < 0
        ):
            raise ValueError("minimum_free_bytes must be a non-negative integer")
        if (
            isinstance(consecutive_samples, bool)
            or not isinstance(consecutive_samples, int)
            or consecutive_samples < 1
            or isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds < 0
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("sample count must be positive and times non-negative")
        deadline = self._clock() + timeout_seconds
        samples: list[dict[str, Any]] = []
        qualifying = 0
        while True:
            process_observation = self.observe_processes(gpu_uuid)
            devices, error = self._gpus()
            device = next((d for d in devices or () if d["uuid"] == gpu_uuid), None)
            free = None if device is None else device["memory_free_bytes"]
            sample = {
                "processes": process_observation,
                "memory_free_bytes": free,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            }
            samples.append(sample)
            if process_observation["state"] == "unknown" or devices is None or device is None or free is None:
                return {
                    "state": "unknown",
                    "gpu_uuid": gpu_uuid,
                    "samples": samples,
                    "reason": process_observation.get("reason") or error or "gpu memory observation unavailable",
                    "memory_enforcement": "accounting-only-observation",
                }
            if process_observation["state"] == "empty" and free >= minimum_free_bytes:
                qualifying += 1
                if qualifying >= consecutive_samples:
                    return {
                        "state": "reclaimed",
                        "gpu_uuid": gpu_uuid,
                        "samples": samples,
                        "consecutive_qualifying_samples": qualifying,
                        "minimum_free_bytes": minimum_free_bytes,
                        "memory_enforcement": "accounting-only-observation",
                        "process_scope": process_observation["scope"],
                        "evidence": {
                            "source": "nvidia-smi-observations",
                            "sample_count": qualifying,
                            "observed_at": samples[-1]["observed_at"],
                            "scope": process_observation["scope"],
                        },
                    }
            else:
                qualifying = 0
            remaining = deadline - self._clock()
            if remaining <= 0:
                return {
                    "state": "unverified",
                    "gpu_uuid": gpu_uuid,
                    "samples": samples,
                    "reason": "reclamation_sample_threshold_not_met_before_deadline",
                    "memory_enforcement": "accounting-only-observation",
                }
            self._sleep(min(interval_seconds, remaining))
