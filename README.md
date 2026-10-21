# Device Scheduler

Device Scheduler is an MCP-first authority for allocating resources and supervising work. It is designed for applications, agents, development tools, and background jobs that need durable resource leases, scheduled execution, policy-controlled admission, and inspectable results.

The first deployment targets one Linux host and one administrative domain. Python and SQLite provide the initial implementation foundation; the host's systemd user manager supervises processes. The design keeps resource decisions independent of a particular device or runtime so adapters can add environments without moving environment-specific behavior into the core.

## Product boundaries

- **device-scheduler-core** defines resource, lease, job, attempt, policy, and handoff lifecycles and validates scheduling decisions.
- **device-scheduler-mcp** exposes the service through MCP, a typed Python client, and a CLI. All surfaces use one daemon and the same application contracts.
- **device-scheduler-adapters** provides persistence, Linux process execution, NVIDIA inventory and controls, managed model backend operations, and continuation integration.

Project Control remains authoritative for project tasks, workspaces, coordination, Todo, and acceptance. Device Scheduler is responsible for physical resource allocation and execution. It may later supervise Project Control agents as workloads, but does not take over their project or task lifecycle. The retired CUDA controller remains historical evidence for GPU reservations, quiescence, profiling isolation and preemption. Device Scheduler owns allocation on this host; known old admission routes are fenced.

FrameShift owns durable agent continuations. Ordinary suspension releases active inference capacity while model weights may remain resident. Model-weight eviction requires an explicit, authorized handoff request. That finite request durably coordinates suspension, resource reclamation, workload execution, restoration, and result delivery at the caller's pending tool boundary.

## Development

Install the workspace and development tools with:

```sh
uv sync --group dev
```

Run the repository suite with:

```sh
uv run pytest
```

For local foreground daemon development, use the example host configuration:

```sh
uv run --locked device-scheduler --config deploy/device-scheduler.example.toml serve
```

The daemon listens on loopback port `8770` by default. Its host-local defaults place configuration and client tokens under `~/.config/device-scheduler/` and ledger/artifacts under `~/.local/state/device-scheduler/`. See [deployment guidance](deploy/README.md) for installing and running the user service. The loopback service exposes `/healthz` and `/readyz`; MCP operations require a scoped token.

## Current state

A working single-host v1 is deployed as `device-scheduler.service`. MCP, SDK and CLI share one SQLite authority. The host suite passes **331 tests**. Live checks cover systemd limits and cleanup, a CPU job surviving daemon restart without relaunch, shared/exclusive GPU allocation, and the complete llama.cpp/FrameShift request–suspend–evict–run–restore–resume journey. Ordinary suspension kept weights resident; the explicit handoff delivered its result once and resumed the actual model.

Project Control is healthy, discovers this workspace and reads its native Todo authority. Its resource facade and the active KDA/KernelBender configurations use Device Scheduler. Both legacy registries are fenced, and the old CUDA controller entrypoint refuses commands. See [deployment evidence](planning/cutover-v1/deployed-v1.md).

The full handoff failure matrix, host-reboot qualification and topology/profiler isolation on a suitable domain remain follow-up work. Production managed-model profiles require operator registration; client connections do not launch models. Native Todo acceptance is separate from delivered code: DS-00 is accepted with an unresolved Git integration; DS-01–DS-14 remain pending.

See [planning/](planning/README.md) for contracts and implementation scope, [the capability matrix](planning/capability-matrix.md) for implemented versus evidenced behavior, and [qualification-v1](planning/qualification-v1/README.md) for qualification evidence. Never infer a live guarantee from implementation code or fake-adapter tests alone.
