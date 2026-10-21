# Device Scheduler

Device Scheduler is an MCP-first authority for allocating resources and supervising work. It is designed for applications, agents, development tools, and background jobs that need durable resource leases, scheduled execution, policy-controlled admission, and inspectable results.

The first deployment targets one Linux host and one administrative domain. Python and SQLite provide the initial implementation foundation; the host's systemd user manager supervises processes. The design keeps resource decisions independent of a particular device or runtime so adapters can add environments without moving environment-specific behavior into the core.

## Product boundaries

- **device-scheduler-core** defines resource, lease, job, attempt, policy, and handoff lifecycles and validates scheduling decisions.
- **device-scheduler-mcp** exposes the service through MCP, a typed Python client, and a CLI. All surfaces use one daemon and the same application contracts.
- **device-scheduler-adapters** provides persistence, Linux process execution, NVIDIA inventory and controls, managed model backend operations, and continuation integration.

Project Control remains authoritative for project tasks, workspaces, coordination, Todo, and acceptance. Device Scheduler is responsible for physical resource allocation and execution. It may later supervise Project Control agents as workloads, but does not take over their project or task lifecycle. The existing CUDA controller is a source of GPU reservation, quiescence, profiling isolation, and preemption evidence; it will be retired after consumers migrate and single-authority operation is qualified.

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

Core scheduling, SQLite persistence, Linux/NVIDIA/backend/FrameShift adapters, MCP client/server, and CLI code are under active development. Live CPU checks verify systemd limits and cleanup, authenticated MCP/SDK/CLI access, and a job surviving daemon restart without relaunch. Real FrameShift Engine/SQLite tests verify durable barriers and delivery. GPU reclamation and the model-backed request–suspend–evict–run–restore–resume journey remain pending. The CUDA controller remains the allocation authority until migrated consumers pass the single-authority cutover gate.

See [planning/](planning/README.md) for contracts and implementation scope, [the capability matrix](planning/capability-matrix.md) for implemented versus evidenced behavior, and [qualification-v1](planning/qualification-v1/README.md) for pending release evidence. Never infer a live guarantee from implementation code or fake-adapter tests alone.
