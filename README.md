# device-scheduler

An MCP-first, environment-agnostic scheduler for coordinating access to resources and supervising work. It is intended to manage resource leases, immediate and scheduled jobs, background work, priorities, configurable scheduling and eviction policies, and execution of processes and AI agents. Future work may include managing Project Control agents.

## Initial architecture

- **device-scheduler-core** owns environment-neutral resource models, lease and job lifecycles, scheduling policies, and eviction decisions.
- **device-scheduler-mcp** provides the primary interface for submitting, scheduling, inspecting, and controlling work. It exposes core capabilities without owning scheduling policy.
- **device-scheduler-adapters** discovers resources and handles environment-specific execution, monitoring, and reclamation. Initial targets are NVIDIA GPUs, processes, and agents.

Keep the core independent of NVIDIA/CUDA, Project Control, and MCP transport details. Resource kinds, environments, policies, and interface transports should remain extensible through clear package boundaries.

Project Control remains the authority for project knowledge, workflow, and native Todo state. Any scheduler connection to Project Control, including agent management, requires a separately authorized design. The existing CUDA controller remains the GPU admission and interlock authority until a deliberate, separately reviewed migration.

This repository is at setup stage. Detailed design and implementation are tracked separately.
