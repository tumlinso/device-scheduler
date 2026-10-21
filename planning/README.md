# Product design baseline

This directory records the accepted first-release product boundary and implementation sequence. These documents state intended contracts; they are not evidence that code, adapters, runtime permissions, or GPU behavior have been implemented or qualified.

- [Product specification](product-spec.md) — responsibilities, lifecycle semantics, policy and security boundaries.
- [Contracts](contracts.md) — public operation families, entities, adapter ports, and request/result behavior.
- [Capability matrix](capability-matrix.md) — intended first-release capabilities and evidence still required.
- [Acceptance](acceptance.md) — implementation and retirement gates.
- [Native implementation plan](implementation-plan.json) — schema-1 Todo plan for DS-00 through DS-14.

Project Control workspace identity is project `5c878e94-2d97-4902-b594-9adb76c3e7fe`, authority `9b19daf8-d7c5-5e56-8b9f-df3630d94961`. The native plan is a reviewable file only; preview and commit it through supported Project Control routes. Never edit Todo state or generated projections directly.

## Ownership boundaries

Project Control owns project tasks, workspaces, agent coordination, Todo authority, and acceptance. Device Scheduler owns host resource allocation, leases, execution attempts, and execution evidence. FrameShift owns logical continuations and model/frame semantics. Integrations use explicit adapters and preserve each system's authority.

The CUDA controller remains the current operational authority until DS-13 cutover. Qualification must establish a single active allocator for each resource pool. CUDA controller retirement includes migrating the narrow Project Control resource-allocation and ownership-query consumers; it does not transfer Project Control agent lifecycle ownership.
