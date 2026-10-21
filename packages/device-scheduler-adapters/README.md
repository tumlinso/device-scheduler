# device-scheduler-adapters

Environment-specific integrations for resource discovery, execution, monitoring, and reclamation. Initial targets are NVIDIA GPUs, processes, and AI agents; support for additional environments and resource kinds should fit behind explicit adapter boundaries.

The existing CUDA controller remains the authority for GPU admission and interlocks until a deliberate migration is separately designed and authorized. Project Control integration, including managing its agents, also requires separate authorization.

This directory is documentation-only scaffolding pending a later authorized design and implementation phase.
