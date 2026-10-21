# Repository guidance

## Product and scope

- Follow the accepted product boundaries and contracts in `planning/`. Keep the core environment-neutral; keep Project Control's project/Todo/agent lifecycle and FrameShift's continuation authority with their respective owners.
- Model-weight eviction requires an explicit authorized finite handoff. Ordinary FrameShift suspension, queueing, conflict, reconnect, or lease cleanup never implies weight unloading. Preserve protected restoration and durable result delivery semantics.
- Treat planned capabilities as unimplemented until there is implementation and qualification evidence. Do not operate the CUDA controller and Device Scheduler as concurrent allocators over the same resource pool; controller retirement requires the DS-13 gate.
- Preserve unknown files, edits, history, active work, and Todo state. Do not reset, clean, or discard state to simplify a task.

## Autonomous engineering

- Use configured Codex subagents for bounded engineering objectives only when delegation materially reduces elapsed work without avoidable coordination. Select an authorized profile explicitly: `scout`, `researcher`, `implementer`, `debugger`, `optimizer`, `verifier`, `reviewer`, `maintainer`, or `principal`. Do not use generic/default profiles or Project Control `delegate_task`/`local-coding-worker` for engineering.
- Workers own coherent objectives, investigate/design/implement/test within their mandate, preserve concurrent work, and report exact evidence and limitations. Root retains workflow lifecycle and final acceptance.
- `principal` uses Sol and is reserved for an exceptional, evidence-backed root allocation. Workers cannot self-promote. Only the user may authorize Astra.
- Users control product direction and strategic choices. Ask when a material product decision is unresolved; do not silently expand scope.

## Project Control and validation

- Use supported Project Control tools for workflow and native Todo operations. Never edit authority databases or generated projections directly.
- For authorized workflow work, select workspace `device-scheduler` with `next_task`; use the returned handle with `inspect_task`, `coordinate_task`, and `finish_task`. Read-only context requests do not require a claim.
- Preview and review a native plan before committing its exact digest and expected revision. Apply only authorized workflow actions; preserve task state and evidence provenance.
- Check affected callers and contracts. Run relevant checks and report only checks that actually ran, plus unavailable checks, migration implications, and evidence limits.

## Optional CUDA kernel task guidance

- For CUDA kernel implementation, profiling, or optimization tasks, consult the current [KDA agent entry](/home/tumlinson/kda/docs/agent-entry.md) and explicitly select only relevant KDA skills. KDA is optional guidance; this repository retains authority over its source, tests, workflow, and evidence.
- Follow Device Scheduler's host resource authority and the active GPU controller until the qualified DS-13 cutover. GPU visibility alone does not authorize execution.
