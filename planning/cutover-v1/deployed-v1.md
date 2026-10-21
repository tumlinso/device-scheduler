# Deployed single-host v1 — 2026-10-21

Device Scheduler is enabled as the systemd user service `device-scheduler.service`, serving authenticated loopback MCP at `http://127.0.0.1:8770/mcp`. Its canonical ledger and artifacts are under `~/.local/state/device-scheduler`; private configuration and scoped credentials are under `~/.config/device-scheduler`. MCP, SDK and CLI share the daemon. Production backend/profile registries are intentionally empty until an operator registers workloads; connecting does not start a model.

Project Control uses its persistent `device_scheduler` resource-authority selector. Its final immutable candidate is `~/.local/state/project-control/venvs/pc-sealed-ds13-budget-a50b617-20261021`; the established `current` pointer and `project-control.service` select it. The sealed Todo 0.2.1 wheel and original project/authority identities are preserved. Inference remains inactive and demand-driven.

## Verified behavior

- [331 repository tests](../qualification-v1/tests-release-final.xml), plus 14 focused Project Control tests (4 subtests) and 11 KDA launcher tests (8 subtests).
- [CPU daemon qualification](../qualification-v1/daemon-receipt-07.json): authenticated MCP/SDK/CLI, real systemd limits/cleanup, the same job attempt and unit invocation surviving daemon SIGKILL/restart with one physical launch.
- [GPU admission qualification](../qualification-v1/gpu-allocations-passed.json): shared coexistence, exclusive wait/reclamation, unmanaged-process blocking and subsequent admission. The topology/profiler case is explicitly skipped on this pool.
- [Real llama.cpp/FrameShift handoff](../qualification-v1/gpu-handoff-passed.json): all 17 checks pass, including resident ordinary suspension, explicit weight eviction, verified reclamation, CUDA work, pinned restoration/readiness, durable result delivery once, model completion at generation 2 and physical fixture cleanup.
- [Project Control workspace/Todo access](project-control-workspace-todo.json): the registered workspace and authority are readable at revision 60; DS-00's candidate remains accepted.
- [Project Control lease smoke](project-control-lease-smoke.json): exclusive GPU lease, process-generation ownership, periodic heartbeat and verified release. A deliberately unrenewed earlier test expired and was reclaimed; its failure record is preserved.
- [KDA production smoke](kda-production-smoke.json): the actual migrated launcher ran a finite 32 MiB CUDA allocation/copy/free probe with the correct physical UUID and verified execution cleanup. KernelBender uses the same bridge with its own scoped identity and site configuration; its separate receipt is retained alongside this record.

## Retired routes and rollback material

[The fence receipt](legacy-fence-receipt.json) records SQLite backups and atomic tombstones at both known legacy registry roots. No authority rows were edited. [The retirement audit](retirement-audit.json) confirms the old immutable Project Control coordinator refuses both paths and the CUDA controller entrypoint refuses commands. KDA and KernelBender site configurations now select Device Scheduler without legacy fallback; no dormant campaign was armed.

The descriptor census inspected ordinary same-user workloads. Procfs hides a small, explicitly identified set of systemd/PAM/SSH session daemons; their exact identities and exclusions are recorded in the fence receipt. Both stopped Project Control units had zero stored descriptors. This is a cooperative single-admin cutover, not hostile same-user containment.

Exact original configuration/controller bytes, candidate preparation and retired SQLite files remain under `~/.local/state/device-scheduler/cutover-backups`. Rollback must first stop new scheduler admissions, reconcile all live jobs/leases/backends and stop Device Scheduler before restoring a legacy allocator. Never activate both allocators on the same pool.

## Remaining scope

The full handoff failure/crash matrix, host-reboot qualification, stdio handshake qualification and topology/profiler isolation on a suitable domain remain follow-up work. GPU budgets are accounting/admission controls. Same-user cgroups are lifecycle/resource controls, not hostile-code isolation. Legacy Project Control foreground preemption and mixed CPU/RAM external lease calls refuse unsupported guarantees; ordinary GPU leases and scheduler-supervised jobs are available. Managed model profiles require explicit registration.

Native Todo acceptance is separate from implementation delivery. DS-00 is accepted but its Git integration effect remains dirty with no resulting source; DS-01–DS-14 are still pending. The implementation, v1 live qualification and host cutover do not claim those native gates have been formally accepted. See [native status](../qualification-v1/native-status.md).

Cross-repository integration edits remain preserved in their source worktrees. Project Control runs an immutable installed candidate; FrameShift, KDA and KernelBender retain their integration edits without overwriting unrelated work. [Source hashes](integrated-source-identities.json) record the observed working-tree identities; no external repository commit or push is claimed.
