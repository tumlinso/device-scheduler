# Qualification v1: evidence record

This record distinguishes automated tests, live CPU service behavior, GPU admission, and the model-backed FrameShift handoff. A pass in one area does not qualify the others. Status below reflects retained evidence dated 2026-10-21; the production cutover is applied; known legacy admission routes are fenced.

## Current evidence

| Area | Result | Limits and status |
|---|---|---|
| Host test suite | **331 passed; 0 failures, errors, or skips** in [tests-release-final.xml](tests-release-final.xml). | Automated repository tests do not establish every live adapter or end-to-end guarantee. |
| CPU daemon, MCP, SDK/CLI and restart | [Receipt 07](daemon-receipt-07.json) passes all 14 checks. The same attempt and systemd invocation survived daemon `SIGKILL`/restart; one physical launch completed and cleanup was verified. | One cooperative finite CPU helper. This does not qualify GPU workloads, model eviction, or host-reboot recovery. |
| Live systemd execution | [Systemd receipt](device-scheduler-systemd-qualification-2gmis0qs/receipt.json) passes 11 named CPU adapter checks, including cgroup limits, process-tree cleanup, and fresh-process reconciliation. | Separate adapter-process reconciliation; daemon restart is covered by receipt 07. No hostile-workload isolation claim. |
| GPU allocation and exclusivity | [GPU receipt](gpu-allocations-passed.json) passes shared coexistence, exclusive wait for verified shared cleanup, exclusive execution after reclamation, unmanaged-process blocking, and resumption after unmanaged cleanup. | Topology/profiler conflict was skipped because the granted GPU has no topology isolation domain. Memory budgets are admission accounting, and process visibility follows `nvidia-smi` compute-process telemetry. This is not model eviction qualification. |
| FrameShift and model handoff | [Real handoff receipt](gpu-handoff-passed.json) passes all 17 checks: resident ordinary suspension, explicit eviction/reclamation, finite CUDA work, restored readiness, exactly-once delivery and actual model completion. | One bounded V100/llama.cpp/FrameShift journey. The prior expired unknown fixture was cancelled and physically cleaned up; its uncertain private scheduler history remains preserved. |
| Project Control and native Todo | DS-00 has an accepted native outcome. Its integration effect is finalized dirty with `resulting_source=null`; DS-01 through DS-14 remain pending. See [native status](native-status.md). | Native acceptance and Git integration are separate. No replacement integration operation is recorded. |
| CUDA-controller retirement | Both old registries are backed up and fenced; the old immutable coordinator refuses them, and the CUDA entrypoint refuses commands. | See [retirement audit](../cutover-v1/retirement-audit.json) and [deployment record](../cutover-v1/deployed-v1.md). |

## Remaining qualification gates

| Gate | Required evidence | Status |
|---|---|---|
| Model backend lifecycle | Pinned llama.cpp configuration, readiness, drain, explicit unload, verified reclamation, restore, and readiness/residency verification. | Passed in the real handoff receipt. |
| Full FrameShift handoff | Resume the saved request at its pending tool boundary after durable result delivery; verify weights remain resident during ordinary suspension and unload only for the authorized finite handoff. | Passed: the actual model completed generation 2 after the finite CUDA result. |
| Handoff failure recovery | Exercise cancellation and restart/failure at each durable phase; preserve results, avoid blind rerun, and verify explicit recovery. | Pending. |
| Topology/profiler isolation | Run the skipped conflict case on a qualified GPU topology/isolation domain. | Pending; current V100 receipt records the scenario as skipped. |
| Project Control consumer migration | Migrate and qualify resource allocation/ownership queries while preserving Project Control's task, workspace, agent, Todo, and acceptance authority. | Resource-facade lease, ownership, heartbeat and release passed live. Native DS-00 Git integration remains unresolved. |
| CUDA-controller cutover | Fence old admissions, reconcile ownership, switch consumers, preserve profiling/quiescence safeguards, and verify one allocator per pool. | Applied; known old routes refuse admissions. |

## Historical evidence and preserved failures

Earlier CPU daemon receipts [01](daemon-receipt-01.json), [02](daemon-receipt-02.json), [03](daemon-receipt-03.json), [04](daemon-receipt-04.json), [05](daemon-receipt-05.json), and [06](daemon-receipt-06.json) remain unchanged. They include incomplete runs and intermediate validations; receipt 07 is the current CPU daemon result. Earlier aggregate test reports [tests-current.xml](tests-current.xml), [tests-pre-gpu.xml](tests-pre-gpu.xml), and [tests-pre-gpu-fixed.xml](tests-pre-gpu-fixed.xml) are retained as historical snapshots.

The prior focused FrameShift evidence is still limited to a bounded Engine/SQLite restart run and host-socket tests: five restart tests passed with a scripted provider, and the separate service/engine/store focused run reported 19 tests passed with two subtests. Neither exercised GPU eviction or model inference. Ordinary pause evidence is [gpu-handoff-05-ordinary-pause-passed.json](evidence/gpu-handoff-05-ordinary-pause-passed.json); partial model-handoff evidence remains in [gpu-handoff-06-model-partial-failure.json](evidence/gpu-handoff-06-model-partial-failure.json) with its cleanup record [gpu-handoff-06-cleanup.json](gpu-handoff-06-cleanup.json).

Three earlier handoff attempts stopped before any model handoff could be qualified: [FrameShift import failure](evidence/gpu-handoff-preallocation-import-failure-2026-10-21.json), [daemon configuration failure](evidence/gpu-handoff-daemon-config-failure-2026-10-21.json), and [resource-status shape failure](evidence/gpu-handoff-resource-status-unwrapped-2026-10-21.json). Their logs and cleanup receipts remain alongside them. The earlier failed systemd adapter run is preserved at [vylywmqg](device-scheduler-systemd-qualification-vylywmqg/receipt.json).

## Evidence entry template

Add a dated subsection for each new gate result. Attach immutable source/runtime identity and the exact evidence artifact; do not replace prior results when behavior changes.

```text
Gate:
Observed at (UTC):
Host and runtime identity:
Source commit / dirty paths / configuration digest:
Resource and service identities:
Command or operation IDs:
Evidence artifact paths and SHA-256 digests:
Result and accepted assertions:
Failures, unknowns, and unsupported guarantees:
Independent reviewer:
```

Keep hardware measurements separate from simulated policy replay. Distinguish ledger accounting from adapter observations and enforceable limits. Keep daemon-restart recovery separate from host-reboot recovery. The retirement audit records refusal by old controller/coordinator routes for the migrated pools; archival source is preserved for rollback and historical evidence.
