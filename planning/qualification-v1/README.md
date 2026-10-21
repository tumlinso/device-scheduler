# Qualification v1: evidence record

This record separates source and unit tests, live CPU service execution, scripted continuation integration, and real GPU evidence. A pass in one class does not qualify the others. The implementation is under active development; the product acceptance journey and CUDA-controller retirement are not complete.

## Current evidence

| Area | Observed evidence | Limits / status |
|---|---|---|
| Live systemd user-unit adapter | CPU-only finite helper passed 11 named checks: deterministic unit identity, invocation binding, whole-tree cgroup match, CPU quota, memory limit, task limit, fresh-process observation and reconciliation of the same invocation, stop acknowledgement, child-tree disappearance, and cgroup cleanup. See [receipt](device-scheduler-systemd-qualification-2gmis0qs/receipt.json) (`verdict: pass`; SHA-256 `3230ae7b86f61d2b4ee42e1561cd68cd282c9c032a38b711587a227a3b803718`). | The stop receipt reports `timeout`, while final cleanup and child-tree disappearance are verified. Fresh Python-process adapter reconciliation only; scheduler-daemon restart was not exercised. No GPU/model or hostile-workload qualification. The earlier failed run [vylywmqg](device-scheduler-systemd-qualification-vylywmqg/receipt.json) remains preserved. |
| Live daemon, HTTP/MCP, SDK/CLI, daemon crash | Receipt 04 passed health, denied unauthorized MCP access, SDK submission, CLI access to the same job, running-before-crash, same attempt and invocation after restart, one physical launch, completion, and verified cleanup. See [receipt](daemon-receipt-04.json) (SHA-256 `2d2cfaef5217e8672454c61df329281a99b08cffa38dd0e866e2ff17278e6efc`). | One cooperative finite CPU helper; GPU allocations were not requested. Earlier incomplete receipts 01–03 are retained. This does not establish model eviction or FrameShift agent resumption. |
| FrameShift host-socket tests | 19 passed, with 2 subtests, using `PYTHONPATH=/home/tumlinson/FrameShift/src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_scheduler_handoff.py tests/test_engine.py tests/test_store.py tests/test_service.py` from `/home/tumlinson/FrameShift`. | Focused test run; not a full FrameShift suite and not the GPU/model handoff journey. |
| FrameShift restart qualification | 5 tests passed in a bounded real-engine, SQLite-backed test with a scripted provider: `PYTHONPATH=/home/tumlinson/FrameShift/src /home/tumlinson/device-scheduler/.venv/bin/python -m pytest -q tests/test_frameshift_restart_qualification.py`. | Command output was reported by the verifier; no separate JSON receipt was retained. It exercised no inference or GPU resource reclamation. |
| Real GPU handoff harness | A plan-only-by-default harness and bounded CUDA allocation/copy/free helper are available in `scripts/qualify_gpu_handoff.py` and `scripts/qualification_gpu_probe.py`; see [runbook](gpu-handoff-harness.md). | Harness unit tests validate grant checks, pinned config, and request bounds. No live GPU/model handoff has been run. |
| Project Control registry and native attempt recovery | Supported operational recovery created the missing host registry after current GPU/process queries were empty, then released the expired DS-00 attempt through the native workflow receipt. The amended native plan now has 15 formal requirements; the new DS-00 gate stored one passing evidence record and seven facts. | Historical ownership was not reconstructed. Compact completion reporting then exceeded its size bound before handback/candidate creation; the expired new attempt requires supported recovery. No native acceptance is claimed. |
| Post-binding CPU daemon qualification | [Receipt 05](daemon-receipt-05.json) passes all ten original transport/restart checks after grant-bound execution changes. | One finite CPU helper only; no GPU/model workload. Credentials and diagnostic ledger remain private in the receipt's `/tmp` directory. |

The latest complete shared-suite run reported 269 passed and one optional FrameShift skip. Subsequent external-lease and admission-precondition changes are still being integrated; the final aggregate baseline must be rerun after those changes. Earlier failed receipts are retained.

## Qualification gates

| Gate | Evidence required | Current status |
|---|---|---|
| Local test baseline | Corrected full `uv run pytest` result; list failures and distinguish fake from real adapters. | Last complete shared run: 269 passed, one optional skip. A final run with FrameShift present is pending current integration changes. |
| Developer install and daemon startup | Clean workspace install; start the foreground daemon using the example config; verify process identity and loopback listener on port `8770`. | Not separately recorded as an install/startup qualification. The live daemon operations above exercised a running service. |
| Live service readiness and MCP/SDK/CLI | Query `/healthz` and `/readyz`; retain build/source identity. Exercise authenticated discovery/read and denied scope; verify SDK/CLI use the same daemon. | Partial: receipts 04–05 verify readiness polling, health, denied MCP authorization, SDK submit, and CLI same-job access. Full capability discovery and persistent credential provisioning remain outstanding. |
| systemd user-unit execution | Bounded command, unit/invocation identity, cgroup, limits, logs, stop and cleanup. Exercise daemon restart separately. | Partial pass: CPU-only adapter checks in the linked receipt passed. Daemon restart and unauthorized systemd-control refusal were not covered by this receipt. |
| NVIDIA inventory and quiescence | Actual adapter inventory, UUID/topology, process telemetry and freshness; repeated quiescence and reclaimed-memory thresholds on an isolated pool. | Pending. Host inventory is only a point-in-time observation in the DS-00 baseline. |
| GPU concurrency and exclusivity | On an explicitly isolated pool, verify shared admission, headroom, explicit whole-device exclusive blocking, topology/profiling conflicts and capacity return. | Pending. No GPU allocation or controller cutover qualification has occurred. |
| Managed model backend | Pin llama.cpp configuration; launch/readiness; attach consumers; drain, explicitly unload, verify reclamation, restore and verify readiness/residency. | Pending. |
| FrameShift continuation | Verify standing consumer contract, durable safe-boundary suspension, idempotent result delivery and resume across restart against the intended live instance. | Partial scripted evidence: focused host-socket tests and five bounded SQLite/scripted-provider restart tests passed. No live model inference or GPU eviction was exercised. |
| Request–suspend–evict–run–restore–resume | Ordinary suspension keeps weights resident; explicit finite authorized handoff drains consumers, evicts and verifies reclamation, runs work, persists results, restores the pinned model and resumes at the pending tool boundary. | Pending. Defining end-to-end acceptance has not been demonstrated. |
| Handoff failure/recovery | Inject restarts/failures at each durable phase; verify unknown/blocked states, no blind rerun, durable results, and explicit recovery. | Partial CPU daemon recovery evidence only; no complete handoff crash matrix. |
| Project Control resource-consumer migration | Inventory callers and bindings; migrate only allocation and ownership-query routes while preserving Project Control task, agent, workspace, coordination, Todo and acceptance ownership. | Pending. Recovery of the registry/native attempt is operational evidence, not consumer migration. |
| CUDA controller retirement | Fence old admissions, reconcile/transfer ownership, migrate clients, preserve profiler/quiescence safeguards and prove one allocator per pool. | Pending. The CUDA controller has not been retired. |

## Evidence entry template

Add a dated subsection for each completed gate. Attach or link immutable source/runtime identity and the exact evidence artifact; do not replace prior results when behavior changes.

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

Keep hardware measurements separate from simulated policy replay. Distinguish ledger accounting from adapter observations and enforceable limits. Keep daemon-restart recovery separate from host-reboot recovery. Do not mark CUDA retirement complete until the old controller and coordinator no longer grant resources to migrated pools and the single-authority audit is recorded.
