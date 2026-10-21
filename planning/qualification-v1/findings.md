# Qualification findings — application and handoff boundaries

This is a bounded adversarial software check for the DS-12 integrated qualification objective. It uses the real SQLite ledger and application services with deterministic fake execution/backend/continuation adapters. It does not qualify systemd, CUDA hardware, llama.cpp, FrameShift runtime behavior, or production authentication transport.

## Recovered authority and scope

Project Control observer access after the service restart succeeded for `device-scheduler`. The native Todo provider reported all four read/workflow providers available at semantic revision 26. Exact task lookup resolves `DS-12` (“Integrated qualification”), still planned, with an objective to independently qualify lifecycle, crash, transport, concurrency, FrameShift handoff, security, and real GPU behavior on an isolated pool. This test strand made no Todo mutations.

Relevant acceptance requirements are in [acceptance.md](../acceptance.md): duplicate requests must not duplicate work; a restart must reconcile stable execution identity without blind relaunch; invalid grants and resource requests cannot bypass validation; and handoff work must preserve its protected round-trip reservation.

## Matrix and result

| Case | Result | What it establishes |
|---|---|---|
| Same idempotency key, same job request | Pass | Returns one durable job identity. |
| Same idempotency key, changed job payload | Pass | Rejects the conflicting request. |
| Handoff from a caller without the eviction grant | Pass | Rejects before recording a handoff. |
| Handoff job working directory outside caller execution roots | Pass | Rejects before recording a handoff. |
| Configured custom scheduling policy | Pass | Application admission invokes it and follows its selected order. |
| Daemon restart with an unresolvable execution identity | Pass | Marks the attempt/job unknown and does not launch a second attempt. |
| Handoff job exceeds protected resource envelope | Pass | Does not start the over-budget command. |
| Handoff job has no declared resources after envelope release is requested | Pass after application fix | The job remains unstarted when its protected reservation is no longer active. |

The first run exposed a reservation-state bypass: once the handoff coordinator persisted its internal job, the test called the allocator's supported `release_envelope()` transition and ran `Scheduler.tick()`. The job had no `resources`, so it launched while the reservation was `release_requested`. The application owner added a live-reservation check in both admission feasibility and the start path. The regression now passes and confirms the command is not launched after release is requested.

The fake GPU inventory uses the same clock as the allocator and includes current inventory/process timestamps, free bytes, process telemetry state, known-process count, unknown/unmanaged counts, and managed PID list. Its stable identity follows the NVIDIA adapter's `gpu:<host>:<UUID>` form; the UUID attribute matches the telemetry identity, while the source is `nvidia-smi`. The fixture publishes through `reconcile_resource_inventory`, so the allocator's availability and source ownership fields are exercised instead of bypassed.

## Reproduction

Environment: Python 3.12.3 from `/home/tumlinson/project-control/.dev-venv/bin/python`; real SQLite files are created in pytest temporary directories; no service, systemd unit, GPU, or model was started. The repository source was a shared, uncommitted implementation worktree when this command ran.

```text
/home/tumlinson/project-control/.dev-venv/bin/python -m pytest -q tests/test_integration_qualification.py
........                                                                 [100%]
8 passed in 2.10s
```

An earlier run of this same command before the live-reservation guard failed only the reservation-loss case (`1 failed, 5 passed`). After the guard was visible in the shared implementation worktree, the command above passed all eight cases. The application change was made by the owning implementation strand; this strand verified the resulting behavior and did not edit application source.

Adjacent compatibility check, rerun after the handoff fixture was updated with its required durable execution handle:

```text
/home/tumlinson/project-control/.dev-venv/bin/python -m pytest -q tests/test_application.py tests/test_handoff.py tests/test_integration_qualification.py
................................                                         [100%]
32 passed in 9.79s
```

An earlier adjacent run found four handoff fixture failures because `FakeBackend.restore()` omitted the now-required durable execution handle. I notified the handoff owner, who corrected the fixture; the rerun above passed the application, handoff, and independent integration suites.

NVIDIA identity/telemetry contract cross-check:

```text
/home/tumlinson/project-control/.dev-venv/bin/python -m pytest -q tests/test_application.py tests/test_handoff.py tests/test_integration_qualification.py tests/test_nvidia.py
................................................                         [100%]
48 passed in 8.90s
```

The adapter tests confirm its canonical resource ID uses host plus UUID and process telemetry is joined by UUID. The integration fixture uses that same identity pattern and a distinct UUID from any device index; no source alias is used to select or match the resource.

The test is at [test_integration_qualification.py](../../tests/test_integration_qualification.py). Relevant implementation paths are `Scheduler._job_can_fit()` and `Scheduler._start_job()` in `packages/device-scheduler-core/src/device_scheduler_core/application.py`.

## Limits

The successful restart case exercises a fake adapter whose stable execution lookup returns no handle. It does not prove systemd invocation identity or host-reboot recovery. The handoff fakes return affirmative drain, reclamation, readiness, delivery, and resume receipts; those receipts test lifecycle ordering and refusal paths but do not prove physical eviction or FrameShift durability. Concurrency races across multiple daemon processes, MCP/SDK/CLI parity, topology, cancellation cleanup, and actual GPU behavior remain outside this strand.
