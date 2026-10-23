# Public and adapter contracts

This file fixes first-release behavior at the contract level. Wire schemas remain versioned API design work in DS-01/DS-09; the operation families and invariants below are authoritative while exact field encoding is implemented.

## Shared request rules

Every mutating request carries a schema version, authenticated scoped principal, idempotency key, and typed payload. Store a canonical request digest. Reusing a key with identical content returns the existing operation; reusing it with different content is a conflict. Stable IDs are opaque and host-qualified where they identify a local resource. Timestamps are UTC instants; recurring schedule definitions additionally carry an IANA timezone. Deadlines are absolute UTC instants, with durations represented separately.

Commands are argument vectors, never shell strings. A command selects a registered execution profile or a permitted command scope, a working directory, and explicitly granted environment/configuration inputs. Secret values are never stored in job specifications. Resource requirements and execution safety declarations are structured data, not policy-changing free-form tags. Priority and eviction authority derive from server-side grants.

All command results distinguish accepted/persisted state from external execution outcome. Long-running operations return a durable operation/job/handoff ID and may be waited on with a bounded wait. Disconnecting or timing out the wait does not cancel work. State inspection reports both authoritative ledger state and last verified adapter state.

## Entities and lifecycle

- **Resource**: stable identity, kind, host, typed attributes, capacity, topology, shared conflict domains, supported modes, and capabilities with enforcement/evidence level.
- **Resource request**: quantities/selectors, hard and soft constraints, placement relationships, requested duration, priority class from a server grant, and isolation intent.
- **Lease/allocation**: owner, request digest, granted bundle, expiry/renewal policy, fencing generation, mode, and state including reclamation/unknown.
- **Job**: durable desired workload, immutable submitted spec version, schedule/dependency source, execution contract, result policy, and current desired state.
- **Attempt**: one execution identity, operation journal, external handle, timestamps, outcome, bounded logs/artifacts, and cleanup receipt.
- **Managed backend**: registered launch/configuration profile, pinned version, readiness check, resource/residency allocation, attached consumers, drain/restore capabilities, and current verified state.
- **Handoff**: explicit finite job batch, caller continuation references, eviction grants, round-trip reservation, target backend, deadline, phase, operation journal, results, and delivery acknowledgements.
- **Schedule/event**: versioned schedule definition or authenticated event identity, occurrence keys, deduplication state, and resulting job IDs.

State changes are validated by core transitions. The API exposes desired state and observed state separately where external effects can lag or be ambiguous. `unknown` is an explicit nonterminal condition, not success/failure and not permission to replay.

## Public operation families

| Family | Operations and required behavior |
|---|---|
| Discovery | List resources and capabilities; inspect resource/topology; obtain health and compatibility information. Read-only. |
| Leases | Request bundle; inspect lease; renew within policy/grants; release; reconcile an uncertain release. Capacity is reusable only after required reclamation is verified. `usage=external` requires a trusted adapter reclamation contract. CPU/memory `usage=supervised` leases are consumed only by `jobs.submit` bound with the lease ID and current fence; that job is launched with the exact grant through the supervised execution adapter. One queued/running job may attach at a time. Release/expiry cancels only when the declared interruption contract permits it, then requires terminal attempt cleanup evidence. An idle lease has no authorized physical work and may be released. Unmanaged client processes, PID attachment, and owner acknowledgements do not qualify as CPU/memory reclamation evidence. |
| Jobs | Submit now or at a time; inspect; wait; stream/paginate bounded logs; cancel; request cooperative suspension/resume; inspect attempts; recover; retry only with authorization and restart-safety proof. |
| Handoffs | Submit explicit authorized finite batch; inspect all phases/results; cancel under cleanup rules; request recovery of blocked restoration/delivery. Submission verifies a feasible round-trip reservation. Ordinary job suspension has no eviction effect. |
| Schedules | Create/revise versioned interval or calendar schedule; inspect; pause/resume; cancel. Nominal occurrence IDs make creation idempotent. |
| Events/dependencies | Publish authenticated deduplicated event; add job-result dependency at submission; report dependency status; reject cycles. |
| Policy/explanation | Explain queue, placement, lease, conflict, and handoff decisions with policy version and bounded evidence; replay recorded observations in side-effect-free mode. |
| Administration | Inspect owner/capability scopes, managed backends, operation reconciliation, artifact metadata, backups, and service health. Destructive or privilege-changing operations require admin scope and audit record. |

## Adapter ports

Adapters are versioned, capability-discoverable ports. Required initial ports are persistence/transaction, resource inventory, lease/reclamation, execution supervision, managed backend lifecycle, artifact storage, clock/scheduler wakeup, and continuation coordination. The core calls interfaces, not adapter implementations. Adapter capability declarations state operations, constraints, and whether a limit is enforced, advisory, or observational. Unsupported requested guarantees fail with a typed capability error; adapters cannot silently downgrade the request.

Before an external action, core persists an operation intent with stable ID and request digest. The adapter operation accepts that idempotency identity and returns an explicit receipt/observation. Crash recovery inspects by operation ID and stable execution/backend identity. An absent, conflicting, or incomplete observation remains unknown for operator or authorized recovery; it is not an automatic retry signal.

Continuation coordination supports: register/revoke a standing drain contract; issue a handoff-specific safe-suspend request; collect durable acknowledgement identifying the continuation and safe boundary; deliver an idempotent result to the pending tool boundary; and inspect/resume pending continuation delivery. FrameShift owns its continuation data and must not expose its internal frame representation to core.

Backend lifecycle supports: inspect pinned profile/readiness/residency; close admission; request consumer drain; verify all required durable acknowledgements and no active generation; stop/unload; verify reclaimed resource capacity; restore exact pinned profile; verify readiness; reopen admission; and recover each operation by stable identity. The adapter never initiates an unload without a core-authorized handoff operation.

## Scheduling and safety decisions

Policies receive a bounded snapshot: resource capacities/observations with freshness, queued request attributes, existing grants/constraints, workload execution contracts, explicit authority grants, and policy version. They return a proposed decision plus reason codes. They cannot execute operations or directly mutate the ledger. Core validates bundle feasibility, authority, interruption safety, minimum headroom, anti-affinity/conflicts, handoff reservation, and lifecycle transition before recording/applying a decision.

Shared GPU admission is the default; per-device whole-resource exclusivity is requested by an explicit exclusive constraint/tag and represented in the canonical request, not inferred from a label. A profile needing profiling isolation may request broader named conflict domains. Required resource estimates, configurable headroom, and freshness are policy inputs; missing or stale required data blocks admission with an explanation. A memory budget is not advertised as hardware enforcement without adapter evidence.

The built-in policy uses interactive preference, weighted per-client fairness, queue aging, starvation protection for large requests, and compatible background filling. Policy code and configuration have stable version identities. Replay evaluates the same decision function on a recorded input snapshot without side effects.

## Handoff invariants

1. A handoff has fixed batch, resource envelope, backend profile/version, caller continuations, deadlines, and explicit eviction grants before acceptance.
2. Feasible round-trip execution/restoration capacity is protected before stopping admissions or requesting consumer suspension.
3. Drain acknowledgement is durable and identifies the safe boundary; timeout means no eviction.
4. Ordinary FrameShift suspension and ordinary lease release do not unload shared model weights.
5. Results are persisted per job before delivery; delivery is idempotent and occurs at the pending logical tool boundary.
6. Workload failure returns a result after required cleanup/restoration. Completed work is not repeated because later phases failed.
7. Failed restoration retains the reservation/continuations as required, marks the backend and handoff blocked, and supports explicit recovery.
8. Conflicting handoffs serialize. No dynamic batch extension or automatic coalescing occurs in the first release.

## Authentication and compatibility

The daemon authenticates loopback MCP clients and maps them to scoped principals. Stdio bridges authenticate to the daemon as clients and do not form a second authority. Separate grants control read, submit, lease ownership, profile invocation, backend registration/control, event publication, recovery, and administration. Agent/user identity propagates to job ownership and audit records.

Protocol and persisted schemas use explicit versions. Old job/attempt records remain inspectable across upgrades; migrations are transactional and tested against backups. Public APIs report unsupported capabilities rather than omit a requested guarantee. Same-user cgroup supervision is documented as lifecycle/resource accounting, never hostile-code isolation.

CPU and host-memory supervised leases provide atomic ledger admission plus per-job systemd cgroup ceilings. They do not preallocate physical memory, reserve CPU time against the OS, or contain unmanaged same-user processes. A client must run lease-backed work through Device Scheduler jobs; direct external consumers remain unsupported unless an adapter can independently prove their reclamation.

## DS-00 baseline: current host and consumer inventory

This read-only snapshot was assembled on 2026-10-21 at approximately 14:36Z. It is an inventory baseline, not proof of Device Scheduler behavior or an authorization to operate a GPU. Host measurements can drift and must be refreshed during qualification and cutover.

### Host snapshot

- Four `V100-SXM2-16GB` devices were visible, with UUID prefixes `GPU-21131915…`, `GPU-cf22c41f…`, `GPU-d745da9c…`, and `GPU-6c1cac7f…`. The read showed `0 MiB` allocated and no compute applications on each device.
- Project Control's service was active with PID `32456`; its inference service was inactive at observation time.
- No Device Scheduler resource inventory, lease ledger, backend lifecycle, or admission path exists yet. The visible GPU state is simply a point-in-time host observation.

### Existing resource and CUDA-controller consumers

| Consumer path | Current behavior to migrate or preserve | Source |
|---|---|---|
| Project Control performance probe | Resolves the configured/pinned skills root and invokes `cuda/scripts/cuda_controller.py` for a registered campaign probe. | [`performance_probe.py`](/home/tumlinson/project-control/src/project_control/services/performance_probe.py:143) |
| Project Control GPU lab | Resolves the selected runtime's controller, then constructs `GpuLabExecutor` with quiesce/resume callbacks and owner inspection. | [`cli.py`](/home/tumlinson/project-control/src/project_control/cli.py:735) |
| Local-worker campaign discovery | Resolves `cuda_controller.py` as its discovery tool and invokes it for campaign registry discovery. | [`controller.py`](/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/controller.py:439) |
| Project Control model-service/local-worker admission | Discovers GPU bundles and calls `reserve_service` while admitting/starting model slots; adapter service also reserves before starting managed adapters. The coordinator and host-resource facade expose the independent Project Control allocator. | [`supervisor.py`](/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/supervisor.py:780), [`service.py`](/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/service.py:76), [`facade.py`](/home/tumlinson/project-control/src/project_control/host_resources/facade.py:76), [`coordinator.py`](/home/tumlinson/project-control/src/project_control/host_resources/coordinator.py:95) |
| Project Control ownership readers and writer fence | Native providers and GPU lab inspect resource owners; `as1_jobs` fences writes against the old coordinator database. These owner and fencing contracts need replacement or explicit retirement before stopping that authority. | [`native_providers.py`](/home/tumlinson/project-control/src/project_control/native_providers.py:1079), [`lab_gpu.py`](/home/tumlinson/project-control/src/project_control/assistance/lab_gpu.py:1086), [`as1_jobs.py`](/home/tumlinson/project-control/src/project_control/as1_jobs.py:102) |

CUDA profiling wrappers, topology discovery, campaign definitions, and historical profiler evidence can remain useful. Any current assumptions that the controller is the sole admission or receipt authority must be replaced or retired as part of DS-13. Project Control retains its agent, task, workspace, coordination, Todo, and acceptance lifecycles; only its resource-allocation and resource-ownership query paths migrate.

### Source identity and evidence limits

The Project Control checkout was at Git commit `07849abd39af476fcd0c6fc3828bf77d6b51b1a9` with a clean working tree at observation. The KDA checkout was at `56ff49242a76d8e7aacb5aff8ff037f00aa3cd33` and had local edits in `CLAUDE.md`, `docs/experiment-engine.md`, `planning/README.md`, and `prompts/basic-flow.md`; do not interpret that commit alone as the complete current KDA tree. Exact SHA-256 values below bind the inspected Project Control/controller files to the bytes read during this inventory.

| Source file | SHA-256 |
|---|---|
| `/home/tumlinson/project-control/src/project_control/services/performance_probe.py` | `9fda247e594a20520e974c4eff1cfc5b65715a476b1d55007585e06074614efa` |
| `/home/tumlinson/project-control/src/project_control/cli.py` | `c1a07a9b6f506ed0bf0d40bba4c62f01f0cc54c551f8feda64f219ffc54121b1` |
| `/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/controller.py` | `cfcb005b19cbc02e0e7ae0d813fb9e4533699fe0d4a39894b93883de6ec47b50` |
| `/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/supervisor.py` | `ee167f80b9a09741007e872d0bc0817308a48c9a5e93fe75eadd9fbb717f5071` |
| `/home/tumlinson/project-control/src/project_control/local_runtime/local_worker/service.py` | `5950a45aeaa8091133f623ecdb5f74c71b7683e54eba67d4f2d54c488b3577eb` |
| `/home/tumlinson/project-control/src/project_control/host_resources/facade.py` | `ae7f65457335bc40869bef5ee2c992e4440e953a910f12b8d6fe63193ba0a439` |
| `/home/tumlinson/project-control/src/project_control/host_resources/coordinator.py` | `6cbbe1f45a6ac62b3f89c2b99a559667b605054443a040804b33f9fe413c2c4c` |
| `/home/tumlinson/project-control/src/project_control/native_providers.py` | `f246a60fb896af12e2762ddb2cd279c91eb90111e00c2a4f356714b3a2cf80ba` |
| `/home/tumlinson/project-control/src/project_control/assistance/lab_gpu.py` | `bf49b190af83db6883a9cb6bd6b70d73bb1b8c33f1bf8706093dd9c21e62bc59` |
| `/home/tumlinson/project-control/src/project_control/as1_jobs.py` | `a276876f84db7bee31b51d05abced6a44c2e92e813476767c23047384f71c6c5` |
| `/home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py` | `d16b5ae270fc63b27e8191e15c6385688e8d9e4799b0e756dfe82e68c08ee245` |

The inspected controller calls `bind_canonical_runtime()` and imports `HostCoordinator`, `RuntimeFacade`, and background runtime modules from `todo_orchestrator`. Project Control model-service code instead imports its own `project_control.host_resources` implementation. Source inspection therefore identifies two allocator interfaces and their potential shared `TODO_BACKGROUND_HOST_RUNTIME_DIR` storage, but does not establish the package/source identity selected by any live controller invocation. A deployment-time runtime binding/source fingerprint must be captured before migration; do not claim that the controller is loading a particular checkout based only on its compatibility locator or this source snapshot.

A read-only `--help` probe using Project Control's installed Python failed before command dispatch with `ModuleNotFoundError: No module named 'coding_workflow_mcp'`. The installed CUDA skill has no `integrations/coding-workflow-mcp` directory at the path that entrypoint inserts. This establishes that this exact invocation cannot currently load the legacy controller; it does not prove that every historical launcher uses the same environment. No GPU workload was started by the probe. The verified Project Control Todo binding is distribution `todo-orchestrator` version `0.2.1`, package `/home/tumlinson/project-control/.compact-todo-venv-20261019/lib/python3.12/site-packages/todo_orchestrator`, fingerprint `613ec4537777eb0c434906f3b87493aea757d97c31e34de9518b9e29bfe0db33`.

The optional downstream CUDA-agent entrypoint is [KDA's agent entry](/home/tumlinson/kda/docs/agent-entry.md), SHA-256 `83b6909d8c39ce2b71ce13f39d087ba7e05232115c6c0dd68ee745728e3b95b2`. It requires the task repository to retain authority over its implementation/evidence and GPU controller, and asks agents to select only relevant KDA skills explicitly. Device Scheduler uses it as optional CUDA kernel work guidance, not as a resource-admission authority.
