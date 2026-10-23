# Product specification

## Goal

Device Scheduler is a standalone, MCP-first authority through which applications, agents, development tools, and background processes request resources, submit work, schedule execution, and control supervised jobs. Its first production deployment supports one Linux host in one administrative domain, using Python, SQLite, and the host's existing systemd user manager.

The system combines durable allocation and execution with configurable scheduling policies. It covers immediate, delayed, recurring, event-triggered, dependency-triggered, and background work; resource leasing; process supervision; managed backend lifecycle; decision explanations; and policy simulation/replay. It should grow through capability-advertising adapters rather than repeated core redesign.

## Responsibilities

The repository has three logical packages and one initially deployable scheduling service:

| Package | Owns | Must not own |
|---|---|---|
| `device-scheduler-core` | Environment-neutral entities, state transitions, transactional application services, scheduling coordinator, policy contracts, validation, and durable handoff orchestration. | NVIDIA, CUDA, systemd, MCP transport, FrameShift internals, Project Control internals. |
| `device-scheduler-mcp` | Daemon composition, authenticated MCP over loopback HTTP, stdio bridge, typed Python SDK, CLI, and service authentication. | Policy decisions or environment-specific resource operations. |
| `device-scheduler-adapters` | SQLite persistence; Linux process supervision; NVIDIA discovery/monitoring; managed backend controls; FrameShift continuation bridge. | Project/task authority or scheduling policy. |

The daemon is the sole scheduling authority and canonical ledger. SDK, CLI, HTTP MCP, and stdio MCP reach the same application contracts. Client disconnects do not cancel accepted work. Startup reconciles the journal and existing executions before conflicting admissions; it never starts a model just because a client connected.

Project Control remains responsible for project tasks, workspaces, agent coordination, Todo authority, and acceptance. Device Scheduler owns physical resource allocations and execution operations. Future Project Control agent execution may be represented as scheduled workloads, but the Project Control agent lifecycle is out of scope. FrameShift retains responsibility for durable agent continuations and inference frames. The existing CUDA controller supplies useful precedents for GPU reservations, quiescence, profiler isolation, and preemption, but its implementation choices are not binding.

## Resource and job model

Resources have stable host-qualified identities, type, attributes, capacities, topology/conflict relationships, supported allocation modes, and enforcement capabilities. Requests contain typed requirements and constraints: quantities, selectors, affinity/anti-affinity, topology, duration, and isolation intent. Atomicity means an allocation bundle is granted as one ledger decision; adapter side effects remain separately journaled and are not physically atomic with SQLite.

A lease records scoped owner, requested/granted bundle, lifetime, fencing identity, sharing mode, and reclamation state. Expiry or release starts reconciliation; expiry alone never proves the resource is free. Cleanup that cannot be verified leaves capacity unavailable and visible.

CPU and host-memory leases have explicit usage. External usage remains unsupported without a trusted adapter that can verify reclamation. Supervised usage authorizes only Device Scheduler jobs submitted with the lease ID and current fencing token. Jobs receive the lease's exact grant through the systemd execution adapter, one queued or running job may attach at a time, and release/expiry requires verified attempt cleanup. This is ledger admission plus per-job cgroup ceilings; it does not preallocate memory, reserve CPU time from the OS, or isolate unmanaged processes.

A job is durable desired work. Each attempt is a uniquely identified execution with immutable request digest, state, operation journal, adapter handle, result, and cleanup evidence. Retrying requires explicit authorization and a safe-restart declaration or established idempotency. Ambiguous effects are never replayed blindly.

The initial resource adapters cover CPU, host memory, NVIDIA devices and memory budgets, shared GPU conflict domains, and profiling/interference domains. GPU concurrency is the default. Whole-device exclusivity is an explicit tag/constraint. Memory estimates and configurable headroom are required for shared GPU admission; absent or uncertain estimates block admission with a reason. GPU budgets are accounting/admission limits unless a reported adapter capability proves enforceable limits. Exclusivity never implicitly unloads model weights and never terminates unmanaged processes.

Policies operate on bounded, recorded observations and propose admission, priority, placement, fairness, preemption, and eviction decisions. The core validates the proposal against authority scopes, declared interruption contracts, capabilities, capacity, lifecycle invariants, and explicit eviction grants. Policy code is operator-installed trusted code, not supplied within a workload request. Built-in defaults favor interactive work, use weighted fairness across clients and queue aging, prevent avoidable starvation of large requests, and fill spare capacity with compatible background work. Record policy identity/version and explanation for every decision.

Workloads declare supported yielding, checkpoint, cancellation, and safe-restart behaviors. The scheduler can interrupt only in declared safe ways. Initially there is no arbitrary process checkpoint/restore and no interruption in the middle of model generation.

## Explicit request–suspend–evict–run–restore–resume

FrameShift ordinary suspension preserves the logical continuation and releases active inference capacity. Model weights may stay resident; this operation does not imply a weight eviction.

An agent submits one durable tool request with a finite, fixed batch of work, bounded resource envelope/duration, deadline, continuation references, result-delivery identity, and explicit eviction authorization. Its logical continuation resumes at the pending tool boundary and receives the durable result. The system does not promise preservation of the same agent process, inference cache, or model residency after suspension.

Agents attach to a shared managed backend under a standing contract that permits a specifically authorized handoff to request drain/suspension within a defined safe boundary and timeout. This standing contract permits asking; it does not itself evict weights. Only an explicit handoff request authorizes unloading or stopping a managed backend. No scheduling conflict, ordinary suspension, queue delay, client reconnect, or cleanup action grants weight-eviction permission.

Before accepting a handoff, the coordinator establishes a feasible round-trip reservation for the execution envelope and restoration capacity, including the requested isolation and compatible constraints. The reservation protects the accepted operation through execution and restoration; it reserves required quantities and constraints rather than automatically occupying every GPU. Conflicting handoffs are serialized. Batches cannot be extended after acceptance.

The durable phases are: persist request and grants; establish reservation; close new backend admissions; request durable safe suspension and drain; stop/unload the targeted managed backend and verify reclamation; run each bounded operation and persist its result; verify cleanup; restore the pinned backend configuration and verify readiness; durably deliver results and resume eligible continuations; settle reservation and retain evidence. Before every external action, persist intent and an idempotent operation identity. Record acknowledgements and receipts.

Test/job failure is an ordinary durable result and normally resumes the caller with that result after restoration. Completed results are retained if later work or restoration fails. A restoration failure leaves the handoff/backend visibly blocked and recoverable; recovery must not rerun completed work. A drain timeout prevents eviction. Cancellation follows cleanup and restoration obligations for remaining live consumers. If restoration fails, preserve continuation and result state until explicit recovery.

## Execution and durability

SQLite is the initial transactional ledger. Use versioned migrations, append-only operation/event history, idempotency records, durable delivery outboxes, explicit state transitions, and separately indexed artifacts. Include consistent backup/restore procedures and startup integrity checks. Persist intent before adapter side effects, then reconcile observed outcomes. Use fencing tokens to reject stale lease operations.

The Linux adapter uses transient systemd user service units with a deterministic unit identity per attempt. It supervises process trees through cgroups and applies CPU, memory, and task limits when the host supports them. A workload may continue across a daemon restart while the host and user manager are running; completion remains inspectable until a terminal receipt is stored. Reconciliation validates invocation identity, containment, result, and cleanup. Missing/ambiguous records become `unknown`, requiring inspection; they never trigger a blind relaunch. Host reboot recovery is separately governed by the workload's declared recovery contract.

MCP uses authenticated loopback HTTP. The stdio MCP endpoint is a thin client bridge to the daemon. SDK and CLI share the same versioned operation contracts. Use scoped identity and grants for observation, submission, lease ownership, execution profile use, backend control, event publication, and administration. A client cannot grant itself priority or eviction rights with labels. Commands use argument vectors and explicit working directories; only registered profiles or authorized execution scopes are runnable. Environment/configuration inputs are granted explicitly, and secrets are host-managed references rather than serialized values.

Same-user cgroups provide process lifecycle and resource control, not hostile-tenant isolation. The initial system is single-host and single-administrative-domain; multi-host scheduling and adversarial tenant isolation are not first-release claims.

## Scheduling, schedules, evidence

Recurring schedules support intervals and calendar times with explicit IANA timezone semantics. Persist each nominal occurrence key. Defaults coalesce missed occurrences into one pending run and prevent overlap. Other overlap/misfire modes must be explicit and supported. Dependencies refer to job results, not Project Control task acceptance. Authenticated events use issuer-scoped identities and deduplication; reject dependency cycles.

Store structured results, ownership transitions, operation evidence, decision explanations, policy snapshots/versions, timestamps, and artifact references. Bound captured output and identify truncation. Do not destructively remove artifacts without an explicit retention policy. Health and reconciliation views must distinguish ledger state from verified adapter state.

Policy simulation/replay is side-effect-free: it consumes recorded observations and policies, allocates no resources, and executes no work. It explains differences without claiming a physical performance result.

## Delivery scope and deferred work

First release: one Linux host; one daemon; SQLite; transient systemd user units; NVIDIA GPU, process, backend, and FrameShift adapters; MCP/SDK/CLI; durable schedules and handoffs. Deferred: multi-host placement, hostile tenant boundaries, arbitrary process checkpointing, interruption during generation, automatic merging/open-ended handoff batches, and ownership of Project Control agent lifecycle.

Completion includes consumer migration and CUDA controller retirement. Migrate Project Control's narrow resource allocation and ownership-query consumers; preserve Project Control tasks, agents, workspaces, coordination, and Todo responsibilities. At cutover, fence the old allocator, reconcile or transfer known leases, switch consumers, and prove one active allocator. Retain rollback materials but never operate both allocators against one pool. Historical profiler evidence and useful wrappers/topology/quiescence safeguards are preserved or replaced by qualified equivalents.
