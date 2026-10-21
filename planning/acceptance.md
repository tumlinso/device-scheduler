# Acceptance and retirement gates

These checks are required for implementation acceptance. They have not run; this document is a test contract, not a results report.

## Core, allocation, and recovery

- Concurrent requests cannot overallocate exclusive capacity or exceed the configured admitted budget. Multi-resource bundles are all-or-none in the ledger.
- Compatible shared allocations coexist. Exclusive placement waits for verified conflict resolution, including topology/interference constraints.
- Expiry, release, process-ID reuse, detached descendants, failed cleanup, stale fencing tokens, and missing execution records preserve uncertainty and keep capacity unavailable until reconciled.
- Duplicate job, schedule occurrence, event, result-delivery, and operation identities do not duplicate authoritative effects. Same key with a changed request is rejected.
- Inject daemon crashes before and after every journaled adapter action, process launch, receipt write, and outbox acknowledgement. Restart finds running work, reconciles by stable identity, and never blindly relaunches unknown work.
- Policy errors, invalid proposals, stale observations, and insufficient grants cannot bypass capacity, lifecycle, or ownership validation.

## Scheduling, interfaces, and security

- Verify weighted fairness, queue aging, interactive preference, compatible background filling, large-request starvation protection, topology placement, and protected handoff restoration.
- Verify delayed jobs, interval/calendar boundary behavior, timezone/DST handling, missed-run coalescing, overlap policy, dependency cycles, event authentication, and duplicate events.
- Verify equal application semantics across HTTP MCP, stdio bridge, SDK, and CLI; bounded waits do not cancel durable work.
- Verify authorization for reads, submit, leases, profiles, backend control, event publication, recovery, and admin. Client labels cannot elevate priority or eviction rights; secrets are absent from stored specs and logs.
- Verify policy replay makes no allocation, process launch, backend operation, or artifact mutation.

## Request–suspend–evict–run–restart journey

Using a scheduler-managed GPU llama.cpp backend and FrameShift agents, demonstrate on an isolated resource pool:

1. Ordinary tool suspension preserves the logical continuation and leaves model weights resident.
2. A specifically authorized, finite handoff obtains a protected reservation and safely drains every affected attached consumer at a durable safe boundary.
3. Actual backend stop/unload releases the requested resource capacity; unmanaged work is neither killed nor treated as evicted.
4. Tests/profiling run under the requested resource and isolation contracts; their results persist before delivery and restoration.
5. A daemon restart at each phase recovers from journal and adapter observations without duplicate execution.
6. Exact pinned backend configuration returns to readiness before eligible continuations resume at their pending tool boundary with the corresponding durable result.
7. Test failure, cancellation, drain timeout, cleanup failure, result-delivery failure, and restoration failure yield inspectable states and explicit recovery without repeating completed work.
8. Ordinary suspension, scheduling conflict, and queue delay never trigger model-weight unloading.

## Deployment and CUDA-controller retirement

- Qualify systemd transient user-unit authorization, cgroup containment, configured limits, logs, cancellation, process-tree cleanup, and daemon restart behavior on the target host.
- Migrate live CUDA/resource consumers and the narrow Project Control resource allocation/ownership-query paths. Keep Project Control task, agent, workspace, coordination, Todo, and acceptance authority intact.
- Fence old controller admissions; reconcile or transfer extant ownership; switch consumers; then verify device-scheduler is the only allocator for that pool. During rollback, preserve exactly one active allocator.
- Preserve historical evidence and retain useful profiler wrappers, topology checks, quiescence, and isolation safeguards or replace them with qualified equivalents.
- Completion requires the old CUDA controller and coordinator no longer grant resources to migrated pools, operational documentation and rollback materials, and verified single-authority state. A committed design or passing unit tests alone do not satisfy retirement.

## Evidence reporting

Record host, adapter/backend version, policy version, request/attempt/handoff IDs, resource identities, time, operation receipts, logs/artifact digests, test command and disposition, and limitations. Separate accounting evidence from enforceable controls, simulated policy outcomes from physical performance, and daemon-restart recovery from host-reboot recovery.
