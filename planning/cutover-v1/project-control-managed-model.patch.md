# Project Control managed-model migration bundle

**Status:** preparation only; unapplied and not yet cutover-ready. This is a source-scoped patch bundle, not a claim that Project Control consumers already use Device Scheduler.

**Baseline:** Project Control `a50b617ea5b948b9594be9021e8cea60c941c4fd`, clean at inspection. See `consumer-map.json` for source fingerprints and the static audit. Refresh the source and deployed runtime identity before applying any patch.

## Patch unit A: inject the Device Scheduler client at composition

1. Change `project_control.host_resources.runtime.LocalWorkerRuntime.__init__` to accept an injected host-resource facade instead of always constructing `HostResourceFacade(HostCoordinator())`.
2. Change `local_runtime.local_worker.supervisor` composition to create one authenticated `DeviceSchedulerClient` from operator-managed endpoint and token configuration, then inject `DeviceSchedulerProjectControlAdapter(client)`.
3. Change `native_providers` and `assistance.lab_gpu.native_owner_reader` to use that same authenticated client for owner/backend reads. Replace `as1_jobs` and `observer_analysis` direct `HostCoordinator(create=False)` owner-count queries with the bridge's authenticated owner/lease observations.
4. Keep the native Todo, workspace, agent-task, and acceptance paths untouched.

The bridge token requires ordinary operation grants plus `leases.request:any_owner`, `leases.read:any_owner`, and `leases.manage:any_owner`. The owner ID must remain in the dedicated `pc:` namespace. `owner_context` is descriptive metadata only and cannot grant access or prove reclamation.

## Patch unit B: move managed model process ownership to Device Scheduler

The current call sites are `Supervisor.admit` and `_start_slot` (GPU bundle selection and `reserve_service`), `_lease` and `AdapterService.run` (priority changes), `AdapterService._monitor_lease` and `Supervisor.poll` (preemption polling and lease renewal), `_record_residency`/`record_residency_process`, and `_evict_slot` (local process stop, quiescence evidence, protected lease release).

Replace these responsibilities as follows:

- Register a stable backend against an operator-registered Device Scheduler backend profile, then use `backends.control(start)` and `backends.get` to start and inspect its process and residency allocation.
- Keep Project Control's inference and agent-coordination requests. Send model RPCs to the ready endpoint returned by the managed backend adapter; do not add an inference RPC to Device Scheduler and do not also launch the model process from Project Control.
- Remove `reserve_service`, `set_priority`, `preempt_requested`, `record_residency_process`, and local protected-residency release from the managed model path. Do not silently translate these into ordinary leases.
- Route any eviction through `handoffs.submit/control` with an explicit authorized `EvictionGrant` and verified consumer-drain/restore contracts. A backend with consumers that cannot participate in the drain contract must be marked non-evictable and the handoff request must return a typed refusal.
- Preserve backend profile/version pinning, model-service request/response behavior, readiness checks, execution identity, and verified stop/restore evidence. Leave Project Control's task and agent authority intact.

## Patch unit C: retire CUDA-controller entrypoints

Replace `performance_probe`'s `probe --spec` invocation with a bounded `jobs.submit/wait/logs` call using an operator-registered execution profile, retaining Project Control campaign validation and evidence. Replace the GPU lab's controller run/quiesce/resume callbacks with a Device Scheduler job or explicit handoff flow; until that consumer patch is separately qualified, disable the controller-dependent lab execution path with a typed unavailable result. Keep registry discovery read-only and independent of the retired controller CLI. Preserve campaign records and existing profiler safeguards or qualify equivalents before removing their old implementation.

## Cutover blockers this bundle does not solve

- The PC runtime needs a supported dependency/configuration path for the Device Scheduler SDK and a protected token provisioned with only the required grants.
- Device Scheduler must expose a stable backend endpoint descriptor to its owner and maintain the backend residency lease for the backend lifetime. The currently reviewed API has `backends.register/get/list/control`, but no standalone backend-renew operation; verify the configured adapter's lease lifecycle before switching a live model service.
- Each managed model profile needs an operator-owned backend profile and a qualified backend adapter. Consumer attachment and eviction safety must be represented in the grant/contract, not inferred from `owner_context`.
- Every running PC/CUDA owner must be reconciled or transferred and old admission paths fenced before switching. The exact live CUDA-controller runtime source is still unresolved.
- No code in Project Control, service, model runtime, CUDA, or GPU state is changed by this bundle. Do not run it before the cutover gate.
