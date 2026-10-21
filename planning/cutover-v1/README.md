# Project Control resource cutover

**Current status:** single-host v1 is deployed; known legacy routes are fenced and the controller is retired. See [the deployment record](deployed-v1.md). The preparation notes below describe the earlier snapshot.

The original preparation artifacts in this folder were read-only snapshots. New deployment receipts record the subsequently authorized and applied consumer switch, service restarts and allocator fence.

`consumer-map.json` records the current resource-authority and controller entrypoint surfaces, their current source fingerprints, the DS operation families intended to replace them, and the unresolved runtime source identity. The source snapshot is useful for preparing a reversible migration; qualification must refresh every fingerprint and inspect the deployed runtime binding.

The bridge under `packages/device-scheduler-adapters` is injected with the public async Device Scheduler MCP client and preserves the Project Control host facade’s reservation and owner DTO boundary for ordinary resource leases. Lease creation, renewal, release, inventory, and owner reads must go through DS operations. Managed model `reserve_service` calls fail closed until those callers migrate to the DS backend lifecycle. Project Control task, workspace, agent coordination, and native Todo responsibilities remain with Project Control.

The bridge intentionally fails closed for old calls that DS lease semantics do not provide: preemption signals, legacy priority mutation, special residency protection, local stale-owner reconciliation, and callback-driven drain. Legacy foreground preemption and mixed CPU/RAM external-lease calls currently refuse unsupported guarantees; the qualified GPU-only lease path does not imply eviction. Managed model services must move to DS backend lifecycle and explicit handoff contracts. They must not be emulated as leases, and a service must not request model eviction without a matching explicit grant. `planning/cutover-v1/project-control-managed-model.patch.md` records the reviewed, unapplied consumer patch bundle and its remaining contract gates.

`scripts/cutover_audit.py` performs a static source-fingerprint check only. It does not query Project Control, Device Scheduler, systemd, the CUDA runtime, or a GPU. A clean static audit means the inspected files still match the recorded source bytes; it does not prove that an allocator is fenced or that only one allocator is active. The consumer map includes the actual `LocalWorkerRuntime` construction seam and the direct physical-release owner readers, not only facade callers.

Cutover may proceed only after known old owners and reservations are reconciled or transferred, old admissions are fenced, every consumer is switched or explicitly retired, ambiguous reclamation remains unavailable, and a single-authority audit passes. Retain rollback materials, but never enable the old and new allocators against the same pool at once.
