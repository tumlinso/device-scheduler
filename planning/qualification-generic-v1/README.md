# Generic v1 qualification — 2026-10-23

The standalone generic service is deployed on the existing canonical ledger. No second allocator, GPU workload, model registration, or FrameShift operation was used in this qualification.

- [342 passing regression tests](tests.xml), including supervised lease admission/reuse/release, stale fencing, uncertain cleanup retention, expiry, the external release race, and replay of pre-upgrade idempotency records.
- A final [lease-policy regression](lease-policy-tests.xml) passes 9 focused checks after adding the guard that prevents custom or built-in policies from taking caller-owned reservations. Normal background preemption remains supported.
- [24 live workflow checks](workflow.json): authenticated SDK/MCP, CLI and stdio canonical reads; CPU/memory jobs; a supervised CPU/memory lease retained after job cleanup, renewed and released; delayed work; interval scheduling; dependencies; event deduplication/wakeup; cancellation; logs; policy explanation; and side-effect-free priority/fairness simulation.
- [Production restart](restart.json): a running CPU job retained the same attempt and systemd invocation, completed with one physical launch, and verified cleanup. The [earlier preparation probe](restart-preparation-timeout.json) exceeded its own bounded wait before validation; its failed job and cleanup evidence were preserved, not replayed.
- [Deployment manifest](deployment.json) records exact changed-file hashes and the private source/config/consistent-ledger backup directory. Credentials were retained; only six own-work lease/schedule/event actions were added. Execution roots and backend rights were not broadened.

Supervised leases reserve CPU/memory capacity exclusively for jobs launched by Device Scheduler. One job may consume a lease at a time; job completion does not discard the reservation. Unmanaged CPU/memory processes require a trusted containment/reclamation adapter and remain unsupported. Cgroup controls provide ceilings and lifecycle containment, not physical memory preallocation, guaranteed host CPU service, or hostile-tenant isolation.

Existing records and the canonical ledger are preserved. Ephemeral legacy tombstones had disappeared since the earlier cutover; the absent paths were re-fenced without touching any database rows. The persistent Project Control selector and retired controller entrypoint remain in effect. Rollback material is retained privately; never restore an old database over new jobs or restart an old allocator alongside Device Scheduler. After new supervised records are admitted, a rollback must preserve/reconcile those records and use a compatible implementation; otherwise fence admissions and repair forward.

Full host-reboot qualification, calendar/DST live boundary matrices, all-phase handoff crash qualification and multi-host execution are not claimed. None is required to use the documented generic v1 workflow. Historical GPU/cutover evidence remains under `../qualification-v1/` and `../cutover-v1/`.

Native Todo verification/acceptance remains distinct from operational delivery. The DS-01 claim is released with references to this evidence; no task acceptance or history rewriting is implied by these receipts.
