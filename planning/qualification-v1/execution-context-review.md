# Execution context binding review

Review scope: binding scheduler grants to application launches, systemd limits and visibility, managed-backend restoration, and protected handoff restoration. This was a read-only source review; the only file created by this reviewer is this report. Project Control read-only context at semantic revision 36 confirmed DS-04, DS-06, and DS-07 remain planned and their outcomes require restart-safe attempts, pinned managed-backend lifecycle, and protected durable handoff.

## Result

No unresolved blocker was found in the bounded review scope.

One gap was found and resolved in the shared working tree while the review was in progress: the managed llama adapter had rejected any backend context without a GPU descriptor, even when its `BackendSpec` declared a valid CPU-only residency requirement. The current adapter validates the protected context against each declared resource kind and quantity, and accepts a CPU-only context. `tests/test_backends.py::test_registered_cpu_only_backend_restores_under_protected_cpu_context` covers that path. A direct reproduction using a CPU `BackendSpec`, a four-core protected allocation, and `ExecutionPurpose.BACKEND` now returns a valid context. No reviewer code changes were made.

The current source establishes these bindings:

- `Scheduler._job_execution_context` validates a live lease or reservation phase inside the ledger transaction, then `_execution_context_from_claim` loads resource descriptors by the granted resource IDs. The same typed context is persisted with the attempt and passed to `executor.start` (`packages/device-scheduler-core/src/device_scheduler_core/application.py:2120-2221, 2223-2306`). The request is not used as a substitute for the actual allocation.
- `SystemdExecutionAdapter._bind_execution_context` requires local-host descriptors, takes GPU UUIDs from the granted resource descriptors, rejects conflicting profile/job `CUDA_VISIBLE_DEVICES`, and sets the exact sorted grant UUID list. With no GPU grant it sets `CUDA_VISIBLE_DEVICES` to the empty string. It derives CPU quota and host-memory limits from the granted amounts and applies the stricter of grant and registered profile limits. GPU memory remains admission accounting only (`packages/device-scheduler-adapters/src/device_scheduler_adapters/systemd.py:477-572`; capabilities at `:216-249`).
- Systemd applies `RuntimeMaxSec` only when `ExecutionContext.purpose` is `JOB`; managed backend launches use `BACKEND` and do not inherit the finite job timeout (`systemd.py:605-609`; covered by `tests/test_systemd.py::test_backend_purpose_uses_grant_binding_without_finite_job_runtime`).
- Handoff restoration builds the execution context from the backend's immutable original allocation and resource snapshots, checks that each source allocation is still covered by the active protected reservation (including exclusive-mode coverage), and uses the reservation's current fence. The wider handoff envelope is not passed to the backend. Failure to bind blocks before `backend_adapter.restore` (`packages/device-scheduler-core/src/device_scheduler_core/handoff.py:571-613, 987-1017`).
- Managed-backend restoration requires backend purpose, a protected claim/fence, and allocation coverage for the backend's declared residency requirements before calling the executor (`packages/device-scheduler-adapters/src/device_scheduler_adapters/backends.py:1251-1306`). The systemd adapter then checks the final GPU descriptors and environment before any unit launch.

## Validation and limits

Checks run against the current working tree:

- `.venv/bin/pytest -q -p no:cacheprovider tests/test_backends.py` — 22 passed.
- `.venv/bin/pytest -q -p no:cacheprovider tests/test_application.py::test_executor_context_uses_only_the_actual_granted_resource_descriptors tests/test_systemd.py::test_start_binds_gpu_visibility_and_enforces_granted_host_limits tests/test_systemd.py::test_grant_limits_cap_a_more_permissive_profile tests/test_systemd.py::test_backend_purpose_uses_grant_binding_without_finite_job_runtime tests/test_handoff.py::test_handoff_drains_evicts_runs_restores_then_delivers_and_resumes tests/test_handoff.py::test_restore_refuses_source_allocation_outside_protected_envelope` — 6 passed.

These tests use fake execution/systemd adapters and synthetic resources. No live systemd unit, GPU, model server, or production handoff was run. `CUDA_VISIBLE_DEVICES` is cooperative visibility, not hostile-code GPU isolation; the adapter advertises the corresponding capabilities as advisory. CPU grants are enforced as aggregate `CPUQuota`, not per-core affinity. NVIDIA external-lease observation/reclamation and its pre-grant baseline work were assigned to a separate stream and are outside this review. Root's final integrated suite remains pending.

## Reviewed source hashes

SHA-256 hashes identify the dirty working-tree source snapshot reviewed here:

| Path | SHA-256 |
|---|---|
| `packages/device-scheduler-core/src/device_scheduler_core/models.py` | `41c901f139ec2c0d58c4ecfa8d29274142eb4be7b17046cb746235f78d27fb2b` |
| `packages/device-scheduler-core/src/device_scheduler_core/application.py` | `7e03ee27382854b020a2195d7b60d650f35c50ee7b6d1f464541186c5dc23de9` |
| `packages/device-scheduler-core/src/device_scheduler_core/handoff.py` | `db8380eee3f1958cd4e5cbf85678d0235ba77d2e2a31809f59d9d5c329098934` |
| `packages/device-scheduler-adapters/src/device_scheduler_adapters/systemd.py` | `6839cfd72665ea3e117afe07b48aca2d20b0362d59b72c80f25462c079238276` |
| `packages/device-scheduler-adapters/src/device_scheduler_adapters/backends.py` | `c4eb910fccc1ff49a6f73135bdff44f310c51ac231e6119b33f05edaf41caec5` |
| `tests/test_application.py` | `6c7602452cdf1d42e760e1449a4e80d5180ff5b8f20901413fbdcea00bdd5b90` |
| `tests/test_systemd.py` | `5cf0f0c2cc2143f2531a211ce70333bdfc0f5c36d0742b7930642adc79727d5e` |
| `tests/test_backends.py` | `1cb8c5fe1640cd51195a61cf0210de2a328273f1f35e5621564bdadab6817566` |
| `tests/test_handoff.py` | `1a1853720d946aee2feef7d1743b435aa3e4c6203b21a3fa207c9782f1629ee6` |
