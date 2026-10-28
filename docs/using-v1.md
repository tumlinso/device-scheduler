# Using Device Scheduler v1

The installed service is `device-scheduler.service`; authenticated MCP is at `http://127.0.0.1:8770/mcp`. The CLI automatically loads `$XDG_CONFIG_HOME/device-scheduler/config.toml` (normally `~/.config/device-scheduler/config.toml`). `--config` overrides it. SDK, CLI and stdio use the same daemon and durable ledger.

From the installed checkout:

```sh
.venv/bin/device-scheduler admin status
.venv/bin/device-scheduler resource capabilities
.venv/bin/device-scheduler job list
```

An MCP client can launch `.venv/bin/device-scheduler mcp-stdio`; this bridges to the existing authenticated daemon and does not start another allocator. Treat token files as private; never paste their contents into requests.

## Check what your caller can do

Every authenticated caller can ask for its own effective grant summary. It
shows the actions, execution roots, backend scope and priority limit attached
to that caller's token. It cannot inspect another caller or reveal token
material:

```sh
.venv/bin/device-scheduler call access.inspect
```

MCP tools advertise the fields expected by each operation. The outer
`arguments` object remains the same as before; for example,
`jobs_submit.arguments` contains `spec` and may also contain `lease_id` and
`fencing_token`.

## Submit and collect a job in one command

Use `job run` for a one-off job. It submits once, waits for at most the chosen
duration, then collects the current job status and bounded logs. A timeout
does not cancel the job: the JSON result includes its durable `job_id`, so you
can inspect it later with `job get` or `job logs`.

```sh
cat job.json | .venv/bin/device-scheduler job run --spec-file - --wait-seconds 20
```

The specification can also come from a JSON or TOML file. Waits are bounded
to 300 seconds. The Python client offers the same composition:

```python
result = await client.run_job(job_spec, wait_seconds=20)
print(result["job_id"], result["timed_out"], result["logs"])
```

## Reuse a supervised reservation

The Python SDK's reservation handle keeps the current fencing token through
renewal and job submission. Releasing may report `unknown` or unverified
cleanup; check `reservation.released` before treating capacity as reclaimed.

```python
reservation = await client.reserve(request)
try:
    await reservation.renew(300)
    result = await reservation.submit_job(job_spec)
finally:
    await reservation.release()
if not reservation.released:
    print("Reservation cleanup is not yet verified", reservation.release_result)
```

The scheduler remains the authority for admission, fencing and reclamation.
The handle is convenience bookkeeping; it does not bypass grants or make
unmanaged processes safe.

## A CPU/memory lease and job

Run this with `.venv/bin/python` from this checkout, which is an approved execution root for the installed operator. For another project, provision a separately scoped token with `scripts/provision-token.py` and load its grant during an administrator-controlled restart.

```python
import asyncio
import sys
import uuid
from pathlib import Path
from device_scheduler_mcp.client import DeviceSchedulerClient

async def main():
    client = DeviceSchedulerClient(
        "http://127.0.0.1:8770",
        Path.home() / ".config/device-scheduler/operator.token",
    )
    lease = await client.request_lease({
        "request_id": "example-" + uuid.uuid4().hex,
        "usage": "supervised",
        "requirements": [
            {"kind": "cpu", "quantities": {"cores": 0.25}},
            {"kind": "memory", "quantities": {"bytes": 128 * 1024**2}},
        ],
        "lease_ttl_seconds": 120,
        "idempotency_key": "lease-" + uuid.uuid4().hex,
    })
    job = await client.submit_job({
        "argv": [sys.executable, "-c", "print('hello from Device Scheduler')"],
        "working_directory": str(Path.cwd()),
        "timeout_seconds": 30,
        "idempotency_key": "job-" + uuid.uuid4().hex,
        # Omit resources: this job inherits its lease's complete allocation.
    }, lease_id=lease["lease_id"], fencing_token=lease["fencing_token"])
    result = await client.wait_job(job["id"], timeout_seconds=60)
    print(result["state"], await client.job_logs(job["id"]))
    if result.get("cleanup_verified") is not True:
        raise RuntimeError("Cleanup is unverified; inspect the job and retained lease")
    released = await client.release_lease(
        lease["lease_id"], fencing_token=lease["fencing_token"],
    )
    assert released["state"] == "released"

asyncio.run(main())
```

A supervised lease is a capacity reservation for **scheduler-launched jobs only**. It supports shared or exclusive allocation and one queued/running job at a time. Completion preserves the lease so it can be renewed, reused or released. Policy preemption cannot take the caller-owned reservation; ordinary scheduler-allocated background jobs remain eligible under their restart contract. Long-lived users must renew before expiry. Release/expiry follows the job's interruption contract and retains unavailable capacity whenever cleanup is uncertain. An unused supervised reservation has no physical execution to reclaim.

For a one-off job, omit the lease and put the same resource requirements directly in the job specification; the scheduler allocates and cleans up automatically. `not_before`, job-result `dependencies`, `event_trigger`, and the `schedules` operations add delayed, conditional and recurring execution. Use `call <operation> --arguments-file <json>` for complete operation arguments; grouped commands also support argument and payload files. The installed operator can submit/control its own jobs, leases and schedules and publish events. Priority and execution roots remain server-granted.

CPU quotas and memory maxima are cgroup ceilings plus ledger admission accounting. They do not preallocate physical RAM or reserve CPU time against unmanaged host work. GPU memory budgets are admission accounting, not hardware memory enforcement. Arbitrary unmanaged CPU/memory processes cannot consume supervised leases. External leases require a trusted reclamation adapter; the built-in NVIDIA path supports exclusive GPU quiescence leases.

Use `job get`, `job logs`, `policy explain`, `admin status` and `admin reconcile` to inspect work. `admin reconcile` currently advances job/handoff reconciliation; periodic ticks additionally reconcile lease expiry, preemption, backend startup and schedules. Explicit process suspend/resume and arbitrary checkpoints are unsupported; safe preemption uses declared cancellation/restart contracts. Unknown execution is never permission for a blind retry or capacity reuse.
