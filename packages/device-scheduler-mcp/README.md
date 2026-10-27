# Device Scheduler MCP, SDK, and CLI

This package connects applications and agents to the Device Scheduler service.
The MCP server, Python client, and `device-scheduler` command use the same
operation names and authenticated service.

## Quick start

Use `device-scheduler job run --spec-file job.json` to submit a job, wait up to
30 seconds, and return its current status and logs. If the wait ends first,
the result still includes the durable job ID. Use `job get JOB_ID` to inspect
it later. The `call access.inspect` operation shows the grants assigned to the
current caller.

Python callers can use `DeviceSchedulerClient.run_job(...)` for the same
submit/wait/log flow, or `client.reserve(request)` for a supervised reservation
whose handle tracks its current fencing token. See
[`docs/using-v1.md`](../../docs/using-v1.md) for copyable examples and the
meaning of an unverified release.

Each MCP operation advertises a concrete input schema generated from the
shared operation registry. The existing tool names and nested `arguments`
shape are retained, and the core remains responsible for authoritative
validation and authorization.
