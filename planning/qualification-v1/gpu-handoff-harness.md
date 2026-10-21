# Real GPU handoff harness

`scripts/qualify_gpu_handoff.py` is a bounded acceptance harness for one local
V100, the pinned llama.cpp executable and GGUF, a fresh private FrameShift
service/frame, and a private Device Scheduler daemon. It starts no services by
default. A real run requires both `--execute` and
`--confirm-live-gpu-handoff`, plus the active parent grant described below.

The harness verifies that ordinary FrameShift suspension keeps the same model
invocation and GPU process resident; a model-produced tool request is then
delivered through one explicit finite handoff. The handoff closes backend
admission, drains the attached continuation, evicts and verifies reclamation,
runs one 128 MiB CUDA allocation/copy/compare/free job, persists its result,
restores the pinned llama.cpp profile, and resumes the real model continuation.
It records identities, receipts, bounded logs, and GPU observations in a
private temporary diagnostic directory. This is a correctness smoke, not a
kernel test, performance measurement, hostile-code isolation claim, service
cutover, or CUDA-controller retirement gate.

The receipt includes SHA-256 maps for the Device Scheduler production Python
packages and harness sources, FrameShift Python sources plus its Git HEAD, and
the llama.cpp executable. It compares those identities before and after the
run and refuses a pass if they changed. The large model is bound by its path,
size, modification time, device, inode, and directory digest hint; its full
contents are not hashed by default.

## Parent grant and invocation

The root-owned `scripts/qualification_pool.py` first obtains a subordinate
single-GPU pool through the existing Project Control host-resource authority.
Use the deployed Project Control Python in an interactive terminal, keep that
parent process alive, and do not release its grant while the child is active:

```bash
/home/tumlinson/.local/state/project-control/current/bin/python \
  scripts/qualification_pool.py --run --gpu GPU-<exact-uuid> \
  --grant-file /tmp/device-scheduler-parent-grant.json
```

The wrapper must report `parent_pool_held`. It records a mode-0600 grant and
continues heartbeating the existing authority. The harness independently
checks that grant against Project Control's public read-only owner API, the
parent process birth identity, the repository root, and exactly one declared
accelerator UUID. It fails closed if the grant is missing, stale, mismatched,
or covers multiple GPUs. Do not run another allocator against this pool.

In a second terminal, use the private workspace environment:

```bash
.venv/bin/python scripts/qualify_gpu_handoff.py \
  --grant-file /tmp/device-scheduler-parent-grant.json \
  --execute --confirm-live-gpu-handoff
```

The default command without the two live-run flags only validates prerequisites
and prints a plan. A failed, blocked, or unknown run preserves its diagnostics
and leaves backend/job state for explicit inspection; it must not trigger an
automatic retry. The parent grant remains held until the exact managed backend
unit is known stopped and GPU quiescence has been independently verified. The
harness's exact-unit systemd cancellation is temporary fixture cleanup only;
it is not a production Device Scheduler backend-stop API.

A signal or run deadline during a submitted handoff requests cancellation and
waits up to ten minutes (capped by the run timeout) for a terminal coordinator
result. A cancelled result is recorded as cancelled rather than passed. A
blocked, unknown, or nonterminal result leaves the parent grant held and
requires explicit inspection and recovery; the harness does not replay it.

The harness requires the local model and binary at the pinned paths in the
script, a fresh NVIDIA UUID/process observation, an empty granted GPU before
launch, a V100-class device with adequate headroom, and llama.cpp `/slots`
support. It deliberately performs no downloads and designs or benchmarks no
CUDA kernel. Its result cannot qualify multi-host placement, restart recovery,
production startup/stop workflows, or retirement of the existing CUDA
controller.
