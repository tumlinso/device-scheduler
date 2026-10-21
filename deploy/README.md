# Host-local daemon deployment

The daemon is a single loopback service. Its default origin is `http://127.0.0.1:8770`, MCP is served at `/mcp`, and the unauthenticated `/healthz` and `/readyz` routes are intended only for local process checks. Operations require a scoped bearer token. The daemon holds an OS file lock in its private state directory, runs a synchronous reconciliation tick before becoming ready, and retains that lock for the process lifetime.

The current code path composes the SQLite ledger, Linux host and NVIDIA inventory adapters, systemd user-unit execution, and operator-configured backend/FrameShift admission and continuation paths. It does not start a model backend during daemon startup or reserve GPUs merely because the service starts. Backend eviction fails closed unless the registered FrameShift-only admission path, exact backend invocation, NVIDIA pre-stop baseline, and post-stop reclamation observation are all available.

Host qualification receipt `planning/qualification-v1/daemon-receipt-04.json` records a successful isolated real-process check: authenticated loopback health/MCP behavior, SDK submission with CLI inspection of the same job, then daemon termination/restart while a systemd CPU job continued. Reconciliation found the same attempt and invocation, observed one physical launch, and verified cleanup. The harness used temporary private configuration and ledger state and requested no GPU allocation. This does not qualify a persistent service installation, a GPU handoff, or reclamation on actual NVIDIA hardware.

## Foreground development run

From the repository root, install the workspace and create a private operator token:

```sh
uv sync --locked
install -d -m 700 ~/.config/device-scheduler ~/.config/device-scheduler/tokens
uv run --locked python scripts/provision-token.py --actions '*'
install -d -m 700 ~/.local/state/device-scheduler
uv run --locked device-scheduler --config deploy/device-scheduler.example.toml serve
```

The example config points to the token paths created by the provisioning command and uses the default state directory. From another terminal, verify the public health routes and authenticated MCP operations:

```sh
uv run --locked python scripts/daemon-smoke.py
```

To run only the daemon health routes, use `curl --fail http://127.0.0.1:8770/healthz` and `curl --fail http://127.0.0.1:8770/readyz`. The smoke command also calls `resources.capabilities`, `resources.list`, and `admin.status`; its token therefore needs those operation grants. Token contents are read from the private client file and are never printed by the scripts.

The example uses a full operator grant (`--actions '*'`) for initial setup. For normal clients, provision a separate token with only the required operation names. Use `--execution-root` and `--backend-id` only when granting those scoped resources.

## Allocation and scheduling policy

`[allocation]` configures GPU admission headroom and the maximum age of inventory observations. `[policy]` configures the trusted scheduler implementation. The built-in `builtin.weighted-fair` accepts `class_weights`, `owner_weights`, `age_quantum_seconds`, `max_age_boost`, `large_request_protection_seconds`, and `fairness_penalty`; `fairness_window_seconds` controls the application’s service-accounting window.

An operator may install a policy extension that registers a Python entry point in the `device_scheduler.policies` group, then set `[policy].extension` to its entry-point name and put constructor arguments under `[policy.options]`. The daemon imports only that installed entry point and requires policy API version `1`, a non-empty policy ID and version, and `choose_next`. Job specifications cannot name policy modules or provide code. Policy changes take effect on daemon restart; no client-side policy editing interface is exposed.

## systemd user service

`device-scheduler.service` is a user unit for a checkout at `$HOME/device-scheduler`. `install-user-service.sh` performs the workspace sync, creates private config/token directories, preserves an existing config, and installs the unit. It does not provision a token, start the service, or modify a live service. From the repository root:

```sh
bash deploy/install-user-service.sh
uv run --locked python scripts/provision-token.py --actions '*'
systemctl --user daemon-reload
systemctl --user enable --now device-scheduler.service
systemctl --user status device-scheduler.service
uv run --locked python scripts/daemon-smoke.py
```

Review `~/.config/device-scheduler/config.toml` and the unit before enabling it. The supplied installer requires the checkout at `$HOME/device-scheduler`; for another location, update `WorkingDirectory` and `ExecStart` in the unit and the matching path guard in the installer. To stop it, use `systemctl --user disable --now device-scheduler.service`.

## Optional execution and continuation profiles

Operator profile files are separate private TOML files owned by the service user with mode `0600`. Add their paths to `[paths]` in the service config only after creating the files. Execution profiles contain fixed argv and working-directory settings. Continuation profiles use a `token_file` reference; the adapter reads the private token into memory and does not put it into job or handoff specifications. Backend profiles and scheduler-only FrameShift controls share one private file:

```toml
[frameshift_controls.local]
base_url = "http://127.0.0.1:9010"
token_file = "~/.config/frameshift/scheduler-admission.token"

[backend_profiles.llama-local]
profile_id = "llama-local"
version = "llama-profile-v1"
execution_profile_id = "llama-server-local"
health_url = "http://127.0.0.1:8080/health"
slots_url = "http://127.0.0.1:8080/slots"
access_mode = "frameshift-only"
frameshift_control_ref = "local"
```

The referenced execution profile must be in the private execution-profile file and launch the managed server with llama.cpp `--slots` enabled. The FrameShift bearer token belongs only in the file named by `token_file`; neither TOML file contains its value. The continuation file uses `[continuations.<reference>]` tables with `base_url`, `frame_id`, and a separate private `token_file` reference. The service config must set `profiles_path`, `backend_profiles_path`, and `continuations_path` to those files.

Backend profiles are not sufficient by themselves to enable eviction: each backend must use the registered `frameshift-only` admission contract, verify closed admission and zero active generations, receive an explicit live eviction grant, and provide resource-reclamation evidence. A health endpoint alone is not that evidence.

The initial deployment is cooperative single-host operation. Same-user processes can bypass the scheduler unless all known consumers use the registered interfaces. This is not hostile-code isolation. Do not run independent allocators against the same resource pool. The isolated daemon restart receipt is not evidence of a persistent systemd user-service installation, CUDA handoff, or GPU reclamation qualification.
