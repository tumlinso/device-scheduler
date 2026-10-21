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

`device-scheduler.service` is a user unit for a checkout at `$HOME/device-scheduler`. `install-user-service.sh` performs the workspace sync, creates private config/token directories, preserves an existing config, and installs the unit. It does not provision a token, start the service, or modify a live service.

The staged production files are in [`production/`](production/). They listen only on loopback port `8770`, use separate, explicit grants for the operator, Project Control, KDA, and KernelBender, and keep the backend and continuation registries empty until their real endpoints and identities have been verified. The principal script creates private token files and hashed grants without printing bearer values; it refuses to overwrite existing token or grant files. The operator token has explicitly listed administration/job/read operations, priority 100, and execution scope limited to this checkout. KDA and KernelBender receive separate project-root scopes and job operations, including per-owner job listing. Project Control receives cross-owner lease rights and read/control/attach rights only for the reserved `project-control-llama` backend ID; eviction authority remains a separate grant. No wildcard resource or backend grant is created. Their example client configuration requests a whole-device exclusive 12 GiB GPU allocation by UUID, but that selector is a cooperative client request, not a server-side UUID permission.

The daemon inventories the host's CPU, memory, and NVIDIA GPUs and currently has no production resource-pool allowlist. Keep the unit disabled until the CUDA controller's admissions are fenced and its outstanding ownership is reconciled. Do not run the daemon as a second allocator over the same live pool. This is the cutover fence; neither the KDA UUID selector nor a separate token provides resource-level authorization.

After that fence is complete, inspect existing config and credential paths before applying anything. If the config path is unused, install the staged files and provision grants, then install the unit:

```sh
set -euo pipefail
config_root="$HOME/.config/device-scheduler"
install -d -m 700 "$config_root" "$config_root/tokens" "$HOME/.local/state/device-scheduler"
for name in config.toml profiles.toml backends.toml continuations.toml; do
  test ! -e "$config_root/$name" && test ! -L "$config_root/$name"
done
install -m 600 deploy/production/config.toml "$config_root/config.toml"
install -m 600 deploy/production/profiles.toml "$config_root/profiles.toml"
install -m 600 deploy/production/backends.toml "$config_root/backends.toml"
install -m 600 deploy/production/continuations.toml "$config_root/continuations.toml"
XDG_CONFIG_HOME="$HOME/.config" bash deploy/install-user-service.sh
bash deploy/provision-production-principals.sh
```

The provisioning script requires the KDA and KernelBender directories to exist. Override `KDA_ROOT` and `KERNELBENDER_ROOT` when their checkout paths differ. It writes `operator.token`, `project-control.token`, `kda.token`, and `kernelbender.token`, with grants stored under `tokens/`. It does not enable or start the service. The bridge templates still need the real isolated GPU UUID and actual checkout paths reviewed before the corresponding clients are configured. Do not claim UUID-specific token authority: authorization limits the principal by operation, execution root, backend ID, and priority only.

Only after the old allocator has been fenced and ownership reconciled, activate and verify the service:

```sh
systemctl --user daemon-reload
systemctl --user enable --now device-scheduler.service
systemctl --user status device-scheduler.service
uv run --locked python scripts/daemon-smoke.py --token-file "$HOME/.config/device-scheduler/operator.token"
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
