#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_root="$HOME/.config/device-scheduler"
grant_root="$config_root/tokens"
python="$repo_root/.venv/bin/python"
kda_root="${KDA_ROOT:-$HOME/kda}"
kernelbender_root="${KERNELBENDER_ROOT:-$HOME/KernelBender}"

if [[ ! -x "$python" ]]; then
  printf 'Missing workspace Python; run deploy/install-user-service.sh first.\n' >&2
  exit 2
fi
if [[ ! -d "$kda_root" || ! -d "$kernelbender_root" ]]; then
  printf 'KDA_ROOT and KERNELBENDER_ROOT must name existing project directories.\n' >&2
  printf 'Current KDA_ROOT=%s KERNELBENDER_ROOT=%s\n' "$kda_root" "$kernelbender_root" >&2
  exit 2
fi

install -d -m 0700 "$config_root" "$grant_root"
for token in operator.token project-control.token kda.token kernelbender.token; do
  if [[ -e "$config_root/$token" || -L "$config_root/$token" ]]; then
    printf 'Refusing to overwrite existing token file: %s\n' "$config_root/$token" >&2
    exit 2
  fi
done
for principal in operator project-control kda kernelbender; do
  if [[ -e "$grant_root/$principal.toml" || -L "$grant_root/$principal.toml" ]]; then
    printf 'Refusing to overwrite existing principal grant: %s\n' "$grant_root/$principal.toml" >&2
    exit 2
  fi
done

provision() {
  local principal="$1" token_file="$2" max_priority="$3"
  shift 3
  "$python" "$repo_root/scripts/provision-token.py" \
    --principal-id "$principal" \
    --token-file "$config_root/$token_file" \
    --token-directory "$grant_root" \
    --max-priority "$max_priority" \
    "$@"
}

provision operator operator.token 100 \
  --actions \
    resources.list resources.capabilities \
    jobs.submit jobs.get jobs.list jobs.wait jobs.logs jobs.control \
    leases.get leases.list \
    backends.get backends.list \
    handoffs.get handoffs.list schedules.get schedules.list \
    policies.explain policies.simulate \
    admin.status admin.reconcile admin.backup \
  --execution-root "$repo_root"

provision project-control project-control.token 50 \
  --actions \
    resources.list resources.capabilities \
    leases.request leases.get leases.list leases.renew leases.release \
    leases.request:any_owner leases.read:any_owner leases.manage:any_owner \
    backends.get backends.list backends.control backends.attach \
    backends.read:any_owner backends.control:any_owner \
  --backend-id project-control-llama

provision kda kda.token 70 \
  --actions resources.list resources.capabilities jobs.submit jobs.get jobs.list jobs.wait jobs.logs jobs.control \
  --execution-root "$kda_root"

provision kernelbender kernelbender.token 70 \
  --actions resources.list resources.capabilities jobs.submit jobs.get jobs.list jobs.wait jobs.logs jobs.control \
  --execution-root "$kernelbender_root"

printf 'Principal tokens created under %s; token values were not printed.\n' "$config_root"
printf 'Do not start the daemon until the previous allocator is fenced and ownership is reconciled.\n'
