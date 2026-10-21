#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_root="${XDG_CONFIG_HOME:-$HOME/.config}/device-scheduler"
unit_root="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ "$repo_root" != "$HOME/device-scheduler" ]]; then
  printf 'The supplied unit expects the checkout at %s/device-scheduler. Edit deploy/device-scheduler.service before installing.\n' "$HOME" >&2
  exit 2
fi

cd "$repo_root"
uv sync --locked
install -d -m 0700 "$config_root" "$config_root/tokens" "$unit_root"
config_path="$config_root/config.toml"
if [[ ! -e "$config_path" ]]; then
  install -m 0600 deploy/device-scheduler.example.toml "$config_path"
else
  printf 'Preserving existing configuration: %s\n' "$config_path"
fi
install -m 0600 deploy/device-scheduler.service "$unit_root/device-scheduler.service"

printf 'Provision an MCP token with: %s/.venv/bin/python scripts/provision-token.py --actions "*"\n' "$repo_root"
printf 'Review %s, then run: systemctl --user daemon-reload && systemctl --user enable --now device-scheduler.service\n' "$config_path"
