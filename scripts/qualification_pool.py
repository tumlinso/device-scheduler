#!/usr/bin/env python3
"""Hold a legacy-authority parent pool for subordinate scheduler qualification.

Run with the deployed Project Control Python, explicitly --run, in a terminal.
Only 'release' after verified GPU quiescence ends the grant. This is temporary
delegation for qualification, not allocator cutover or controller retirement.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


def gpu_idle(uuid):
    process = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True)
    if any(line.split(",", 1)[0].strip() == uuid for line in process.stdout.splitlines()):
        return False
    memory = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    values = memory.stdout.strip().split(",")
    return len(values) == 2 and float(values[0]) <= 16 and float(values[1]) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu")
    parser.add_argument("--grant-file", type=Path)
    args = parser.parse_args()
    if not args.run:
        print("Plan: reserve one explicit GPU and its topology domains through the existing host authority; keep the parent process alive until verified release.")
        return 0
    if not args.gpu or not args.grant_file or not sys.stdin.isatty():
        parser.error("--run requires --gpu UUID, --grant-file NEW_PATH, and an interactive terminal")
    if args.grant_file.exists():
        parser.error("grant file exists; preserve it and choose a new path")
    os.environ.setdefault("TODO_BACKGROUND_HOST_RUNTIME_DIR", "/tmp/codex-todo-orchestrator-1000")
    from project_control.host_resources import HostCoordinator, HostResourceFacade

    facade = HostResourceFacade(HostCoordinator(create=False))
    inventory = facade.discover_gpus()
    resource_id = f"accelerator:{args.gpu}"
    if resource_id not in {item["id"] for item in inventory}:
        raise RuntimeError("requested GPU not present in the current authority inventory")
    selected = next(item for item in inventory if item["id"] == resource_id)
    tags = selected.get("tags", {})
    domains = [resource_id]
    for tag, prefix in (("pcie_root", "pcie"), ("nvlink_domain", "nvlink")):
        if tags.get(tag):
            domains.append(f"interference:{prefix}:{tags[tag]}")
    if facade.conflicts(domains):
        raise RuntimeError("qualification pool has known owners; refusing to request their preemption")
    for _ in range(3):
        if not gpu_idle(args.gpu):
            raise RuntimeError("GPU quiescence is not established; no parent grant requested")
        time.sleep(0.2)
    reservation = facade.begin_foreground(project_root=Path(__file__).resolve().parents[1], pid=os.getpid(),
        resource_request={"schema_version": 1, "ids": [resource_id], "isolate_pcie_root": True,
                          "isolate_nvlink_domain": True, "cpu_threads": 8, "ram_bytes": 48 * 1024**3})
    owner_id = reservation["owner_id"]
    if resource_id not in reservation["resource_ids"] or not facade.activate_foreground(reservation):
        facade.release(owner_id)
        raise RuntimeError("existing authority could not grant the exact qualification pool")
    owner = facade.owner(owner_id)
    if not owner or owner["state"] != "active" or not gpu_idle(args.gpu):
        raise RuntimeError("grant is not active and quiescent; inspect the retained owner before proceeding")
    grant = {"schema_version": 1, "owner_id": owner_id, "gpu_uuid": args.gpu,
             "resource_ids": owner["resources"], "project_root": owner["project_root"],
             "pid": owner["pid"], "process_start": owner["process_start"], "state": "active",
             "created_at": datetime.now(timezone.utc).isoformat(), "authority": "project-control-host-resources",
             "purpose": "subordinate-device-scheduler-qualification", "independent_allocator_cutover": False}
    args.grant_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.grant_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(grant, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": "parent_pool_held", "grant_file": str(args.grant_file), "owner_id": owner_id,
                      "gpu_uuid": args.gpu, "release": "type release only after child execution and backend cleanup"}), flush=True)
    while True:
        facade.heartbeat(owner_id, pid=os.getpid())
        if select.select([sys.stdin], [], [], 2)[0]:
            line = sys.stdin.readline()
            if not line:
                print("Terminal input closed; parent grant remains held for inspection.", flush=True)
                time.sleep(2)
                continue
            if line.strip() == "release":
                if not all(gpu_idle(args.gpu) for _ in range(3)):
                    print("Refusing release: GPU processes/memory/utilization are not quiescent.", flush=True)
                    continue
                facade.release(owner_id)
                after = facade.owner(owner_id)
                print(json.dumps({"status": "released", "owner_state": after.get("state") if after else None}), flush=True)
                return 0
            print(json.dumps({"status": "parent_pool_held", "owner_state": facade.owner(owner_id)["state"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
