#!/usr/bin/env python3
"""Add generic own-work operations to an existing principal; preserve its token and scopes."""
import argparse,json,os,re,stat,tomllib
from pathlib import Path
EXTRA={"leases.request","leases.renew","leases.release","schedules.put","schedules.control","events.publish"}
def upgrade(path,backup,apply=False):
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
  raise ValueError("principal grant must be a private current-user regular file")
 raw=path.read_text();before=tomllib.loads(raw);grant=before.get("principal",{})
 if grant.get("principal_id")!="operator":raise ValueError("expected operator principal")
 actions=grant.get("actions")
 if not isinstance(actions,list) or not all(isinstance(x,str) for x in actions):raise ValueError("invalid actions")
 updated,n=re.subn(r"(?ms)^actions\s*=\s*\[.*?\]", "actions = "+json.dumps(sorted(set(actions)|EXTRA)),raw,count=1)
 if n!=1:raise ValueError("expected one actions array")
 parsed=tomllib.loads(updated);expected={**before,"principal":{**grant,"actions":sorted(set(actions)|EXTRA)}}
 if parsed!=expected:raise ValueError("unexpected grant change")
 if apply:
  if not backup.is_dir() or backup.stat().st_mode&0o077:raise ValueError("private backup directory required")
  archived=backup/"operator-before-generic.toml"
  if archived.exists():raise ValueError("backup already exists")
  fd=os.open(archived,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,"w") as f:f.write(raw);f.flush();os.fsync(f.fileno())
  temporary=path.with_name(path.name+".generic-v1.tmp")
  fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,"w") as f:f.write(updated);f.flush();os.fsync(f.fileno())
  os.replace(temporary,path)
 return {"status":"applied" if apply else "preview","added_actions":sorted(EXTRA-set(actions)),"credentials_rotated":False,"execution_roots_changed":False,"backend_grants_changed":False}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--grant",type=Path,default=Path.home()/".config/device-scheduler/tokens/operator.toml");p.add_argument("--backup",type=Path,required=True);p.add_argument("--apply",action="store_true");a=p.parse_args();print(json.dumps(upgrade(a.grant,a.backup,a.apply)))
