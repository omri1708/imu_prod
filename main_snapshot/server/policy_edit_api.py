# עריכת מדיניות YAML ועוד: get/set + eval (net/fs) + apply מיידי.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import yaml

from policy.policy_hotload import _apply_cfg
from security.network_policies import POLICY_DB, UserNetPolicy, NetRule
from security.filesystem_policies import FS_DB, UserFsPolicy, PathRule

POLICY_FILE = Path("security/policy_rules.yaml")
router = APIRouter(prefix="/policy", tags=["policy"])

@router.get("/yaml")
def get_yaml():
    if not POLICY_FILE.exists(): return {"ok": True, "yaml": ""}
    return {"ok": True, "yaml": POLICY_FILE.read_text(encoding="utf-8")}

class SetReq(BaseModel):
    yaml_text: str

@router.post("/yaml")
def set_yaml(req: SetReq):
    try:
        cfg = yaml.safe_load(req.yaml_text) or {}
    except Exception as e:
        raise HTTPException(400, f"yaml parse error: {e}")
    # write atomically
    tmp = POLICY_FILE.with_suffix(".yaml.tmp")
    tmp.write_text(req.yaml_text, encoding="utf-8")
    tmp.replace(POLICY_FILE)
    # apply immediately
    try:
        _apply_cfg(cfg)
    except Exception as e:
        raise HTTPException(400, f"apply failed: {e}")
    return {"ok": True}

@router.get("/eval/net")
def eval_net(user_id: str, host: str, port: int):
    pol = POLICY_DB.get(user_id)
    if not pol:
        return {"ok": False, "reason": "no policy for user"}
    # replicate logic
    allow=False
    for r in pol.rules:
        from security.network_policies import _host_matches
        if _host_matches(r.host, host) and port in r.ports:
            allow = True
    if not allow and not pol.default_deny:
        allow=True
    return {"ok": True, "allow": allow}

@router.get("/eval/fs")
def eval_fs(user_id: str, path: str, write: bool=False):
    pol = FS_DB.get(user_id)
    if not pol:
        return {"ok": False, "reason": "no policy"}
    ap = Path(path).resolve()
    allow = False
    for r in pol.rules:
        base = Path(r.path).expanduser().resolve()
        if str(ap).startswith(str(base)):
            if write and r.mode!="rw": allow=False; break
            allow=True
    if not allow and not pol.default_deny:
        allow=True
    return {"ok": True, "allow": allow}