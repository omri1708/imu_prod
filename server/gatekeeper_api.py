# server/gatekeeper_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os, json, urllib.request

from server.gitops_checks_api import _gh  # משתמשים באותה פונקציה
from runtime.p95 import GATES

router = APIRouter(prefix="/gatekeeper", tags=["gatekeeper"])

class EvidenceReq(BaseModel):
    digest: str
    min_trust: float = Field(0.5, ge=0.0, le=1.0)

class ChecksReq(BaseModel):
    owner: str
    repo: str
    ref: Optional[str] = None
    pr_number: Optional[int] = None
    required: List[str] = Field(default_factory=list)
    mode: str = "all"  # all|any
    token_env: str = "GITHUB_TOKEN"

class P95Req(BaseModel):
    keys: List[str] = Field(default_factory=lambda: ["adapters.run:unity.build"])
    ceiling_ms: int = 5000

class EvaluateReq(BaseModel):
    evidences: List[EvidenceReq] = Field(default_factory=list)
    checks: Optional[ChecksReq] = None
    p95: Optional[P95Req] = None

def _verify_evidence(ev: EvidenceReq) -> bool:
    # אימות בסיסי: קיום מעטפה ב-.imu/provenance עם digest, ו-trust מינימלי (אם רשום במטא).
    # לצורך הפשטות — נבדוק מטא JSON אם קיים: .imu/provenance/meta/<digest>.json (אם פיתחת)
    meta_path = f".imu/provenance/meta/{ev.digest}.json"
    if not os.path.exists(meta_path):
        return False
    try:
        m = json.loads(open(meta_path,"r",encoding="utf-8").read())
        trust = float(m.get("trust", 0.0))
        return trust >= ev.min_trust
    except Exception:
        return False

def _verify_checks(ch: ChecksReq) -> bool:
    token = os.environ.get(ch.token_env)
    if not token:
        # אי אפשר לאשר — חסר משאב
        return False
    if ch.pr_number:
        pr = _gh(token, f"/repos/{ch.owner}/{ch.repo}/pulls/{ch.pr_number}")
        ref = pr["head"]["sha"]
    else:
        ref = ch.ref
    runs = _gh(token, f"/repos/{ch.owner}/{ch.repo}/commits/{ref}/check-runs").get("check_runs",[])
    summary = {cr["name"]: (cr["conclusion"] or cr["status"]) for cr in runs}
    missing = [n for n in ch.required if n not in summary]
    bad = [n for n,v in summary.items() if (n in ch.required or not ch.required) and v not in ("success","neutral","skipped")]
    return (len(missing)==0) and ((ch.mode=="all" and len(bad)==0) or (ch.mode=="any" and len(bad)<len(summary)))

def _verify_p95(p: P95Req) -> bool:
    for k in p.keys:
        try:
            GATES.ensure(k, p.ceiling_ms)
        except Exception:
            return False
    return True

@router.post("/evaluate")
def evaluate(req: EvaluateReq):
    reasons=[]
    # evidences
    if req.evidences:
        ok_all=True
        for ev in req.evidences:
            ok=_verify_evidence(ev); ok_all = ok_all and ok
            if not ok: reasons.append(f"evidence:{ev.digest}:insufficient")
        if not ok_all: return {"ok": False, "reasons": reasons}
    # checks
    if req.checks:
        if not _verify_checks(req.checks):
            reasons.append("checks:failed")
            return {"ok": False, "reasons": reasons}
    # p95
    if req.p95:
        if not _verify_p95(req.p95):
            reasons.append("p95:exceeded")
            return {"ok": False, "reasons": reasons}
    return {"ok": True}