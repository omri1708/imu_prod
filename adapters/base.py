# adapters/base.py
# -*- coding: utf-8 -*-
import os, subprocess, shutil, json
from typing import Dict, Any, Tuple, List
from grounded.evidence_contracts import EvidenceIndex
from governance.policy import RespondPolicy
from audit.log import AppendOnlyAudit

from storage import cas
from common.exc import ResourceRequired

AUDIT = AppendOnlyAudit("var/audit/adapters.jsonl")


def _which(x: str) -> str:
    p = shutil.which(x)
    return p or ""

def _need(tool: str, how: str):
    if not _which(tool):
        raise ResourceRequired(kind=f"tool:{tool}", items=[tool], how_to=how)

def run(cmd: List[str], cwd: str = None, env: dict = None) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def put_artifact_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    h, _ = cas.put_text(text, meta={"path": path})
    return h

def evidence_from_text(kind: str, text: str) -> dict:
    h, _ = cas.put_text(text, meta={"kind": kind})
    return {"kind": kind, "hash": h}


class BuildResult:
    def __init__(self, artifacts: Dict[str, bytes], claims: List[Dict[str,Any]], evidence: List[Dict[str,Any]]):
        self.artifacts = artifacts
        self.claims = claims
        self.evidence = evidence

class BuildAdapter:
    """ממשק בסיס לאדפטרים."""
    KIND = "base"

    def detect(self) -> bool:
        """האם הכלים הזמינים במכונה?"""
        return True

    def requirements(self) -> Tuple[str, list, str]:
        """מה צריך כדי לרוץ בפועל."""
        return (self.KIND, [], "ready")

    def build(self, job: Dict[str,Any], *, user: str, workspace: str,
              policy: RespondPolicy, ev_index: EvidenceIndex) -> BuildResult:
        raise NotImplementedError

    def _audit(self, **k):
        AUDIT.append(dict(kind=self.KIND, **k))


