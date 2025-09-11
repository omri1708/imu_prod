# engine/respond.py
# -*- coding: utf-8 -*-
import time
from typing import List, Dict, Any, Optional

from governance.policy import RespondPolicy
from grounded.evidence_contracts import EvidenceIndex
from engine.contracts_gate import enforce_respond_contract
from perf.monitor import monitor_global
from audit.log import AppendOnlyAudit
from governance.slo_gate import gate_p95

AUDIT = AppendOnlyAudit("var/audit/respond.jsonl")

def respond_grounded_json(text: str,
                          claims: Optional[List[Dict[str,Any]]],
                          evidence: Optional[List[Dict[str,Any]]],
                          policy: RespondPolicy,
                          ev_index: EvidenceIndex,
                          user: str = "anonymous") -> Dict[str,Any]:
    """
    מחזיר תשובה רק אם עומדת במדיניות: claims+evidence מחויבים
    אימות Evidences מול EvidenceIndex, ושער p95 לפני החזרה.
    """
    t0 = time.time()
    enforce_respond_contract(text=text, claims=claims, evidence=evidence, policy=policy, ev_index=ev_index)
    # אם הגענו לפה, המדיניות נאכפה; אפשר להחזיר.
    elapsed_ms = (time.time() - t0)*1000.0
    monitor_global.observe_ms(elapsed_ms)
    # עמידה ב-SLO לפני יציאה (חותך אם לא עומד)
    gate_p95(max_ms=policy.evidence.max_age_sec * 1000.0 if policy.evidence.max_age_sec < 60 else 250.0)
    AUDIT.append({"kind":"respond_ok","user":user,"ms":elapsed_ms,"claims":len(claims or []),"evidence":len(evidence or [])})
    return {"ok": True, "text": text, "claims": claims or [], "evidence": evidence or [], "latency_ms": elapsed_ms}