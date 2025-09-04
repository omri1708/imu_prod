# imu_repo/engine/runtime_bridge.py
from __future__ import annotations
from typing import Dict, Any
from runtime.metrics import metrics
from engine.gates.runtime_budget import RuntimeBudgetGate
from engine.gates.slo_gate import SLOGate
from engine.gates.ui_gate import UIGate
from engine.gates.grounding_gate import GroundingGate

def apply_runtime_gates(extras: Dict[str,Any] | None, *, bundle: Dict[str,Any] | None=None) -> Dict[str,Any]:
    """
    מפעיל Gates מערכתיים בהתאם ל-spec.extras:
      - runtime_budget: {"p95": {...}, "counters_max": {...}}
      - slo_gate: {"p95_ms": {...}, "error_rate_max": 0.05, "min_requests": 10}
      - ui_gate: {"path": "/mnt/data/imu_repo/site", "min_contrast": 4.5}
      - grounding: {"allowed_domains": [...], "require_signature": true, "min_good_evidence": 1}
    פרמ' bundle: אובייקט תשובה עם claims/text לבדיקת Grounding.
    """
    out = {"runtime_budget": None, "slo_gate": None, "ui_gate": None, "grounding": None}
    if not extras: return out

    if extras.get("runtime_budget"):
        rbcfg = extras["runtime_budget"]
        gate = RuntimeBudgetGate(p95=rbcfg.get("p95"), counters_max=rbcfg.get("counters_max"))
        res = gate.check()
        out["runtime_budget"] = res
        if not res["ok"]:
            raise RuntimeError(f"runtime_budget_failed:{res['violations']}")

    if extras.get("slo_gate"):
        scfg = extras["slo_gate"]
        gate = SLOGate(p95_ms=scfg.get("p95_ms"),
                       error_rate_max=scfg.get("error_rate_max", 0.05),
                       min_requests=scfg.get("min_requests", 10))
        res = gate.check()
        out["slo_gate"] = res
        if not res["ok"]:
            raise RuntimeError(f"slo_gate_failed:{res['violations']}")

    if extras.get("ui_gate"):
        ucfg = extras["ui_gate"]
        gate = UIGate(path=ucfg["path"], min_contrast=ucfg.get("min_contrast", 4.5))
        res = gate.check()
        out["ui_gate"] = res
        if not res["ok"]:
            raise RuntimeError(f"ui_accessibility_failed:{res['violations']}")

    if extras.get("grounding"):
        if bundle is None:
            raise RuntimeError("grounding_enabled_but_no_bundle")
        gcfg = extras["grounding"]
        gate = GroundingGate(allowed_domains=gcfg.get("allowed_domains"),
                             require_signature=gcfg.get("require_signature", True),
                             min_good_evidence=gcfg.get("min_good_evidence", 1))
        res = gate.check(bundle)
        out["grounding"] = res
        if not res["ok"]:
            raise RuntimeError(f"grounding_failed:{res['violations']}")

    return out