# imu_repo/engine/runtime_bridge.py
from __future__ import annotations
from typing import Dict, Any
from runtime.metrics import metrics
from engine.gates.runtime_budget import RuntimeBudgetGate
from engine.gates.slo_gate import SLOGate
from engine.gates.ui_gate import UIGate
from engine.gates.grounding_gate import GroundingGate
from engine.gates.distributed_gate import DistributedGate
from engine.gates.streaming_gate import StreamingGate
from realtime.metrics_stream import StreamMetrics

def apply_runtime_gates(extras: Dict[str,Any] | None, *, bundle: Dict[str,Any] | None=None,
                        stream_metrics: StreamMetrics | None=None) -> Dict[str,Any]:
    """
    מפעיל Gates מערכתיים בהתאם ל-spec.extras:
      - runtime_budget / slo_gate / ui_gate / grounding / distributed / streaming
    """
    out = {"runtime_budget": None, "slo_gate": None, "ui_gate": None, "grounding": None,
           "distributed": None, "streaming": None}
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
        res = gate.check(); out["ui_gate"] = res
        if not res["ok"]:
            raise RuntimeError(f"ui_accessibility_failed:{res['violations']}")

    if extras.get("grounding"):
        if bundle is None:
            raise RuntimeError("grounding_enabled_but_no_bundle")
        gcfg = extras["grounding"]
        gate = GroundingGate(allowed_domains=gcfg.get("allowed_domains"),
                             require_signature=gcfg.get("require_signature", True),
                             min_good_evidence=gcfg.get("min_good_evidence", 1))
        res = gate.check(bundle); out["grounding"] = res
        if not res["ok"]:
            raise RuntimeError(f"grounding_failed:{res['violations']}")

    if extras.get("distributed"):
        dcfg = extras["distributed"]
        gate = DistributedGate(require_quorum=dcfg.get("require_quorum", True),
                               require_leader=dcfg.get("require_leader", True))
        res = gate.check(); out["distributed"] = res
        if not res["ok"]:
            raise RuntimeError(f"distributed_gate_failed:{res['violations']}")

    if extras.get("streaming"):
        if stream_metrics is None:
            raise RuntimeError("streaming_enabled_but_no_metrics")
        scfg = extras["streaming"]
        gate = StreamingGate(stream_metrics,
                             p95_rtt_ms_max=scfg.get("p95_rtt_ms_max", 200.0),
                             max_queue_depth=scfg.get("max_queue_depth", 200))
        res = gate.check(); out["streaming"] = res
        if not res["ok"]:
            raise RuntimeError(f"streaming_gate_failed:{res['violations']}")

    return out