# imu_repo/tests/test_stage44_regression_block.py
from __future__ import annotations
import json, time, os
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy
from guard.anti_regression import HIST_PATH
from grace.grace_manager import refill

def run():
    # מכניס היסטוריית KPI גבוהה כדי לייצר רגרסיה
    os.makedirs(os.path.dirname(HIST_PATH), exist_ok=True)
    with open(HIST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time()-3600, "service": "stage44_regress", "kpi_score": 95.0, "p95_ms": 120.0}) + "\n")
    # לא נותנים גרייס
    refill("shay", tokens=0)
    SourcePolicy.set_allowlist(["internal.test"])

    schema = {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1000}},"required":["p95_ms"]},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":80}},"required":["score"]}
        },
        "required":["tests","perf","ui"]
    }

    spec = BuildSpec(
        name="stage44_regress",
        kind="web_service",
        language_pref=["python"],
        ports=[18989],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[]
    )

    s = run_pipeline(spec, user_id="shay")
    ok = (not s["rollout"]["approved"]) and ("anti_regression" in s["rollout"])
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())