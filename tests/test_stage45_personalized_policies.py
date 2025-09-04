# imu_repo/tests/test_stage45_personalized_policies.py
from __future__ import annotations
import json, os
from user_model.policies import set_for_user, set_for_app
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy
from grace.grace_manager import refill

def _schema():
    return {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1500}},"required":["p95_ms"]},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":70}},"required":["score"]}
        },
        "required":["tests","perf","ui"]
    }

def run():
    # נאפשר רק internal evidence
    SourcePolicy.set_allowlist(["internal.test"])
    # מילוי grace לשני המשתמשים
    refill("conservative_user", tokens=2)
    refill("fast_user", tokens=2)

    # 1) משתמש שמרן: עקביות חשובה, משקולות KPI לטובת consistency
    set_for_user("conservative_user", {
        "kpi_weights": {"tests":0.25,"latency":0.15,"ui":0.10,"consistency":0.40,"resolution":0.10},
        "min_consistency_score": 90.0,
        "anti_regression": {"max_regression_pct":5.0, "min_kpi":75.0}
    })

    # 2) מדיניות פר־יישום (גוברת) — כאן נעדיף מהירות/latency
    set_for_app("stage45_app", {
        "kpi_weights": {"tests":0.20,"latency":0.40,"ui":0.10,"consistency":0.20,"resolution":0.10},
        "min_consistency_score": 75.0,
        "trust_cut_for_resolution": 0.75,
        "anti_regression": {"max_regression_pct":10.0, "min_kpi":65.0}
    })

    # A. ריצה לשמרן — app לא זהה, אז המדיניות היא של המשתמש
    specA = BuildSpec(
        name="not_the_app",
        kind="web_service",
        language_pref=["python"],
        ports=[19090],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    sA = run_pipeline(specA, user_id="conservative_user")
    okA = sA["tests"]["passed"] and sA["verify"]["ok"] and sA["rollout"]["approved"]
    print("A_OK" if okA else "A_FAIL")

    # B. ריצה למשתמש מהיר על אפליקציה עם מדיניות app — app policy גוברת
    specB = BuildSpec(
        name="stage45_app",
        kind="web_service",
        language_pref=["python"],
        ports=[19191],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    sB = run_pipeline(specB, user_id="fast_user")
    okB = sB["tests"]["passed"] and sB["verify"]["ok"] and sB["rollout"]["approved"]
    print("B_OK" if okB else "B_FAIL")

    return 0 if (okA and okB) else 1

if __name__=="__main__":
    raise SystemExit(run())