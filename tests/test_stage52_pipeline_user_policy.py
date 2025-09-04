# imu_repo/tests/test_stage52_pipeline_user_policy.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from user_model.identity import ensure_user, load_policy, save_policy

def _schema():
    return {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1500}}},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":65}}}
        },
        "required":["tests","perf","ui"]
    }

def run():
    user = ensure_user("strict@corp.com"); uid = user["uid"]
    pol = load_policy(uid)
    pol["quality"] = "strict"
    pol["latency_p95_ms"] = 1200
    save_policy(uid, pol)

    spec = BuildSpec(
        name="stage52_user_policy",
        kind="web_service",
        language_pref=["python"],
        ports=[19898],
        endpoints={"/hello":"hello_json","/ui":"static_ui"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    s = run_pipeline(spec, user_id=uid)
    # בהנחה שהפייפליין שלך משתמש ב-base_targets["p95_ms"] בזמן מדידת perf,
    # אנו בודקים שהריצה מאושרת והמדדים קיימים.
    ok = s["rollout"]["approved"] and s["kpi"]["score"] >= 70.0
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())