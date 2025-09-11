# imu_repo/tests/test_stage46_engine_integration.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy
from user_model.consolidation import Consolidator

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
    SourcePolicy.set_allowlist(["internal.test"])
    uid = "omer"
    cons = Consolidator()
    # קובעים מראש העדפת שפה T2: "python"
    cons.add_event(uid, "preference", {"key":"lang_pref","value":"python"}, confidence=0.9, trust=0.9, stable_hint=True)
    cons.consolidate(uid)

    spec = BuildSpec(
        name="stage46_integration",
        kind="web_service",
        language_pref=["go","rust","python"],  # python יעלה לראש מתוך T2
        ports=[19292],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )

    s = run_pipeline(spec, user_id=uid)
    # הצלחה = עבר טסטים ו-rollout אושר; ובנוסף שפת ה-generation היא python
    ok = s["tests"]["passed"] and s["rollout"]["approved"] and (s["generated"]["language"]=="python")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())