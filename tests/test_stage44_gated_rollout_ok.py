# imu_repo/tests/test_stage44_gated_rollout_ok.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy
from grace.grace_manager import refill

def run():
    # מרוקנים/ממלאים Grace למשתמש
    refill("dana", tokens=2)
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
        name="stage44_service",
        kind="web_service",
        language_pref=["python"],
        ports=[18888],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[]
    )

    s = run_pipeline(spec, user_id="dana")
    ok = s["tests"]["passed"] and s["verify"]["ok"] and s["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())