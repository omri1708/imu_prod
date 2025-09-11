# imu_repo/tests/test_stage39_accessibility.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline

def run():
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
        name="stage39_ui_access",
        kind="web_service",
        language_pref=["python"],
        ports=[18282],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc_perf_ui_ok", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    summary = run_pipeline(spec, user_id="noa")
    ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())