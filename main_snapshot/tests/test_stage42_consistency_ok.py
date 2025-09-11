# imu_repo/tests/test_stage42_consistency_ok.py
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
        name="stage42_ok",
        kind="web_service",
        language_pref=["python"],
        ports=[18686],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[]  # אין ראיות חוץ → הציון יישאר 100 או קרוב (אין pairwise)
    )
    s = run_pipeline(spec, user_id="omer")
    ok = s["tests"]["passed"] and s["verify"]["ok"] and s["rollout"]["approved"] and s["consistency"]["ok"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())