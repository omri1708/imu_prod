# imu_repo/run_pipeline_multi.py
from __future__ import annotations
import sys, json
from synth.specs import BuildSpec, Contract
from engine.pipeline_multi import run_pipeline_multi

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

def main():
    spec = BuildSpec(
        name="stage48_suite",
        kind="web_service",
        language_pref=["python"],
        ports=[19393],
        endpoints={"/hello":"hello_json","/bg_sum":"bg_sum","/ui":"static_ui"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    out = run_pipeline_multi(spec, user_id="multi_user")
    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()