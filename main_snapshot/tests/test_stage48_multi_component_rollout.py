# imu_repo/tests/test_stage48_multi_component_rollout.py
from __future__ import annotations
import json
from synth.specs import BuildSpec, Contract
from engine.pipeline_multi import run_pipeline_multi
from grounded.source_policy import policy_singleton as SourcePolicy

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
    SourcePolicy.set_allowlist(["internal.test"])  # ראיות פנימיות בלבד
    spec = BuildSpec(
        name="stage48_multi",
        kind="web_service",
        language_pref=["python"],
        ports=[19494],
        endpoints={
            "/hello":"hello_json",          # ילך ל-api
            "/ui":"static_ui",              # api
            "/bg_task":"bg_sum"             # ילך ל-worker
        },
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )
    res = run_pipeline_multi(spec, user_id="u48")
    agg = res["aggregate"]
    ok = bool(agg["approved"]) and agg["score"] > 70.0 and len(res["components"])>=1
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())