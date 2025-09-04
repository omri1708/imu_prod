from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline

def run():
    perf_schema = {
        "type":"object",
        "properties": {
            "tests": {"type":"object"},
            "perf": {
                "type":"object",
                "properties": {"p95_ms":{"type":"number","maximum": 1000}},
                "required": ["p95_ms"]
            }
        },
        "required": ["tests","perf"]
    }
    spec = BuildSpec(
        name="perf_grounded",
        kind="web_service",
        language_pref=["python","node"],
        ports=[18080],
        endpoints={"/hello":"hello_json","/t":"echo_time"},
        contracts=[Contract(name="svc_ok", schema=perf_schema)],
        evidence_requirements=["service_tests","perf_summary"]
    )
    summary = run_pipeline(spec, user_id="alice")
    ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())