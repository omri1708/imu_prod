from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline

def run():
    schema = {
        "type":"object",
        "properties":{
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1000}},"required":["p95_ms"]},
            "tests":{"type":"object"}
        },
        "required":["perf","tests"]
    }
    spec = BuildSpec(
        name="sandbox_db_ui",
        kind="web_service",
        language_pref=["python"],
        ports=[18080],
        endpoints={"/hello":"hello_json"},  # /ui יתווסף אוטומטית ע"י הגנרטור
        contracts=[Contract(name="svc_perf_ok", schema=schema)],
        evidence_requirements=["service_tests","perf_summary"]
    )
    summary = run_pipeline(spec, user_id="bob")
    ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())