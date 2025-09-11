from __future__ import annotations

from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline

def run():
    spec = BuildSpec(
        name="e2e_demo",
        kind="web_service",
        language_pref=["python","node"],
        ports=[18080],
        endpoints={"/hello":"hello_json","/healthz":"OK"},
        contracts=[Contract(name="health_ok", schema={"type":"object"})],
        evidence_requirements=["service_tests"]
    )
    summary = run_pipeline(spec)
    ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())