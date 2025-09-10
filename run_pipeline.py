# run_pipeline.py
from __future__ import annotations
import json, sys, os
from synth.specs import BuildSpec, Contract
from pipeline.synthesis import SynthesisPipeline

SP = SynthesisPipeline
def default_spec() -> BuildSpec:
    return BuildSpec(
        name="hello_service",
        kind="web_service",
        language_pref=["python","node"],
        ports=[18080],
        endpoints={"/hello":"hello_json","/time":"echo_time"},
        contracts=[
            Contract(name="health_ok", schema={"type":"object"}),
            Contract(name="status_ok", schema={"type":"object"}),
        ],
        evidence_requirements=["service_tests"]
    )

def main():
    spec = default_spec()
    summary = SP.run(spec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()