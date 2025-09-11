from __future__ import annotations
import os
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.evidence_policy import policy_singleton as EvidencePolicy

def run():
    # מקשיחים את הספים – לדוגמה דורשים אמון ~0.97 ל-service_tests (גבוה)
    EvidencePolicy.batch_update({
        "service_tests": 0.97,
        "perf_summary":  0.90,
        "ui_accessibility": 0.80
    })

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
        name="stage40_trust_policy",
        kind="web_service",
        language_pref=["python"],
        ports=[18383],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc_perf_ui_ok", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"]
    )

    ok=True
    try:
        summary = run_pipeline(spec, user_id="lev")
        # אם עבר — סימן שלא נאכף סף האמון → כשל בבדיקה
        ok = False
    except Exception as e:
        # מצופה ליפול עם evidence_low_trust:service_tests
        msg = str(e)
        ok = ("evidence_low_trust:service_tests" in msg) or ("provenance" in msg)

    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())