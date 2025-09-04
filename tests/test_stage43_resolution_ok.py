# imu_repo/tests/test_stage43_resolution_ok.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy

def run():
    # מאפשרים מקורות פנימיים וגם "user.example" (נמוך אמון)
    SourcePolicy.set_allowlist(["internal.test", "user.example"])

    schema = {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1000}},"required":["p95_ms"]},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":80}},"required":["score"]}
        },
        "required":["tests","perf","ui"]
    }

    # נפעיל evidence חיצונית "חלשה" בכוונה: תיצור סתירה ב-perf מול הפנימית
    spec = BuildSpec(
        name="stage43_rescue",
        kind="web_service",
        language_pref=["python"],
        ports=[18787],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc", schema=schema)],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[{"key":"ext_perf","url":"user.example://perf"}]
    )

    # הפייפליין יזהה סתירה, יבצע resolve_contradictions(trust_cut=0.8),
    # יפיל ראיות נמוכות אמון, ייצור resolution_proof, ויתקדם.
    s = run_pipeline(spec, user_id="noa")
    ok = s["tests"]["passed"] and s["verify"]["ok"] and s["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())