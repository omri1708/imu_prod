from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy

def run():
    # מאפשרים גם דומיינים חיצוניים לבדיקה (כאן user.example "חלש")
    SourcePolicy.set_allowlist(["internal.test", "example.com", "user.example"])

    schema = {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1000}},"required":["p95_ms"]},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":80}},"required":["score"]}
        },
        "required":["tests","perf","ui"]
    }

    # מקרה א׳: דורש min_trust=0.8 לעדות חיצונית "ext_user" → ייפול (user.example≈0.4)
    spec_bad = BuildSpec(
        name="stage41_trust_fail",
        kind="web_service",
        language_pref=["python"],
        ports=[18484],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="reqs", schema=schema, evidence_min_trust={"ext_user": 0.8})],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[{"key":"ext_user","url":"https://user.example/data"}]
    )
    failed_ok = False
    try:
        run_pipeline(spec_bad, user_id="avi")
        failed_ok = False
    except Exception as e:
        failed_ok = "evidence_low_trust:ext_user" in str(e) or "evidence_low_trust" in str(e)

    # מקרה ב׳: internal.test עם אמון גבוה → יעבור
    spec_ok = BuildSpec(
        name="stage41_trust_pass",
        kind="web_service",
        language_pref=["python"],
        ports=[18585],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="reqs", schema=schema, evidence_min_trust={"ext_int": 0.9})],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        external_evidence=[{"key":"ext_int","url":"internal.test://doc"}]
    )
    passed_ok = False
    try:
        summary = run_pipeline(spec_ok, user_id="avi")
        passed_ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]
    except Exception:
        passed_ok = False

    ok = failed_ok and passed_ok
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())