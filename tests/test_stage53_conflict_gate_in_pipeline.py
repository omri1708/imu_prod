# imu_repo/tests/test_stage53_conflict_gate_in_pipeline.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from user_model.identity import ensure_user
from user_model.memory_store import put_event

def _schema():
    return {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":2000}}},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":60}}}
        },
        "required":["tests","perf","ui"]
    }

def _spec(extras=None):
    return BuildSpec(
        name="stage53_conflict_gate",
        kind="web_service",
        language_pref=["python"],
        ports=[19999],
        endpoints={"/hello":"hello_json","/ui":"static_ui"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility"],
        extras=extras or {}
    )

def run():
    u = ensure_user("gate_conflict@corp.com"); uid = u["uid"]
    # צור אמביגואיות מכוונת סביב dark_mode (mu≈0.5)
    put_event(uid, "pref","dark_mode", True,  confidence=0.6, source="ui")
    put_event(uid, "pref","dark_mode", False, confidence=0.6, source="settings")

    spec_fail = _spec(extras={
        "user_conflict_gate": {"keys":["dark_mode"], "max_ambiguity": 0.25, "min_strength": 0.5}
    })

    failed = False
    try:
        run_pipeline(spec_fail, user_id=uid)
    except Exception as e:
        failed = "user_conflict_gate_failed" in str(e)

    # חיזוק ראיות — החלטה ברורה
    put_event(uid, "pref","dark_mode", True, confidence=0.95, source="ui")
    spec_ok = _spec(extras={
        "user_conflict_gate": {"keys":["dark_mode"], "max_ambiguity": 0.25, "min_strength": 0.5}
    })
    s = run_pipeline(spec_ok, user_id=uid)
    ok = failed and s["rollout"]["approved"]
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())