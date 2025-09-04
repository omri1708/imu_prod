# imu_repo/tests/test_grounded_end2end.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from user_model.memory import UserMemory

def run():
    # 1) בנה שירות עם חובת עדות "service_tests"
    spec = BuildSpec(
        name="grounded_service",
        kind="web_service",
        language_pref=["python","node"],
        ports=[18080],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="health_ok", schema={"type":"object"})],
        evidence_requirements=["service_tests"]
    )
    summary = run_pipeline(spec)
    assert summary["rollout"]["approved"], "rollout gate failed"
    # 2) זיכרון משתמש: תעדוף "פייתון" ו"system" + פיוס סתירות
    mem = UserMemory()
    for v in ("python","python","go"):
        mem.put_episode("alice","preference",{"key":"lang_pref","value":v}, confidence=0.8)
    mem.consolidate("alice")
    prof = mem.read_profile("alice")
    print("PROFILE:", prof["prefs"].get("lang_pref"))

    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())