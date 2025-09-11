# imu_repo/tests/test_adaptive_and_chain.py
from __future__ import annotations
import os, shutil
from grounded.claims import current
from ui.dsl import Page, Component
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from policy.adaptive import AdaptivePolicyController
from ui.package import build_ui_artifact

ROOT = "/mnt/data/imu_stage98"

def setup_module(module=None):
    if os.path.exists(ROOT): shutil.rmtree(ROOT)
    # אפס ראיות קודמות
    try: current().clear()
    except Exception: pass

def test_adaptive_policy_and_signed_evidences():
    # קבע אמון/מקורות
    os.environ["IMU_TRUST_PATH"]  = "/mnt/data/.imu_trust98.json"
    os.environ["IMU_POLICY_PATH"] = "/mnt/data/.imu_policy98.json"
    os.environ["IMU_KEYS_PATH"]   = "/mnt/data/.imu_keys98.json"

    # מדיניות קיימת
    ctrl = AdaptivePolicyController(os.environ["IMU_POLICY_PATH"])
    before = ctrl.current()["risk_levels"]["prod"]

    # אסוף 2 מקורות → אמור להיכשל ב-prod (דורש min_sources>=3)
    current().add_evidence("ui", {"source_url":"imu://ui/sandbox","payload":{"ok":True}, "ttl_s":86400})
    current().add_evidence("spec", {"source_url":"https://docs.example/spec","payload":{"v":1}, "ttl_s":86400, "trust":0.8})

    try:
        enforce_evidence_gate(current().snapshot(), domain="payments")  # mapped→prod
        assert False, "expected GateFailure (not enough sources)"
    except GateFailure:
        pass

    # הוסף מקור שלישי
    current().add_evidence("table", {"source_url":"imu://ui/table","payload":{"ok":True}, "ttl_s":86400})

    g = enforce_evidence_gate(current().snapshot(), domain="payments")
    assert g["policy"]["min_sources"] >= 3

    # הפק ארטיפקט עם evidences+signature
    page = Page(title="Stage98", components=[Component(kind="text", id="t", props={"text":"ok"})])
    pkg = build_ui_artifact(page, key_id="default", cas_root=ROOT, min_trust=g["policy"]["min_trust"])

    # אימות שרשרת
    cas = CAS(ROOT)
    store = ProvenanceStore(cas, min_trust=g["policy"]["min_trust"])
    man = cas.resolve("latest/manifest")["sha256"]
    vr = store.verify_chain(man)
    assert vr["ok"]

    # נניח ביצועים גרועים → קשיחת מדיניות
    res = ctrl.update_with_metrics("prod", p95_ms=800, error_rate=0.05)
    assert res["ok"]
    after = res["new"]
    assert after["min_trust"] >= before["min_trust"]
    assert after["max_ttl_s"]  <= before["max_ttl_s"]

def run():
    test_adaptive_policy_and_signed_evidences()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())