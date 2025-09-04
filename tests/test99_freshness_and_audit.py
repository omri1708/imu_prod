# imu_repo/tests/test_freshness_and_audit.py
from __future__ import annotations
import os, shutil, json
from grounded.claims import current
from ui.dsl import Page, Component
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from engine.audit_log import verify_chain, AUDIT_PATH
from ui.package import build_ui_artifact

ROOT = "/mnt/data/imu_stage99"

def setup_module(module=None):
    if os.path.exists(ROOT): shutil.rmtree(ROOT)
    try: current().clear()
    except Exception: pass
    if os.path.exists(AUDIT_PATH): os.remove(AUDIT_PATH)

def test_freshness_profiles_and_audit_chain():
    os.environ["IMU_KEYS_PATH"]   = "/mnt/data/.imu_keys99.json"
    os.environ["IMU_TRUST_PATH"]  = "/mnt/data/.imu_trust99.json"
    os.environ["IMU_POLICY_PATH"] = "/mnt/data/.imu_policy99.json"

    # הזרקת ראיות: אחת "news" ישנה מדי, שתיים רעננות ("ui","docs")
    current().add_evidence("old_news", {"kind":"news", "source_url":"https://news.example/1",
                                        "ts": 0, "ttl_s": 999999, "payload":{"h":"x"}})
    current().add_evidence("ui", {"kind":"ui", "source_url":"imu://ui/table",
                                  "ttl_s": 30*24*3600, "payload":{"ok":True}})
    current().add_evidence("docs", {"kind":"docs", "source_url":"https://docs.example/spec",
                                    "ttl_s": 7*24*3600, "payload":{"v":2}})

    # payments→prod במדיניות ברירת־מחדל (דורש min_sources>=3) → עדיין חסר מקור שלישי
    try:
        enforce_evidence_gate(current().snapshot(), domain="payments")
        assert False, "expected GateFailure"
    except GateFailure:
        pass

    # הוסף מקור שלישי (identity)
    current().add_evidence("idp", {"kind":"identity","source_url":"https://idp.example/.well-known",
                                   "ttl_s":24*3600, "payload":{"jwks":"..."}})

    gate = enforce_evidence_gate(current().snapshot(), domain="payments")
    page = Page(title="Stage99", components=[Component(kind="text", id="t", props={"text":"ok"})])
    pkg = build_ui_artifact(page, key_id="default", cas_root=ROOT, min_trust=gate["policy"]["min_trust"])

    cas = CAS(ROOT)
    store = ProvenanceStore(cas, min_trust=gate["policy"]["min_trust"])
    man = cas.resolve("latest/manifest")["sha256"]
    v = store.verify_chain(man)
    assert v["ok"]

    # בדיקת שרשרת Audit
    a = verify_chain()
    assert a["ok"] and a["count"] >= 1

def run():
    test_freshness_profiles_and_audit_chain()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())