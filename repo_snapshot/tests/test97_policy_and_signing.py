# imu_repo/tests/test_policy_and_signing.py
from __future__ import annotations
import os, shutil, json, subprocess, sys
from policy.policy_engine import PolicyEngine
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from grounded.claims import current
from ui.dsl import Page, Component
from ui.package import build_ui_artifact
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore
from security.signing import ensure_ed25519_key

ROOT = "/mnt/data/imu_stage97"

def setup_module(module=None):
    if os.path.exists(ROOT): shutil.rmtree(ROOT)

def test_policy_levels_and_ed25519_or_hmac():
    # תן ראיות מכמה מקורות
    current().clear()
    current().add_evidence("ui_render", {"source_url":"imu://ui/sandbox","trust":0.96,"ttl_s":86400,"payload":{"ok":True}})
    current().add_evidence("ui_logic",  {"source_url":"imu://ui/table",  "trust":0.95,"ttl_s":86400,"payload":{"ok":True}})
    pe = PolicyEngine()
    # high: דורש min_sources >=3 → צריך להיכשל
    try:
        enforce_evidence_gate(current().snapshot(), domain="ui_admin", policy_engine=pe)
        assert False, "expected GateFailure"
    except GateFailure:
        pass
    # הוסף מקור שלישי
    current().add_evidence("cfg", {"source_url":"https://example.com/spec","trust":0.80,"ttl_s":86400,"payload":{"v":"1"}})
    g = enforce_evidence_gate(current().snapshot(), domain="ui_admin", policy_engine=pe)
    assert g["policy"]["min_sources"] >= 3
    # הפק ארטיפקט; נסה להשתמש ב־Ed25519 אם קיים pynacl
    os.environ["IMU_KEYS_PATH"] = "/mnt/data/.imu_keys97.json"
    try:
        ensure_ed25519_key("prodKey")
        key_id = "prodKey"
    except Exception:
        key_id = "default"  # HMAC fallback
    page = Page(title="Stage97", components=[Component(kind="text", id="t", props={"text":"ok"})])
    pkg = build_ui_artifact(page, key_id=key_id, cas_root=ROOT, min_trust=g["policy"]["min_trust"])
    cas = CAS(ROOT)
    store = ProvenanceStore(cas, min_trust=g["policy"]["min_trust"])
    man = cas.resolve("latest/manifest")["sha256"]
    res = store.verify_chain(man)
    assert res["ok"] is True

def run():
    test_policy_levels_and_ed25519_or_hmac()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())