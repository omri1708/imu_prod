# imu_repo/tests/test_provenance_gate_and_cli.py
from __future__ import annotations
import os, json, shutil, subprocess, sys
from ui.dsl import Page, Component
from ui.package import build_ui_artifact
from grounded.claims import current
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore

ROOT = "/mnt/data/imu_prov96"

def setup_module(module=None):
    if os.path.exists(ROOT): shutil.rmtree(ROOT)

def test_gate_and_cli():
    os.environ["IMU_KEYS_PATH"] = "/mnt/data/.imu_keys_prov96.json"
    # אסוף ראיות "אמינות"
    current().add_evidence("ui_render", {"source_url":"imu://ui/sandbox","trust":0.95,"ttl_s":3600,"payload":{"ok":True}})
    current().add_evidence("ui_table_render", {"source_url":"imu://ui/table","trust":0.94,"ttl_s":3600,"payload":{"ok":True}})
    # gate
    g = enforce_evidence_gate(current().snapshot(), min_trust=0.7)
    assert g["agg_trust"] >= 0.7

    # בנה ארטיפקט וכתוב ל-CAS
    page = Page(title="GateUI", components=[Component(kind="text", id="t", props={"text":"ok"})])
    pkg = build_ui_artifact(page, nonce="X", key_id="gk", cas_root=ROOT, min_trust=0.7)
    cas = CAS(ROOT)
    man = cas.resolve("latest/manifest")["sha256"]

    # CLI
    code = subprocess.call([sys.executable, "/mnt/data/imu_repo/tools/imu_verify.py", "--cas", ROOT, "--manifest-sha", man, "--min-trust", "0.7"])
    assert code == 0

def run():
    test_gate_and_cli()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())