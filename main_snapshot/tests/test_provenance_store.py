# imu_repo/tests/test_provenance_store.py
from __future__ import annotations
import os, json, shutil
from ui.dsl import Page, Component
from ui.package import build_ui_artifact
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore

ROOT = "/mnt/data/imu_prov_test"

def setup_module(module=None):
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

def test_end2end_provenance_chain():
    os.environ["IMU_KEYS_PATH"] = "/mnt/data/.imu_keys_prov.json"
    page = Page(title="Prov UI",
                components=[Component(kind="text", id="t", props={"text":"hello prov"})])
    pkg = build_ui_artifact(page, nonce="X", key_id="provKey", cas_root=ROOT, min_trust=0.6)
    assert "provenance" in pkg and pkg["provenance"].get("artifact_sha")
    cas = CAS(ROOT)
    store = ProvenanceStore(cas, min_trust=0.6)
    man_sha = cas.resolve("latest/manifest")["sha256"]
    result = store.verify_chain(man_sha)
    assert result["ok"] and result["artifact_sha"] == pkg["provenance"]["artifact_sha"]

def run():
    test_end2end_provenance_chain()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())