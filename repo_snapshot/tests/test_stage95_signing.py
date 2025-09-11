# imu_repo/tests/test_stage95_signing.py
from __future__ import annotations
import os, json, hashlib
from ui.dsl import Page, Component
from ui.package import build_ui_artifact
from security.signing import verify_manifest, VerifyError

def _sha256_hex(s: str) -> str:
    h = hashlib.sha256(); h.update(s.encode("utf-8")); return h.hexdigest()

def test_signed_manifest_verification():
    os.environ["IMU_KEYS_PATH"] = "/mnt/data/.imu_keys_test.json"
    page = Page(
        title="Signed",
        components=[Component(kind="text", id="t", props={"text":"Hello"})],
        permissions={}
    )
    pkg = build_ui_artifact(page, nonce="X", key_id="k1")
    # sha matches
    assert pkg["sha256"] == _sha256_hex(pkg["html"])
    # verify signature OK
    verify_manifest(pkg["manifest"])
    # tamper
    bad = dict(pkg["manifest"])
    bad["sha256_hex"] = "deadbeef"
    try:
        verify_manifest(bad)
        assert False, "expected VerifyError"
    except VerifyError:
        pass

def run():
    test_signed_manifest_verification()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())