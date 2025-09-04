# imu_repo/tools/imu_rotate_key.py
from __future__ import annotations
import argparse, json, os
from security.signing import _load_keys, _save_keys, ensure_ed25519_key

def main():
    ap = argparse.ArgumentParser(description="Rotate default signing key to Ed25519 key-id")
    ap.add_argument("--new-key-id", required=True)
    args = ap.parse_args()
    ensure_ed25519_key(args.new_key_id)
    doc = _load_keys()
    # הצבע את ברירת־המחדל ל־key החדש ע"י העתקת קישור ל-"default"
    doc["default"] = {"alg":"Ed25519","pub":doc[args.new_key_id]["pub"],"priv":doc[args.new_key_id]["priv"]}
    _save_keys(doc)
    print(json.dumps({"ok":True,"default":"Ed25519","key_id":args.new_key_id}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())