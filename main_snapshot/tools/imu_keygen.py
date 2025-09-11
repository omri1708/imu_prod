# imu_repo/tools/imu_keygen.py
from __future__ import annotations
import argparse, json
from security.signing import ensure_ed25519_key, _load_keys

def main():
    ap = argparse.ArgumentParser(description="Generate Ed25519 key in IMU_KEYS_PATH")
    ap.add_argument("--key-id", default="prodKey")
    args = ap.parse_args()
    ensure_ed25519_key(args.key_id)
    keys = _load_keys()
    print(json.dumps({"ok":True,"created":args.key_id,"alg":keys[args.key_id]["alg"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())