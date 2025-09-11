# imu_repo/tools/imu_verify.py
from __future__ import annotations
import argparse, json, sys
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore, TrustError

def main():
    ap = argparse.ArgumentParser(description="Verify IMU artifact manifest/evidences in CAS")
    ap.add_argument("--cas", required=True, help="CAS root dir")
    ap.add_argument("--manifest-sha", help="manifest sha256 (if omitted: latest/manifest)")
    ap.add_argument("--min-trust", type=float, default=0.75)
    args = ap.parse_args()

    cas = CAS(args.cas)
    if not args.manifest_sha:
        args.manifest_sha = cas.resolve("latest/manifest")["sha256"]
    store = ProvenanceStore(cas, min_trust=args.min_trust)
    try:
        res = store.verify_chain(args.manifest_sha)
    except TrustError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    print(json.dumps({"ok": True, **res}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())