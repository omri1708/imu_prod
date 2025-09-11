# imu_repo/tools/imu_trust.py
from __future__ import annotations
import argparse, json
from provenance.trust_registry import TrustRegistry

def main():
    ap = argparse.ArgumentParser(description="IMU Trust Registry CLI")
    ap.add_argument("--set-source", help="full source url")
    ap.add_argument("--set-prefix", help="prefix like https:// or imu://")
    ap.add_argument("--trust", type=float, help="0..1")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    reg = TrustRegistry()
    if args.set_source and args.trust is not None:
        reg.set_source_trust(args.set_source, args.trust)
    if args.set_prefix and args.trust is not None:
        reg.set_prefix_trust(args.set_prefix, args.trust)
    if args.show:
        print(json.dumps({"ok":True,"registry": reg._read()}, indent=2))
    else:
        print(json.dumps({"ok":True}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())