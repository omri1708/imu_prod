# imu_repo/tools/imu_policy_tune.py
from __future__ import annotations
import argparse, json
from policy.adaptive import AdaptivePolicyController

def main():
    ap = argparse.ArgumentParser(description="IMU adaptive policy tuning")
    ap.add_argument("--risk", required=True, choices=["low","medium","high","prod"])
    ap.add_argument("--p95-ms", required=True, type=float)
    ap.add_argument("--error-rate", required=True, type=float)
    args = ap.parse_args()
    ctrl = AdaptivePolicyController()
    res = ctrl.update_with_metrics(args.risk, args.p95_ms, args.error_rate)
    print(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())