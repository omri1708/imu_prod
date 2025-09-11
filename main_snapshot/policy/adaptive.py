# imu_repo/policy/adaptive.py
from __future__ import annotations
import os, json, threading
from typing import Dict, Any

POLICY_PATH = os.environ.get("IMU_POLICY_PATH", "/mnt/data/.imu_policy.json")

# גבולות קשיחים – כדי לא "להשתולל" בלמידה
HARD_LIMITS = {
    "min_trust": (0.50, 0.98),
    "max_ttl_s": (3600, 30*24*3600),
    "min_sources": (1, 5)
}

class AdaptivePolicyController:
    """
    מעדכן את המדיניות לפי מדדים אמפיריים:
      • אם p95_latency>target או error_rate>target ⇒ העלה min_trust/min_sources והורד max_ttl_s.
      • אם מצוין ביצועים טובים לאורך זמן ⇒ הורד מעט min_trust/העלה ttl להגדלת yield.
    השינויים קטנים ומוגבלים בטווח קשיח.
    """
    def __init__(self, path: str = POLICY_PATH):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            self._write({
                "risk_levels": {
                    "low":    {"min_trust": 0.65, "max_ttl_s": 7*24*3600,  "min_sources": 1},
                    "medium": {"min_trust": 0.75, "max_ttl_s": 72*3600,    "min_sources": 2},
                    "high":   {"min_trust": 0.85, "max_ttl_s": 24*3600,    "min_sources": 3},
                    "prod":   {"min_trust": 0.90, "max_ttl_s": 6*3600,     "min_sources": 3},
                },
                "domain_overrides": {"default": {"risk": "medium"}},
                "targets": {
                    "p95_ms": {"low":600, "medium":500, "high":400, "prod":300},
                    "error_rate": {"low":0.05,"medium":0.03,"high":0.02,"prod":0.01}
                }
            })

    def _read(self) -> Dict[str,Any]:
        with open(self.path, "r", encoding="utf-8") as f: return json.load(f)

    def _write(self, doc: Dict[str,Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def update_with_metrics(self, risk: str, p95_ms: float, error_rate: float) -> Dict[str,Any]:
        with self._lock:
            doc = self._read()
            rl = doc["risk_levels"].get(risk)
            if not rl: return {"ok":False,"reason":"unknown risk"}
            targets = doc["targets"]
            p95_t = float(targets["p95_ms"][risk])
            err_t = float(targets["error_rate"][risk])

            def clamp(k: str, v: float) -> float:
                lo, hi = HARD_LIMITS[k]; return max(lo, min(hi, v))

            new = dict(rl)
            # התאמה פשוטה: חריגה → קשיחה יותר; עמידה טובה → ריכוך
            if p95_ms > p95_t or error_rate > err_t:
                new["min_trust"]   = clamp("min_trust",  rl["min_trust"] + 0.02)
                new["min_sources"] = int(clamp("min_sources", rl["min_sources"] + 1))
                new["max_ttl_s"]   = int(clamp("max_ttl_s", rl["max_ttl_s"] * 0.75))
            else:
                new["min_trust"]   = clamp("min_trust",  rl["min_trust"] - 0.01)
                new["min_sources"] = int(clamp("min_sources", rl["min_sources"]))
                new["max_ttl_s"]   = int(clamp("max_ttl_s", rl["max_ttl_s"] * 1.10))

            doc["risk_levels"][risk] = new
            self._write(doc)
            return {"ok": True, "risk": risk, "old": rl, "new": new}

    def current(self) -> Dict[str,Any]:
        return self._read()