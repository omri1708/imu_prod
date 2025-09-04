# imu_repo/grounded/evidence_policy.py
from __future__ import annotations
from typing import Dict, Any, Optional
from grounded.provenance import ProvenanceStore
from grounded.trust import trust_score
from grounded.source_policy import policy_singleton as Policy

class EvidencePolicyError(Exception): ...

class EvidencePolicy:
    """
    שומר כללי min_trust לראיות (לפי שם הראיה, לדוגמה: 'service_tests', 'perf_summary', 'ui_accessibility').
    ניתן לעדכן בזמן ריצה.
    """
    def __init__(self):
        # ברירת מחדל: בדיקות פנימיות דורשות אמון גבוה; UI/Perf דיפולט.
        self.min_trust_by_key: Dict[str, float] = {
            "service_tests": 0.90,    # פנימי/מאומת
            "perf_summary":  0.80,
            "db_migration":  0.80,
            "ui_accessibility": 0.70,
            # אפשר להוסיף/לדרוס מבחוץ
        }

    def set_min_trust(self, key: str, value: float) -> None:
        self.min_trust_by_key[key] = float(value)

    def batch_update(self, mapping: Dict[str, float]) -> None:
        for k, v in mapping.items():
            self.set_min_trust(k, v)

    def required_trust(self, key: str) -> float:
        return float(self.min_trust_by_key.get(key, 0.0))

    def check(self, pv: ProvenanceStore, keys: list[str]) -> None:
        """
        מאמת שהראיות קיימות, חתומות, טריות, ועומדות ב-min_trust שנקבע במדיניות.
        זורק חריגה אם משהו אינו עומד במדיניות.
        """
        for k in keys:
            rec = pv.get(k)
            if not rec:
                raise EvidencePolicyError(f"evidence_missing:{k}")
            if not rec.get("_sig_ok", False):
                raise EvidencePolicyError(f"evidence_bad_sig:{k}")
            if not rec.get("_fresh", False):
                raise EvidencePolicyError(f"evidence_stale:{k}")
            # דרישת אמון:
            required = self.required_trust(k)
            # trust אפקטיבי: מקסימום בין class_score ל-trust הידני ששמנו בעת put
            eff_trust = max(float(rec.get("class_score", 0.0)), float(rec.get("trust", 0.0)))
            if eff_trust < required:
                raise EvidencePolicyError(f"evidence_low_trust:{k}:{eff_trust:.2f}<{required:.2f}")

policy_singleton = EvidencePolicy()