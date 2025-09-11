# imu_repo/engine/gates/privacy_gate.py
from __future__ import annotations
from typing import Dict, Any
from user_model.consent import check as check_consent

class PrivacyGate:
    """
    בודק הסכמה (Consent) לפני פעולת קריאה/כתיבה של user store:
      cfg = {"user_key":"...", "purpose":"preferences"}
    """
    def __init__(self, user_key: str, purpose: str):
        self.user_key = user_key
        self.purpose = purpose

    def check(self) -> Dict[str,Any]:
        res = check_consent(self.user_key, self.purpose)
        return {"ok": res.get("ok", False), "consent": res}