# imu_repo/engine/respond_strict.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable, Tuple
from engine.strict_mode import strict_package_response
from engine.policy_overrides import apply_user_overrides

GenerateFn = Callable[[Dict[str,Any]], Tuple[str, Optional[List[Dict[str,Any]]]]]
# contract: generate(ctx) → (response_text, claims|None)

class RespondStrict:
    """
    מתאם תגובה שמחייב claims+evidence (או compute-claim דטרמיניסטי),
    ממזג מדיניות עם פרופיל משתמש, ומחזיר proof חתום.
    """
    def __init__(self, *, base_policy: Dict[str,Any],
                 http_fetcher: Optional[Callable[[str,str], tuple]] = None,
                 sign_key_id: Optional[str] = None):
        self.base_policy = base_policy or {}
        self.http_fetcher = http_fetcher
        self.sign_key_id = sign_key_id or "root"

    def respond(self, *, ctx: Dict[str,Any], generate: GenerateFn) -> Dict[str,Any]:
        user = (ctx or {}).get("user") or {}
        effective_policy = apply_user_overrides(self.base_policy, user)
        text, claims = generate(ctx)
        # אריזת תשובה עם אכיפה קשיחה
        proof = strict_package_response(
            response_text=text, claims=claims, policy=effective_policy,
            http_fetcher=self.http_fetcher, sign_key_id=self.sign_key_id
        )
        # מחזיר bundle לשכבות הבאות (verify/rollout/observe)
        return {"ok": True, "bundle": proof, "policy": effective_policy, "text": text}