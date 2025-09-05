# imu_repo/realtime/strict_ws.py
from __future__ import annotations
import json, time, hashlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from engine.respond_strict import RespondStrict

class StrictWSMux:
    """
    מולטיפלקסר לוגי לשידור הודעות "ריל־טיים" (abstract):
      - send() תמיד אורז הודעה עם claims (אם חסר → compute-claim).
      - מוכן לחיבור ל־WS אמיתי/HTTP SSE — כאן נשארת שכבה טהורה שאינה תלויה ברשת.
    """
    def __init__(self, *, base_policy: Dict[str,Any], http_fetcher=None, sign_key_id: str="root"):
        self.responder = RespondStrict(base_policy=base_policy, http_fetcher=http_fetcher, sign_key_id=sign_key_id)

    def send(self, *, ctx: Dict[str,Any], channel: str, payload: Dict[str,Any], claims: Optional[List[Dict[str,Any]]]=None) -> Dict[str,Any]:
        msg = {"ch": channel, "ts": time.time(), "payload": payload}
        def _gen(_ctx: Dict[str,Any]):
            return (json.dumps(msg, ensure_ascii=False), claims)
        return self.responder.respond(ctx=ctx, generate=_gen)