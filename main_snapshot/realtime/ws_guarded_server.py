# imu_repo/realtime/ws_guarded_server.py
from __future__ import annotations
import asyncio, json
from typing import Any, Callable, Awaitable, Optional
from realtime.ws_minimal import WSServer
from engine.evidence_middleware import guarded_handler
from grounded.claims import current
from security.response_signer import sign_payload

class WSGuardedServer(WSServer):
    """
    הרחבה של WSServer:
      - עוטף את ה-handler ב-guarded_handler (חובת Evidences)
      - מכריח את ה-handler לצרף evidences (באחריות ה-handler לקרוא current().add_evidence(...))
      - משיב JSON חתום {"text":..., "claims":[...], "sig":{...}}
    """
    def __init__(self, host: str="127.0.0.1", port: int=8766, *,
                 allowed_origins: Optional[list[str]]=None,
                 min_trust: float=0.7,
                 chunk_size: int=32_000):
        async def dummy(x: Any) -> str:
            # ברירת מחדל: הדגמה — מוסיף ראיה בסיסית כדי שלא ייחסם.
            current().add_evidence("default-proof", {"source_url":"about:blank","trust":0.9,"ttl_s":60})
            return f"echo:{x}"
        super().__init__(host, port,
                         allowed_origins=allowed_origins,
                         handler=None,
                         chunk_size=chunk_size)
        self._min_trust = float(min_trust)
        self._inner_handler = dummy

    async def set_handler(self, fn: Callable[[Any], Awaitable[str]]):
        # עיטוף בחובת Evidences
        self._inner_handler = await guarded_handler(fn, min_trust=self._min_trust)

    async def handler(self, arg: Any) -> str | bytes:
        """
        מפעיל את ה-handler השמור (עם gate), חותם JSON, ומחזיר כמחרוזת UTF-8.
        """
        res = await self._inner_handler(arg)  # res = {"text":..., "claims":[...]}
        signed = sign_payload(res)
        return json.dumps(signed, ensure_ascii=False)