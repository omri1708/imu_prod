# ui_dsl/live_bindings.py
# -*- coding: utf-8 -*-
"""
חיבור UI-DSL ל-/run_adapter ולברוקר ה-WS כך ש-Progress/Timeline מתעדכנים חי.
אין תלות חיצונית; שימוש ב-websocket-client אם זמין, אחרת fallback ל-HTTP polling קצר (מוגבל ע"י Policy).
"""
from __future__ import annotations
import json, time, threading
from typing import Callable, Dict, Any, Optional
try:
    import websocket  # type: ignore
except Exception:
    websocket = None

from governance.enforcement import guard_ws, guard_rate

class LiveStream:
    def __init__(self, user_id: str, ws_url: str, on_event: Callable[[Dict[str, Any]], None]):
        self.user_id = user_id
        self.ws_url = ws_url
        self.on_event = on_event
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self):
        host = self.ws_url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
        guard_ws(self.user_id, host)

        if websocket is None:
            # Fallback ל-polling (כל 1.5s) – נשלט ע"י rate limit 'ui_push'
            def _poll():
                import urllib.request
                while not self._stop.is_set():
                    guard_rate(self.user_id, "ui_push", 1)
                    try:
                        with urllib.request.urlopen(self.ws_url.replace("wss://", "https://") + "?poll=1", timeout=3) as r:
                            data = json.loads(r.read().decode("utf-8"))
                            for ev in data.get("events", []):
                                self.on_event(ev)
                    except Exception:
                        pass
                    time.sleep(1.5)
            self._t = threading.Thread(target=_poll, daemon=True)
            self._t.start()
            return

        def _run_ws():
            try:
                def _on_message(_, message):
                    guard_rate(self.user_id, "ui_push", 1)
                    try:
                        ev = json.loads(message)
                        self.on_event(ev)
                    except Exception:
                        pass
                ws = websocket.WebSocketApp(self.ws_url, on_message=_on_message)
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
        self._t = threading.Thread(target=_run_ws, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)