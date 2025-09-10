from engine.recovery.freeze_window import start_freeze, is_frozen
from engine.recovery.backoff import allow, clear
def test_freeze_backoff(tmp_path, monkeypatch):
    out = start_freeze("svcA", minutes=1, reason="fail")
    fr = is_frozen("svcA"); assert fr["frozen"]
    bk1 = allow("svcA", attempts_max=1); assert bk1["ok"]
    bk2 = allow("svcA", attempts_max=1); assert not bk2["ok"] and bk2["escalate"]
    clear("svcA")
