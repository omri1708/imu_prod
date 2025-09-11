# imu_repo/tests/test_stage74_capabilities_guarded.py
from __future__ import annotations
import os, asyncio, tempfile, glob, json
from typing import Any, Dict, List

from engine.config import load_config, save_config
from user_model.policy import set_profile
from engine.capability_wrap import registry
from engine.capability_wrap import guard_text_capability_for_user
from grounded.provenance import STORE

# ייבוא היכולות
from capabilities.http_fetch import fetch_text
from capabilities.fs_read import read_text
from capabilities.db_memory import DB, db_get_text

def _reset_env():
    os.makedirs(STORE, exist_ok=True)

async def _call_guarded(name: str, payload: Dict[str,Any], *, user_id: str) -> Dict[str,Any]:
    cap = await guard_text_capability_for_user(registry.get(name), user_id=user_id)
    return await cap(payload)

async def test_all_caps_are_guarded_and_emit_claims():
    _reset_env()
    # אוכפים Evidences חובה
    cfg = load_config()
    cfg.setdefault("evidence", {}).update({"required": True})
    cfg.setdefault("guard", {}).update({"min_trust": 0.7, "max_age_s": 3600})
    save_config(cfg)

    # פרופיל משתמש
    set_profile("u_caps", min_trust=0.7, max_age_s=3600, strict_grounded=True)

    # רשום יכולות
    registry.register("http_fetch", fetch_text)
    registry.register("fs_read", read_text)
    registry.register("db_get_text", db_get_text)

    # הכנת קובץ מקומי
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("hello-fs")
        p = f.name

    # הכנת DB בזיכרון
    DB.set("greet", "hello-db")

    # קריאות עטופות — חוזר dict עם text+claims חתומים
    out1 = await _call_guarded("http_fetch", {"url":"https://example/data","content":"hello-http"}, user_id="u_caps")
    out2 = await _call_guarded("fs_read", {"path": p}, user_id="u_caps")
    out3 = await _call_guarded("db_get_text", {"key": "greet"}, user_id="u_caps")

    for out in (out1,out2,out3):
        assert isinstance(out, dict) and "text" in out and "claims" in out
        assert isinstance(out["claims"], list) and len(out["claims"])>0

    # אימות שקבצי CAS של ראיות נוצרו
    files = glob.glob(os.path.join(STORE, "*.json"))
    assert files, "expected evidence CAS json files"
    # אין אימות של HMAC כאן כי נעשה בתוך ה-middleware עם assert

def run():
    asyncio.get_event_loop().run_until_complete(test_all_caps_are_guarded_and_emit_claims())
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())