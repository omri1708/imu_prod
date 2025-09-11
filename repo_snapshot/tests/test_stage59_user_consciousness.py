# imu_repo/tests/test_stage59_user_consciousness.py
from __future__ import annotations
import os, json

from user_model.identity import ensure_master_key, user_dir
from user_model.consent import set_consent, revoke, check as check_consent
from user_model.semantic_store import add_memory, get_memory, search, consolidate
from user_model.crypto_utils import open_sealed
from user_model.conflict_resolution import resolve_preferences

U1 = "user:alice@example.com"
U2 = "user:bob@example.com"

def assert_true(x, msg=""):
    if not x: 
        print("ASSERT FAIL:", msg)
        raise SystemExit(1)

def test_encryption_and_consent():
    ensure_master_key(U1)
    set_consent(U1, "preferences", granted=True, ttl_s=3600)
    # כתוב רשומה
    sha = add_memory(U1, text="pref: theme=dark", purpose="preferences", tier="T1", confidence=0.6, ttl_s=3600)
    # בדוק שהקובץ מוצפן (אי-אפשר למצוא את המחרוזת בטקסט הקובץ)
    up = user_dir(U1)
    blob_p = os.path.join(up, "mem", "blobs", sha+".json")
    raw = open(blob_p,"r",encoding="utf-8").read()
    assert_true("theme=dark" not in raw, "ciphertext must not contain plaintext")
    # פענוח דרך API
    rec = get_memory(U1, sha)
    assert_true("theme=dark" in rec["text"])

def test_search_and_consolidate_and_conflict():
    set_consent(U1, "preferences", granted=True, ttl_s=3600)
    add_memory(U1, text="pref: theme=light", purpose="preferences", tier="T1", confidence=0.55, ttl_s=3600)
    add_memory(U1, text="pref: theme=dark",  purpose="preferences", tier="T1", confidence=0.70, ttl_s=3600)
    # קונסולידציה (יחשב הופעות ויקדם ל-T2)
    res = consolidate(U1, min_hits=2)
    # חיפוש
    hits = search(U1, "theme preference", topk=5, purpose="preferences")
    assert_true(len(hits)>=1)
    # איחוד סתירות — נבנה מועמדים סינתטיים (מדמים kv במטא)
    # כאן, מאחר והדאטה ב-index לא כולל kv אמיתי, נייצר מבנים ידניים לבדיקה:
    cands = [
        {"kv":{"key":"theme","value":"dark"},"confidence":0.8,"tier":"T2","added_at":0},
        {"kv":{"key":"theme","value":"light"},"confidence":0.6,"tier":"T1","added_at":0},
    ]
    resolve = resolve_preferences(cands)
    assert_true(resolve["decided"] and resolve["winner"]["kv"]["value"] in ("dark","light"))

def test_revoke_consent_blocks_new_writes():
    revoke(U1, "preferences")
    ok = check_consent(U1, "preferences")["ok"]
    assert_true(not ok, "consent should be revoked")
    blocked = False
    try:
        add_memory(U1, text="pref: language=he", purpose="preferences")
    except PermissionError:
        blocked = True
    assert_true(blocked, "write should block without consent")

def run():
    test_encryption_and_consent()
    test_search_and_consolidate_and_conflict()
    test_revoke_consent_blocks_new_writes()
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())