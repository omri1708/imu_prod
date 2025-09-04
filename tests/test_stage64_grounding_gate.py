# imu_repo/tests/test_stage64_grounding_gate.py
from __future__ import annotations
import asyncio, time
from grounded.claims import current, respond_with_evidence
from grounded.gate import GateDenied
from engine.evidence_middleware import guarded_handler
from grounded.provenance_store import add_evidence
from db.sandbox_multi import create_namespace_multi, exec_write, exec_read, DBAclError

def assert_true(b, msg=""):
    if not b:
        print("ASSERT FAIL:", msg); raise SystemExit(1)

# -------- Evidence Gate: חייב ראיות תקפות --------

async def _raw_handler_echo(x: str) -> str:
    # מדמה "מודול שעובד נכון": מוסיף ראיה עבור תוכן נלווה (למשל השיטה/נוסחה/מקור)
    cur = current()
    cur.add_evidence("arithmetics:2+2=4", {"source_url": "https://example.calc", "trust": 0.95, "ttl_s": 60})
    return f"answer:{x}"

async def _raw_handler_missing(x: str) -> str:
    # לא מוסיף ראיות — אמור להיכשל בשער
    return f"answer:{x}"

async def test_gate_enforced():
    ok_handler = await guarded_handler(_raw_handler_echo, min_trust=0.7)
    out = await ok_handler("query")
    assert_true(out["text"]=="answer:query", "ok_handler_text")
    assert_true(len(out["claims"])==1, "ok_handler_claims")

    bad_handler = await guarded_handler(_raw_handler_missing, min_trust=0.7)
    try:
        await bad_handler("q2")
        assert_true(False, "gate_should_fail")
    except Exception as e:
        assert_true(isinstance(e, GateDenied), "expected_gate_denied")

# -------- DB multi-user + הצפנת שדות --------

def test_db_multi_encryption_acl():
    ns="userspace"
    schema = """
    CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    """
    create_namespace_multi(ns, schema, owners=["alice"], readers=["bob"], quota_rows=100, ttl_seconds=60,
                           enc_columns={"NOTES": ["DATA"]})  # case-insensitive בהיגיון ה-SQL שלנו (אנו בודקים upper)

    # כתיבה ע"י alice — data יוצפן
    now = int(time.time())
    n = exec_write(ns, "INSERT INTO notes(owner,data,created_at) VALUES(?,?,?)", ("alice", "secret:hello", now), user="alice")
    assert_true(n==1, "insert_alice")

    # קריאה ע"י bob (יש לו read) — מפוענח
    rows = exec_read(ns, "SELECT id,owner,data,created_at FROM notes", (), user="bob")
    assert_true(len(rows)==1 and rows[0][2]=="secret:hello", "decrypt_for_reader")

    # כתיבה ע"י bob — אמורה להיכשל (אין לו write)
    try:
        exec_write(ns, "INSERT INTO notes(owner,data,created_at) VALUES(?,?,?)", ("bob", "nope", now), user="bob")
        assert_true(False, "bob_write_should_fail")
    except Exception as e:
        assert_true(isinstance(e, DBAclError), "expected_acl_error")

def run():
    asyncio.run(test_gate_enforced())
    test_db_multi_encryption_acl()
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())