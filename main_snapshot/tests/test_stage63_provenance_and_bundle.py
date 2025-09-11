# imu_repo/tests/test_stage63_provenance_and_bundle.py
from __future__ import annotations
import os, time, json, socket
from typing import List, Tuple
from db.sandbox import create_namespace, exec_write, exec_read, grant_access, DB_ROOT, META_ROOT
from grounded.provenance_store import add_evidence, get_evidence, get_meta, verify
from packaging.html_bundle import build_html_bundle, serve_html_bundle, DIST

def assert_true(cond, msg=""):
    if not cond:
        print("ASSERT FAIL:", msg)
        raise SystemExit(1)

# -------- DB Sandbox --------

def test_db_sandbox_ttl_quota_acl():
    ns = "events_test"
    schema = """
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ns TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    """
    # צור namespace עם quota=5 ו-ttl=1s
    create_namespace(ns, schema_sql=schema, owners=["system"], readers=["system"], quota_rows=5, ttl_seconds=1)

    # כתיבה + קריאה
    now = int(time.time())
    for i in range(3):
        exec_write(ns, "INSERT INTO events(ns,data,created_at) VALUES(?,?,?)", ("ns", f"d{i}", now))
    rows = exec_read(ns, "SELECT COUNT(1) FROM events", ())
    assert_true(rows[0][0]==3, "insert_count_wrong")

    # TTL: נחכה שיעבור הזמן ונכניס עוד — המנגנון ינקה ישנים
    time.sleep(1.2)
    for i in range(4):
        exec_write(ns, "INSERT INTO events(ns,data,created_at) VALUES(?,?,?)", ("ns", f"n{i}", int(time.time())))
    # quota=5 => אחרי ניקוי TTL אמורים להיות לכל היותר 5
    cnt = exec_read(ns, "SELECT COUNT(1) FROM events", ())[0][0]
    assert_true(cnt <= 5, f"quota_not_enforced:{cnt}")

# -------- Provenance Store --------

def test_provenance_end2end():
    dg = add_evidence(b"hello-evidence", {"source_url": "https://example.test/info", "ttl_s": 5, "trust": 0.9})
    ok = verify(dg, require_hmac=True, min_trust=0.5)
    assert_true(ok["ok"], f"verify_failed:{ok}")
    # בדוק קבלת התוכן + מטא
    content = get_evidence(dg)
    meta = get_meta(dg)
    assert_true(content == b"hello-evidence", "content_mismatch")
    assert_true(meta.get("source_url")=="https://example.test/info", "meta_missing")
    # בדיקת תפוגה
    time.sleep(0.5)
    still_ok = verify(dg)
    assert_true(still_ok["ok"], "should_still_be_valid")

# -------- HTML Bundle --------

def test_html_bundle_build_and_serve():
    p = build_html_bundle({"extra.js": "console.log('imu');"})
    assert_true(os.path.exists(os.path.join(p,"index.html")), "index_missing")
    assert_true(os.path.exists(os.path.join(p,"manifest.json")), "manifest_missing")
    t = serve_html_bundle()
    time.sleep(0.2)
    # בדיקת 'שרת חי' ע"י ניסיון לפתוח סוקט
    s = socket.socket()
    try:
        s.settimeout(0.5)
        s.connect(("127.0.0.1", 8999))
        assert_true(True)
    finally:
        s.close()

def run():
    test_db_sandbox_ttl_quota_acl()
    test_provenance_end2end()
    test_html_bundle_build_and_serve()
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())