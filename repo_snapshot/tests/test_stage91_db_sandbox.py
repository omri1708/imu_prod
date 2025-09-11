# imu_repo/tests/test_stage91_db_sandbox.py
from __future__ import annotations
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.caps_db import db_tx_capability

def _cfg():
    cfg = load_config()
    # Guard מינימלי כדי לא לחסום (יש Evidences internal)
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    # Φ: תקציב נינוח ובמיוחד על db.tx
    cfg["phi"] = {"max_allowed": 200.0, "per_capability_cost": {"db.tx": 2.0}}
    save_config(cfg)

SCHEMA = {
    "users": {
        "columns": [
            {"name":"id","type":"INTEGER","pk":True,"not_null":True},
            {"name":"name","type":"TEXT","not_null":True},
            {"name":"age","type":"INTEGER","not_null":False},
        ],
        "uniques": [],
        "indexes": []
    },
    "orders": {
        "columns": [
            {"name":"id","type":"INTEGER","pk":True,"not_null":True},
            {"name":"user_id","type":"INTEGER","not_null":True},
            {"name":"total","type":"REAL","not_null":True},
        ],
        "uniques": [],
        "indexes": []
    }
}

def test_db_tx_commit_and_rollback_and_limit():
    _cfg()
    current().reset()
    with user_scope("mila"):
        cap = db_tx_capability("mila")

        # 1) טרנזקציה תקינה: יצירת נתונים וקריאה מוגבלת
        ok_payload = {
            "schema_contract": SCHEMA,
            "ops": [
                ("INSERT INTO users(id,name,age) VALUES(?,?,?)", (1,"Ana",30)),
                ("INSERT INTO users(id,name,age) VALUES(?,?,?)", (2,"Ben",22)),
                ("INSERT INTO orders(id,user_id,total) VALUES(?,?,?)", (10,1,99.5)),
                ("SELECT id,name FROM users ORDER BY id LIMIT 10", ()),
            ]
        }
        out1 = cap.sync(ok_payload)  # כל היכולות עטופות ב-text_capability_for_user עם .sync נוח לבדיקה
        assert "db_tx_ok" in out1["text"], out1

        # 2) טרנזקציה שתידחה: SELECT ללא LIMIT
        bad_payload = {
            "schema_contract": SCHEMA,
            "ops": [
                ("INSERT INTO users(id,name,age) VALUES(?,?,?)", (3,"Cid",44)),
                ("SELECT id,name FROM users ORDER BY id", ()),  # אין LIMIT → יידחה → ROLLBACK לכל הטרנזקציה
            ]
        }
        out2 = cap.sync(bad_payload)
        assert "db_tx_failed" in out2["text"], out2

        # בדיקת Evidences
        evs = current().snapshot()
        kinds = {e["kind"] for e in evs}
        assert "db_tx_begin" in kinds
        assert "db_tx_commit" in kinds
        assert "db_tx_rollback" in kinds
        # Grounding: לכל הרצות נוצרו db_exec evidences
        assert any(e["kind"] == "db_exec" for e in evs)

def run():
    test_db_tx_commit_and_rollback_and_limit()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())