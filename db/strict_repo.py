# imu_repo/db/strict_repo.py
from __future__ import annotations
import os, sqlite3, hashlib, time, json
from typing import Dict, Any, List, Tuple

class StrictRepo:
    """
    מעטפת SQLite בסיסית:
      - מאחסנת קובץ db תחת .state אם אין מסלול.
      - כל קריאה מחזירה rows ו־claim על השאילתה (hash), כולל Evidence על קובץ ה־DB.
    """
    def __init__(self, *, path: str|None=None):
        if path is None:
            d = os.environ.get("IMU_STATE_DIR") or ".state"
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "repo.sqlite3")
        self.path = path
        self._ensure()

    def _ensure(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            c = conn.cursor()
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("CREATE TABLE IF NOT EXISTS sample (id INTEGER PRIMARY KEY, name TEXT, score REAL);")
            conn.commit()
            # הזרע דמו אם ריק
            cur = c.execute("SELECT COUNT(1) FROM sample;")
            n = cur.fetchone()[0]
            if n == 0:
                c.executemany("INSERT INTO sample(name,score) VALUES(?,?)", [
                    ("Alice", 91.5), ("Bob", 77.0), ("Carol", 88.2)
                ])
                conn.commit()
        finally:
            conn.close()

    def _hash_sql(self, sql: str, params: Tuple[Any,...]) -> str:
        b = (sql + "::" + json.dumps(params)).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def query(self, sql: str, params: Tuple[Any,...]=()) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
        qh = self._hash_sql(sql, params)
        claim = {
            "id": f"db:{qh[:16]}",
            "type": "db_query",
            "text": f"sqlite query {sql}",
            "schema": {"type":"tabular","unit":""},
            "value": len(rows),
            "evidence": [{
                "kind":"sqlite_file","path": self.path, "ts": time.time()
            },{
                "kind":"query_hash","sha256": qh
            }],
            "consistency_group": "db_rows"
        }
        return rows, [claim]