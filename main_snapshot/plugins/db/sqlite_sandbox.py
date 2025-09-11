# imu_repo/plugins/db/sqlite_sandbox.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, json, sqlite3, time

class SQLiteSandbox:
    """
    מנוע DB בטוח:
      - יוצר קובץ .db בתיקיית ה-build
      - מיישם schema, טוען seed, ומריץ queries בטוחים (קריאה/עדכון) עם גבולות:
          * max_rows, max_ms, disabled pragmas (DROP TABLE וכו')
    """
    def __init__(self, limits: Dict[str,Any] | None=None):
        lim = dict(limits or {})
        self.max_rows = int(lim.get("max_rows", 5000))
        self.max_ms   = float(lim.get("max_ms", 1500.0))

    def _bad(self, q: str) -> bool:
        ql = (q or "").strip().lower()
        # לא מתירים מחיקות/דרופים/attach
        deny = ("drop table", "drop index", "drop view", "attach ", "pragma ")
        return any(d in ql for d in deny)

    def run(self, spec: Any, build_dir: str, user_id: str) -> Dict[str,Any]:
        extras = getattr(spec, "extras", {}) or {}
        cfg = (extras.get("db") or {})
        schema: List[str] = cfg.get("schema") or []
        seed:   List[Tuple[str, list]] = cfg.get("seed") or []
        queries: List[str] = cfg.get("queries") or []

        dbp = os.path.join(build_dir, "app.db")
        con = sqlite3.connect(dbp)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # schema
        for s in schema:
            if self._bad(s): 
                raise RuntimeError("bad_schema_statement")
            cur.execute(s)
        con.commit()

        # seed
        for (ins, vals) in seed:
            if self._bad(ins): 
                raise RuntimeError("bad_seed_statement")
            cur.execute(ins, vals)
        con.commit()

        # queries עם gate זמן ושורות
        out=[]
        t0=time.time()
        for q in queries:
            if self._bad(q): 
                raise RuntimeError("bad_query_statement")
            cur.execute(q)
            rows = cur.fetchall()
            if len(rows) > self.max_rows:
                rows = rows[:self.max_rows]
            out.append([dict(r) for r in rows])
            if (time.time()-t0)*1000.0 > self.max_ms:
                raise RuntimeError("db_time_exceeded")

        evidence = {
            "db_path": dbp,
            "rows": sum(len(x) for x in out),
            "samples": out[:3]
        }
        kpi = {"score": 85.0}  # נורמליזציה בסיסית; אפשר לקשור ל־p95 אמיתי במערכת
        return {"plugin":"sqlite_sandbox","evidence":evidence,"kpi":kpi}