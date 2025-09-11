# imu_repo/db/sandbox_sqlite.py
from __future__ import annotations
import sqlite3, os, re
from typing import Dict, Any, List, Tuple, Optional, Iterable

ALLOWED_PREFIX = ("SELECT","PRAGMA","CREATE","INSERT","UPDATE","DELETE","DROP","ALTER","BEGIN","COMMIT","ROLLBACK")

class DBPolicyError(Exception): ...
class DBError(Exception): ...

def _check(sql: str):
    head = (sql.strip().split(None,1)[0] or "").upper()
    if head not in ALLOWED_PREFIX:
        raise DBPolicyError(f"sql_not_allowed:{head}")

def run_script(db_path: str, statements: Iterable[str], transactional: bool = True) -> List[Tuple[str, Any]]:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out=[]
    try:
        if transactional: conn.execute("BEGIN")
        for st in statements:
            st = st.strip()
            if not st: continue
            _check(st)
            cur = conn.execute(st)
            if cur.description:
                out.append((st, [dict(r) for r in cur.fetchall()]))
            else:
                out.append((st, cur.rowcount))
        if transactional: conn.execute("COMMIT")
    except Exception as e:
        if transactional:
            try: conn.execute("ROLLBACK")
            except Exception: pass
        raise DBError(str(e))
    finally:
        conn.close()
    return out

def ensure_schema(db_path: str, ddl: str) -> None:
    # מפרק ל־statements לפי ';' (נאיבי אך מספיק כאן)
    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    run_script(db_path, stmts, transactional=True)

def migrate(db_path: str, migrations: List[str]) -> List[Tuple[str,Any]]:
    return run_script(db_path, migrations, transactional=True)