from __future__ import annotations
from typing import List, Tuple, Any
import os, sqlite3, re

DB_ROOT = "/mnt/data/imu_repo/dbs"
os.makedirs(DB_ROOT, exist_ok=True)

ALLOWED = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE")

_sql_re = re.compile(r"^\s*([A-Za-z]+)")

def _check_sql(sql: str) -> None:
    m = _sql_re.match(sql or "")
    if not m: raise RuntimeError("sql_empty")
    op = m.group(1).upper()
    if op not in ALLOWED:
        raise RuntimeError(f"sql_op_not_allowed:{op}")

def db_path(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-","_"))
    return os.path.join(DB_ROOT, f"{safe}.sqlite")

def execute(dbname: str, sql: str, params: Tuple[Any,...]=()) -> List[Tuple[Any,...]]:
    _check_sql(sql)
    p = db_path(dbname)
    con = sqlite3.connect(p, timeout=2.0)
    try:
        cur = con.cursor()
        cur.execute(sql, params)
        if _sql_re.match(sql).group(1).upper()=="SELECT":
            rows = cur.fetchall()
            con.commit()
            return rows
        con.commit()
        return []
    finally:
        con.close()