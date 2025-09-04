# imu_repo/db/sandbox.py
from __future__ import annotations
import os, json, sqlite3, time, re, threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

DB_ROOT = "/mnt/data/imu_repo/db"
META_ROOT = "/mnt/data/imu_repo/db/meta"
os.makedirs(DB_ROOT, exist_ok=True)
os.makedirs(META_ROOT, exist_ok=True)

_SQL_OK = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
_SQL_FORBIDDEN = re.compile(r"\b(ATTACH|DETACH|PRAGMA|VACUUM|ALTER|DROP|CREATE\s+TRIGGER|CREATE\s+VIEW)\b", re.IGNORECASE)

_lock = threading.RLock()

class DBPolicyError(Exception): ...
class DBAclError(Exception): ...
class DBQuotaError(Exception): ...
class DBTtlError(Exception): ...

def _ns_path(ns: str) -> str:
    return os.path.join(DB_ROOT, f"{ns}.db")

def _meta_path(ns: str) -> str:
    return os.path.join(META_ROOT, f"{ns}.json")

def _now_s() -> int:
    return int(time.time())

def _load_meta(ns: str) -> Dict[str, Any]:
    p = _meta_path(ns)
    if not os.path.exists(p):
        raise DBPolicyError(f"namespace_meta_missing:{ns}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_meta(ns: str, meta: Dict[str, Any]) -> None:
    tmp = _meta_path(ns) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _meta_path(ns))

def create_namespace(ns: str, schema_sql: str, *,
                     owners: Optional[List[str]] = None,
                     readers: Optional[List[str]] = None,
                     quota_rows: int = 10000,
                     ttl_seconds: int = 0) -> None:
    """
    יוצר namespace חדש עם schema קשיח, ACL בסיסי, TTL ו-Quota לפי שורות.
    דרישה: כל הטבלאות שזקוקות ל-TTL יכילו עמודה 'created_at INTEGER'.
    """
    with _lock:
        dbp = _ns_path(ns)
        if os.path.exists(dbp):
            raise DBPolicyError(f"namespace_exists:{ns}")
        # צרוב DB ו-schema (מותר CREATE TABLE בלבד בשלב ההקמה)
        con = sqlite3.connect(dbp)
        try:
            con.executescript(schema_sql)
            con.commit()
        finally:
            con.close()
        meta = {
            "owners": owners or ["system"],
            "readers": readers or ["system"],
            "quota_rows": int(quota_rows),
            "ttl_seconds": int(ttl_seconds),
            "tables": _introspect_tables(ns),
        }
        _save_meta(ns, meta)

def grant_access(ns: str, *, user: str, read: bool=False, own: bool=False) -> None:
    with _lock:
        m = _load_meta(ns)
        if read and user not in m["readers"]:
            m["readers"].append(user)
        if own and user not in m["owners"]:
            m["owners"].append(user)
        _save_meta(ns, m)

def _introspect_tables(ns: str) -> List[str]:
    dbp = _ns_path(ns)
    con = sqlite3.connect(dbp)
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()

def _assert_acl(ns: str, user: str, write: bool) -> None:
    m = _load_meta(ns)
    if write:
        if user not in m["owners"]:
            raise DBAclError(f"write_denied:{ns}:{user}")
    else:
        if user not in m["owners"] and user not in m["readers"]:
            raise DBAclError(f"read_denied:{ns}:{user}")

def _assert_sql_safe(sql: str) -> None:
    if _SQL_FORBIDDEN.search(sql):
        raise DBPolicyError("forbidden_sql")
    if not _SQL_OK.search(sql):
        raise DBPolicyError("only_select_insert_update_delete_allowed")

def _enforce_ttl(ns: str, con: sqlite3.Connection) -> None:
    m = _load_meta(ns)
    ttl = int(m.get("ttl_seconds", 0))
    if ttl <= 0:
        return
    now_cut = _now_s() - ttl
    for t in m["tables"]:
        # ננקה רק אם קיימת עמודת created_at
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if "created_at" in cols:
                con.execute(f"DELETE FROM {t} WHERE created_at < ?", (now_cut,))
        except sqlite3.Error:
            continue
    con.commit()

def _total_rows(ns: str, con: sqlite3.Connection) -> int:
    m = _load_meta(ns)
    total = 0
    for t in m["tables"]:
        try:
            total += con.execute(f"SELECT COUNT(1) FROM {t}").fetchone()[0]
        except sqlite3.Error:
            continue
    return int(total)

def _evict_oldest(ns: str, con: sqlite3.Connection, target_total: int) -> None:
    """
    מפנה רשומות ישנות (על בסיס created_at) מכל הטבלאות עד שיורדים מתחת לסף.
    """
    m = _load_meta(ns)
    tables = []
    for t in m["tables"]:
        # רק טבלאות עם created_at
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if "created_at" in cols:
                tables.append(t)
        except sqlite3.Error:
            continue
    if not tables:
        raise DBQuotaError("quota_exceeded_no_evict_strategy")
    while _total_rows(ns, con) > target_total:
        # מצא מועמדים ישנים
        for t in tables:
            try:
                con.execute(f"DELETE FROM {t} WHERE rowid IN (SELECT rowid FROM {t} ORDER BY created_at ASC LIMIT 1)")
            except sqlite3.Error:
                pass
        con.commit()

def _enforce_quota(ns: str, con: sqlite3.Connection) -> None:
    m = _load_meta(ns)
    q = int(m["quota_rows"])
    rows = _total_rows(ns, con)
    if rows <= q:
        return
    # העדפה: לפנות רשומות עתיקות
    _evict_oldest(ns, con, target_total=q)

def exec_write(ns: str, sql: str, params: Iterable[Any] | None = None, *, user: str="system") -> int:
    """
    הרצה מבוקרת של כתיבה. אוכף ACL/TTL/Quota ו-SQL Safe.
    מחזיר מספר שורות שהושפעו.
    """
    with _lock:
        _assert_acl(ns, user, write=True)
        _assert_sql_safe(sql)
        con = sqlite3.connect(_ns_path(ns))
        try:
            _enforce_ttl(ns, con)
            cur = con.execute(sql, tuple(params or []))
            con.commit()
            _enforce_quota(ns, con)
            return cur.rowcount if cur.rowcount is not None else 0
        finally:
            con.close()

def exec_read(ns: str, sql: str, params: Iterable[Any] | None = None, *, user: str="system") -> List[Tuple[Any,...]]:
    """
    הרצה מבוקרת של קריאה. אוכף ACL ו-SQL Safe (SELECT בלבד / ללא פקודות מסוכנות).
    """
    with _lock:
        _assert_acl(ns, user, write=False)
        _assert_sql_safe(sql)
        con = sqlite3.connect(_ns_path(ns))
        try:
            cur = con.execute(sql, tuple(params or []))
            return cur.fetchall()
        finally:
            con.close()