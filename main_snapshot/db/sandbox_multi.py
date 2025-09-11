# imu_repo/db/sandbox_multi.py
from __future__ import annotations
import os, sqlite3, json, time, threading, base64, hashlib, hmac
from typing import Any, Dict, Iterable, List, Tuple
from db.sandbox import (_ns_path, _meta_path, _now_s, _introspect_tables,
                        DBPolicyError, DBAclError, DBQuotaError,
                        _SQL_FORBIDDEN, _SQL_OK)

ROOT = "/mnt/data/imu_repo/db"
META_ROOT = "/mnt/data/imu_repo/db/meta"
SECRET = "/mnt/data/imu_repo/db/enc.key"
os.makedirs(ROOT, exist_ok=True); os.makedirs(META_ROOT, exist_ok=True)
_lock = threading.RLock()

def _key() -> bytes:
    if not os.path.exists(SECRET):
        k = os.urandom(32)
        open(SECRET, "wb").write(k)
        return k
    return open(SECRET, "rb").read()

def _enc(b: bytes) -> str:
    # "XOR+HMAC-tag" פשוט (ללא תלויות): לא AES, אבל הצפנה קלה עם אימות תקינות.
    k = _key()
    x = bytes([b[i] ^ k[i % len(k)] for i in range(len(b))])
    tag = hmac.new(k, x, hashlib.sha256).digest()[:12]
    return base64.b64encode(tag + x).decode("ascii")

def _dec(s: str) -> bytes:
    k = _key()
    raw = base64.b64decode(s.encode("ascii"))
    tag, x = raw[:12], raw[12:]
    if not hmac.compare_digest(tag, hmac.new(k, x, hashlib.sha256).digest()[:12]):
        raise DBPolicyError("enc_tag_mismatch")
    return bytes([x[i] ^ k[i % len(k)] for i in range(len(x))])

def create_namespace_multi(ns: str, schema_sql: str, *,
                           owners: List[str],
                           readers: List[str] | None=None,
                           quota_rows: int=20000,
                           ttl_seconds: int=0,
                           enc_columns: Dict[str, List[str]] | None=None) -> None:
    """
    דומה ל-create_namespace, אך עם בעלי-זכויות מרובים + רשימת עמודות מוצפנות (per table).
    enc_columns: {"table": ["colA","colB",...]}
    """
    dbp = _ns_path(ns)
    if os.path.exists(dbp):
        raise DBPolicyError(f"namespace_exists:{ns}")
    con = sqlite3.connect(dbp)
    try:
        con.executescript(schema_sql); con.commit()
    finally:
        con.close()
    meta = {
        "owners": owners,
        "readers": list(set((readers or [])+owners)),
        "quota_rows": int(quota_rows),
        "ttl_seconds": int(ttl_seconds),
        "tables": _introspect_tables(ns),
        "enc_columns": enc_columns or {}
    }
    open(_meta_path(ns), "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=2))

def _load_meta(ns: str) -> Dict[str,Any]:
    p = _meta_path(ns)
    if not os.path.exists(p): raise DBPolicyError(f"namespace_meta_missing:{ns}")
    return json.load(open(p, "r", encoding="utf-8"))

def _assert_acl(ns: str, user: str, write: bool):
    m = _load_meta(ns)
    if write:
        if user not in m["owners"]: raise DBAclError(f"write_denied:{ns}:{user}")
    else:
        if user not in m["owners"] and user not in m["readers"]:
            raise DBAclError(f"read_denied:{ns}:{user}")

def _assert_sql_safe(sql: str):
    if _SQL_FORBIDDEN.search(sql): raise DBPolicyError("forbidden_sql")
    if not _SQL_OK.search(sql): raise DBPolicyError("only_select_insert_update_delete_allowed")

def _enforce_ttl(ns: str, con: sqlite3.Connection):
    ttl = int(_load_meta(ns).get("ttl_seconds",0))
    if ttl<=0: return
    cut = int(time.time())-ttl
    for t in _load_meta(ns)["tables"]:
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if "created_at" in cols:
                con.execute(f"DELETE FROM {t} WHERE created_at < ?", (cut,))
        except sqlite3.Error: ...
    con.commit()

def _total_rows(ns: str, con: sqlite3.Connection) -> int:
    tot=0
    for t in _load_meta(ns)["tables"]:
        try: tot += con.execute(f"SELECT COUNT(1) FROM {t}").fetchone()[0]
        except sqlite3.Error: ...
    return int(tot)

def _evict(ns: str, con: sqlite3.Connection, target: int):
    tabs=[]
    for t in _load_meta(ns)["tables"]:
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if "created_at" in cols: tabs.append(t)
        except sqlite3.Error: ...
    if not tabs: raise DBQuotaError("quota_exceeded_no_evict_strategy")
    while _total_rows(ns, con) > target:
        for t in tabs:
            try:
                con.execute(f"DELETE FROM {t} WHERE rowid IN (SELECT rowid FROM {t} ORDER BY created_at ASC LIMIT 1)")
            except sqlite3.Error: ...
        con.commit()

def _maybe_encrypt_params(ns: str, sql: str, params: Iterable[Any] | None) -> Iterable[Any]:
    p = list(params or [])
    meta = _load_meta(ns)
    # אם זוהי INSERT/UPDATE לטבלה שהוגדרה עם עמודות מוצפנות — הצפן את הערכים לפי סדר placeholders
    # נישען על פורמט "INSERT INTO T(col1,col2,...) VALUES(?,?,?)" או "UPDATE T SET col=?..."
    up = sql.strip().upper()
    try:
        if up.startswith("INSERT INTO"):
            t = sql.split()[2]
            cols_part = sql.split("(",1)[1].split(")",1)[0]
            cols = [c.strip() for c in cols_part.split(",")]
            enc_cols = set(meta.get("enc_columns", {}).get(t, []))
            for i, c in enumerate(cols):
                if c in enc_cols and isinstance(p[i], str):
                    p[i] = _enc(p[i].encode("utf-8"))
        elif up.startswith("UPDATE"):
            t = sql.split()[1]
            enc_cols = set(meta.get("enc_columns", {}).get(t, []))
            # מפושט: נניח "SET col=? ..." — נרוץ על פרמטרים לפי סדר ההופעה
            # TODO (DX: עבור שימושים מורכבים מומלץ לבנות בעצמך map col->index)
            # כאן קו בטיחותי — אם יש עמודות מוצפנות, נצפין כל str בפרמטרים
            if enc_cols:
                for i,val in enumerate(p):
                    if isinstance(val, str):
                        p[i] = _enc(val.encode("utf-8"))
    except Exception:
        pass
    return tuple(p)

def _maybe_decrypt_rows(ns: str, table: str, rows: List[Tuple[Any,...]], col_names: List[str]) -> List[Tuple[Any,...]]:
    meta = _load_meta(ns)
    enc_cols = set(meta.get("enc_columns", {}).get(table, []))
    if not enc_cols: return rows
    out=[]
    for r in rows:
        r2=list(r)
        for i,col in enumerate(col_names):
            if col in enc_cols and isinstance(r2[i], str):
                try:
                    r2[i] = _dec(r2[i]).decode("utf-8")
                except Exception:
                    # לא נשבור — נחזיר ערך מקורי
                    ...
        out.append(tuple(r2))
    return out

def exec_write(ns: str, sql: str, params: Iterable[Any] | None=None, *, user: str) -> int:
    with _lock:
        _assert_acl(ns, user, write=True); _assert_sql_safe(sql)
        con = sqlite3.connect(_ns_path(ns))
        try:
            _enforce_ttl(ns, con)
            p = _maybe_encrypt_params(ns, sql, params)
            cur = con.execute(sql, p)
            con.commit()
            # Quota
            q = int(_load_meta(ns).get("quota_rows", 20000))
            if _total_rows(ns, con) > q:
                _evict(ns, con, q)
            return cur.rowcount if cur.rowcount is not None else 0
        finally:
            con.close()

def exec_read(ns: str, sql: str, params: Iterable[Any] | None=None, *, user: str) -> List[Tuple[Any,...]]:
    with _lock:
        _assert_acl(ns, user, write=False); _assert_sql_safe(sql)
        con = sqlite3.connect(_ns_path(ns))
        try:
            cur = con.execute(sql, tuple(params or []))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            # נסה לזהות טבלה מתוך FROM הראשון (פשטני)
            tbl = None
            up = sql.strip().upper()
            if " FROM " in up:
                try: tbl = sql.upper().split(" FROM ",1)[1].split()[0]
                except Exception: tbl = None
            if tbl:
                rows = _maybe_decrypt_rows(ns, tbl, rows, cols)
            return rows
        finally:
            con.close()