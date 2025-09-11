# imu_repo/db/contracts.py
from __future__ import annotations
import re, sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------- מודל חוזה סכימה ----------
@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str  # "INTEGER" | "TEXT" | "REAL" | "BLOB"
    not_null: bool = False
    pk: bool = False
    default: str | None = None

@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Tuple[ColumnSpec, ...]  # סדר העמודות חשוב ל-PK מרוכב
    uniques: Tuple[Tuple[str, ...], ...] = ()
    indexes: Tuple[Tuple[str, ...], ...] = ()

SchemaContract = Dict[str, TableSpec]

class SchemaMismatchError(Exception): ...
class QueryRejected(Exception): ...

# --------- אימות סכימה בפועל מול חוזה ----------
def _fetch_table_info(conn: sqlite3.Connection, table: str) -> List[dict]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = []
    for cid, name, ctype, notnull, dflt_value, pk in cur.fetchall():
        cols.append({
            "name": name,
            "type": (ctype or "").upper().strip(),
            "notnull": bool(notnull),
            "pk": pk and True or False,
            "default": dflt_value
        })
    return cols

def _exists_table(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None

def _create_table_sql(ts: TableSpec) -> str:
    defs = []
    for c in ts.columns:
        line = f"{c.name} {c.type}"
        if c.not_null: line += " NOT NULL"
        if c.default is not None: line += f" DEFAULT {c.default}"
        if c.pk: line += " PRIMARY KEY"
        defs.append(line)
    col_sql = ", ".join(defs)
    # UNIQUEs
    uniq_sql = []
    for u in ts.uniques:
        uniq_sql.append(f", UNIQUE ({', '.join(u)})")
    return f"CREATE TABLE IF NOT EXISTS {ts.name} ({col_sql}{''.join(uniq_sql)});"

def ensure_schema(conn: sqlite3.Connection, contract: SchemaContract, *, create_if_missing: bool=True) -> None:
    """
    מוודא שהסכימה תואמת לחוזה.
    - אם טבלה חסרה: תיווצר (אם create_if_missing=True).
    - אם עמודה/טיפוס/NOT NULL/PK לא תואמים: ייזרק SchemaMismatchError (לא מבצעים ALTER אוטומטי).
    """
    for tname, ts in contract.items():
        if not _exists_table(conn, tname):
            if not create_if_missing:
                raise SchemaMismatchError(f"missing table: {tname}")
            conn.execute(_create_table_sql(ts))
        # אימות
        actual = _fetch_table_info(conn, tname)
        want = list(ts.columns)
        if len(actual) != len(want):
            raise SchemaMismatchError(f"column count mismatch in {tname}: got {len(actual)} != {len(want)}")
        for a, w in zip(actual, want):
            if a["name"] != w.name: raise SchemaMismatchError(f"{tname}: column name mismatch {a['name']} != {w.name}")
            if a["type"] != w.type.upper(): raise SchemaMismatchError(f"{tname}.{w.name}: type mismatch {a['type']} != {w.type}")
            if bool(a["notnull"]) != bool(w.not_null): raise SchemaMismatchError(f"{tname}.{w.name}: notnull mismatch")
            if bool(a["pk"]) != bool(w.pk): raise SchemaMismatchError(f"{tname}.{w.name}: pk mismatch")

# --------- כללי ולידציה לשאילתות ----------
_ALLOWED_VERBS = ("SELECT","INSERT","UPDATE","DELETE")
# נחפש שמות טבלאות אחרי מילות מפתח נפוצות
_TABLE_TOKENS = ("FROM","INTO","UPDATE","JOIN","DELETE FROM")

def _extract_idents(sql: str) -> List[str]:
    s = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    s = re.sub(r"--.*?$", "", s, flags=re.M)
    tokens = re.split(r"(\s+|,|\(|\))", s)
    out, last = [], ""
    for tok in tokens:
        if not tok or tok.isspace(): continue
        up = tok.upper()
        if last.upper() == "DELETE" and up == "FROM":
            last = "DELETE FROM"
            continue
        if up in _TABLE_TOKENS:
            last = up
            continue
        if last in ("FROM","INTO","UPDATE","JOIN","DELETE FROM"):
            ident = tok.strip().strip("`[]\"")
            # הסר אליאסים בסוף
            ident = re.split(r"\s+", ident)[0]
            ident = ident.rstrip(";")
            if ident:
                out.append(ident)
            last = ""
        else:
            last = tok
    return out

def validate_query(sql: str, contract: SchemaContract, *, require_limit_on_select: bool=True, max_limit: int=1000) -> None:
    sql = sql.strip().rstrip(";")
    if not sql: raise QueryRejected("empty query")
    verb = sql.split(None,1)[0].upper()
    if verb not in _ALLOWED_VERBS:
        raise QueryRejected(f"verb not allowed: {verb}")
    # בדיקת טבלאות מול חוזה
    tables = _extract_idents(sql)
    unknown = [t for t in tables if t not in contract]
    if unknown:
        raise QueryRejected(f"unknown tables: {unknown}")

    # אכיפת LIMIT ל-SELECT
    if verb == "SELECT" and require_limit_on_select:
        m = re.search(r"\bLIMIT\s+(\d+)\b", sql, flags=re.I)
        if not m:
            raise QueryRejected("SELECT without LIMIT")
        lim = int(m.group(1))
        if lim > max_limit:
            raise QueryRejected(f"LIMIT too high ({lim} > {max_limit})")