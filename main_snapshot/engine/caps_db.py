# imu_repo/engine/caps_db.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from grounded.claims import current
from engine.capability_wrap import text_capability_for_user
from engine.policy_ctx import get_user
from db.sandbox import DBSandbox
from db.contracts import TableSpec, ColumnSpec, SchemaContract

# נשמר אינסטנס יחיד ללייף־טיים התהליך (Sandbox פר־תהליך)
_DB: DBSandbox | None = None

def _db() -> DBSandbox:
    global _DB
    if _DB is None:
        _DB = DBSandbox(memory=True)
    return _DB

async def _db_tx_impl(payload: Dict[str, Any]) -> str:
    """
    payload:
      schema_contract: {table: spec}
      ops: [(sql, params_tuple)]
    """
    uid = get_user() or "anon"
    sc_in = payload.get("schema_contract") or {}
    ops_in = payload.get("ops") or []
    # המרה ל־SchemaContract
    contract: SchemaContract = {}
    for tname, tdesc in sc_in.items():
        cols = tuple(ColumnSpec(**c) for c in tdesc["columns"])
        uniques = tuple(tuple(u) for u in tdesc.get("uniques", []))
        idxs = tuple(tuple(ix) for ix in tdesc.get("indexes", []))
        contract[tname] = TableSpec(name=tname, columns=cols, uniques=uniques, indexes=idxs)

    db = _db()
    db.ensure_contract(contract)

    # טרנזקציה
    ops: List[Tuple[str, Tuple]] = []
    for item in ops_in:
        sql = str(item[0])
        params = tuple(item[1]) if len(item) > 1 else tuple()
        ops.append((sql, params))

    res = db.transaction(ops)
    current().add_evidence("db_tx_summary", {
        "source_url":"sqlite:///sandbox","trust":0.98,"ttl_s":600,
        "payload":{"user": uid, "ok": res["ok"], "ops": len(ops)}
    })
    if res["ok"]:
        return f"db_tx_ok ops={len(ops)}"
    else:
        return f"[FALLBACK] db_tx_failed: {res['error']}"

def db_tx_capability(user_id: str):
    # cost תחת Φ; ניתן להקשיח ב-config per_capability_cost
    return text_capability_for_user(_db_tx_impl, user_id=user_id, capability_name="db.tx", cost=2.0)