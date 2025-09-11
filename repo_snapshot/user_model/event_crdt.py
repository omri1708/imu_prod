# imu_repo/user_model/event_crdt.py
from __future__ import annotations
from typing import Dict, Any, List
import hashlib, json

def event_id(ev: Dict[str,Any]) -> str:
    """
    מזהה יציב לאירוע T0: hash של (kind,key,value,ts_approx,source).
    נזהר משדות דינמיים (confidence/ttl_s/evidence_id אינם בזהות).
    """
    core = {
        "kind": ev.get("kind"),
        "key":  ev.get("key"),
        "value": ev.get("value"),
        "source": ev.get("source",""),
        "ts_approx": round(float(ev.get("ts",0.0)), 3),
    }
    s = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",",":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def gset_union(a: List[Dict[str,Any]], b: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    CRDT G-Set: איחוד לפי event_id; אם שתי גרסאות עם אותו id — נעדיף ts הגבוה.
    """
    out: Dict[str,Dict[str,Any]] = {}
    def add_all(arr):
        for ev in arr:
            eid = ev.get("id") or event_id(ev)
            cur = out.get(eid)
            if (cur is None) or (float(ev.get("ts",0.0)) > float(cur.get("ts",0.0))):
                nev = dict(ev)
                nev["id"] = eid
                out[eid] = nev
    add_all(a); add_all(b)
    return list(out.values())