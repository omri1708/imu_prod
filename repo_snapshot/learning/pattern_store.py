# learning/pattern_store.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, List

STATE = ".imu_state"
LOG   = os.path.join(STATE, "patterns.jsonl")
INDEX = os.path.join(STATE, "patterns_index.json")

def _h(s:str)->str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

def record_episode(request_text:str, spec:Dict[str,Any], tools_ok:Dict[str,bool],
                   blueprint:str, ok:bool, latency_ms:int|None=None):
    if not os.path.exists(STATE): os.makedirs(STATE, exist_ok=True)
    rec = {"ts": int(time.time()*1000), "h": _h(request_text.strip()[:512]),
           "domain": _dominant(spec), "blueprint": blueprint, "ok": bool(ok),
           "tools": tools_ok, "latency_ms": latency_ms or 0}
    with open(LOG,"a",encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    idx = {}
    if os.path.exists(INDEX):
        try: idx = json.loads(open(INDEX,"r",encoding="utf-8").read())
        except Exception: pass
    d = idx.setdefault(rec["domain"], {})
    b = d.setdefault(blueprint, {"n":0,"ok":0,"ms_sum":0})
    b["n"] += 1; b["ok"] += int(ok); b["ms_sum"] += rec["latency_ms"]
    d[blueprint]=b; idx[rec["domain"]]=d
    open(INDEX,"w",encoding="utf-8").write(json.dumps(idx, ensure_ascii=False, indent=2))

def suggest(spec:Dict[str,Any]) -> Dict[str,Any]:
    domain = _dominant(spec)
    if not os.path.exists(INDEX): return {}
    try: idx = json.loads(open(INDEX,"r",encoding="utf-8").read())
    except Exception: return {}
    d = idx.get(domain) or {}
    # בחר Blueprint עם יחס הצלחה גבוה ביותר (וברירת מחדל אם אין)
    best=None; best_score=-1.0
    for bp, m in d.items():
        n = max(1, int(m.get("n",1)))
        ok = int(m.get("ok",0))
        score = ok / n
        if score > best_score:
            best, best_score = bp, score
    return {"suggested_blueprint": best, "domain": domain, "score": best_score}

def _dominant(spec:Dict[str,Any]) -> str:
    types = [c.get("type","") for c in (spec.get("components") or [])]
    if "realtime" in types: return "realtime"
    if "game" in types:     return "game"
    if "mobile" in types:   return "mobile"
    if "web" in types:      return "web"
    if "api" in types:      return "api"
    return "custom"