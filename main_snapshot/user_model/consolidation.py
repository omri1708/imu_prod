# imu_repo/user_model/consolidation.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time
from grounded.personal_evidence import PersonalProvenance
from user_model.conflict_resolution import resolve_records
from grounded.ttl import TTLPolicy

USERS_ROOT = "/mnt/data/imu_repo/users"

def _ensure(p: str): os.makedirs(p, exist_ok=True)

class Consolidator:
    """
    שכבות:
      - T0: אירועים גולמיים (episodes.jsonl)
      - T1: צבירה קצרה (counts.json)
      - T2: העדפות/אמונות/מטרות יציבות (profile.json)
    """
    def __init__(self, root: str = USERS_ROOT):
        self.root = root
        _ensure(root)
        self.pp = PersonalProvenance(root)

    def _dir(self, user_id: str) -> str:
        p = os.path.join(self.root, user_id)
        _ensure(p); _ensure(os.path.join(p,"evidence"))
        return p

    def add_event(self, user_id: str, kind: str, payload: Dict[str,Any], *,
                  confidence: float=0.7, trust: float=0.8, stable_hint: bool=False) -> Dict[str,Any]:
        """
        רושם אירוע ל-T0 + שומר כראיה פרסונלית חתומה עם TTL דינמי.
        """
        d = self._dir(user_id)
        ev = {"ts": time.time(), "kind": kind, "payload": payload, "confidence":confidence, "trust":trust}
        with open(os.path.join(d, "episodes.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # עדכון T1 counts
        counts_p = os.path.join(d, "counts.json")
        counts = {}
        if os.path.exists(counts_p):
            with open(counts_p,"r",encoding="utf-8") as f: counts = json.load(f)
        cnt = counts.get(kind, {"n":0, "last_ts":0})
        cnt["n"] += 1; cnt["last_ts"] = ev["ts"]
        counts[kind] = cnt
        with open(counts_p,"w",encoding="utf-8") as f: json.dump(counts,f,ensure_ascii=False,indent=2)

        # ראיה חתומה
        rec = self.pp.put(user_id, key=f"user:{kind}", payload=payload, kind=kind,
                          confidence=confidence, trust=trust, seen_count=cnt["n"], stable=stable_hint,
                          source_url="user://event")
        return {"event": ev, "evidence": rec}

    def consolidate(self, user_id: str) -> Dict[str,Any]:
        """
        מקדם העדפות/אמונות ל-T2 (profile.json) לפי מופעים ופיוס סתירות.
        """
        d = self._dir(user_id)
        profile_p = os.path.join(d, "profile.json")
        prof = {"preferences":{}, "beliefs":{}, "goals":{}}
        if os.path.exists(profile_p):
            with open(profile_p,"r",encoding="utf-8") as f: prof = json.load(f)

        # אוספים ראיות טריות בלבד
        evs = self.pp.scan(user_id)["records"]
        fresh = [r for r in evs if r.get("_fresh", False)]
        # ממפים לפי preference keys
        buckets: Dict[str, List[Dict[str,Any]]] = {}
        for r in fresh:
            if r["kind"] == "preference":
                v = dict(r.get("payload") or {})
                if "key" in v and "value" in v:
                    rec = {"value": v["value"], "trust": r.get("trust",0.5),
                           "confidence": r.get("confidence",0.5), "_ts": r.get("ts", r.get("expire_ts", time.time()))}
                    buckets.setdefault(v["key"], []).append(rec)

        # רזולוציה לכל preference key
        resolutions = {}
        for k, arr in buckets.items():
            res = resolve_records(arr, value_key="value")
            if res["ok"]:
                prof["preferences"][k] = {"value": res["chosen"], "proof": {"weights": res["weights"]}}
                resolutions[k] = res

        # כתיבה חזרה
        with open(profile_p,"w",encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)

        return {"updated": list(resolutions.keys()), "profile": prof, "counts": len(fresh)}

    def snapshot(self, user_id: str) -> Dict[str,Any]:
        d = self._dir(user_id)
        out = {"episodes":0,"counts":{},"profile":{}}
        ep = os.path.join(d, "episodes.jsonl")
        if os.path.exists(ep):
            with open(ep,"r",encoding="utf-8") as f:
                out["episodes"] = sum(1 for _ in f)
        cp = os.path.join(d, "counts.json")
        if os.path.exists(cp):
            with open(cp,"r",encoding="utf-8") as f: out["counts"] = json.load(f)
        pp = os.path.join(d, "profile.json")
        if os.path.exists(pp):
            with open(pp,"r",encoding="utf-8") as f: out["profile"] = json.load(f)
        return out