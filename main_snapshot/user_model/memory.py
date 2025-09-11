# imu_repo/user_model/memory.py
from __future__ import annotations
import os, json, time, secrets
from typing import Dict, Any, List, Tuple

class UserMemory:
    """
    T0 (session), T1 (episodic), T2 (consolidated). Contradiction resolution via weighted confidence.
    Very lightweight persistence (JSON) + XOR mask "at rest".
    """
    def __init__(self, root: str = ".imu_state/users"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _paths(self, user_id: str):
        udir = os.path.join(self.root, user_id)
        os.makedirs(udir, exist_ok=True)
        return {
            "meta": os.path.join(udir, "meta.json"),
            "t1":   os.path.join(udir, "t1_episodes.jsonl"),
            "t2":   os.path.join(udir, "t2_model.json"),
            "key":  os.path.join(udir, ".xor_key"),
        }

    def _key(self, path: str) -> bytes:
        if not os.path.exists(path):
            with open(path,"wb") as f: f.write(secrets.token_bytes(32))
        with open(path,"rb") as f: return f.read()

    def _xor(self, b: bytes, k: bytes) -> bytes:
        return bytes([b[i] ^ k[i % len(k)] for i in range(len(b))])

    def put_episode(self, user_id: str, kind: str, data: Dict[str,Any], confidence: float = 0.7):
        P = self._paths(user_id)
        key = self._key(P["key"])
        rec = {"ts": time.time(), "kind": kind, "data": data, "confidence": float(confidence)}
        raw = (json.dumps(rec, ensure_ascii=False)+"\n").encode()
        enc = self._xor(raw, key)
        with open(P["t1"], "ab") as f: f.write(enc)

    def consolidate(self, user_id: str, ttl_days: int = 90):
        """
        Move stable preferences to T2 with weighted average; resolve contradictions by confidence and recency.
        """
        P = self._paths(user_id); key=self._key(P["key"])
        # load episodes
        eps=[]
        if os.path.exists(P["t1"]):
            with open(P["t1"],"rb") as f:
                for line in f:
                    try:
                        dec = self._xor(line, key)
                        rec = json.loads(dec.decode())
                        eps.append(rec)
                    except Exception:
                        pass
        # aggregate simple preferences
        prefs={}
        now=time.time()
        half_life = 60*60*24*ttl_days
        for e in eps:
            if e.get("kind")!="preference": continue
            k = e["data"]["key"]; v = e["data"]["value"]; conf=float(e.get("confidence",0.5))
            age = now - float(e.get("ts", now))
            w = conf * pow(0.5, max(0.0, age)/half_life)
            prefs.setdefault(k, {})
            prefs[k][v] = prefs[k].get(v, 0.0) + w
        model={}
        for k,dist in prefs.items():
            best = max(dist, key=lambda vv: dist[vv])
            model[k] = {"value": best, "confidence": dist[best] / (sum(dist.values()) or 1.0), "dist": dist}
        with open(P["t2"],"w",encoding="utf-8") as f:
            json.dump({"ts":now,"prefs":model}, f, ensure_ascii=False, indent=2)

    def read_profile(self, user_id: str) -> Dict[str,Any]:
        P = self._paths(user_id)
        if not os.path.exists(P["t2"]): return {"prefs": {}}
        with open(P["t2"],"r",encoding="utf-8") as f: return json.load(f)