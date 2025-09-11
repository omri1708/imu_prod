# engine/llm/cache.py
from __future__ import annotations
import os, json, time, hashlib, hmac, threading, difflib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

# עץ קאש בסגנון Merkle: קבצי JSON לפי prefix/sha, + אינדקס ומטא חתום (אופציונלי)

DEFAULT_ROOT = os.environ.get("IMU_CACHE_ROOT", "var/llm_cache")
DEFAULT_HMAC_KEY = (os.environ.get("IMU_CACHE_KEY") or "imu_default_key").encode("utf-8")

@dataclass
class CacheEntry:
    key: str
    created: float
    ttl_s: int
    model: str
    payload: Dict[str,Any]  # output מה‑LLM + meta

class LLMCache:
    def __init__(self, root: str = DEFAULT_ROOT, hmac_key: bytes = DEFAULT_HMAC_KEY):
        self.root = root; os.makedirs(root, exist_ok=True)
        self.hmac_key = hmac_key
        self._lock = threading.RLock()

    # --- keying ---
    def make_key(self, *, model: str, system_v: str, template_v: str,
                 tools_set: str, user_text_norm: str, ctx_ids: str,
                 persona_v: str, policy_v: str) -> str:
        s = "|".join([model, system_v, template_v, tools_set, user_text_norm, ctx_ids, persona_v, policy_v])
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    # --- paths ---
    def _path(self, key: str) -> str:
        pref = os.path.join(self.root, key[:2], key[2:4])
        os.makedirs(pref, exist_ok=True)
        return os.path.join(pref, f"{key}.json")

    # --- integrity ---
    def _sign(self, blob: bytes) -> str:
        return hmac.new(self.hmac_key, blob, hashlib.sha256).hexdigest()

    # --- get/put ---
    def get(self, key: str) -> Tuple[bool, Optional[CacheEntry]]:
        p = self._path(key)
        if not os.path.exists(p):
            return False, None
        with self._lock:
            try:
                obj = json.loads(open(p,"r",encoding="utf-8").read())
                sig_ok = (obj.get("_sig") == self._sign(obj.get("_blob","{}" ).encode("utf-8")))
                if not sig_ok: return False, None
                if obj["created"] + obj["ttl_s"] < time.time():
                    return False, None
                return True, CacheEntry(key=obj["key"], created=obj["created"], ttl_s=obj["ttl_s"], model=obj["model"], payload=obj["payload"])
            except Exception:
                return False, None

    def put(self, key: str, *, model: str, payload: Dict[str,Any], ttl_s: int = 3600) -> CacheEntry:
        entry = CacheEntry(key=key, created=time.time(), ttl_s=ttl_s, model=model, payload=payload)
        blob = json.dumps({"key": key, "created": entry.created, "ttl_s": entry.ttl_s, "model": model, "payload": payload}, ensure_ascii=False)
        obj = {"_blob": blob, "_sig": self._sign(blob.encode("utf-8")), **json.loads(blob)}
        p = self._path(key)
        with self._lock:
            with open(p, "w", encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False))
        return entry

    # --- near-hit (approximate) ---
    def near_hit(self, *, query: str, model: str, top_k: int = 3, threshold: float = 0.92) -> List[CacheEntry]:
        hits: List[CacheEntry] = []
        # חיפוש נאיבי לפי ratio של difflib על user_text_norm
        for d1 in os.listdir(self.root):
            d1p = os.path.join(self.root, d1)
            if not os.path.isdir(d1p): continue
            for d2 in os.listdir(d1p):
                d2p = os.path.join(d1p, d2)
                for fn in os.listdir(d2p):
                    if not fn.endswith('.json'): continue
                    try:
                        obj = json.loads(open(os.path.join(d2p,fn),"r",encoding="utf-8").read())
                        if obj.get("model") != model: continue
                        src = obj.get("payload",{}).get("_user_text_norm","")
                        if not src: continue
                        ratio = difflib.SequenceMatcher(a=query, b=src).ratio()
                        if ratio >= threshold:
                            hits.append(CacheEntry(key=obj["key"], created=obj["created"], ttl_s=obj["ttl_s"], model=obj["model"], payload=obj["payload"]))
                    except Exception:
                        continue
        hits.sort(key=lambda e: e.created, reverse=True)
        return hits[:top_k]

# singleton
_default: Optional[LLMCache] = None

def default_cache() -> LLMCache:
    global _default
    if _default is None:
        _default = LLMCache()
    return _default