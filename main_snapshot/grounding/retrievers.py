# -*- coding: utf-8 -*-
from __future__ import annotations
import requests, time, json, os
from typing import Dict, Any, List, Tuple
from assurance.cas import CAS
from assurance.errors import ResourceRequired

class Retriever:
    def __init__(self, cas: CAS): self.cas = cas

class HTTPRetriever(Retriever):
    def fetch_json(self, url: str, max_age_seconds: int = 3600, trust: float = 0.6) -> Tuple[str, Dict[str,Any], float, float]:
        try:
            r = requests.get(url, timeout=10); r.raise_for_status()
        except Exception:
            raise ResourceRequired("tool:requests", "pip install requests")
        data = r.json()
        digest = self.cas.put_bytes(json.dumps(data, sort_keys=True).encode("utf-8"),
                                    meta={"provider":"http","url":url,"headers":dict(r.headers)})
        ts = time.time(); ttl = ts + max_age_seconds
        return digest, data, ts, ttl

class LocalGraph(Retriever):
    """גרף ידע לוקאלי (קבצי JSON בנתיב)."""
    def __init__(self, cas: CAS, store_path: str = "./grounding_graph.json"):
        super().__init__(cas); self.path = store_path

    def lookup(self, key: str) -> Tuple[str, Any, float, float]:
        if not os.path.exists(self.path):
            raise ResourceRequired("graph:missing", "populate grounding_graph.json")
        data = json.loads(open(self.path,"r",encoding="utf-8").read())
        if key not in data:
            raise ResourceRequired(f"graph:key:{key}", "extend grounding_graph.json")
        value = data[key]
        digest = self.cas.put_bytes(json.dumps(value).encode("utf-8"),
                                    meta={"provider":"graph","key":key})
        ts = time.time(); ttl = ts + 24*3600
        return digest, value, ts, ttl
