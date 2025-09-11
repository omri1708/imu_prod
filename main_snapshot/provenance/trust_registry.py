# imu_repo/provenance/trust_registry.py
from __future__ import annotations
import os, json, threading
from typing import Dict, Any, Optional

DEFAULT_PATH = os.environ.get("IMU_TRUST_PATH", "/mnt/data/.imu_trust.json")

_DEFAULTS = {
    # בסיסים לדוגמה; ניתן לשנות עם CLI
    "prefix_trust": {
        "imu://": 0.95,
        "https://": 0.70,
        "http://": 0.50,
    },
    "sources": {
        # דוגמאות: שם מקור מלא → ציון אמון
        "imu://ui/sandbox": 0.95,
        "imu://ui/table": 0.94,
    }
}

class TrustRegistry:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            self._write(_DEFAULTS)
        self._cache = self._read()

    def _read(self) -> Dict[str,Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return dict(_DEFAULTS)

    def _write(self, doc: Dict[str,Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def reload(self) -> None:
        with self._lock:
            self._cache = self._read()

    def set_source_trust(self, source_url: str, trust: float) -> None:
        trust = float(max(0.0, min(1.0, trust)))
        with self._lock:
            doc = self._read()
            doc.setdefault("sources", {})[source_url] = trust
            self._write(doc)
            self._cache = doc

    def set_prefix_trust(self, prefix: str, trust: float) -> None:
        trust = float(max(0.0, min(1.0, trust)))
        with self._lock:
            doc = self._read()
            doc.setdefault("prefix_trust", {})[prefix] = trust
            self._write(doc)
            self._cache = doc

    def trust_for(self, url: str) -> float:
        s = self._cache.get("sources", {})
        if url in s: return float(s[url])
        p = self._cache.get("prefix_trust", {})
        # חפש prefix ארוך תחילה
        candidates = sorted(p.items(), key=lambda kv: len(kv[0]), reverse=True)
        for pref, val in candidates:
            if url.startswith(pref): return float(val)
        return 0.5