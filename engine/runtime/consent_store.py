# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, threading
from pathlib import Path
from typing import Dict, Any, List

class ConsentStore:
    def __init__(self, path: str = ".imu/consents.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text(json.dumps({}, ensure_ascii=False, indent=2), "utf-8")

    def _load(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text("utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def _user(self, uid: str) -> Dict[str, Any]:
        db = self._load()
        return db.setdefault(uid, {"tools": [], "secrets": {}, "flags": {}})

    def has_tool(self, uid: str, tool: str) -> bool:
        u = self._user(uid); return tool in u["tools"]

    def grant_tools(self, uid: str, tools: List[str]) -> None:
        db = self._load(); u = db.setdefault(uid, {"tools": [], "secrets": {}, "flags": {}})
        for t in tools:
            if t not in u["tools"]: u["tools"].append(t)
        self._save(db)

    def has_secret(self, uid: str, name: str) -> bool:
        u = self._user(uid); return bool(u["secrets"].get(name))

    def put_secret(self, uid: str, name: str, value: str) -> None:
        db = self._load(); u = db.setdefault(uid, {"tools": [], "secrets": {}, "flags": {}})
        u["secrets"][name] = value
        self._save(db)

    def get_secret(self, uid: str, name: str, default: str = "") -> str:
        u = self._user(uid); return str(u["secrets"].get(name) or default)

    def set_flag(self, uid: str, key: str, value: bool) -> None:
        db = self._load(); u = db.setdefault(uid, {"tools": [], "secrets": {}, "flags": {}})
        u["flags"][key] = bool(value); self._save(db)

    def get_flag(self, uid: str, key: str) -> bool:
        u = self._user(uid); return bool(u["flags"].get(key, False))
