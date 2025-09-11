# engine/capabilities/registry.py
from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class CapabilityNeed:
    kind: str
    reason: str

class CapabilityRegistry:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f: json.dump({"built":[], "seen":[]}, f)

    def detect_missing(self, spec: Any, ctx: Dict[str,Any]) -> List[CapabilityNeed]:
        needs: List[CapabilityNeed] = []
        if isinstance(spec, dict):
            for k in (spec.get("adapters") or []):
                kind = k.get("kind") if isinstance(k, dict) else str(k)
                if not self._is_built(kind):
                    needs.append(CapabilityNeed(kind=kind, reason="adapter_required"))
        return needs

    def _is_built(self, kind: str) -> bool:
        try:
            d = json.loads(open(self.path,"r",encoding="utf-8").read())
            return any(x.get("kind") == kind for x in d.get("built",[]))
        except Exception:
            return False

    def register_built(self, need: CapabilityNeed, res) -> None:
        d = json.loads(open(self.path,"r",encoding="utf-8").read())
        built = d.get("built",[])
        built.append({"kind": need.kind, "ok": bool(getattr(res,"ok",False))})
        d["built"] = built
        with open(self.path,"w",encoding="utf-8") as f: json.dump(d, f, ensure_ascii=False, indent=2)