# imu_repo/ui/dsl_runtime_rt.py
from __future__ import annotations
from typing import Any, Dict, List, Callable, Optional
import json, hashlib

class GroundingViolation(Exception):
    pass

def _distinct_sources(evidence: List[Dict[str, Any]]) -> int:
    seen = set()
    for ev in evidence:
        src = ev.get("source") or ev.get("url") or ev.get("sha256") or ev.get("kind")
        if src: seen.add(str(src))
    return len(seen)

def _verify_grounded_bundle(bundle: Dict[str, Any], *, min_sources: int = 1, min_trust: float = 1.0) -> Dict[str, Any]:
    if "text" not in bundle or "claims" not in bundle:
        raise GroundingViolation("bundle_missing_fields")
    if not isinstance(bundle["text"], str): raise GroundingViolation("text_not_string")
    claims = bundle["claims"]
    if not isinstance(claims, list) or not claims: raise GroundingViolation("empty_claims")
    trust = 0.0
    for i, c in enumerate(claims):
        if not isinstance(c, dict): raise GroundingViolation(f"claim_{i}_not_object")
        if "type" not in c or "text" not in c: raise GroundingViolation(f"claim_{i}_missing_core")
        ev = c.get("evidence", [])
        if not isinstance(ev, list) or not ev: raise GroundingViolation(f"claim_{i}_no_evidence")
        ds = _distinct_sources(ev)
        if ds < min_sources: raise GroundingViolation(f"claim_{i}_insufficient_sources")
        # ניקוד פשוט: מקור ייחודי=1, sha256=+0.5, https=+0.25
        score = ds
        for e in ev:
            if "sha256" in e: score += 0.5
            url = e.get("url") or ""
            if isinstance(url, str) and url.startswith("https://"): score += 0.25
        trust += score
    if trust < min_trust: raise GroundingViolation("low_total_trust")
    return {"trust": trust, "claims": claims}

# רישום ווידג’טים וצנרת עדכונים
class Widget:
    def apply(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

class TableWidget(Widget):
    """
    טבלה עם סינון/מיון בצד הלקוח (מושתת על שלב 94–95). כאן נוסיף update() מסטרים.
    payload צפוי: {"rows":[{...}], "schema":{"columns":[...]}, "ops": [{"op":"upsert","key":...,"row":{...}}, ...]}
    """
    def __init__(self, *, key_field: str):
        self.key_field = key_field
        self.rows: Dict[Any, Dict[str, Any]] = {}
        self.sort_key: Optional[str] = None
        self.sort_reverse: bool = False
        self.filters: Dict[str, Callable[[Any], bool]] = {}

    def set_sort(self, col: str, reverse: bool = False):
        self.sort_key, self.sort_reverse = col, reverse

    def set_filter(self, col: str, fn: Callable[[Any], bool]):
        self.filters[col] = fn

    def _filtered_sorted(self) -> List[Dict[str, Any]]:
        vals = list(self.rows.values())
        for col, fn in self.filters.items():
            vals = [r for r in vals if fn(r.get(col))]
        if self.sort_key:
            vals.sort(key=lambda r: r.get(self.sort_key), reverse=self.sort_reverse)
        return vals

    def apply(self, payload: Dict[str, Any]) -> None:
        # תמיכה ב-upsert/batch
        ops = payload.get("ops")
        rows = payload.get("rows")
        if rows:
            for r in rows:
                k = r.get(self.key_field)
                if k is None: continue
                self.rows[k] = r
        if ops:
            for op in ops:
                if op.get("op") == "upsert":
                    r = op.get("row", {})
                    k = r.get(self.key_field)
                    if k is None: continue
                    self.rows[k] = {**self.rows.get(k, {}), **r}
                elif op.get("op") == "delete":
                    k = op.get("key")
                    if k in self.rows: del self.rows[k]

    def to_list(self) -> List[Dict[str, Any]]:
        return self._filtered_sorted()

class GridWidget(Widget):
    """
    Grid מתקדם (areas/nested) – כאן מנהל רק מודל מצב (לא CSS בפועל),
    ומתממשק למנוע הרנדרינג של שלב 95+ (קיים בצד הדפדפן).
    payload: {"areas":[{"name":"header","x":0,"y":0,"w":12,"h":1},...], "widgets":[...]}
    """
    def __init__(self):
        self.areas: List[Dict[str, Any]] = []
        self.widgets: List[Dict[str, Any]] = []

    def apply(self, payload: Dict[str, Any]) -> None:
        if "areas" in payload: self.areas = payload["areas"]
        if "widgets" in payload: self.widgets = payload["widgets"]

class UISession:
    def __init__(self, *, min_sources=1, min_trust=1.0):
        self._widgets: Dict[str, Widget] = {}
        self._min_sources = min_sources
        self._min_trust = min_trust

    def register(self, name: str, widget: Widget):
        self._widgets[name] = widget

    def handle_stream_message(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """
        envelope = {"op": "...", "bundle": {"text": "...", "claims":[...], "meta": {...}, "ui": {...}}}
        """
        op = envelope.get("op")
        bundle = envelope.get("bundle", {})
        # אכיפת Grounded-Strict בצד קליינט
        _verify_grounded_bundle(bundle, min_sources=self._min_sources, min_trust=self._min_trust)
        ui = bundle.get("ui", {})
        for target, payload in ui.items():
            w = self._widgets.get(target)
            if w: w.apply(payload)
        return {"ok": True}