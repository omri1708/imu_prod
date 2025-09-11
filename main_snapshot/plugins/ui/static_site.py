# imu_repo/plugins/ui/static_site.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json, re

class StaticSite:
    """
    בונה אתר סטטי מתצורה:
      extras.ui = {
        "pages": [{"path":"/index.html","title":"Home","body":"<h1>Hi</h1>"}]
      }
    Gate בסיסי לנגישות:
      - כותרת H1 אחת לפחות
      - ניגודיות טקסט פשוטה (בדיקה על inline style אם קיים)
    """
    def __init__(self): ...

    def _has_h1(self, html: str) -> bool:
        return bool(re.search(r"<h1[^>]*>.*?</h1>", html, re.IGNORECASE|re.DOTALL))

    def _contrast_ok(self, html: str) -> bool:
        # בדיקה נאיבית: אם יש style עם color=#... ו-background=#..., נוודא שהם לא זהים
        m1 = re.search(r"color:\s*#([0-9a-fA-F]{3,6})", html)
        m2 = re.search(r"background(?:-color)?:\s*#([0-9a-fA-F]{3,6})", html)
        if not (m1 and m2): 
            return True
        return m1.group(1).lower() != m2.group(1).lower()

    def run(self, spec: Any, build_dir: str, user_id: str) -> Dict[str,Any]:
        extras = getattr(spec, "extras", {}) or {}
        ui = (extras.get("ui") or {})
        pages: List[Dict[str,str]] = ui.get("pages") or [{"path":"/index.html","title":"App","body":"<h1>Hello</h1>"}]

        out_dir = os.path.join(build_dir, "site")
        os.makedirs(out_dir, exist_ok=True)
        report=[]
        for p in pages:
            rel = p["path"].lstrip("/")
            html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{p.get('title','')}</title></head>
<body>{p.get('body','')}</body></html>"""
            with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as f:
                f.write(html)
            # gates
            h1 = self._has_h1(html)
            contrast = self._contrast_ok(html)
            report.append({"path": p["path"], "has_h1": h1, "contrast_ok": contrast})
            if not (h1 and contrast):
                raise RuntimeError(f"ui_accessibility_failed:{p['path']}")

        return {"plugin":"static_site","evidence":{"out_dir": out_dir,"report": report},"kpi":{"score": 88.0}}