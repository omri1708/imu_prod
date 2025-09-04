# imu_repo/packaging/html_bundle.py
from __future__ import annotations
import os, shutil, json, http.server, socketserver, threading, time
from typing import Dict, Any
from ui.toolkits_bridge import ensure_static_ui

DIST = "/mnt/data/imu_repo/dist/html_bundle"

def build_html_bundle(extra_assets: Dict[str,str] | None=None) -> str:
    """
    יוצר חבילת UI סטטית תחת dist/html_bundle (index.html + נכסים),
    מוסיף manifest.json עם חותמת זמן וגרסה.
    extra_assets: מיפוי {שם-קובץ: תוכן}
    """
    if os.path.exists(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST, exist_ok=True)
    src = ensure_static_ui()
    # העתק את הסטטי
    for fn in os.listdir(src):
        sp = os.path.join(src, fn)
        dp = os.path.join(DIST, fn)
        if os.path.isfile(sp):
            shutil.copyfile(sp, dp)
    # הוסף נכסים נוספים לפי בקשה
    if extra_assets:
        for name, content in extra_assets.items():
            open(os.path.join(DIST, name), "w", encoding="utf-8").write(content)
    # כתוב מניפסט
    manifest = {
        "name": "IMU HTML Bundle",
        "version": "63.0",
        "built_at": int(time.time()),
        "files": sorted(os.listdir(DIST)),
    }
    open(os.path.join(DIST, "manifest.json"), "w", encoding="utf-8").write(json.dumps(manifest, ensure_ascii=False, indent=2))
    return DIST

class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k): pass

def serve_html_bundle(host: str="127.0.0.1", port: int=8999) -> threading.Thread:
    """
    מרים שרת קבצים סטטי על תיקיית ה-bundle.
    """
    os.chdir(DIST)
    httpd = socketserver.TCPServer((host, port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t