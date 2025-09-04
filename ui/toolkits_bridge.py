# PATH: imu_repo/ui/unified_static.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import http.server
import socketserver
import threading
import os
import shutil
import textwrap

# בסיס יחיד לכל ה־assets הסטטיים של ה־UI
UI_STATIC_ROOT = "/mnt/data/imu_repo/ui_static"
PROOFS_SRC_JS = "/mnt/data/imu_repo/ui/proofs_view.js"  # אם קיים – נעתיק; אחרת ניצור סטאב


# --------------------------- HTML TEMPLATES --------------------------- #
INDEX_HTML = textwrap.dedent("""
<!doctype html>
<html lang="en">
  <meta charset="utf-8" />
  <title>IMU UI</title>
  <style>
    :root { --fg:#222; --muted:#666; --border:#ddd; --bg:#fff; --link:#0a66c2; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: var(--fg); background: var(--bg); }
    h1 { margin: 0 0 6px 0; }
    p { color: var(--muted); margin: 0 0 18px 0; }
    ul { margin: 12px 0 0 18px; }
    a { color: var(--link); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card { border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin: 14px 0; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  </style>
  <body>
    <h1>IMU – Unified UI</h1>
    <p>בחרו כלי:</p>

    <div class="row">
      <div class="card">
        <h2>Console</h2>
        <p>קונסול בזמן אמת (WebSocket).</p>
        <a href="./console/" rel="noopener">פתח Console →</a>
      </div>
      <div class="card">
        <h2>Proofs</h2>
        <p>שליחת הודעה וקבלת תשובה חתומה עם שרשרת הוכחות.</p>
        <a href="./proofs/" rel="noopener">פתח Proofs →</a>
      </div>
    </div>
  </body>
</html>
""")

CONSOLE_HTML = textwrap.dedent("""
<!DOCTYPE html>
<html lang="en">
  <meta charset="UTF-8" />
  <title>IMU – Real-Time Console</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
    #log { width: 100%; height: 360px; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }
    #st { font-weight: 600; }
    input { width: 100%; padding: 8px; margin-top: 8px; }
  </style>
  <body>
    <h1>IMU – Real-Time Console</h1>
    <div>סטטוס: <span id="st">connecting…</span></div>
    <textarea id="log" readonly></textarea>
    <input id="inp" placeholder="type and press Enter" />
    <script>
      const st = document.getElementById('st');
      const log = document.getElementById('log');
      const inp = document.getElementById('inp');
      const host = location.host.replace(/:\\d+$/, ':8976');
      const url  = (location.protocol==='https:'?'wss':'ws') + '://' + host + '/chat';
      let ws = new WebSocket(url);
      ws.onopen    = () => { st.textContent = 'open';  log.value += '[open]\n'; };
      ws.onmessage = (ev) => { log.value += '[recv] ' + ev.data + '\n'; log.scrollTop = log.scrollHeight; };
      ws.onclose   = () => { st.textContent = 'closed'; log.value += '[closed]\n'; };
      inp.addEventListener('keydown', (e)=>{
        if (e.key==='Enter' && ws.readyState===1) {
          ws.send(inp.value);
          log.value += '[send] ' + inp.value + '\n';
          inp.value='';
        }
      });
    </script>
  </body>
</html>
""")

PROOFS_HTML = textwrap.dedent("""
<!doctype html>
<html lang="en">
  <meta charset="utf-8" />
  <title>IMU Realtime Proofs</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
    #out { white-space: pre; border: 1px solid #ddd; padding: 10px; height: 300px; overflow: auto; border-radius: 8px; }
    input { padding: 6px 8px; }
    button { padding: 6px 10px; }
  </style>
  <body>
    <h1>IMU Realtime Proofs</h1>
    <p>שלח הודעה וקבל תשובה חתומה עם שרשרת הוכחות.</p>
    <p>
      <input id="msg" value="hello evidence"/>
      <button id="send">send</button>
      <a href=".." style="margin-left:12px">← Back</a>
    </p>
    <div id="out"></div>
    <script src="proofs_view.js" defer></script>
  </body>
</html>
""")


# --------------------------- FILE WRITERS --------------------------- #
def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ensure_console_ui(root: str = UI_STATIC_ROOT) -> str:
    """יוצר/מעדכן את דפי ה־Console תחת {root}/console/"""
    console_dir = os.path.join(root, "console")
    os.makedirs(console_dir, exist_ok=True)
    _write(os.path.join(console_dir, "index.html"), CONSOLE_HTML)
    return console_dir


def ensure_proofs_ui(root: str = UI_STATIC_ROOT) -> str:
    """יוצר/מעדכן את דפי ה־Proofs תחת {root}/proofs/ ומעתיק proofs_view.js אם קיים."""
    proofs_dir = os.path.join(root, "proofs")
    os.makedirs(proofs_dir, exist_ok=True)
    _write(os.path.join(proofs_dir, "index.html"), PROOFS_HTML)
    dst_js = os.path.join(proofs_dir, "proofs_view.js")
    if os.path.exists(PROOFS_SRC_JS):
        shutil.copy2(PROOFS_SRC_JS, dst_js)
    elif not os.path.exists(dst_js):
        _write(dst_js, "console.warn('proofs_view.js source not found; using stub');")
    return proofs_dir


def ensure_unified_ui(root: str = UI_STATIC_ROOT) -> str:
    """יוצר אינדקס מאוחד וקישורים לשני הכלים."""
    os.makedirs(root, exist_ok=True)
    ensure_console_ui(root)
    ensure_proofs_ui(root)
    _write(os.path.join(root, "index.html"), INDEX_HTML)
    return root


# --------------------------- STATIC SERVER --------------------------- #
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs) -> None:  # שקט בלוגים
        pass


def serve_unified_ui(host: str = "127.0.0.1", port: int = 8975,
                     root: str = UI_STATIC_ROOT) -> Tuple[threading.Thread, socketserver.BaseServer]:
    """מרים שרת סטטי שמשרת את כל התיקייה המאוחדת (index + תתי־כלים).
    מחזיר (thread, httpd) כדי שתוכלו לכבות נקי: httpd.shutdown(); thread.join().
    """
    root = ensure_unified_ui(root)

    # העדפה: להשתמש בפרמטר 'directory' של ה־Handler (ללא chdir)
    try:
        from functools import partial
        Handler = partial(http.server.SimpleHTTPRequestHandler, directory=root)
    except TypeError:
        # תאימות לאחור: נשתמש ב־chdir אם אין תמיכה בפרמטר 'directory'
        os.chdir(root)
        Handler = _QuietHandler

    class ThreadingTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    httpd = ThreadingTCPServer((host, port), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t, httpd


# --------------------------- CLI --------------------------- #
if __name__ == "__main__":
    thread, server = serve_unified_ui()
    url = f"http://127.0.0.1:8975/"
    print(f"[UI] Unified static server running at: {url}")
    print("[UI] Press Ctrl+C to stop…")
    try:
        while True:
            # שמירה על תהליך חי
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[UI] Shutting down…")
        server.shutdown()
        thread.join(timeout=2.0)
        print("[UI] Bye")
