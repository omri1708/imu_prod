# imu_repo/ui/toolkits_bridge.py
from __future__ import annotations
import http.server, socketserver, threading, os, time
from typing import Dict, Any, Optional

DEFAULT_UI_DIR = "/mnt/data/imu_repo/ui_static"

INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><meta charset="UTF-8"><title>IMU UI</title>
<body>
  <h1>IMU – Real-Time Console</h1>
  <div>Status: <span id="st">connecting…</span></div>
  <textarea id="log" cols="100" rows="16" readonly></textarea><br/>
  <input id="inp" placeholder="type and press Enter"/>
<script>
const st=document.getElementById('st'); const log=document.getElementById('log'); const inp=document.getElementById('inp');
const url = (location.protocol==='https:'?'wss':'ws') + '://' + location.host.replace(/:\\d+$/,':8976') + '/chat';
let ws = new WebSocket(url);
ws.onopen = ()=>{ st.textContent='open'; log.value+='[open]\\n'; };
ws.onmessage = (ev)=>{ log.value += '[recv] '+ ev.data + '\\n'; log.scrollTop=log.scrollHeight; };
ws.onclose = ()=>{ st.textContent='closed'; log.value+='[closed]\\n'; };
inp.addEventListener('keydown', (e)=>{
  if (e.key==='Enter' && ws.readyState===1) { ws.send(inp.value); log.value+='[send] '+inp.value+'\\n'; inp.value=''; }
});
</script>
</body></html>
"""

def ensure_static_ui(dirpath: str=DEFAULT_UI_DIR) -> str:
    os.makedirs(dirpath, exist_ok=True)
    index = os.path.join(dirpath, "index.html")
    if not os.path.exists(index):
        open(index, "w", encoding="utf-8").write(INDEX_HTML)
    return dirpath

class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k): pass

def serve_static_ui(host: str="127.0.0.1", port: int=8975, dirpath: str=DEFAULT_UI_DIR) -> threading.Thread:
    dirpath = ensure_static_ui(dirpath)
    os.chdir(dirpath)
    httpd = socketserver.TCPServer((host, port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t

def console_render(msg: str) -> None:
    print(f"[UI] {msg}")