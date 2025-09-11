# imu_repo/ui/static_pack.py
from __future__ import annotations
import os

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>IMU UI</title>
<meta name="description" content="IMU generated static UI">
<style>body{font-family:sans-serif;margin:2rem}code{background:#f6f6f6;padding:.2rem .4rem}</style>
</head><body>
<h1>IMU App</h1>
<p>Static UI packaged by <code>ui/static_pack.py</code>.</p>
<div>
  <label>Key <input id="k" placeholder="key" aria-label="key"/></label>
  <label>Value <input id="v" placeholder="value" aria-label="value"/></label>
  <button onclick="postKV()" aria-label="Post KV">POST /kv</button>
</div>
<pre id="out" aria-live="polite"></pre>
<script src="/app.js"></script>
</body></html>"""

JS = r"""async function postKV(){
  const k = document.getElementById('k').value;
  const v = document.getElementById('v').value;
  const res = await fetch('/kv', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({k,v})});
  const txt = await res.text();
  document.getElementById('out').textContent = txt;
}"""

def write_basic_ui(dst_root: str) -> str:
    ui_dir = os.path.join(dst_root, "ui")
    os.makedirs(ui_dir, exist_ok=True)
    index = os.path.join(ui_dir, "index.html")
    appjs = os.path.join(ui_dir, "app.js")
    with open(index,"w",encoding="utf-8") as f: f.write(HTML)
    with open(appjs,"w",encoding="utf-8") as f: f.write(JS)
    return index