# PATH: engine/blueprints/auto_writer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import re, os, textwrap
from pathlib import Path

BLUEPRINTS_ROOT = Path(__file__).parent

TEMPLATES: Dict[str, Dict[str, str]] = {
    "ui.chat_console": {
        "ui/chat.html": """<!doctype html><html lang="he" dir="rtl"><head>
<meta charset="utf-8"><title>IMU Chat Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root { --bg:#0b0f14; --panel:#11161d; --text:#e8eef6; --muted:#9db0c3; --acc:#4cc3ff; }
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Helvetica,Arial}
header{padding:12px 16px;background:var(--panel);border-bottom:1px solid #18202a;display:flex;gap:12px;align-items:center}
header input,header select{background:#0f141b;border:1px solid #1a2330;color:var(--text);padding:8px;border-radius:8px}
header button{background:var(--acc);border:0;color:#001422;padding:10px 14px;border-radius:10px;font-weight:600;cursor:pointer}
#main{display:grid;grid-template-columns:280px 1fr;min-height:calc(100dvh - 56px)}
#side{border-left:1px solid #18202a;background:#0f141b;padding:12px;overflow:auto}
#side h3{margin:8px 0 12px 0;font-size:14px;color:var(--muted)}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.kv label{font-size:12px;color:var(--muted)}
.kv input,.kv select{width:100%;background:#0b1016;border:1px solid #1a2330;border-radius:8px;padding:6px;color:var(--text)}
#chat{display:flex;flex-direction:column;height:100%}
#log{flex:1;overflow:auto;padding:16px}
.msg{max-width:900px;margin:0 0 14px 0;padding:12px 14px;border-radius:12px;line-height:1.45;white-space:pre-wrap}
.user{background:#132133;border:1px solid #1f2c3a}.bot{background:#0f141b;border:1px solid #1a2330}
#composer{display:flex;gap:8px;padding:12px;border-top:1px solid #18202a;background:#0f141b}
#composer textarea{flex:1;min-height:56px;max-height:30dvh;resize:vertical;background:#0b1016;border:1px solid #1a2330;border-radius:12px;padding:10px;color:var(--text)}
#composer button{background:var(--acc);border:0;color:#001422;padding:12px 16px;border-radius:12px;font-weight:700;cursor:pointer}
.meta{margin-top:6px;font-size:12px;color:var(--muted)} code{background:#0b1016;border:1px solid #1a2330;border-radius:6px;padding:2px 4px}
</style></head><body>
<header><strong>IMU Chat</strong>
<label>API Base:<input id="apiBase" value="http://localhost:8000" style="width:240px"></label>
<label>User:<input id="userId" value="u1" style="width:100px"></label>
<label>Mode:<select id="mode"><option value="live" selected>live</option><option value="poc">poc</option><option value="prod">prod</option></select></label>
<button id="resetBtn">אפס שיחה</button></header>
<div id="main"><aside id="side"><h3>אפשרויות</h3><div class="kv">
<label><input type="checkbox" id="proceed" checked> proceed</label>
<label><input type="checkbox" id="autobuild" checked> autobuild</label>
<label><input type="checkbox" id="persist" checked> persist ל-out/</label>
<label><input type="checkbox" id="emit_ci"> emit_ci</label>
<label><input type="checkbox" id="emit_iac"> emit_iac</label>
<label><input type="checkbox" id="autodeploy"> deploy_script</label>
</div><h3>מידע</h3><div id="lastMeta" class="meta">—</div></aside>
<section id="chat"><div id="log"></div><div id="composer">
<textarea id="prompt" placeholder="כתוב כאן בשפה רגילה. לדוגמה: בני CRUD ל-Users ושמרי קוד ל-out/."></textarea>
<button id="sendBtn">שלח</button></div></section></div>
<script>
const $=s=>document.querySelector(s);const apiBase=$('#apiBase'),userId=$('#userId'),mode=$('#mode');
const proceed=$('#proceed'),autobuild=$('#autobuild'),persist=$('#persist'),emit_ci=$('#emit_ci'),emit_iac=$('#emit_iac'),autodeploy=$('#autodeploy');
const log=$('#log'),promptEl=$('#prompt');
function addMsg(t,who="bot",meta=""){const d=document.createElement('div');d.className="msg "+(who==="user"?"user":"bot");d.textContent=t;log.appendChild(d);if(meta){const m=document.createElement('div');m.className="meta";m.innerHTML=meta;log.appendChild(m)}log.scrollTop=log.scrollHeight;}
async function send(){const msg=promptEl.value.trim();if(!msg)return;addMsg(msg,"user");promptEl.value="";
const body={user_id:userId.value||"u1",message:msg,mode:mode.value||"live",proceed:proceed.checked,autobuild:autobuild.checked};
if(persist.checked)body.persist="./out";if(emit_ci.checked)body.emit_ci=true;if(emit_iac.checked)body.emit_iac=true;if(autodeploy.checked)body.autodeploy=true;
const t0=performance.now();try{const r=await fetch(apiBase.value.replace(/\/$/,"")+"/chat/send",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const j=await r.json();const t1=performance.now();
const meta=[];meta.push(`⏱ ${(t1-t0).toFixed(0)}ms`);if(j.extra?.build){const b=j.extra.build;meta.push(`build: ${b.ok?"OK":"FAIL"} | compile=${b.compile_rc} test=${b.test_rc}`)}
if(j.extra?.files_written?.length){const files=j.extra.files_written.map(x=>`<code>${x}</code>`).join("<br>");meta.push(`<div>files_written:<br>${files}</div>`)}
$('#lastMeta').innerHTML=meta.join(" · ")||"—";addMsg(j.text||"(no text)","bot",meta.join(" · "));}catch(e){addMsg("שגיאת רשת/‏API: "+e,"bot");}}
$('#sendBtn').onclick=send;promptEl.addEventListener('keydown',e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});$('#resetBtn').onclick=async()=>{try{await fetch(apiBase.value.replace(/\/$/,"")+"/chat/reset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({user_id:userId.value||"u1"})});addMsg("ההקשר אופס.","bot");$('#lastMeta').textContent="—"}catch(e){addMsg("שגיאה באיפוס: "+e,"bot")}};
</script></body></html>""",
        "docker/ui/nginx.conf": """server {
  listen 80; server_name _; root /usr/share/nginx/html;
  location / { try_files $uri /chat.html; }
  location /chat/ {
    proxy_pass http://api:8000;
    proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr;
    proxy_http_version 1.1; proxy_set_header Connection "";
  }
}""",
        "docker/ui/Dockerfile": """FROM nginx:alpine
WORKDIR /usr/share/nginx/html
COPY ui/chat.html ./chat.html
COPY docker/ui/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80"""
    }
}

def synthesize_blueprint(name: str, spec: Dict[str, Any]) -> Path:
    """
    Creates engine/blueprints/<safe>.py with generate() that returns {path: bytes}.
    Currently supports 'ui.chat_console' out-of-the-box; can be extended.
    """
    safe = re.sub(r"[^A-Za-z0-9_\.]", "_", name).strip(".")
    target = BLUEPRINTS_ROOT / f"{safe.replace('.','_')}.py"
    if target.exists():
        return target

    if name not in TEMPLATES:
        # default: minimal skeleton returning empty mapping
        code = "def generate(spec):\n    return {}\n"
    else:
        mapping = TEMPLATES[name]
        pairs = []
        for path, content in mapping.items():
            pairs.append(f'    "{path}": """{content}""".encode("utf-8"),')
        code = "def generate(spec):\n    return {\n" + "\n".join(pairs) + "\n}\n"

    target.write_text(code, encoding="utf-8")
    return target
