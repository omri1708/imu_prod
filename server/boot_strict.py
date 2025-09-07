# -*- coding: utf-8 -*-
from __future__ import annotations
import os, contextlib, importlib, pkgutil, asyncio, tempfile, json
from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse, JSONResponse

APP = FastAPI(title="IMU • Full System (STRICT)")

# ---------- UI: שיחה אחת ----------
CHAT_HTML = os.path.join("ui", "index.html")
os.makedirs("ui", exist_ok=True)
if not os.path.exists(CHAT_HTML):
    with open(CHAT_HTML, "w", encoding="utf-8") as f:
        f.write("""<!doctype html><meta charset="utf-8"><title>IMU Chat</title>
<style>body{background:#0b1020;color:#e6edf3;font-family:system-ui}#log{border:1px solid #233259;border-radius:12px;margin:16px;padding:12px;min-height:300px}input,button{padding:10px;border-radius:8px;border:1px solid #233259;background:#0e1630;color:#e6edf3}button{background:#4ea1ff;color:#001b3f;font-weight:700}</style>
<div id="log"></div><div style="display:flex;gap:8px;margin:16px"><input id="user" value="demo-user" style="max-width:160px"><input id="msg" style="flex:1" placeholder="כתוב חופשי."><button onclick="send()">שלח</button></div>
<script>
const log=document.getElementById('log');const $$=s=>document.querySelector(s);
function line(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=1e9}
async function send(){const u=$$('#user').value||'user',m=$$('#msg').value||'';if(!m.trim())return;line('me',m);$$('#msg').value='';
  const r=await fetch('/chat/send',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({user_id:u,message:m})});
  const j=await r.json().catch(()=>({ok:false,text:'שגיאה'})); line('bot', j.text+(j.extra?('\\n\\n'+JSON.stringify(j.extra,null,2)):''));
}
</script>""")

@APP.get("/", include_in_schema=False)
def root():  return RedirectResponse(url="/chat/")
@APP.get("/chat", include_in_schema=False)
def chat1(): return RedirectResponse(url="/chat/")
@APP.get("/chat/", include_in_schema=False)
def chat2(): return FileResponse(CHAT_HTML)

# ---------- Auto-Discovery: חיבור כל ה־APIs בפרויקט ----------
_LOADED: List[str] = []

def _discover_and_include(package: str):
    try:
        pkg = importlib.import_module(package)
        if not hasattr(pkg, "__path__"): return
    except Exception:
        return
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        r = getattr(mod, "router", None)
        if r is not None and getattr(r, "__class__", None).__name__ == "APIRouter":
            APP.include_router(r); _LOADED.append(m.name); continue
        if hasattr(mod, "get_router"):
            try:
                r2 = mod.get_router()
                if getattr(r2, "__class__", None).__name__ == "APIRouter":
                    APP.include_router(r2); _LOADED.append(m.name + ".get_router"); continue
            except Exception:
                pass
        for attr in ("APP","app","App"):
            sub = getattr(mod, attr, None)
            if sub is not None and sub.__class__.__name__ == "FastAPI":
                APP.mount("/" + m.name.replace(".", "/"), sub)
                _LOADED.append(m.name + ":mounted")
                break

# קודם: שרת השיחה (חובה)
from server.routers.chat_api import router as chat_api_router
APP.include_router(chat_api_router); _LOADED.append("server.routers.chat_api")

# כל החבילות בעץ שלך:
for pkg in ("server.routers","server","api","http","stream","streams","streaming","realtime"):
    _discover_and_include(pkg)

# ---------- בריאות / דיאגנוסטיקה ----------
@APP.get("/healthz", include_in_schema=False, response_class=PlainTextResponse)
def healthz(): return "ok"

@APP.get("/diag/routes", include_in_schema=False)
def routes():
    return {"count": len(APP.routes),
            "paths": sorted({getattr(r,"path","") for r in APP.routes}),
            "loaded": _LOADED}

@APP.get("/diag/full", include_in_schema=False)
def diag_full():
    out: Dict[str, Any] = {"loaded": _LOADED, "checks": {}}
    try:
        from assurance.assurance import AssuranceKernel
        AssuranceKernel("./assurance_store_diag")
        out["checks"]["kernel"]="ok"
    except Exception as e:
        out["checks"]["kernel"]=f"fail:{e}"
    try:
        from executor.policy import Policy
        Policy.load("./executor/policy.yaml")
        out["checks"]["policy"]="ok"
    except Exception as e:
        out["checks"]["policy"]=f"fail:{e}"
    try:
        from assurance.respond_text import GroundedResponder
        tf=tempfile.NamedTemporaryFile(delete=False); tf.write(b"evidence"); tf.close()
        gr=GroundedResponder("./assurance_store_text_diag")
        r=gr.respond_from_sources("diag",[{"file":tf.name}])
        out["checks"]["grounded"]="ok" if r.get("ok") else "fail"
        with contextlib.suppress(Exception): os.unlink(tf.name)
    except Exception as e:
        out["checks"]["grounded"]=f"fail:{e}"
    return JSONResponse(out)

# ---------- STRICT: חייבים נתיבי ליבה ----------
REQUIRED = {
    "POST:/chat/send",
    "POST:/adapters/secure/run",
    "POST:/program/build",
    "POST:/respond/grounded",
    "POST:/consent/grant",
}
def _sig(rt) -> str:
    m = next(iter(getattr(rt,"methods",{"GET"})))
    return f"{m}:{getattr(rt,'path','')}"
present = { _sig(r) for r in APP.routes }
missing = sorted(REQUIRED - present)
if missing:
    raise RuntimeError("STRICT COVERAGE FAILED – missing: " + ", ".join(missing))

# ---------- Self-Improving (אם קיים) ----------
try:
    from learning.supervisor import LearningSupervisor
    sup = LearningSupervisor(
        policy_path="./executor/policy.yaml",
        adapters_root="./adapters/generated",
        audit_roots=[
            "./assurance_store", "./assurance_store_text",
            "./assurance_store_programs", "./assurance_store_adapters"
        ],
        rr_log_path="./logs/resource_required.jsonl"
    )
    @APP.on_event("startup")
    async def _startup():
        os.makedirs("./logs", exist_ok=True)
        APP.state.learn_task = asyncio.create_task(sup.run_forever())
    @APP.on_event("shutdown")
    async def _shutdown():
        t = getattr(APP.state, "learn_task", None)
        if t:
            t.cancel()
            with contextlib.suppress(Exception):
                await t
except Exception:
    pass