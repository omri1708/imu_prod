# -*- coding: utf-8 -*-
from __future__ import annotations
import os, contextlib, importlib, pkgutil, re, asyncio, tempfile, json
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse, JSONResponse
from engine.pipeline_events import AUDIT as _ensure_pipeline_events  # noqa: F401

# ===== UI (שיחה אחת) =====
APP = FastAPI(title="IMU • Full System")

CHAT_HTML = os.path.join("ui", "index.html")
os.makedirs("ui", exist_ok=True)
if not os.path.exists(CHAT_HTML):
    # fallback דק לשיחה אם אין UI
    with open(CHAT_HTML, "w", encoding="utf-8") as f:
        f.write("""<!doctype html><meta charset="utf-8"><title>IMU Chat</title>
<style>body{background:#0b1020;color:#e6edf3;font-family:system-ui}#log{border:1px solid #233259;border-radius:12px;margin:16px;padding:12px;min-height:300px}input,button{padding:10px;border-radius:8px;border:1px solid #233259;background:#0e1630;color:#e6edf3}button{background:#4ea1ff;color:#001b3f;font-weight:700}</style>
<div id="log"></div><div style="display:flex;gap:8px;margin:16px"><input id="user" value="demo-user" style="max-width:160px"><input id="msg" style="flex:1" placeholder="כתוב חופשי."><button onclick="send()">שלח</button></div>
<script>
const log=document.getElementById('log');const $$=s=>document.querySelector(s);
function line(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=1e9;}
async function send(){const u=$$('#user').value||'user',m=$$('#msg').value||'';if(!m.trim())return;line('me',m);$$('#msg').value='';
  const r=await fetch('/chat/send',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({user_id:u,message:m})}); const j=await r.json().catch(()=>({ok:false,text:'שגיאה'}));
  line('bot', j.text+(j.extra?('\\n\\n'+JSON.stringify(j.extra,null,2)):''));
}
</script>""")

@APP.get("/", include_in_schema=False)
def _root():        return RedirectResponse(url="/chat/")

@APP.get("/chat", include_in_schema=False)
def _chat1():       return RedirectResponse(url="/chat/")

@APP.get("/chat/", include_in_schema=False)
def _chat2():       return FileResponse(CHAT_HTML)

# ===== טעינה אוטומטית של כל ה-routers בכל המערכת =====
_LOADED: List[str] = []

def _include_router(router, name: str):
    global _LOADED
    APP.include_router(router)
    _LOADED.append(name)

def _mount_fastapi_app(subapp, name: str):
    # מיפוי מודול לנתיב יציב (server.http_api -> /server/http_api)
    path = "/" + name.replace(".", "/")
    APP.mount(path, subapp)
    _LOADED.append(name + ":mounted")

def _discover_and_include(package: str, patterns: Tuple[str, ...] = ("router",), also_apps: bool = True):
    """
    סורק חבילה (וכל תתי-החבילות) ומחפש:
    - משתנה בשם 'router' מסוג APIRouter -> include
    - FastAPI בשם APP/app -> mount תחת /<שם־מודול>
    - פונקציה get_router() -> include
    """
    try:
        pkg = importlib.import_module(package)
        if not hasattr(pkg, "__path__"):
            return
    except Exception:
        return
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        # router משתנה
        r = getattr(mod, "router", None)
        if r is not None and getattr(r, "__class__", None).__name__ == "APIRouter":
            _include_router(r, m.name)
            continue
        # get_router()
        if hasattr(mod, "get_router"):
            try:
                r2 = mod.get_router()
                if getattr(r2, "__class__", None).__name__ == "APIRouter":
                    _include_router(r2, m.name + ".get_router")
                    continue
            except Exception:
                pass
        # FastAPI APP/app
        if also_apps:
            for attr in ("APP", "App", "app"):
                a = getattr(mod, attr, None)
                if a is not None and a.__class__.__name__ == "FastAPI":
                    _mount_fastapi_app(a, m.name)
                    break

# 1) שרת השיחה – תחילה כדי להבטיח קיים
try:
    from server.routers.chat_api import router as chat_api_router
    _include_router(chat_api_router, "server.routers.chat_api")
except Exception:
    pass

# 2) כל routers תחת server/routers/*
_discover_and_include("server.routers")

# 3) כל קובץ *_api.py ודומיו תחת server/*
_discover_and_include("server")  # למשל server/provenance_api.py, server/http_api.py וכו'

# 4) חבילת api/ (api/http_api.py, api/stream_http.py …)
_discover_and_include("api")

# 5) אם קיימים APIs דומים בחבילות אחרות – הוסף כאן עוד:
for extra_pkg in ("http", "streaming", "streams"):
    _discover_and_include(extra_pkg)

# ===== בריאות ודיאגנוסטיקה =====
@APP.get("/healthz", include_in_schema=False, response_class=PlainTextResponse)
def _healthz():     return "ok"

@APP.get("/diag/routes", include_in_schema=False)
def _routes():
    return {"count": len(APP.routes),
            "paths": sorted({getattr(r, "path", "") for r in APP.routes})}

@APP.get("/diag/full", include_in_schema=False)
def _diag_full():
    out: Dict[str, Any] = {"loaded": _LOADED, "checks": {}}
    # בדיקות ליבה (לא חובה להצלחה – רק עזרה)
    try:
        from assurance.assurance import AssuranceKernel
        k = AssuranceKernel("./assurance_store_diag")
        out["checks"]["kernel"] = "ok"
    except Exception as e:
        out["checks"]["kernel"] = f"fail:{e}"

    try:
        from executor.policy import Policy
        Policy.load("./executor/policy.yaml")
        out["checks"]["policy"] = "ok"
    except Exception as e:
        out["checks"]["policy"] = f"fail:{e}"

    try:
        from assurance.respond_text import GroundedResponder
        tf = tempfile.NamedTemporaryFile(delete=False); tf.write(b"evidence"); tf.close()
        gr = GroundedResponder("./assurance_store_text_diag")
        r = gr.respond_from_sources("diag", [{"file": tf.name}])
        out["checks"]["grounded"] = "ok" if r.get("ok") else "fail"
        with contextlib.suppress(Exception): os.unlink(tf.name)
    except Exception as e:
        out["checks"]["grounded"] = f"fail:{e}"
    return JSONResponse(out)

# ===== Self-Improving Supervisor =====
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
        task = getattr(APP.state, "learn_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(Exception):
                await task
except Exception:
    # לא מפיל את השרת אם מודול הלמידה לא קיים – פשוט מדלג
    pass
