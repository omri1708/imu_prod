# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, os, contextlib
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse

# ה־API-ים שצריך (כבר קיימים אצלך)
from server.routers.consent_api import router as consent_router
from server.routers.adapters_secure import router as adapters_secure_router
from server.routers.respond_api import router as respond_router
from server.routers.program_api import router as program_router
from server.routers.orchestrate_api import router as orchestrate_router
from server.routers.chat_api import router as chat_api_router  # שרת השיחה
from learning.supervisor import LearningSupervisor
from engine.pipeline_events import AUDIT as _ensure_pipeline_events  # noqa: F401
APP = FastAPI(title="IMU Core (Chat)")


# מחברים יכולות
APP.include_router(consent_router)
APP.include_router(adapters_secure_router)
APP.include_router(respond_router)
APP.include_router(program_router)
APP.include_router(chat_api_router)  # /chat/send (API)
APP.include_router(orchestrate_router)

# דף הבית = /chat/
@APP.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/chat/")

@APP.get("/chat", include_in_schema=False)
def chat_noslash():
    return RedirectResponse(url="/chat/")

@APP.get("/chat/", include_in_schema=False)
def chat_index():
    # מגיש ישירות את דף השיחה (קובץ HTML אחד)
    return FileResponse("ui/index.html")

@APP.get("/healthz", response_class=PlainTextResponse, include_in_schema=False)
def healthz():
    return "ok"

# Self-Improving Supervisor (לולאת הלמידה ברקע)
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
