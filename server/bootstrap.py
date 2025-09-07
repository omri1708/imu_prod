# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, os, contextlib
from fastapi import FastAPI
from server.routers.consent_api import router as consent_router
from server.routers.adapters_secure import router as adapters_secure_router
from server.routers.respond_api import router as respond_router
from server.routers.program_api import router as program_router
from learning.supervisor import LearningSupervisor

APP = FastAPI(title="IMU Core")

# Routers = כל היכולות
APP.include_router(consent_router)         # /consent/grant
APP.include_router(adapters_secure_router) # /adapters/secure/run
APP.include_router(respond_router)         # /respond/grounded
APP.include_router(program_router)         # /program/build

# Self-Improving Supervisor (לולאת הלמידה)
sup = LearningSupervisor(
    policy_path="./executor/policy.yaml",
    adapters_root="./adapters/generated",
    audit_roots=[
        "./assurance_store", "./assurance_store_text", "./assurance_store_programs", "./assurance_store_adapters"
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
