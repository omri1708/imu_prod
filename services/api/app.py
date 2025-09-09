# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import logging
from typing import Dict, Any, Type
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import PlainTextResponse

from sqlmodel import SQLModel, Session, select
from services.api.db import init_engine, get_session, try_connect_db, run_alembic_upgrade_if_needed
from .models import build_models_from_entities_spec, BaseEntity
from .spec_loader import load_entities_spec, load_behavior_spec
from .compute import WeightedScore

"""
FastAPI App:
- Dynamic entities (from spec JSON if exists), CRUD on SQLModel DB
- Optional compute endpoint (weighted score) if behavior spec exists
- Health/Ready/Metrics endpoints for orchestration/observability
- Minimal migrations strategy: auto-create tables; optional Alembic upgrade
"""





LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("api")

app = FastAPI(title="IMU Domain Backend (SQLModel)")

# --------- Load spec & models ---------
entities_spec = load_entities_spec()      # List[ {name, fields:[[name,type],...]} ]
behavior_spec = load_behavior_spec()      # {name, inputs:[...], weights:[...], tests:[...]}

ModelRegistry: Dict[str, Type[BaseEntity]] = build_models_from_entities_spec(entities_spec)

# --------- DB init & migrations ----------
engine = init_engine()
if os.getenv("DB_AUTO_CREATE", "1").lower() not in ("0","false","no","off"):
    SQLModel.metadata.create_all(engine)

if os.getenv("DB_AUTO_MIGRATE", "0").lower() in ("1","true","yes","on"):
    try:
        run_alembic_upgrade_if_needed()
    except Exception as e:
        logger.warning("alembic upgrade failed; proceeding with create_all fallback: %s", e)

# --------- Health / Ready / Metrics ----------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True}

@app.get("/readyz")
def readyz() -> Dict[str, Any]:
    ok, msg = try_connect_db(engine)
    return {"ready": ok, "db": msg}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    # very small set; Prometheus plaintext exposition
    return "# HELP app_up 1 if app is up\n# TYPE app_up gauge\napp_up 1\n"

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "api", "entities": list(ModelRegistry.keys())}

# --------- CRUD factories ----------
def _plural(name: str) -> str:
    return name.lower() + "s"

def register_crud(model_name: str, model_cls: Type[BaseEntity]) -> None:
    plural = _plural(model_name)

    @app.post(f"/{plural}")
    def create_item(item: model_cls, session: Session = Depends(get_session)):  # type: ignore
        if getattr(item, "id", None) is None:
            raise HTTPException(422, "id required")
        existing = session.get(model_cls, item.id)
        if existing is not None:
            raise HTTPException(409, "id exists")
        session.add(item)
        session.commit()
        session.refresh(item)
        return {"ok": True, model_name.lower(): item}

    @app.get(f"/{plural}/{{oid}}")
    def get_item(oid: int, session: Session = Depends(get_session)):
        obj = session.get(model_cls, oid)
        if obj is None:
            raise HTTPException(404, "not found")
        return obj

    @app.get(f"/{plural}")
    def list_items(session: Session = Depends(get_session)):
        stmt = select(model_cls)
        return list(session.exec(stmt))

    @app.put(f"/{plural}/{{oid}}")
    def update_item(oid: int, item: model_cls, session: Session = Depends(get_session)):  # type: ignore
        obj = session.get(model_cls, oid)
        if obj is None:
            raise HTTPException(404, "not found")
        for k, v in item.model_dump().items():
            setattr(obj, k, v)
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return {"ok": True, model_name.lower(): obj}

    @app.delete(f"/{plural}/{{oid}}")
    def delete_item(oid: int, session: Session = Depends(get_session)):
        obj = session.get(model_cls, oid)
        if obj is None:
            raise HTTPException(404, "not found")
        session.delete(obj)
        session.commit()
        return {"ok": True}

# Register all entities
for name, cls in ModelRegistry.items():
    register_crud(name, cls)

# --------- Optional compute endpoint (behavior.json) ----------
ENABLE_BEHAVIOR = os.getenv("ENABLE_BEHAVIOR", "true").lower() not in ("0","false","no","off")
if ENABLE_BEHAVIOR and behavior_spec:
    score = WeightedScore.from_spec(behavior_spec)

    @app.post(f"/compute/{score.name}")
    def compute(payload: Dict[str, float]) -> Dict[str, Any]:
        val = score.eval(payload)
        return {"ok": True, "score": val}
