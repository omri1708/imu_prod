from __future__ import annotations
import os
from typing import Dict, Any, Type
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import PlainTextResponse
from sqlmodel import SQLModel, Session, select
from prometheus_client import CollectorRegistry, Gauge, generate_latest

from .db import init_engine, get_session, try_connect_db, run_alembic_upgrade_if_needed
from .models import build_models_from_entities_spec, BaseEntity
from .spec_loader import load_entities_spec, load_behavior_spec
from .compute import WeightedScore

LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()
app = FastAPI(title="IMU API (SQLModel)")
entities_spec = load_entities_spec()
behavior_spec = load_behavior_spec()
ModelReg: Dict[str, Type[BaseEntity]] = build_models_from_entities_spec(entities_spec)

engine = init_engine()
if os.getenv("DB_AUTO_CREATE","1").lower() not in ("0","false","no","off"):
    SQLModel.metadata.create_all(engine)
if os.getenv("DB_AUTO_MIGRATE","0").lower() in ("1","true","yes","on"):
    try:
        run_alembic_upgrade_if_needed()
    except Exception:
        pass

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/readyz")
def readyz():
    ok, msg = try_connect_db(engine)
    return {"ready": ok, "db": msg}

_registry = CollectorRegistry()
_g_up = Gauge("app_up", "1 if app is up", registry=_registry)
_g_up.set(1)

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest(_registry).decode("utf-8")

@app.get("/")
def root(): return {"ok": True, "entities": list(ModelReg.keys())}

def _plural(n: str) -> str: return n.lower()+"s"

def register_crud(model_name: str, model_cls: Type[BaseEntity]) -> None:
    p = _plural(model_name)
    
    @app.post(f"/{p}")
    def create(obj: model_cls, s: Session = Depends(get_session)):  # type: ignore
        if getattr(obj, "id", None) is None:
            raise HTTPException(422, "id required")
        if s.get(model_cls, obj.id) is not None:
            raise HTTPException(409, "id exists")
        s.add(obj)
        s.commit()
        s.refresh(obj)
        return {"ok": True, model_name.lower(): obj}
    
    @app.get(f"/{p}/{{oid}}")
    def read(oid: int, s: Session = Depends(get_session)):
        o = s.get(model_cls, oid) 
        if o is None:
            raise HTTPException(404, "not found")
        return o
    
    @app.get(f"/{p}")
    def list_(s: Session = Depends(get_session)):
        return list(s.exec(select(model_cls)))
    
    @app.put(f"/{p}/{{oid}}")
    def update(oid: int, obj: model_cls, s: Session = Depends(get_session)):  # type: ignore
        o = s.get(model_cls, oid) 
        if o is None:
            raise HTTPException(404, "not found")
        for k,v in obj.model_dump().items():
            setattr(o,k,v)
        s.add(o)
        s.commit()
        s.refresh(o)
        return {"ok": True, model_name.lower(): o}
    
    @app.delete(f"/{p}/{{oid}}")
    def delete(oid: int, s: Session = Depends(get_session)):
        o = s.get(model_cls, oid) 
        if o is None:
            raise HTTPException(404, "not found")
        s.delete(o)
        s.commit()
        return {"ok": True}

for name, cls in ModelReg.items():
    register_crud(name, cls)

if os.getenv("ENABLE_BEHAVIOR","true").lower() not in ("0","false","no","off") and behavior_spec:
    score = WeightedScore.from_spec(behavior_spec)
    @app.post(f"/compute/{score.name}")
    def compute(payload: Dict[str, float]) -> Dict[str, Any]:
        return {"ok": True, "score": score.eval(payload)}
