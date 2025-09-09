# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import subprocess
from typing import Tuple
from sqlmodel import create_engine, Session

"""
DB bootstrap:
- Reads DB_URL from environment (e.g., sqlite:///data/app.db or postgres)
- Provides SQLModel engine + session dependency
- Health/ready helpers and optional Alembic migration
"""



def _db_url() -> str:
    return os.getenv("DB_URL", "sqlite:///data/app.db")

def init_engine():
    url = _db_url()
    if url.startswith("sqlite"):
        return create_engine(url, echo=False, connect_args={"check_same_thread": False})
    return create_engine(url, echo=False)

def get_session():
    from sqlmodel import SQLModel
    engine = init_engine()
    with Session(engine) as session:
        yield session

def try_connect_db(engine) -> Tuple[bool, str]:
    try:
        with Session(engine) as s:
            s.exec("SELECT 1")
        return True, "ok"
    except Exception as e:
        return False, str(e)

def run_alembic_upgrade_if_needed():
    # runs 'alembic upgrade head' if alembic config is present
    if not os.path.exists("alembic.ini"):
        return
    subprocess.run(["alembic", "upgrade", "head"], check=True)
