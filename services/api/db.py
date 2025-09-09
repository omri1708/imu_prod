from __future__ import annotations
import os, subprocess
from typing import Tuple
from sqlmodel import create_engine, Session

def _db_url() -> str:
    return os.getenv("DB_URL", "sqlite:///data/app.db")

def init_engine():
    url = _db_url()
    if url.startswith("sqlite"):
        return create_engine(url, echo=False, connect_args={"check_same_thread": False})
    return create_engine(url, echo=False)

def get_session():
    engine = init_engine()
    with Session(engine) as s:
        yield s

def try_connect_db(engine) -> Tuple[bool, str]:
    try:
        with Session(engine) as s:
            s.exec("SELECT 1")
        return True, "ok"
    except Exception as e:
        return False, str(e)

def run_alembic_upgrade_if_needed():
    if not os.path.exists("alembic.ini"):
        return
    subprocess.run(["alembic", "upgrade", "head"], check=True)
