# services/api/db.py
from __future__ import annotations
import os, subprocess
from typing import Tuple
from urllib.parse import urlparse
from sqlmodel import create_engine, Session

def _db_url() -> str:
    # אם לא הוגדר DB_URL: ברירת-מחדל בטוחה לטסטים/לוקאל (in-memory)
    # אם תרצה קובץ, הגדר: DB_URL=sqlite:///absolute/or/relative/path/to/app.db
    return os.getenv("DB_URL", "sqlite:///:memory:")

def _ensure_sqlite_dir(url: str) -> None:
    # sqlite:///:memory:  → אין צורך בתיקייה
    if url.strip().lower().startswith("sqlite") and ":memory:" not in url:
        # פורק את הנתיב מה-URL
        path = url.replace("sqlite:///", "", 1) if "sqlite:///" in url else url.replace("sqlite:", "", 1)
        path = path.strip()
        # אם הוא יחסי, נבנה ממנו נתיב מוחלט עבור התהליך הנוכחי
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        dir_ = os.path.dirname(path) or "."
        os.makedirs(dir_, exist_ok=True)

def init_engine():
    url = _db_url()
    _ensure_sqlite_dir(url)
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
