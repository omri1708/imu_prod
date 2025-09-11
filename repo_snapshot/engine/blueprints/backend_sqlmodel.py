# NEW: engine/blueprints/backend_sqlmodel.py
from __future__ import annotations
from typing import Dict, Any, List
import json, re

def _py_type(t: str) -> str:
    return {"int":"int","float":"float","str":"str","bool":"bool"}.get(t,"str")

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name or "Entity")

def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    entities: List[Dict[str, Any]] = spec.get("entities") or []
    # db.py
    db_py = """from __future__ import annotations
import os, subprocess
from typing import Tuple
from sqlmodel import SQLModel, Session, create_engine

def _db_url() -> str:
    # default: in-memory for tests; set DB_URL=sqlite:///... for file
    return os.getenv("DB_URL", "sqlite:///:memory:")

def _ensure_sqlite_dir(url: str) -> None:
    if url.startswith("sqlite") and ":memory:" not in url:
        path = url.replace("sqlite:///", "", 1)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

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
"""
    # app.py
    lines: List[str] = []
    lines.append("import time")
    lines.append("from typing import Dict, Any, List, Optional")
    lines.append("from fastapi import FastAPI, HTTPException, Depends")
    lines.append("from pydantic import BaseModel")
    lines.append("from sqlmodel import SQLModel, Field, Session, select")
    lines.append("from .db import init_engine, get_session")
    lines.append("")
    lines.append("app = FastAPI(title='IMU API (SQLModel)')")
    lines.append("engine = init_engine()")
    lines.append("SQLModel.metadata.create_all(engine)")
    lines.append("")
    lines.append("@app.get('/healthz')")
    lines.append("def healthz(): return {'ok': True, 'ts': time.time()}")
    lines.append("@app.get('/metrics')")
    lines.append("def metrics(): return ('# HELP app_up 1\\n# TYPE app_up gauge\\napp_up 1\\n', 200, {'Content-Type':'text/plain; version=0.0.4'})")
    lines.append("")
    # entities
    ent_names = []
    for e in entities:
        en = _sanitize(e.get("name","Entity"))
        ent_names.append(en)
        fields = e.get("fields") or [["id","int"]]
        lines.append(f"class {en}(SQLModel, table=True):")
        has_id = any((f and f[0]=="id") for f in fields)
        if not has_id: fields = [["id","int"]] + fields
        for fname, ftype in fields:
            if fname == "id":
                lines.append("    id: Optional[int] = Field(default=None, primary_key=True)")
            else:
                lines.append(f"    { _sanitize(fname) }: { _py_type(ftype) } | None = None")
        lines.append("")
        # CRUD
        low = en.lower()
        lines.append(f"@app.post('/{low}s')")
        lines.append(f"def create_{low}(obj: {en}, s: Session = Depends(get_session)):")
        lines.append("    s.add(obj); s.commit(); s.refresh(obj); return {'ok': True, '%s': obj}" % low)
        lines.append(f"@app.get('/{low}s')")
        lines.append(f"def list_{low}(s: Session = Depends(get_session)):")
        lines.append(f"    return list(s.exec(select({en})))")
        lines.append(f"@app.get('/{low}s/{{oid}}')")
        lines.append(f"def get_{low}(oid: int, s: Session = Depends(get_session)):")
        lines.append(f"    o = s.get({en}, oid);")
        lines.append("    if not o: raise HTTPException(404,'not found') ; return o")
        lines.append("")
    # root
    lines.append("@app.get('/')")
    lines.append(f"def root(): return {{'ok': True, 'entities': {ent_names or ['Entity']} }}")
    lines.append("")
    app_py = "\n".join(lines)

    # tests
    t = []
    t.append("from fastapi.testclient import TestClient")
    t.append("import services.api.app as appmod")
    t.append("c = TestClient(appmod.app)")
    t.append("")
    t.append("def test_healthz():")
    t.append("    r = c.get('/healthz'); assert r.status_code==200 and r.json().get('ok') is True")
    t.append("")
    t.append("def test_root_lists_entities():")
    t.append("    r = c.get('/'); assert r.status_code==200 and isinstance(r.json().get('entities'), list)")
    test_py = "\n".join(t)

    req = b"fastapi\npydantic\nuvicorn[standard]\nsqlmodel\n"

    return {
      "services/api/db.py": db_py.encode("utf-8"),
      "services/api/app.py": app_py.encode("utf-8"),
      "services/api/tests/test_acceptance_generated.py": test_py.encode("utf-8"),
      "services/api/requirements.txt": req,
      "services/__init__.py": b"# package",
      "services/api/__init__.py": b"# package",
    }
