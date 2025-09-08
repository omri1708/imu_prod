# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List
import json, re

def _py_type(t: str) -> str:
    return {"int": "int", "float": "float", "str": "str", "bool": "bool"}.get(t, "str")

def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name or "X")

def generate_backend(spec: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Legal / Use Anchor:
    - No violation of any provider ToS. Lawful use under user control.
    - No hidden installation or network actions here; files only.
    - Output is auditable and intentionally minimal on external deps.

    Generates a practical multi-service scaffold from SPEC:
      - services/api/app.py  (FastAPI: CRUD from entities + /compute + health/ready/metrics)
      - tests (pytest) based on core_behavior tests
      - server/stream_wfq_ws.py (minimal WebSocket service; echo/broadcast)
      - docker/prod/api/Dockerfile, docker/ws/Dockerfile, docker/ui/Dockerfile
      - ui/index.html (static)
      - __init__.py packages to enable module imports
      - services/api/requirements.txt
    """
    entities: List[Dict[str, Any]] = spec.get("entities") or []
    beh = spec.get("core_behavior") or {}
    b_name = _sanitize_name(beh.get("name", "score"))
    inputs = [_sanitize_name(x) for x in (beh.get("inputs") or [])]
    weights = list(beh.get("weights") or [1.0] * len(inputs))
    tests = beh.get("tests") or []

    # ---------- services/api/app.py ----------
    lines: List[str] = []
    lines.append("import os, logging, time")
    lines.append("from typing import Dict, List, Any")
    lines.append("from fastapi import FastAPI, HTTPException")
    lines.append("from pydantic import BaseModel")
    lines.append("")
    lines.append("LOG_LEVEL = os.getenv('LOG_LEVEL','INFO').upper()")
    lines.append("logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))")
    lines.append("app = FastAPI(title='IMU Domain Backend')")
    lines.append("")
    lines.append("@app.get('/healthz')")
    lines.append("def healthz():")
    lines.append("    return {'ok': True, 'ts': time.time()}")
    lines.append("")
    lines.append("@app.get('/readyz')")
    lines.append("def readyz():")
    lines.append("    return {'ready': True}")
    lines.append("")
    lines.append("@app.get('/metrics')")
    lines.append("def metrics():")
    lines.append("    # minimal Prometheus plaintext without external deps")
    lines.append("    body = '# HELP app_up 1 if app is up\\n# TYPE app_up gauge\\napp_up 1\\n'")
    lines.append("    return (body, 200, {'Content-Type': 'text/plain; version=0.0.4'})")
    lines.append("")

    # Pydantic models per entity + in-memory store + CRUD
    for e in entities:
        en = _sanitize_name(e.get("name", "Entity"))
        fields = e.get("fields") or [["id", "int"]]
        lines.append(f"class {en}(BaseModel):")
        has_id = any((f and f[0] == "id") for f in fields)
        if not has_id:
            fields = [["id", "int"]] + fields
        for fname, ftype in fields:
            lines.append(f"    {_sanitize_name(fname)}: {_py_type(ftype)}")
        lines.append("")
        lines.append(f"DB_{en.upper()}: Dict[int, {en}] = {{}}")
        lines.append("")
        # Create
        lines.append(f"@app.post('/{en.lower()}s')")
        lines.append(f"def create_{en.lower()}(obj: {en}):")
        lines.append(f"    if obj.id in DB_{en.upper()}:")
        lines.append("        raise HTTPException(409, 'id exists')")
        lines.append(f"    DB_{en.upper()}[obj.id] = obj")
        lines.append(f"    return {{'ok': True, '{en.lower()}': obj}}")
        # Read one
        lines.append(f"@app.get('/{en.lower()}s/{{oid}}')")
        lines.append(f"def get_{en.lower()}(oid: int):")
        lines.append(f"    if oid not in DB_{en.upper()}:")
        lines.append("        raise HTTPException(404, 'not found')")
        lines.append(f"    return DB_{en.upper()}[oid]")
        # List
        lines.append(f"@app.get('/{en.lower()}s')")
        lines.append(f"def list_{en.lower()}():")
        lines.append(f"    return list(DB_{en.upper()}.values())")
        lines.append("")

    # Behavior compute endpoint (weighted sum)
    if inputs:
        lines.append("class ComputeIn(BaseModel):")
        for i in inputs:
            lines.append(f"    {i}: float = 0.0")
        lines.append("")
        lines.append(f"WEIGHTS_{b_name.upper()} = {json.dumps(weights)}")
        lines.append(f"INPUTS_{b_name.upper()}  = {json.dumps(inputs)}")
        lines.append("")
        lines.append(f"@app.post('/compute/{b_name}')")
        lines.append("def compute(inp: ComputeIn):")
        lines.append(f"    xs = [getattr(inp, k, 0.0) for k in INPUTS_{b_name.upper()}]")
        lines.append(f"    ws = WEIGHTS_{b_name.upper()}")
        lines.append("    if len(xs) != len(ws):")
        lines.append("        raise HTTPException(422, 'weights mismatch')")
        lines.append("    score = sum(x*w for x, w in zip(xs, ws))")
        lines.append("    return {'ok': True, 'score': score}")
        lines.append("")
    else:
        lines.append("# no core_behavior inputs provided; compute endpoint skipped")
        lines.append("")

    lines.append("@app.get('/')")
    lines.append("def root():")
    lines.append("    return {'ok': True, 'service': 'api'}")
    lines.append("")
    app_py = "\n".join(lines)

    # ---------- tests (pytest) ----------
    tlines: List[str] = []
    tlines.append("from fastapi.testclient import TestClient")
    tlines.append("import services.api.app as apiapp")
    tlines.append("c = TestClient(apiapp.app)")
    tlines.append("")
    tlines.append("def test_healthz():")
    tlines.append("    r = c.get('/healthz'); assert r.status_code == 200 and r.json().get('ok') is True")
    if entities:
        en0 = _sanitize_name(entities[0].get("name", "Entity"))
        tlines.append("")
        tlines.append("def test_crud_entity():")
        tlines.append(f"    r = c.post('/{en0.lower()}s', json={{'id': 1}})")
        tlines.append("    assert r.status_code in (200, 409)")
        tlines.append(f"    r = c.get('/{en0.lower()}s'); assert r.status_code == 200")
    if inputs and tests:
        tlines.append("")
        tlines.append("def test_behavior_cases():")
        for idx, tc in enumerate(tests[:5], start=1):
            tinp = tc.get("inputs") or {}
            exp = float(tc.get("expected", 0.0))
            tlines.append(f"    r = c.post('/compute/{b_name}', json={json.dumps(tinp)})")
            tlines.append("    assert r.status_code == 200")
            tlines.append("    val = r.json()['score']")
            tlines.append(f"    assert abs(val - {exp}) < max(1.0, 0.05*abs({exp}))")
    test_py = "\n".join(tlines)

    # ---------- server/stream_wfq_ws.py ----------
    ws_py = """# -*- coding: utf-8 -*-
import os, asyncio, logging
try:
    import websockets  # minimal dependency in container only
except Exception:
    websockets = None

HOST = "0.0.0.0"
PORT = int(os.getenv("WS_PORT", "8766"))

_clients = set()

async def _handler(ws, path):
    _clients.add(ws)
    try:
        async for msg in ws:
            # minimal broadcast (placeholder for WFQ)
            await asyncio.gather(*(c.send(msg) for c in _clients if c is not ws), return_exceptions=True)
    finally:
        _clients.discard(ws)

def main():
    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL","INFO").upper(), logging.INFO))
    if websockets is None:
        raise RuntimeError("websockets not installed")
    start_server = websockets.serve(_handler, HOST, PORT)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()
"""

    # ---------- dockerfiles ----------
    docker_api = """# docker/prod/api/Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY services/api /app
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir fastapi pydantic uvicorn[standard]
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
"""

    docker_ws = """# docker/ws/Dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY server /app/server
RUN pip install --no-cache-dir websockets
EXPOSE 8766
CMD ["python","-m","server.stream_wfq_ws"]
"""

    docker_ui = """# docker/ui/Dockerfile
FROM nginx:alpine
COPY ui /usr/share/nginx/html
EXPOSE 80
"""

    # ---------- ui/index.html ----------
    index_html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>IMU UI</title></head>
<body style="font-family: system-ui, sans-serif">
<h3>IMU UI</h3>
<p>Minimal static UI. Connects to API on <code>/</code> and WS via <code>WS_URL</code> from .env.</p>
</body></html>
"""

    # ---------- packages & requirements ----------
    pkg_root = b"# package"
    requirements = b"fastapi\npydantic\nuvicorn[standard]\n"

    # return mapping
    return {
        "services/api/app.py": app_py.encode("utf-8"),
        "services/api/test_app.py": test_py.encode("utf-8"),
        "services/api/requirements.txt": requirements,
        "services/__init__.py": pkg_root,
        "services/api/__init__.py": pkg_root,
        "server/__init__.py": pkg_root,
        "server/stream_wfq_ws.py": ws_py.encode("utf-8"),
        "docker/prod/api/Dockerfile": docker_api.encode("utf-8"),
        "docker/ws/Dockerfile": docker_ws.encode("utf-8"),
        "docker/ui/Dockerfile": docker_ui.encode("utf-8"),
        "ui/index.html": index_html.encode("utf-8"),
    }
