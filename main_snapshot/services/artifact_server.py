# services/artifact_server.py (Artifact-Server פשוט לקליטה/אחסון ושילוב Provenance)
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, time
from provenance.cas import put_blob, EvidenceMeta

ARTIFACT_ROOT = os.environ.get("IMU_ARTIFACT_ROOT", ".imu_artifacts")
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...), source: str = Form("unity"), ttl: int = Form(30*24*3600), trust: float = Form(0.8)):
    b = await file.read()
    h = put_blob(b, EvidenceMeta(source=source, retrieved_at=time.time(), ttl_seconds=ttl, trust=trust, content_type=file.content_type or "application/octet-stream"))
    path = os.path.join(ARTIFACT_ROOT, h)
    with open(path, "wb") as f: f.write(b)
    return JSONResponse({"ok": True, "hash": h, "path": path})