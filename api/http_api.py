# api/http_api.py (שרת HTTP טהור stdlib + SSE)
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, threading, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
from engine.synthesis_pipeline import run_pipeline
from synth.rollout import gated_rollout
from stream.broker import BROKER
from engine.events import emit_progress, emit_timeline
from common.exc import ResourceRequired
import asyncio, json
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Query
from fastapi.responses import JSONResponse
from stream.broker import Broker
from engine.enforcer import enforce_claims, GroundingError
from adapters.unity_cli import run_unity_build, ActionRequired as UnityReq
from adapters.k8s_uploader import upload_dir_with_tar, deploy_k8s_job, ActionRequired as K8sReq
from __future__ import annotations
from typing import Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from broker.stream_bus import bus
from engine.adapter_runner import enforce_policy, ResourceRequired
from adapters import android, ios, unity, cuda, k8s
from adapters.contracts import AdapterResult

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()


app = FastAPI()
BROKERS: Dict[str, Broker] = {}


ADAPTERS = {
    "android": lambda p: android.run_android_build(**p),
    "ios":     lambda p: ios.run_ios_build(**p),
    "unity":   lambda p: unity.run_unity_cli(**p),
    "cuda":    lambda p: cuda.run_cuda_job(**p),
    "k8s":     lambda p: k8s.deploy_k8s_manifest(**p),
}

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: Dict[str,Any]):
        self.send_response(code); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        if self.path != "/run_adapter":
            return self._json(404, {"error":"not found"})
        ln = int(self.headers.get("Content-Length","0"))
        body = json.loads(self.rfile.read(ln) or b"{}")
        user = body.get("user_id","anon")
        adapter = body.get("adapter")
        params = body.get("params",{})
        topic = f"adapter.{adapter}.run"
        try:
            enforce_policy(user_id=user, topic=topic, trust=body.get("trust","unknown"))
        except ResourceRequired as rr:
            bus.publish("timeline", {"type":"policy","user":user,"topic":topic,"required":rr.required, "priority":"high"})
            return self._json(200, {"status":"awaiting_consent","required": rr.required})

        if adapter not in ADAPTERS:
            return self._json(400, {"status":"error","message":"unknown adapter"})
        bus.publish("progress", {"phase":"start","adapter":adapter,"percent":0,"priority":"high"})
        res: AdapterResult = ADAPTERS[adapter](params)
        if res.status == "awaiting_consent":
            bus.publish("timeline", {"type":"resource","adapter":adapter,"required":res.required,"priority":"high"})
            return self._json(200, {"status":"awaiting_consent","required":res.required})
        if res.status == "ok":
            bus.publish("progress", {"phase":"done","adapter":adapter,"percent":100,"priority":"high"})
            bus.publish("timeline", {"type":"adapter_ok","adapter":adapter,"outputs":res.outputs})
        else:
            bus.publish("timeline", {"type":"adapter_error","adapter":adapter,"message":res.message})
        return self._json(200, {"status":res.status,"message":res.message,"outputs":res.outputs})

def serve(addr="127.0.0.1", port=8089):
    httpd = HTTPServer((addr,port), Handler)
    httpd.serve_forever()


def broker_for(uid: str) -> Broker:
    b = BROKERS.get(uid)
    if not b:
        b = Broker(uid)
        BROKERS[uid] = b
    return b

@app.websocket("/events")
async def events(ws: WebSocket, user: str = Query("default")):
    await ws.accept()
    b = broker_for(user)
    sub_timeline = b.subscribe("timeline")
    sub_progress = b.subscribe("progress")
    try:
        while True:
            # שולחים לפי עדיפויות שכבר נאכפות ב-broker
            done, _ = await asyncio.wait(
                [asyncio.create_task(sub_timeline.get()), asyncio.create_task(sub_progress.get())],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    payload = task.result()
                    await ws.send_text(json.dumps(payload))
                except Exception:
                    pass
    except WebSocketDisconnect:
        return

@app.post("/run_adapter")
async def run_adapter(
    user: str = Query("default"),
    kind: str = Query(...),         # "unity_k8s"
    claims: Dict[str, Any] = Body({"claims":[]}),
    project_path: str = Body(...),
    k8s_image: str = Body("alpine:3.19"),
    artifact_server_url: str = Body("http://localhost:8089")
):
    b = broker_for(user)
    try:
        enforce_claims(user, claims.get("claims", []))
    except GroundingError as e:
        return JSONResponse({"ok": False, "error":"grounding", "detail": str(e)}, status_code=412)

    if kind != "unity_k8s":
        return JSONResponse({"ok": False, "error":"unknown_kind"}, status_code=400)

    async def _run():
        # 1) Unity build
        try:
            async for ev in run_unity_build(project_path):
                await b.publish(ev["type"], ev)
        except UnityReq as r:
            await b.publish("timeline", {"type":"timeline","event":f"action_required: {r.what}"})
            return JSONResponse({"ok": False, "action_required":{"what":r.what, "how": r.how}}, status_code=428)
        # 2) upload to Artifact-Server
        try:
            from adapters.k8s_uploader import upload_dir_with_tar
            res = upload_dir_with_tar(artifact_server_url, project_path + "/Builds/StandaloneLinux64")
            await b.publish("timeline", {"type":"timeline","event":f"artifact: {res.get('hash')}"})
        except Exception as e:
            await b.publish("timeline", {"type":"timeline","event":f"artifact_upload_failed: {e}"})
            return JSONResponse({"ok": False, "error":"artifact_upload_failed", "detail": str(e)}, status_code=500)
        # 3) deploy job to k8s
        try:
            from adapters.k8s_uploader import deploy_k8s_job
            env = {"ARTIFACT_HASH": res.get("hash","")}
            dres = deploy_k8s_job("unity-runner", k8s_image, env)
            await b.publish("timeline", {"type":"timeline","event":f"k8s_job: {dres}"})
        except K8sReq as r:
            await b.publish("timeline", {"type":"timeline","event":f"action_required: {r.what}"})
            return JSONResponse({"ok": False, "action_required":{"what":r.what, "how": r.how}}, status_code=428)
        except Exception as e:
            return JSONResponse({"ok": False, "error":"k8s_deploy_failed", "detail": str(e)}, status_code=500)

        await b.publish("timeline", {"type":"timeline","event":"done"})
        return JSONResponse({"ok": True})

    # מריצים אסינכרוני ומחזירים תשובה ראשונית מידית
    loop = asyncio.get_event_loop()
    fut = loop.create_task(_run())
    return JSONResponse({"ok": True, "started": True})


def _new_job_id() -> str:
    import time, secrets
    return f"job_{int(time.time()*1000)}_{secrets.token_hex(6)}"

def _set_job(job_id: str, payload: Dict[str, Any]):
    with _JOBS_LOCK:
        _JOBS[job_id] = payload

def _get_job(job_id: str) -> Dict[str, Any]:
    with _JOBS_LOCK:
        return dict(_JOBS.get(job_id) or {})

def _json(self: BaseHTTPRequestHandler, code: int, obj: Dict[str, Any]):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type","application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

def _bad(self: BaseHTTPRequestHandler, msg: str, code: int = 400):
    _json(self, code, {"ok": False, "error": msg})

def _parse_body(self: BaseHTTPRequestHandler) -> Dict[str, Any]:
    ln = int(self.headers.get("Content-Length","0"))
    raw = self.rfile.read(ln) if ln>0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

class Handler(BaseHTTPRequestHandler):
    server_version = "IMU-HTTP/1.0"

    def do_GET(self):
        if self.path.startswith("/v1/jobs/"):
            job_id = self.path.split("/")[-1]
            return _json(self, 200, _get_job(job_id))
        if self.path.startswith("/v1/events"):
            # SSE: /v1/events?topic=pipeline.progress
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            topic = (qs.get("topic") or ["pipeline.progress"])[0]
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream; charset=utf-8")
            self.send_header("Cache-Control","no-cache")
            self.send_header("Connection","keep-alive")
            self.end_headers()
            for ev in BROKER.subscribe_iter(topic):
                line = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                try:
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    break
            return
        _bad(self, "not_found", 404)

    def do_POST(self):
        if self.path == "/v1/pipeline/run":
            body = _parse_body(self)
            spec = body.get("spec","")
            user = body.get("user","anonymous")
            job_id = _new_job_id()
            _set_job(job_id, {"ok": None, "stage":"queued"})
            emit_timeline("job_queued", user=user, job_id=job_id)

            def _worker():
                try:
                    emit_progress("start", user=user, job_id=job_id)
                    res = run_pipeline(user=user, spec_text=spec)
                    _set_job(job_id, res)
                    emit_progress("done", user=user, job_id=job_id, ok=res.get("ok",False))
                except ResourceRequired as rr:
                    payload = {"ok": False, "stage":"resource_required",
                               "kind": rr.kind, "items": rr.items, "how_to": rr.how_to}
                    _set_job(job_id, payload)
                    emit_progress("resource_required", user=user, job_id=job_id, **payload)
                except Exception as e:
                    _set_job(job_id, {"ok": False, "stage":"error", "error": str(e)})
                    emit_progress("error", user=user, job_id=job_id, error=str(e))

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            return _json(self, 202, {"ok": True, "job": job_id})

        if self.path == "/v1/rollout/k8s":
            body = _parse_body(self)
            manifest = body.get("manifest","")
            percent = float(body.get("canary_percent", 5.0))
            user = body.get("user","anonymous")
            from engine.adapter_registry import get_adapter
            from adapters.k8s import K8sAdapter
            ad = get_adapter("k8s")
            try:
                emit_progress("k8s_canary_start", user=user, percent=percent)
                # בדיקת dry-run כבר נעשית באדפטר בזמן build; כאן rollout אמיתי
                # נפתח פריסת canary לפי אחוז (למשל ע"י label/annotation; כאן חד פעמי)
                ad.rollout(manifest)
                emit_progress("k8s_canary_applied", user=user, percent=percent)
                return _json(self, 200, {"ok": True})
            except ResourceRequired as rr:
                return _json(self, 428, {"ok": False, "needs": rr.items, "how_to": rr.how_to})
            except Exception as e:
                return _json(self, 500, {"ok": False, "error": str(e)})

        _bad(self, "not_found", 404)


if __name__ == "__main__":
    serve()