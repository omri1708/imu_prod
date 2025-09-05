# api/http_api.py (שרת HTTP טהור stdlib + SSE)
# -*- coding: utf-8 -*-
import json, threading, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any
from engine.synthesis_pipeline import run_pipeline
from synth.rollout import gated_rollout
from stream.broker import BROKER
from engine.events import emit_progress, emit_timeline
from common.exc import ResourceRequired

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.RLock()

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

def serve(addr: str = "127.0.0.1", port: int = 8088):
    httpd = HTTPServer((addr, port), Handler)
    print(f"[IMU-HTTP] serving on http://{addr}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()