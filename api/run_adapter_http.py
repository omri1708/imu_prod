# api/run_adapter_http.py
from __future__ import annotations
import asyncio, os, json, time
from typing import Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from adapters.unity_cli import run_unity_headless, UnityBuildError
from infra.artifact_server import ArtifactServer
from broker.ws_server import BROKER

ART = ArtifactServer(artifacts_dir="./.artifacts")

def _ok(d: Dict[str, Any]) -> bytes:
    return json.dumps({"ok": True, **d}).encode("utf-8")
def _err(msg: str) -> bytes:
    return json.dumps({"ok": False, "error": msg}).encode("utf-8")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            ln = int(self.headers.get("content-length","0"))
            raw = self.rfile.read(ln)
            body = json.loads(raw or b"{}")
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(_err("bad_json")); return

        if self.path == "/run_adapter/unity_build":
            topic = body.get("topic") or "unity-job"
            proj = body["project_path"]; target = body["target"]; outdir = body.get("output_dir","./.unity_out")
            asyncio.run(BROKER.publish(topic, "progress", {"pct": 0.0, "stage":"start"}))
            try:
                arts = run_unity_headless(project_path=proj, target=target, output_dir=outdir)
                asyncio.run(BROKER.publish(topic, "progress", {"pct": 60.0, "stage":"built"}))
                # העלאה ל־ArtifactServer
                shas = []
                for p in arts:
                    sha, _ = ART.put_file(p, {"target": target, "project": proj})
                    shas.append(sha)
                    asyncio.run(BROKER.publish(topic, "timeline", {"event":"artifact_uploaded","sha":sha,"file":os.path.basename(p)}))
                asyncio.run(BROKER.publish(topic, "progress", {"pct": 85.0, "stage":"uploaded"}))
                # דוגמת פריסה ל־K8s (לצורך הדגמה, מפרסמים אירוע; חיבור אמיתי ל־kubectl אפשר להשלים כאן)
                asyncio.run(BROKER.publish(topic, "timeline", {"event":"k8s_deploy_start"}))
                time.sleep(0.4)
                asyncio.run(BROKER.publish(topic, "timeline", {"event":"k8s_deploy_done","replicas":1}))
                asyncio.run(BROKER.publish(topic, "progress", {"pct": 100.0, "stage":"done"}))
                self.send_response(200); self.end_headers(); self.wfile.write(_ok({"artifacts": shas}))
            except UnityBuildError as e:
                asyncio.run(BROKER.publish(topic, "timeline", {"event":"unity_error","msg":str(e)}))
                self.send_response(500); self.end_headers(); self.wfile.write(_err(str(e)))
            return

        self.send_response(404); self.end_headers(); self.wfile.write(_err("not_found"))

def serve_http(host="0.0.0.0", port=8089):
    httpd = HTTPServer((host, port), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    # run WS broker and HTTP server
    t = Thread(target=serve_http, kwargs={"host":"0.0.0.0","port":8089}, daemon=True)
    t.start()
    asyncio.run(BROKER.serve(host="0.0.0.0", port=8765))