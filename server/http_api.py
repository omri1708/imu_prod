# server/http_api.py

from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
import json, os, mimetypes
from broker.stream import broker, _Sub
from broker.policy import DropPolicy
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from engine.policy import AskAndProceedPolicy, UserSubspace, RequestContext
from engine.provenance import ProvenanceStore
from capabilities.manager import CapabilityManager
from streaming.broker import StreamBroker

STATIC_DIR = os.environ.get("IMU_STATIC_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui_dsl")))
REGISTRY = {
    "android_sdk": {"installer": ["echo installing_android_sdk"], "min_trust": 1, "needs_network": False, "ttl_hint": 86400},
    "ios_xcode":  {"installer": ["echo installing_xcode_tools"], "min_trust": 2, "needs_network": False, "ttl_hint": 86400},
    "unity_cli":  {"installer": ["echo installing_unity_cli"], "min_trust": 1, "needs_network": False},
    "cuda_toolkit":{"installer": ["echo installing_cuda_toolkit"], "min_trust": 2, "needs_network": False},
    "k8s_cli":    {"installer": ["echo installing_kubectl"], "min_trust": 1, "needs_network": False},
}

policy = AskAndProceedPolicy(REGISTRY)
prov = ProvenanceStore()
capman = CapabilityManager(policy, prov)
broker = StreamBroker()  # ישות משותפת ל-HTTP ול-WS


def _json(self: BaseHTTPRequestHandler, code: int, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/capabilities/status":
            return _json(self, 200, {"status": capman.status})
        return _json(self, 404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/capabilities/request":
            ln = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(ln).decode() if ln > 0 else "{}"
            try:
                req = json.loads(body or "{}")
                name = req["name"]
                dry_run = bool(req.get("dry_run", True))
                user = UserSubspace(
                    user_id=req.get("user_id", "anon"),
                    trust_level=int(req.get("trust_level", 1)),
                    ttl_seconds=int(req.get("ttl", 3600)),
                    allow_exec=bool(req.get("allow_exec", False)),
                    allow_network=bool(req.get("allow_network", False)),
                    strict_provenance=bool(req.get("strict_provenance", True)),
                )
                ctx = RequestContext(user=user, reason=f"request_capability:{name}")
                res = capman.request(name, ctx, dry_run=dry_run)
                # דחיפת אירוע התקדמות ל-WS timeline
                broker.publish("timeline", {
                    "kind": "capability_request",
                    "capability": name,
                    "user": user.user_id,
                    "dry_run": dry_run,
                    "result": res,
                })
                return _json(self, 200, res)
            except Exception as e:
                broker.publish("timeline", {"kind": "error", "where": "capabilities/request", "msg": str(e)})
                return _json(self, 400, {"error": str(e)})
        
        if self.path == "/adapters/dry_run":
            ln = int(self.headers.get("Content-Length","0"))
            req = json.loads(self.rfile.read(ln) or b"{}")
            from engine.adapters_runner import AdaptersService
            user = UserSubspace(
                user_id=req.get("user_id","anon"),
                trust_level=int(req.get("trust_level",1)),
                ttl_seconds=int(req.get("ttl",3600)),
                allow_exec=False,  # dry-run בלבד
                allow_network=False,
                strict_provenance=True,
            )
            ctx = RequestContext(user=user, reason=f"adapter_dry_run:{req.get('adapter')}")
            svc = AdaptersService(policy, broker)
            plan = svc.dry_run(req["adapter"], req.get("spec",{}), ctx)
            return _json(self, 200, {"ok": True, "plan": {"commands": plan.commands, "env": plan.env, "notes": plan.notes}})

        return _json(self, 404, {"error": "not_found"})


def serve_http(host="0.0.0.0", port=8081):
    httpd = HTTPServer((host, port), Handler)
    print(f"[HTTP] listening on http://{host}:{port}")
    httpd.serve_forever()


def app(env, start):
    path = env.get('PATH_INFO','/')
    method = env.get('REQUEST_METHOD','GET')

    # SSE
    if path == "/events":
        qs = parse_qs(env.get('QUERY_STRING',''))
        topic = qs.get('topic', ['timeline'])[0]
        # כל מנוי מקבל תור עם drop-policy בטוח (החלפת נמוכים)
        sub: _Sub = broker.subscribe(topic, drop_policy=DropPolicy.LOWEST_PRIORITY_REPLACE)
        start("200 OK", [('Content-Type','text/event-stream'),
                         ('Cache-Control','no-cache'),
                         ('Connection','keep-alive')])
        return broker.sse_iter(sub)

    # פרסום אירוע
    if path == "/publish" and method == "POST":
        size = int(env.get('CONTENT_LENGTH','0') or 0)
        raw = env['wsgi.input'].read(size) if size>0 else b"{}"
        try:
            data = json.loads(raw.decode('utf-8') or "{}")
            topic = data.get("topic","timeline")
            prio = data.get("priority","telemetry")
            ev = data.get("event",{})
            ok = broker.publish(topic, ev, priority=prio)
            return _json(start, 200, {"ok": ok})
        except Exception as e:
            return _json(start, 400, {"ok": False, "error": str(e)})

    # סטטוס/מדדים
    if path == "/stats":
        return _json(start, 200, broker.stats())

    # עדכון קונפיג נושא (rps/burst/max_queue/weight/policy)
    if path == "/topic/config" and method == "POST":
        size = int(env.get('CONTENT_LENGTH','0') or 0)
        raw = env['wsgi.input'].read(size) if size>0 else b"{}"
        data = json.loads(raw.decode('utf-8') or "{}")
        topic = data["topic"]
        cfg = {}
        for k in ("rps","burst","max_queue","weight"):
            if k in data: cfg[k] = type(broker._topics[topic]["bucket"].rps if k=="rps" else
                                        broker._topics[topic]["bucket"].burst if k=="burst" else
                                        broker._topics[topic]["max_queue"] if k=="max_queue" else
                                        broker._topics[topic]["weight"])(data[k])
        policy = data.get("drop_policy")
        if policy: cfg["drop_policy"] = policy
        broker.configure_topic(topic, **cfg)
        return _json(start, 200, {"ok": True, "topic": topic})

    # סטטיק
    fpath = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
    if path == "/" or not os.path.isfile(fpath):
        fpath = os.path.join(STATIC_DIR, "index.html")
    try:
        ctype, _ = mimetypes.guess_type(fpath); ctype = ctype or "text/plain"
        with open(fpath,'rb') as fh:
            buff = fh.read()
        start("200 OK", [('Content-Type', f"{ctype}; charset=utf-8"),
                         ('Cache-Control','no-store')])
        return [buff]
    except FileNotFoundError:
        start("404 NOT FOUND", [('Content-Type','text/plain')])
        return [b'not found']


if __name__ == "__main__":
    serve_http()