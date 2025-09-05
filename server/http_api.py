from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
import json, os, mimetypes
from broker.stream import broker, _Sub
from broker.policy import DropPolicy

STATIC_DIR = os.environ.get("IMU_STATIC_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui_dsl")))

def _json(start, code, body):
    start(f"{code} OK", [('Content-Type','application/json; charset=utf-8'),
                         ('Cache-Control','no-store')])
    return [json.dumps(body, ensure_ascii=False).encode('utf-8')]

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
    port = int(os.environ.get("IMU_HTTP_PORT","8080"))
    httpd = make_server("", port, app)
    print(f"* http://127.0.0.1:{port}  (SSE: /events?topic=progress)")
    httpd.serve_forever()