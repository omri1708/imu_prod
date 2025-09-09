# app/http_api.py
# -*- coding: utf-8 -*-
# HTTP API מינימלי על ספריית התקן בלבד (wsgiref) + JSON
import json, time
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
from typing import Dict, Any

from realtime.integrations import start_realtime, push_progress, push_timeline
from governance.user_policy import get_user_policy, ensure_user
from engine.respond import respond_grounded_json
from audit.log import AppendOnlyAudit

AUDIT = AppendOnlyAudit("var/audit/http_api.jsonl")

def _json(environ) -> Dict[str, Any]:
    try:
        ln = int(environ.get('CONTENT_LENGTH') or 0)
    except:
        ln = 0
    body = environ['wsgi.input'].read(ln) if ln > 0 else b"{}"
    try:
        return json.loads(body.decode('utf-8') or "{}")
    except:
        return {}

def _ok(start_response, obj):
    data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    start_response('200 OK', [('Content-Type','application/json; charset=utf-8'),
                              ('Content-Length', str(len(data)))])
    return [data]

def _bad(start_response, code, msg, detail=None):
    payload = {"error": msg, "detail": detail or {}, "code": code}
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    start_response('400 Bad Request', [('Content-Type','application/json; charset=utf-8'),
                                       ('Content-Length', str(len(data)))])
    return [data]

def application(environ, start_response):
    path = environ.get('PATH_INFO','/')
    method = environ.get('REQUEST_METHOD','GET')
    qs = parse_qs(environ.get('QUERY_STRING',''))
    user_id = (qs.get("user") or ["anonymous"])[0]
    ensure_user(user_id)

    if path == "/health":
        return _ok(start_response, {"ok": True, "ts": int(time.time())})

    if path == "/progress" and method == "POST":
        body = _json(environ)
        task = str(body.get("task","default"))
        value = int(body.get("value",0))
        push_progress(task, value)
        AUDIT.append({"kind":"progress","user":user_id,"task":task,"value":value})
        return _ok(start_response, {"ok": True})

    if path == "/timeline" and method == "POST":
        body = _json(environ)
        stream = str(body.get("stream","default"))
        event = body.get("event",{})
        push_timeline(stream, event, priority=5)
        AUDIT.append({"kind":"timeline","user":user_id,"stream":stream,"event":event})
        return _ok(start_response, {"ok": True})

    if path == "/respond" and method == "POST":
        # מחייב Evidences/Claims לפי מדיניות המשתמש (רמות אמון/TTL/דומיינים)
        body = _json(environ)
        text = str(body.get("text",""))
        claims = body.get("claims")  # [{"id","text"}]
        evidence = body.get("evidence")  # [{"sha256","ts","trust","url","sig_ok"}]

        try:
            policy, ev_index = get_user_policy(user_id)  # מדיניות קשיחה לפי user
            out = respond_grounded_json(text=text, claims=claims, evidence=evidence,
                                        policy=policy, ev_index=ev_index, user=user_id)
            AUDIT.append({"kind":"respond","user":user_id,"claims":claims,"evidence_count":len(evidence or [])})
            return _ok(start_response, out)
        except Exception as e:
            AUDIT.append({"kind":"respond_error","user":user_id,"type":e.__class__.__name__,"msg":str(e)})
            return _bad(start_response, e.__class__.__name__, str(e))

    return _bad(start_response, "not_found", f"path {path} not found")

if __name__ == "__main__":
    # מפעיל ברוקר ריל־טיים עם QoS + שרת HTTP #TODO
    start_realtime(host="127.0.0.1", ws_port=8766)
    httpd = make_server('127.0.0.1', 8080, application)
    print("HTTP API on http://127.0.0.1:8080 , WS broker on ws://127.0.0.1:8766")
    httpd.serve_forever()