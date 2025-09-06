# api/stream_http.py (שרת poll פשוט ל־Broker)
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, urllib.parse, threading
from ..stream.broker import StreamBroker

BROKER: StreamBroker = None

class StreamHandler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        b = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/stream/poll"):
            q = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(q)
            topic = (params.get("topic") or [""])[0]
            max_items = int((params.get("max") or ["100"])[0])
            if not topic:
                return self._send(400, {"error":"missing_topic"})
            evs = BROKER.poll(topic, max_items=max_items)
            return self._send(200, evs)
        return self._send(404, {"error":"not_found"})

def serve_stream(broker: StreamBroker, port: int=8090):
    global BROKER
    BROKER = broker
    httpd = HTTPServer(("0.0.0.0", port), StreamHandler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd