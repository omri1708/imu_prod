# run_servers.py
import threading
from streaming.ws_server import WSServer
from api.http_api import serve_http, broker_for

if __name__ == "__main__":
    ws = WSServer(broker=broker_for)
    t = threading.Thread(target=ws.run, daemon=True)
    t.start()
    serve_http(host="0.0.0.0", port=8081)