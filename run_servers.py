# run_servers.py
import threading
from streaming.ws_server import WSServer
from server.http_api import serve_http, broker

if __name__ == "__main__":
    ws = WSServer(broker=broker)
    t = threading.Thread(target=ws.run, daemon=True)
    t.start()
    serve_http(host="0.0.0.0", port=8081)