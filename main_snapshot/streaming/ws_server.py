# streaming/ws_server.py
import socket, threading, base64, hashlib, struct, json
from streaming.broker import StreamBroker

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSServer:
    def __init__(self, host="0.0.0.0", port=8080, broker: StreamBroker | None = None):
        self.host, self.port = host, port
        self.broker = broker
        self.clients = []
        if self.broker:
            self.broker.subscribe("timeline", self.broadcast_json)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)
        print(f"[WS] listening on ws://{self.host}:{self.port}/ws")
        while True:
            conn, addr = sock.accept()
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handshake(self, conn: socket.socket) -> bool:
        data = conn.recv(2048).decode("utf-8", errors="ignore")
        if "Upgrade: websocket" not in data:
            return False
        lines = data.split("\r\n")
        key = ""
        for ln in lines:
            if ln.lower().startswith("sec-websocket-key:"):
                key = ln.split(":")[1].strip()
        if not key:
            return False
        accept = base64.b64encode(hashlib.sha1((key+GUID).encode()).digest()).decode()
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        conn.send(resp.encode("utf-8"))
        return True

    def _read_frame(self, conn: socket.socket) -> str | None:
        hdr = conn.recv(2)
        if not hdr:
            return None
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            ext = conn.recv(2)
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = conn.recv(8)
            length = struct.unpack("!Q", ext)[0]
        mask = conn.recv(4) if masked else b""
        payload = conn.recv(length) if length else b""
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            return None
        return payload.decode("utf-8", errors="ignore")

    def _send_text(self, conn: socket.socket, s: str):
        payload = s.encode("utf-8")
        header = bytes([0x81])
        n = len(payload)
        if n < 126:
            header += bytes([n])
        elif n <= 0xFFFF:
            header += bytes([126]) + struct.pack("!H", n)
        else:
            header += bytes([127]) + struct.pack("!Q", n)
        conn.send(header + payload)

    def _handle(self, conn: socket.socket, addr):
        if not self._handshake(conn):
            conn.close()
            return
        self.clients.append(conn)
        try:
            while True:
                msg = self._read_frame(conn)
                if msg is None:
                    break
                # כאן אפשר לפענח פקודות לקוח אם תרצה (subscribe ספציפי וכו')
        finally:
            try: conn.close()
            except: pass
            if conn in self.clients:
                self.clients.remove(conn)

    def broadcast_json(self, obj):
        s = json.dumps(obj, ensure_ascii=False)
        dead = []
        for c in self.clients:
            try:
                self._send_text(c, s)
            except Exception:
                dead.append(c)
        for c in dead:
            try: c.close()
            except: pass
            if c in self.clients:
                self.clients.remove(c)

if __name__ == "__main__":
    WSServer().run()