# realtime/ws_broker.py
# -*- coding: utf-8 -*-
# WebSocket broker מינימלי ללא תלות חיצונית: RFC6455 (Handshake + Text frames)
import base64, hashlib, selectors, socket, threading, time, json
from typing import Dict, Any, List

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class _Conn:
    __slots__ = ("sock","addr","alive")
    def __init__(self, sock, addr):
        self.sock = sock; self.addr = addr; self.alive = True

def _handshake(client):
    data = client.recv(4096).decode("latin1", "ignore")
    headers = {}
    for line in data.split("\r\n"):
        if ":" in line:
            k,v = line.split(":",1); headers[k.strip().lower()] = v.strip()
    key = headers.get("sec-websocket-key")
    if not key: return False
    accept = base64.b64encode(hashlib.sha1((key+_GUID).encode()).digest()).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    client.send(resp.encode("latin1"))
    return True

def _encode_frame_text(s: str) -> bytes:
    # FIN + opcode=1 (text)
    b = s.encode("utf-8")
    header = bytearray([0x81])
    l = len(b)
    if l <= 125:
        header.append(l)
    elif l < (1<<16):
        header.append(126); header += (l).to_bytes(2,"big")
    else:
        header.append(127); header += (l).to_bytes(8,"big")
    return bytes(header)+b

class WSBroker:
    """ברוקר WS פשוט: משדר לכל החיבורים; הקליינט מסנן לפי topic בצד הלקוח."""
    def __init__(self, host="127.0.0.1", port=8766):
        self.host = host; self.port = port
        self.sel = selectors.DefaultSelector()
        self.conns: List[_Conn] = []
        self._lock = threading.Lock()
        self._srv = None
        self._bg = None
        self._running = False

    def start(self):
        if self._running: return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(64); srv.setblocking(False)
        self.sel.register(srv, selectors.EVENT_READ, data="srv")
        self._srv = srv
        self._running = True
        self._bg = threading.Thread(target=self._loop, daemon=True); self._bg.start()
        print(f"[imu] ws broker on ws://{self.host}:{self.port}/ws")

    def _loop(self):
        while self._running:
            for key, _ in self.sel.select(timeout=0.2):
                if key.data == "srv":
                    client, addr = key.fileobj.accept()
                    # Upgrade path '/ws' לא נדרשת כאן — אנו מבצעים Handshake ישיר
                    if not _handshake(client):
                        client.close(); continue
                    client.setblocking(False)
                    conn = _Conn(client, addr)
                    with self._lock: self.conns.append(conn)
                else:
                    # לא קוראים הודעות מהלקוח (publish הוא בצד השרת); אם נסגר — מנקים.
                    pass
            # ניקוי חיבורים מתים
            dead = []
            with self._lock:
                for c in self.conns:
                    try:
                        c.sock.send(b"")  # no-op keepalive
                    except Exception:
                        dead.append(c)
                for c in dead:
                    try: c.sock.close()
                    except: pass
                    self.conns.remove(c)

    def publish(self, topic: str, payload: Dict[str,Any]):
        msg = json.dumps({"topic": topic, **payload}, ensure_ascii=False)
        frame = _encode_frame_text(msg)
        with self._lock:
            for c in list(self.conns):
                try:
                    c.sock.send(frame)
                except Exception:
                    try: c.sock.close()
                    except: pass
                    self.conns.remove(c)

    def stop(self):
        self._running = False
        try: self.sel.close()
        except: pass
        try: self._srv.close()
        except: pass

# Singleton לשימוש גלובלי
_broker = WSBroker()
def start_ws_broker(host="127.0.0.1", port=8766):
    _broker.host = host; _broker.port = port; _broker.start(); return _broker
def publish(topic: str, payload: Dict[str,Any]):
    _broker.publish(topic, payload)
