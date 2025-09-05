# tests/test_ws_broker_local.py
# -*- coding: utf-8 -*-
import json, socket, threading, time
from realtime.ws_broker import start_ws_broker, publish, _encode_frame_text

def _fake_client():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 8766))
    # handshake
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    req = ("GET /ws HTTP/1.1\r\n"
           "Host: localhost\r\n"
           "Upgrade: websocket\r\n"
           "Connection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n")
    s.send(req.encode())
    s.recv(4096)  # handshake resp
    s.settimeout(2.0)
    return s

def _read_frame_text(sock: socket.socket) -> str:
    # דקוד טקסט (פשוט): מניחים len<=125, ללא מסיכה (צד שרת שולח ללא מסיכה)
    h = sock.recv(2)
    if not h: return ""
    l = h[1] & 0x7F
    data = sock.recv(l)
    return data.decode("utf-8")

def test_broker_broadcast():
    start_ws_broker("127.0.0.1", 8766)
    c = _fake_client()
    publish("progress/build1", {"value": 42})
    time.sleep(0.1)
    msg = _read_frame_text(c)
    assert "progress/build1" in msg and "42" in msg
    c.close()