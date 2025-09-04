# imu_repo/adapters/redis_resp.py
from __future__ import annotations
import socket, time
from typing import Optional, Tuple, List, Any



class RESPError(Exception): ...


def _enc_bulk(s: str) -> bytes:
    b = s.encode("utf-8")
    return b"$%d\r\n%s\r\n" % (len(b), b)


def _enc_array(cmd: List[str]) -> bytes:
    out = b"*%d\r\n" % len(cmd)
    for c in cmd: out += _enc_bulk(c)
    return out


def _readline(sock) -> bytes:
    buf=b""
    while not buf.endswith(b"\r\n"):
        x=sock.recv(1)
        if not x: break
        buf+=x
    return buf[:-2]


def _readbulk(sock, n: int) -> bytes:
    data=b""
    while len(data)<n:
        data+=sock.recv(n-len(data))
    # read CRLF
    sock.recv(2)
    return data


def _parse(sock) -> Any:
    t = sock.recv(1)
    if not t: raise RESPError("eof")
    if t == b"+":  # simple string
        return _readline(sock).decode()
    if t == b"-":  # error
        raise RESPError(_readline(sock).decode())
    if t == b":":  # integer
        return int(_readline(sock).decode())
    if t == b"$":  # bulk
        n = int(_readline(sock).decode())
        if n==-1: return None
        return _readbulk(sock, n).decode()
    if t == b"*":  # array
        n = int(_readline(sock).decode())
        arr=[]
        for _ in range(n): arr.append(_parse(sock))
        return arr
    raise RESPError("unknown_type")



class RedisResp:
    """
    לקוח RESP מינימלי ל-Redis (PING, SET, GET, LPUSH, BRPOP).
    לא דורש redis-py.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 6379, timeout: float = 1.0):
        self.addr=(host,port); 
        self.host=host; self.port=int(port); self.timeout=float(timeout)
    
    def _call(self, *cmd: str) -> Any:
        sock=socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.sendall(_enc_array(list(cmd)))
        res=_parse(sock)
        sock.close()
        return res

    def ping(self) -> str:
        return self._call("PING")

    def set(self, key: str, val: str) -> str:
        return self._call("SET", key, val)

    def get(self, key: str) -> Optional[str]:
        return self._call("GET", key)
    
    def _cmd(self, *args: bytes) -> bytes:
        with socket.create_connection(self.addr, timeout=self.timeout) as s:
            arr = [f"*{len(args)}\r\n".encode()]
            for a in args:
                arr.append(f"${len(a)}\r\n".encode()); arr.append(a + b"\r\n")
            s.sendall(b"".join(arr))
            return self._read_resp(s)

    def _readline(self, s: socket.socket) -> bytes:
        buf=bytearray()
        while True:
            b=s.recv(1)
            if not b: break
            buf.extend(b)
            if len(buf)>=2 and buf[-2:]==b"\r\n": break
        return bytes(buf)

    def _read_bulk(self, s: socket.socket, n: int) -> bytes:
        data=b""
        while len(data) < n:
            ch=s.recv(n-len(data))
            if not ch: break
            data+=ch
        s.recv(2) # \r\n
        return data

    def _read_resp(self, s: socket.socket) -> bytes:
        line=self._readline(s)
        if not line: return b""
        t=line[:1]
        if t==b"+":  # Simple String
            return line[1:-2]
        if t==b"-":  # Error
            return line
        if t==b":":  # Integer
            return line[1:-2]
        if t==b"$":  # Bulk
            ln=int(line[1:-2])
            if ln==-1: return b""
            return self._read_bulk(s, ln)
        if t==b"*":  # Array
            count=int(line[1:-2])
            out=[]
            for _ in range(count):
                head=self._readline(s)
                if head[:1]==b"$":
                    ln=int(head[1:-2]); out.append(self._read_bulk(s, ln) if ln!=-1 else b"")
                else:
                    out.append(head.strip())
            return b"||".join(out)
        return line

    # public ops
    def lpush(self,key:str,val:str)->bool: return self._cmd(b"LPUSH", key.encode(), val.encode()).isdigit()
    
    def brpop(self,key:str, timeout_s:int=1)->Optional[str]:
        rv=self._cmd(b"BRPOP", key.encode(), str(timeout_s).encode())
        if not rv: return None
        parts=rv.split(b"||")
        if len(parts)>=2:
            return parts[-1].decode()
        return None
