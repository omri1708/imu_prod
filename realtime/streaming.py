# imu_repo/realtime/streaming.py
from __future__ import annotations
import asyncio, time
from typing import Callable, Any, Dict, List

class StreamError(Exception): ...

class Stream:
    """Async publish/subscribe stream with throttling."""
    def __init__(self,name:str,max_rate:float=100.0):
        self.name=name
        self.subscribers:List[Callable[[Any],None]]=[]
        self.last_emit=0.0
        self.max_rate=max_rate  # messages per second

    def subscribe(self,cb:Callable[[Any],None]):
        if cb not in self.subscribers:
            self.subscribers.append(cb)

    async def publish(self,msg:Any):
        now=time.time()
        if now-self.last_emit < 1.0/self.max_rate:
            raise StreamError("throttled")
        self.last_emit=now
        for cb in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(msg)
                else:
                    cb(msg)
            except Exception as e:
                raise StreamError(f"subscriber_failed:{e}")

class StreamManager:
    """Manage multiple named streams."""
    def __init__(self):
        self.streams:Dict[str,Stream]={}

    def get(self,name:str)->Stream:
        if name not in self.streams:
            self.streams[name]=Stream(name)
        return self.streams[name]
