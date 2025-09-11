# server/events/bus.py
import asyncio
from enum import Enum
from typing import Dict, Any, Callable, Awaitable, Optional

class Topic(str, Enum):
    TELEMETRY = "telemetry"

class EventBus:
    def __init__(self):
        self._subs = {Topic.TELEMETRY: []}
        self._push_hook: Optional[Callable[[Dict[str,Any]], Awaitable[None]]] = None

    def subscribe(self, topic: Topic) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[topic].append(q)
        return q

    def set_push_hook(self, hook):
        self._push_hook = hook

    def emit(self, topic: Topic, event: Dict[str, Any]):
        # local fanout
        for q in self._subs.get(topic, []):
            q.put_nowait(event)
        # push hook (e.g., WS)
        if self._push_hook:
            asyncio.create_task(self._push_hook(event))