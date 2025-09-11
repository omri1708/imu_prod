# imu_repo/engine/realtime_and_dist.py
from __future__ import annotations
import asyncio, json, time
from typing import Dict, Any, Callable, Awaitable
from rt.async_runtime import AsyncSupervisor
from rt.queue import DurableQueue
from dist.service_registry import ServiceRegistry
from dist.router import Router
from dist.lease_quorum import InMemoryNet, Node

class MicroRuntime:
    """
    Runtime משותף:
      - תור עמיד requests
      - רג'יסטרי שירותים + ראוטינג
      - consensus מינימלי (lease quorum) לרשומות מערכתיות
    """
    def __init__(self, *, q_root="/mnt/data/imu_repo/rtq", q_name="requests"):
        self.q = DurableQueue(root=q_root, name=q_name)
        self.sup = AsyncSupervisor(default_deadline_s=2.0, max_concurrency=200)
        self.reg = ServiceRegistry()
        self.router = Router(self.reg)
        self.net = InMemoryNet()
        self.nodes: Dict[str, Node] = {}

    def spawn_node(self, nid: str, peers: list[str], *, lease_s: float = 2.0) -> Node:
        n = Node(nid, peers, self.net, lease_s=lease_s)
        self.nodes[nid] = n
        return n

    async def elect_leader(self) -> str:
        # מנסים לבחור אחד מהצמתים (תקף לסימולציה בתהליך)
        for nid, node in self.nodes.items():
            ok = await node.elect()
            if ok:
                return nid
        # אם לא הצליח, ננסה שוב
        for nid, node in self.nodes.items():
            ok = await node.elect()
            if ok:
                return nid
        raise RuntimeError("no_leader")

    async def write_consensus(self, entry: Dict[str,Any]) -> None:
        # דורש מנהיג עם lease
        # נבחר/נאשר מנהיג
        leader_id = None
        for nid, node in self.nodes.items():
            if node.leader == nid and node._lease_active():
                leader_id = nid; break
        if leader_id is None:
            leader_id = await self.elect_leader()
        node = self.nodes[leader_id]
        ok = await node.append(entry)
        if not ok:
            raise RuntimeError("consensus_append_failed")

    def register_service(self, name: str, inst_id: str, addr: str, *, meta: Dict[str,Any]|None=None):
        self.reg.register(name, inst_id, addr, meta=meta or {})

    async def submit(self, payload: Dict[str,Any]) -> str:
        return self.q.put(payload)

    async def worker(self, service: str, handler: Callable[[Dict[str,Any]], Awaitable[Dict[str,Any]]], *, poll_ms: int=50):
        while True:
            item = self.q.get()
            if not item:
                await asyncio.sleep(poll_ms/1000.0)
                continue
            mid, payload = item
            try:
                await self.sup.retry(lambda: handler(payload), attempts=3, initial_backoff_s=0.05)
                self.q.ack(mid)
            except Exception:
                self.q.nack(mid)
                await asyncio.sleep(0.05)