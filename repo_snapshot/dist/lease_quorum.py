# imu_repo/dist/lease_quorum.py
from __future__ import annotations
import asyncio, time, random
from typing import Dict, Any, List, Optional, Callable

class QuorumError(Exception): ...
class LeaseRejected(QuorumError): ...
class NotLeader(QuorumError): ...

class InMemoryNet:
    """סימולציית רשת בתהליך יחיד: ערוצים בין-צמתים."""
    def __init__(self): 
        self.channels: Dict[str, asyncio.Queue] = {}

    def get(self, nid: str) -> asyncio.Queue:
        if nid not in self.channels: self.channels[nid] = asyncio.Queue()
        return self.channels[nid]

    async def send(self, to: str, msg: Dict[str,Any]) -> None:
        await self.get(to).put(msg)

class Node:
    """
    קונצנזוס פשטני:
    - כל צומת מצביע על מועמד עם (term, candidate_id).
    - בחירה: דורשת רוב (quorum) ו-lease עד T.
    - anti split-brain: מונוטוניות term + כבוד ל-lease פעיל (לא תבחר אם לא פג).
    - כתיבה: דורשת ack מרוב עם אותו term.
    """
    def __init__(self, nid: str, peers: List[str], net: InMemoryNet, *, lease_s: float = 2.5):
        self.nid = nid
        self.peers = [p for p in peers if p != nid]
        self.net = net
        self.lease_s = float(lease_s)
        self.term = 0
        self.leader: Optional[str] = None
        self.lease_until = 0.0
        self.log: List[Dict[str,Any]] = []
        self.running = True

    def _quorum(self) -> int:
        return (len(self.peers)+1)//2 + 1

    def _lease_active(self) -> bool:
        return time.time() < self.lease_until

    async def _broadcast(self, msg: Dict[str,Any]) -> List[Dict[str,Any]]:
        out=[]
        for p in self.peers:
            await self.net.send(p, msg)
        return out

    async def start(self):
        """לולאת קליטת הודעות."""
        ch = self.net.get(self.nid)
        while self.running:
            try:
                msg = await asyncio.wait_for(ch.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            t = msg.get("type")
            if t == "vote_req":
                await self._on_vote_req(msg)
            elif t == "append":
                await self._on_append(msg)

    async def _on_vote_req(self, msg: Dict[str,Any]):
        cand = msg["candidate"]
        term = int(msg["term"])
        now = time.time()
        if self._lease_active() and self.leader and self.leader != cand:
            # מכבדים lease קיים → דוחים
            await self.net.send(cand, {"type":"vote_resp","from":self.nid,"granted":False,"term":self.term})
            return
        if term > self.term:
            self.term = term
            self.leader = None
        granted = (term >= self.term)
        if granted:
            self.term = term
        await self.net.send(cand, {"type":"vote_resp","from":self.nid,"granted":granted,"term":self.term})

    async def _on_append(self, msg: Dict[str,Any]):
        leader = msg["leader"]; term= int(msg["term"]); entry = msg["entry"]
        if term < self.term:
            await self.net.send(leader, {"type":"append_ack","from":self.nid,"ok":False,"term":self.term})
            return
        self.term = term
        self.leader = leader
        self.lease_until = time.time() + self.lease_s
        self.log.append(entry)
        await self.net.send(leader, {"type":"append_ack","from":self.nid,"ok":True,"term":self.term})

    async def elect(self) -> bool:
        """מבקש קולות; מצליח רק אם אין lease מתחרה פעיל ורוב מצביעים."""
        if self._lease_active() and self.leader == self.nid:
            return True  # כבר מנהיג
        self.term += 1
        granted = 1  # הקול של עצמי
        futs=[]
        for p in self.peers:
            await self.net.send(p, {"type":"vote_req","candidate":self.nid,"term":self.term})
        # אוספים תשובות
        deadline = time.time() + 0.8
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self.net.get(self.nid).get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "vote_resp" and msg.get("term") == self.term:
                if msg.get("granted"): granted += 1
        if granted >= self._quorum():
            self.leader = self.nid
            self.lease_until = time.time() + self.lease_s
            return True
        return False

    async def append(self, entry: Dict[str,Any]) -> bool:
        """כתיבה עם quorum acks. חייב להיות מנהיג עם lease פעיל."""
        if self.leader != self.nid or not self._lease_active():
            raise NotLeader(f"nid={self.nid} leader={self.leader} lease_active={self._lease_active()}")
        acks = 1  # עצמי
        for p in self.peers:
            await self.net.send(p, {"type":"append","leader":self.nid,"term":self.term,"entry":entry})
        deadline = time.time() + 0.8
        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self.net.get(self.nid).get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if msg.get("type") == "append_ack" and msg.get("term") == self.term and msg.get("ok"):
                acks += 1
                if acks >= self._quorum():
                    self.log.append(entry)
                    self.lease_until = time.time() + self.lease_s
                    return True
        return False