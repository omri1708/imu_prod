# distributed/raft.py
# imu_repo/distributed/raft.py
from __future__ import annotations
import asyncio, random, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from grounded.claims import current

# -----------------------------
# Raft Types
# -----------------------------
@dataclass
class LogEntry:
    term: int
    cmd: Tuple[str, str, str]  # ("put", key, val)

@dataclass
class AppendEntries:
    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int

@dataclass
class AppendResp:
    term: int
    success: bool
    match_index: int

@dataclass
class RequestVote:
    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int

@dataclass
class VoteResp:
    term: int
    vote_granted: bool

# -----------------------------
# In-memory transport (cluster)
# -----------------------------
class Transport:
    def __init__(self):
        self.queues: Dict[int, asyncio.Queue] = {}

    def register(self, node_id: int):
        self.queues[node_id] = asyncio.Queue()

    async def send(self, to_id: int, msg):
        await self.queues[to_id].put(msg)

    async def recv(self, node_id: int):
        return await self.queues[node_id].get()

# -----------------------------
# Node
# -----------------------------
FOLLOWER = "follower"
CANDIDATE = "candidate"
LEADER = "leader"

class Node:
    def __init__(self, node_id: int, peers: List[int], transport: Transport, *,
                 election_range_ms: Tuple[int,int]=(250, 400), hb_ms: int=75):
        self.id = node_id
        self.peers = [p for p in peers if p != node_id]
        self.t = transport
        # Raft persistent/volatile
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[LogEntry] = [LogEntry(0, ("noop","_","_"))]  # index 0 sentinel
        self.commit_index = 0
        self.last_applied = 0

        # Leader state
        self.next_index: Dict[int,int] = {}
        self.match_index: Dict[int,int] = {}

        # State & timers
        self.state = FOLLOWER
        self.leader_id: Optional[int] = None
        self._election_range_ms = election_range_ms
        self._hb_ms = hb_ms

        # KV state machine
        self.kv: Dict[str,str] = {}

        # control
        self._stop = asyncio.Event()
        self._task = None

    def last_log_index(self) -> int:
        return len(self.log) - 1

    def last_log_term(self) -> int:
        return self.log[-1].term

    def reset_election_deadline(self):
        ms = random.randint(*self._election_range_ms)
        self._deadline = time.time() + ms/1000.0

    async def start(self):
        self.reset_election_deadline()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self):
        while not self._stop.is_set():
            # role loop
            if self.state in (FOLLOWER, CANDIDATE):
                await self._step_follower_candidate()
            elif self.state == LEADER:
                await self._step_leader()
            else:
                await asyncio.sleep(0.01)

    async def _step_follower_candidate(self):
        try:
            timeout = max(0.0, self._deadline - time.time())
            msg = await asyncio.wait_for(self.t.recv(self.id), timeout=timeout)
            await self._handle(msg)
        except asyncio.TimeoutError:
            # election timeout
            await self._start_election()

    async def _step_leader(self):
        # periodic heartbeat
        await self._broadcast_heartbeat()
        # process messages quickly
        await asyncio.sleep(self._hb_ms/1000.0)
        while True:
            try:
                msg = self.t.queues[self.id].get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._handle(msg)

    async def _start_election(self):
        self.state = CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.leader_id = None
        votes = 1  # self
        total = len(self.peers) + 1
        current().add_evidence("raft_election_start", {
            "source_url": f"raft://{self.id}", "trust": 0.9, "ttl_s": 600,
            "payload": {"term": self.current_term}
        })
        rv = RequestVote(self.current_term, self.id, self.last_log_index(), self.last_log_term())
        for p in self.peers:
            await self.t.send(p, rv)
        self.reset_election_deadline()
        # collect votes until win/lose/timeout
        while time.time() < self._deadline and self.state == CANDIDATE:
            try:
                msg = await asyncio.wait_for(self.t.recv(self.id), timeout=0.02)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, VoteResp):
                if msg.term > self.current_term:
                    self._become_follower(msg.term, None)
                    return
                if msg.vote_granted and self.state == CANDIDATE:
                    votes += 1
                    if votes > total//2:
                        await self._become_leader()
                        return
            else:
                await self._handle(msg)

    async def _become_leader(self):
        self.state = LEADER
        self.leader_id = self.id
        # init leader state
        last = self.last_log_index()
        self.next_index = {p: last+1 for p in self.peers}
        self.match_index = {p: 0 for p in self.peers}
        current().add_evidence("raft_elected_leader", {
            "source_url": f"raft://{self.id}", "trust": 0.95, "ttl_s": 600,
            "payload": {"term": self.current_term}
        })

    def _become_follower(self, term: int, leader_id: Optional[int]):
        self.state = FOLLOWER
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        self.leader_id = leader_id
        self.reset_election_deadline()

    async def _broadcast_heartbeat(self):
        ae = AppendEntries(
            term=self.current_term,
            leader_id=self.id,
            prev_log_index=self.last_log_index(),
            prev_log_term=self.last_log_term(),
            entries=[],  # heartbeat
            leader_commit=self.commit_index
        )
        for p in self.peers:
            await self.t.send(p, ae)

    async def _handle(self, msg):
        if isinstance(msg, RequestVote):
            await self._handle_request_vote(msg)
        elif isinstance(msg, VoteResp):
            # handled in election loop
            pass
        elif isinstance(msg, AppendEntries):
            await self._handle_append_entries(msg)
        elif isinstance(msg, AppendResp):
            # handled in replicate/propose
            pass

    async def _handle_request_vote(self, m: RequestVote):
        # term checks
        if m.term < self.current_term:
            await self.t.send(m.candidate_id, VoteResp(self.current_term, False))
            return
        if m.term > self.current_term:
            self._become_follower(m.term, None)
        up_to_date = (m.last_log_term > self.last_log_term()) or (
            m.last_log_term == self.last_log_term() and m.last_log_index >= self.last_log_index()
        )
        can_vote = (self.voted_for in (None, m.candidate_id)) and up_to_date
        if can_vote:
            self.voted_for = m.candidate_id
            self.reset_election_deadline()
        await self.t.send(m.candidate_id, VoteResp(self.current_term, can_vote))

    async def _handle_append_entries(self, m: AppendEntries):
        if m.term < self.current_term:
            await self.t.send(m.leader_id, AppendResp(self.current_term, False, self.last_log_index()))
            return
        # step down to follower if needed
        self._become_follower(m.term, m.leader_id)
        # check prev
        if m.prev_log_index > self.last_log_index():
            await self.t.send(m.leader_id, AppendResp(self.current_term, False, self.last_log_index()))
            return
        if m.prev_log_index >= 0 and self.log[m.prev_log_index].term != m.prev_log_term:
            # conflict — truncate
            del self.log[m.prev_log_index+1:]
            await self.t.send(m.leader_id, AppendResp(self.current_term, False, self.last_log_index()))
            return
        # append new
        if m.entries:
            self.log.extend(m.entries)
            current().add_evidence("raft_append", {
                "source_url": f"raft://{self.id}", "trust": 0.92, "ttl_s": 600,
                "payload": {"count": len(m.entries), "last_index": self.last_log_index(), "term": self.current_term}
            })
        # advance commit
        if m.leader_commit > self.commit_index:
            self.commit_index = min(m.leader_commit, self.last_log_index())
            await self._apply_committed()
        await self.t.send(m.leader_id, AppendResp(self.current_term, True, self.last_log_index()))

    async def _apply_committed(self):
        # apply log entries to KV
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            e = self.log[self.last_applied]
            if e.cmd[0] == "put":
                _, k, v = e.cmd
                self.kv[k] = v
                current().add_evidence("raft_apply", {
                    "source_url": f"raft://{self.id}", "trust": 0.93, "ttl_s": 600,
                    "payload": {"index": self.last_applied, "k": k}
                })

    async def propose_put(self, key: str, val: str) -> bool:
        """
        לקוח מציע פקודת PUT. רק מנהיג יקבל.
        משכפל לרוב, מקדם commit, מחזיר True אם בוצע.
        """
        if self.state != LEADER:
            return False
        entry = LogEntry(self.current_term, ("put", key, val))
        self.log.append(entry)
        my_index = self.last_log_index()
        current().add_evidence("raft_propose", {
            "source_url": f"raft://{self.id}", "trust": 0.94, "ttl_s": 600,
            "payload": {"index": my_index, "k": key}
        })
        # replicate to followers until majority matches
        acks = 1  # self
        majority = (len(self.peers)+1)//2 + 1
        pending = set(self.peers)
        # initialize next_index if not yet
        for p in self.peers:
            self.next_index.setdefault(p, my_index)
        # send loop
        while pending and not self._stop.is_set():
            tasks = []
            for p in list(pending):
                prev_i = my_index-1
                ae = AppendEntries(self.current_term, self.id, prev_i, self.log[prev_i].term, [entry], self.commit_index)
                tasks.append((p, ae))
            # send batch
            for p, ae in tasks:
                await self.t.send(p, ae)
            # wait a bit for responses
            deadline = time.time()+0.2
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(self.t.recv(self.id), timeout=0.02)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, AppendResp):
                    if msg.term > self.current_term:
                        self._become_follower(msg.term, None)
                        return False
                    if msg.success and msg.match_index >= my_index and msg.term == self.current_term:
                        if msg in pending:
                            # won't happen; track by peer count
                            pass
                        # count an ack per response that matches index
                        acks += 1
                        pending.discard(next((p for p,_ in tasks), None))
                        if acks >= majority:
                            # commit & apply
                            self.commit_index = my_index
                            await self._apply_committed()
                            current().add_evidence("raft_commit", {
                                "source_url": f"raft://{self.id}", "trust": 0.96, "ttl_s": 600,
                                "payload": {"index": my_index, "k": key, "majority": acks}
                            })
                            return True
                else:
                    await self._handle(msg)
        return False

# -----------------------------
# Cluster helper
# -----------------------------
class Cluster:
    def __init__(self, n: int = 3):
        self.t = Transport()
        self.nodes: List[Node] = []
        ids = list(range(n))
        for i in ids:
            self.t.register(i)
        for i in ids:
            node = Node(i, ids, self.t)
            self.nodes.append(node)

    async def start(self):
        await asyncio.gather(*(n.start() for n in self.nodes))

    async def stop(self):
        await asyncio.gather(*(n.stop() for n in self.nodes))

    def leader(self) -> Optional[Node]:
        for n in self.nodes:
            if n.state == LEADER:
                return n
        return None

    async def wait_for_leader(self, timeout_s: float = 3.0) -> Optional[Node]:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ld = self.leader()
            if ld:
                return ld
            await asyncio.sleep(0.02)
        return None