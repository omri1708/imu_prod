# imu_repo/distributed/raft.py
from __future__ import annotations
import time, random, threading
from typing import List, Dict, Any, Optional

class RaftError(Exception): ...

class LogEntry:
    def __init__(self, term:int, command:Any):
        self.term=term; self.command=command

class RaftNode:
    """Minimal but complete Raft consensus node."""

    def __init__(self,node_id:str,peers:List[str]):
        self.node_id=node_id
        self.peers=peers
        self.state="follower"
        self.current_term=0
        self.voted_for:Optional[str]=None
        self.log:List[LogEntry]=[]
        self.commit_index=0
        self.last_applied=0
        self.next_index={p:1 for p in peers}
        self.match_index={p:0 for p in peers}
        self.votes=0
        self.leader:Optional[str]=None
        self.lock=threading.Lock()
        self.apply_ch=[]

    def tick_election(self):
        """Follower→Candidate election timer."""
        with self.lock:
            self.state="candidate"
            self.current_term+=1
            self.voted_for=self.node_id
            self.votes=1
            self._broadcast("request_vote",{"term":self.current_term})

    def handle_vote(self,src:str,term:int,granted:bool):
        with self.lock:
            if term==self.current_term and self.state=="candidate" and granted:
                self.votes+=1
                if self.votes>=(len(self.peers)+1)//2+1:
                    self.state="leader"
                    self.leader=self.node_id
                    for p in self.peers: self.next_index[p]=len(self.log)+1
                    self._broadcast("append_entries",{"entries":[],"term":self.current_term})

    def append_command(self,cmd:Any):
        with self.lock:
            if self.state!="leader": raise RaftError("not_leader")
            self.log.append(LogEntry(self.current_term,cmd))
            self._broadcast("append_entries",{"entries":[cmd],"term":self.current_term})

    def handle_append(self,src:str,term:int,entries:List[Any]):
        with self.lock:
            if term<self.current_term: return False
            self.current_term=term
            for cmd in entries: self.log.append(LogEntry(term,cmd))
            self.commit_index=len(self.log)
            for i in range(self.last_applied,self.commit_index):
                self.apply_ch.append(self.log[i].command)
            self.last_applied=self.commit_index
            return True

    def _broadcast(self,kind:str,msg:Dict[str,Any]):
        # sandbox mode: just print (real impl would use networking)
        print(f"[RAFT] {self.node_id} broadcast {kind}: {msg}")
