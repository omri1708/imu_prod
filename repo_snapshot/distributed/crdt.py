# imu_repo/distributed/crdt.py
from __future__ import annotations
import time
from typing import Dict, Any, Set

class CRDTError(Exception): ...

class GCounter:
    """Grow-only counter."""
    def __init__(self,node_id:str):
        self.node_id=node_id
        self.counts:Dict[str,int]={node_id:0}

    def inc(self,n:int=1):
        self.counts[self.node_id]=self.counts.get(self.node_id,0)+n

    def value(self)->int:
        return sum(self.counts.values())

    def merge(self,other:"GCounter"):
        for k,v in other.counts.items():
            self.counts[k]=max(self.counts.get(k,0),v)

class GSet:
    """Grow-only set."""
    def __init__(self):
        self.items:Set[Any]=set()

    def add(self,v:Any): self.items.add(v)
    def value(self)->Set[Any]: return self.items
    def merge(self,other:"GSet"): self.items|=other.items

class LWWMap:
    """Last-Write-Wins map."""
    def __init__(self):
        self.store:Dict[str,tuple[Any,float]]={}

    def put(self,key:str,value:Any):
        self.store[key]=(value,time.time())

    def get(self,key:str)->Any:
        return self.store.get(key,(None,0))[0]

    def merge(self,other:"LWWMap"):
        for k,(v,t) in other.store.items():
            if k not in self.store or self.store[k][1]<t:
                self.store[k]=(v,t)
