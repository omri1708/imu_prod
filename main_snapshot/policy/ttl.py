# policy/ttl.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, Literal
from policy.policies import UserPolicy

Kind = Literal["claim","evidence","artifact","session"]

def ttl_for(policy: UserPolicy, kind: Kind) -> int:
    return policy.ttl_seconds.get(kind, 24*3600)

def is_expired(now_ts: float, created_ts: float, ttl_sec: int) -> bool:
    return (now_ts - created_ts) > ttl_sec

def enforce_ttl(index, policy: UserPolicy, now_ts: float):
    """
    index implements .iter_docs(kind)->Iterable[(doc_id, created_ts)] and .delete(doc_id)
    Hard delete expired entries according to per-kind ttl
    """
    removed = {"claim":0,"evidence":0,"artifact":0,"session":0}
    for kind in removed.keys():
        ttl_sec = ttl_for(policy, kind) 
        for doc_id, created_ts in index.iter_docs(kind):
            if is_expired(now_ts, created_ts, ttl_sec):
                index.delete(doc_id)
                removed[kind]+=1
    return removed