# policy/policy_engine.py
# -*- coding: utf-8 -*-
"""
Policy engine: per-user subspace policies (TTL/Trust/p95/Net/FS caps)
Hard enforcement via decorators and explicit checks.
"""
from __future__ import annotations
import time, ipaddress, os, re, threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

class PolicyViolation(Exception): pass
class RateLimited(Exception): pass

TRUST_LEVELS = ("low","medium","high","system")

@dataclass
class Limits:
    cpu_ms_budget: int = 5_000            # wall-clock budget per op
    mem_mb_budget: int = 512              # soft cap; enforce by accounting hooks
    io_read_mb: int = 50
    io_write_mb: int = 50
    net_out_mb: int = 25
    p95_ms_max: int = 800                 # perf SLO per op type
    ttl_seconds: int = 7*24*3600

@dataclass
class NetRule:
    allow: bool
    cidr: Optional[str] = None
    host_regex: Optional[str] = None
    ports: Optional[Set[int]] = None

@dataclass
class FsRule:
    allow: bool
    path_regex: str

@dataclass
class UserPolicy:
    user_id: str
    trust: str = "medium"
    limits: Limits = field(default_factory=Limits)
    net_rules: List[NetRule] = field(default_factory=list)
    fs_rules: List[FsRule] = field(default_factory=list)
    priority: int = 100  # smaller = higher priority in queues

    def check_trust(self):
        if self.trust not in TRUST_LEVELS:
            raise PolicyViolation(f"unknown trust level {self.trust}")

class PolicyStore:
    """In-memory + on-disk json store (atomic replace)."""
    def __init__(self, path: str = ".imu/policies.json"):
        self.path = path
        self._mux = threading.Lock()
        self._cache: Dict[str,UserPolicy] = {}

    def load(self):
        import json, os
        with self._mux:
            if os.path.exists(self.path):
                obj = json.load(open(self.path, "r", encoding="utf-8"))
                self._cache.clear()
                for uid, rec in obj.items():
                    limits = Limits(**rec["limits"])
                    net = [NetRule(**n) for n in rec.get("net_rules",[])]
                    fs = [FsRule(**f) for f in rec.get("fs_rules",[])]
                    self._cache[uid] = UserPolicy(user_id=uid, trust=rec["trust"],
                                                  limits=limits, net_rules=net, fs_rules=fs,
                                                  priority=rec.get("priority",100))
            else:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self._persist()

    def _persist(self):
        import json, tempfile, os, shutil
        tmp = self.path + ".tmp"
        obj = {}
        for uid, p in self._cache.items():
            obj[uid] = {
                "trust": p.trust,
                "limits": p.limits.__dict__,
                "net_rules": [nr.__dict__ for nr in p.net_rules],
                "fs_rules": [fr.__dict__ for fr in p.fs_rules],
                "priority": p.priority
            }
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,indent=2)
        shutil.move(tmp, self.path)

    def get(self, user_id: str) -> UserPolicy:
        with self._mux:
            p = self._cache.get(user_id)
            if not p:
                p = UserPolicy(user_id=user_id)
                self._cache[user_id]=p
                self._persist()
            return p

    def put(self, p: UserPolicy):
        with self._mux:
            p.check_trust()
            self._cache[p.user_id] = p
            self._persist()

policy_store = PolicyStore()

# ---------- Enforcement helpers ----------

def enforce_net(user: UserPolicy, host: str, port: int):
    # allow by explicit allow rule; deny by default
    for r in user.net_rules:
        if r.allow:
            ok_host = True
            if r.host_regex and not re.search(r.host_regex, host, re.I):
                ok_host = False
            if r.cidr:
                try:
                    ipaddress.ip_network(r.cidr)
                    # don't DNS here; treat host as ip string if looks like ip
                    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
                        if ipaddress.ip_address(host) not in ipaddress.ip_network(r.cidr):
                            ok_host = False
                    # if hostname & cidr specified, we allow hostname (policy owner guarantees mapping)
                except Exception:
                    raise PolicyViolation("bad cidr in policy")
            if r.ports and port not in r.ports:
                ok_host = False
            if ok_host:
                return
    raise PolicyViolation(f"net denied: {host}:{port}")

def enforce_fs(user: UserPolicy, path: str, is_write: bool):
    allowed = False
    for r in user.fs_rules:
        if re.search(r.path_regex, path):
            allowed = r.allow
    if not allowed:
        raise PolicyViolation(f"fs denied: {'write' if is_write else 'read'} {path}")

class Budget:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.start = time.time()
        self.io_r = 0
        self.io_w = 0
        self.net_o = 0

    def tick_cpu(self):
        if (time.time()-self.start)*1000 > self.limits.cpu_ms_budget:
            raise PolicyViolation("cpu budget exceeded")

    def add_read(self, mb):
        self.io_r += mb
        if self.io_r > self.limits.io_read_mb: raise PolicyViolation("read budget exceeded")

    def add_write(self, mb):
        self.io_w += mb
        if self.io_w > self.limits.io_write_mb: raise PolicyViolation("write budget exceeded")

    def add_net_out(self, mb):
        self.net_o += mb
        if self.net_o > self.limits.net_out_mb: raise PolicyViolation("net out budget exceeded")