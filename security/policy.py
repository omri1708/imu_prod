# security/policy.py
from __future__ import annotations
import ipaddress, os, re, json, hashlib, time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal

NetAction = Literal["connect", "bind", "listen"]
FSAction  = Literal["read", "write", "exec", "list"]

@dataclass
class NetworkRule:
    action: NetAction
    proto: Literal["tcp","udp","any"] = "tcp"
    host_pattern: str = "*"     # supports wildcard or CIDR
    port_range: Tuple[int,int] = (1,65535)

@dataclass
class FileRule:
    action: FSAction
    path_glob: str   # e.g. /var/log/**, ~/projects/**, /tmp/*
    allow_exec: bool = False

@dataclass
class UserPolicy:
    user_id: str
    default_net: Literal["deny","allow"] = "deny"
    default_fs:  Literal["deny","allow"] = "deny"
    net_allow: List[NetworkRule] = field(default_factory=list)
    fs_allow:  List[FileRule]    = field(default_factory=list)
    ttl_seconds: int = 86400
    trust_level: Literal["low","medium","high"] = "medium"
    p95_latency_budget_ms: int = 2500

class PolicyError(RuntimeError): ...
class Denied(RuntimeError): ...

def _host_ok(pattern: str, host: str) -> bool:
    if pattern == "*" or pattern == host: return True
    # CIDR?
    try:
        ip = ipaddress.ip_address(host)
        return ip in ipaddress.ip_network(pattern, strict=False)
    except: pass
    # wildcard
    pat = re.escape(pattern).replace("\\*",".*")
    return re.fullmatch(pat, host) is not None

def check_network(policy: UserPolicy, action: NetAction, host: str, port: int, proto="tcp"):
    for r in policy.net_allow:
        if r.action == action and (r.proto==proto or r.proto=="any") and \
           _host_ok(r.host_pattern, host) and r.port_range[0] <= port <= r.port_range[1]:
            return
    if policy.default_net == "allow": return
    raise Denied(f"network {action} to {host}:{port}/{proto} denied by policy")

def _path_ok(glob_pat: str, path: str) -> bool:
    from fnmatch import fnmatch
    return fnmatch(os.path.abspath(path), os.path.abspath(os.path.expanduser(glob_pat)))

def check_fs(policy: UserPolicy, action: FSAction, path: str, require_exec=False):
    for r in policy.fs_allow:
        if r.action == action and _path_ok(r.path_glob, path):
            if require_exec and not r.allow_exec:
                break
            return
    if policy.default_fs == "allow" and not require_exec:
        return
    raise Denied(f"fs {action} {path} denied by policy")

# provenance helpers
def sha256_file(path:str)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def record_audit(event_type:str, user:str, payload:dict, sink:str="audit.log"):
    payload=dict(payload)
    payload.update({"ts":time.time(),"user":user,"event":event_type})
    with open(sink,"a",encoding="utf-8") as f:
        f.write(json.dumps(payload,ensure_ascii=False)+"\n")