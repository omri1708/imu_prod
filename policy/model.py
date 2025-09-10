# policy/model.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import ipaddress


Trust = int  # 0..100

@dataclass
class RateBudget:
    qps: float
    burst: int

@dataclass
class P95Budget:
    # סף זמן תגובה מותר במילישניות – לפי סוג פעולה
    limits_ms: Dict[str, int]  # e.g. {"respond": 900, "adapter.run": 2500}

@dataclass
class TTLPolicy:
    # TTL תכנים/ראיות פר סוג וחוזק־מקור
    # example: {("evidence","high"): 86400, ("profile","low"): 3600}
    sec: Dict[Tuple[str,str], int]

@dataclass
class NetPolicy:
    allow_hosts: List[str] = field(default_factory=list)  # דומיינים/‏IP/CIDR מותרים
    deny_hosts: List[str] = field(default_factory=list)
    max_concurrent: int = 64
    per_host_qps: float = 10.0
    per_host_burst: int = 5
    outbound_block_default: bool = True  # ברירת מחדל: לחסום כלום עד whitelist

@dataclass
class FilePolicy:
    # allow-list מדוייק – absolute prefixes בלבד
    allow_paths: List[str] = field(default_factory=list)
    deny_paths: List[str] = field(default_factory=list)
    max_file_mb: int = 128
    read_only: bool = True

@dataclass
class UserPolicy:
    user_id: str
    trust: Trust
    rate: RateBudget
    p95: P95Budget
    ttl: TTLPolicy
    net: NetPolicy
    files: FilePolicy
    strict_grounding: bool = True           # ללא Evidence → בלוק
    min_source_trust: str = "medium"        # low/medium/high
    provenance_required: bool = True
    require_signed_artifacts: bool = True

def _host_in_list(host: str, items: List[str]) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        for it in items:
            if "/" in it:
                if ip in ipaddress.ip_network(it, strict=False):
                    return True
            else:
                # literal IP match
                if host == it:
                    return True
        return False
    except ValueError:
        # not an IP – treat as domain suffix match
        for it in items:
            it = it.lower()
            if host.lower()==it or host.lower().endswith("."+it):
                return True
        return False

def check_host_allowed(host: str, pol: NetPolicy) -> bool:
    if pol.outbound_block_default:
        if _host_in_list(host, pol.allow_hosts):
            return True
        return False
    else:
        if _host_in_list(host, pol.deny_hosts):
            return False
        return True

# תבניות מדיניות מומלצות:
def default_user_policy(user_id: str, trust: Trust=70) -> UserPolicy:
    return UserPolicy(
        user_id=user_id,
        trust=trust,
        rate=RateBudget(qps=2.0, burst=10),
        p95=P95Budget(limits_ms={"respond": 1200, "adapter.run": 4000, "adapter.dry_run": 2000}),
        ttl=TTLPolicy(sec={
            ("evidence","high"): 7*24*3600,
            ("evidence","medium"): 24*3600,
            ("evidence","low"): 12*3600,
            ("profile","high"): 90*24*3600,
            ("profile","low"): 7*24*3600,
        }),
        net=NetPolicy(
            allow_hosts=[
                "127.0.0.1", "::1", "localhost",
                "api.github.com", "storage.googleapis.com",
                "registry.npmjs.org", "pypi.org", "files.pythonhosted.org",
            ],
            deny_hosts=[],
            max_concurrent=64,
            per_host_qps=5.0,
            per_host_burst=3,
            outbound_block_default=True
        ),
        files=FilePolicy(
            allow_paths=["/tmp/imu/", "./artifacts/", "./workspace/"],
            deny_paths=["/etc/", "/var/lib/"],
            max_file_mb=256,
            read_only=False
        ),
        strict_grounding=True,
        min_source_trust="medium",
        provenance_required=True,
        require_signed_artifacts=True
    )