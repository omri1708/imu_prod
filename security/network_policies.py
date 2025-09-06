# security/network_policies.py
# Deny-by-default רשת, עם allowlist פר־משתמש + פרמטרי throttling בסיסיים.

from __future__ import annotations
import ipaddress, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class NetRule:
    host: str                 # hostname או CIDR/IP (תומך גם "*.example.com")
    ports: List[int]          # פורטים מותרים
    tls_only: bool = False    # אם True—מאפשר https/wss בלבד

@dataclass
class UserNetPolicy:
    user_id: str
    default_deny: bool = True
    rules: List[NetRule] = field(default_factory=list)
    max_outbound_qps: int = 5
    max_concurrent: int = 10

class NetPolicyDB:
    def __init__(self):
        self._by_user: Dict[str, UserNetPolicy] = {}
    def put(self, policy: UserNetPolicy):
        self._by_user[policy.user_id] = policy
    def get(self, user_id: str) -> Optional[UserNetPolicy]:
        return self._by_user.get(user_id)

POLICY_DB = NetPolicyDB()

def _host_matches(rule_host: str, target_host: str) -> bool:
    # CIDR?
    try:
        net = ipaddress.ip_network(rule_host, strict=False)
        ip = ipaddress.ip_address(target_host)
        return ip in net
    except ValueError:
        # wildcard "*.example.com"
        if rule_host.startswith("*."):
            return target_host.endswith(rule_host[1:])
        return rule_host == target_host

def is_allowed(user_id: str, host: str, port: int, scheme: str = "tcp") -> bool:
    pol = POLICY_DB.get(user_id)
    if not pol:  # אין פוליסי → deny
        return False
    for r in pol.rules:
        if _host_matches(r.host, host) and port in r.ports:
            if r.tls_only and scheme not in ("tls","https","wss"):
                return False
            return True
    return not pol.default_deny

# ברירת מחדל קשיחה למשתמש דמו:
POLICY_DB.put(UserNetPolicy(
    user_id="demo-user",
    default_deny=True,
    rules=[
        NetRule("127.0.0.1", [8000, 8765]),
        NetRule("localhost", [8000, 8765]),
    ],
    max_outbound_qps=10,
    max_concurrent=20
))