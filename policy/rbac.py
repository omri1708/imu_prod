# policy/rbac.py
# RBAC מינימלי: משתמש -> רשימת תפקידים; תפקיד -> סט הרשאות (עם תמיכת * וויילדקארד).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import fnmatch

@dataclass
class Role:
    name: str
    permissions: List[str] = field(default_factory=list)  # e.g., "adapter:run:*", "runbook:unity_k8s", "keys:*"

@dataclass
class UserRoles:
    user_id: str
    roles: List[str] = field(default_factory=list)

class RBAC:
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, UserRoles] = {}

    def put_role(self, name: str, perms: List[str]):
        self.roles[name] = Role(name=name, permissions=list(sorted(set(perms))))

    def grant(self, user_id: str, role_name: str):
        ur = self.users.setdefault(user_id, UserRoles(user_id=user_id))
        if role_name not in ur.roles:
            ur.roles.append(role_name)

    def list_user_perms(self, user_id: str) -> List[str]:
        ur = self.users.get(user_id)
        if not ur: return []
        perms: List[str] = []
        for r in ur.roles:
            perms += self.roles.get(r, Role(r,[])).permissions
        # normalize + dedup
        seen=set(); out=[]
        for p in perms:
            if p not in seen: out.append(p); seen.add(p)
        return out

    def allow(self, user_id: str, permission: str) -> bool:
        perms = self.list_user_perms(user_id)
        for pat in perms:
            if fnmatch.fnmatch(permission, pat):
                return True
        return False

RBAC_DB = RBAC()

# תפקידי ברירת מחדל:
RBAC_DB.put_role("admin", ["*"])
RBAC_DB.put_role("dev", [
    "capabilities:request",
    "adapter:dry_run:*",
    "adapter:run:*",
    "runbook:*",
    "events:publish",
    "metrics:read",
    "sbom:view",
])
RBAC_DB.put_role("viewer", [
    "adapter:dry_run:*",
    "metrics:read",
    "events:poll",
    "sbom:view",
])

# הענקות ברירת מחדל: DEMO
RBAC_DB.grant("demo-user", "admin")

def require_perm(user_id: str, permission: str):
    if not RBAC_DB.allow(user_id, permission):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail=f"rbac_denied:{permission}")