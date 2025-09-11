# imu_repo/engine/key_delegation.py
from __future__ import annotations
import hmac, hashlib, os, time, json
from typing import Dict, Any, Iterable, Tuple, Optional, List

class DelegationError(Exception): ...

def _digest(algo: str):
    try:
        return getattr(hashlib, algo)
    except AttributeError:
        raise DelegationError(f"unsupported hash algo: {algo}")

def _canon(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode("utf-8")

def derive_child_secret_hex(parent_secret_hex: str, child_kid: str, *, salt_hex: Optional[str]=None, algo: str="sha256") -> str:
    if salt_hex is None:
        salt_hex = os.urandom(16).hex()
    parent = bytes.fromhex(parent_secret_hex)
    data = (child_kid + ":" + salt_hex).encode("utf-8")
    mac = hmac.new(parent, data, _digest(algo))
    return mac.hexdigest()

def issue_delegation(parent_kid: str, parent_secret_hex: str, *, child_kid: str, scopes: Iterable[str], exp_epoch: float, salt_hex: Optional[str]=None, algo: str="sha256") -> Dict[str,Any]:
    if salt_hex is None:
        salt_hex = os.urandom(16).hex()
    stmt = {
        "v": 1,
        "parent": parent_kid,
        "child": child_kid,
        "algo": algo,
        "salt_hex": salt_hex,
        "scopes": list(scopes),
        "exp": float(exp_epoch)
    }
    parent = bytes.fromhex(parent_secret_hex)
    mac = hmac.new(parent, _canon(stmt), _digest(algo))
    stmt["sig"] = mac.hexdigest()
    return stmt

def verify_delegation(stmt: Dict[str,Any], parent_secret_hex: str) -> bool:
    algo = (stmt.get("algo") or "sha256").lower()
    sig = (stmt.get("sig") or "").lower()
    m = dict(stmt); m.pop("sig", None)
    mac = hmac.new(bytes.fromhex(parent_secret_hex), _canon(m), _digest(algo))
    return hmac.compare_digest(mac.hexdigest().lower(), sig)

def expand_keyring_with_chain(root_keyring: Dict[str,Dict[str,str]], chain: Iterable[Dict[str,Any]]) -> Dict[str,Dict[str,str]]:
    out = dict(root_keyring)
    parent_secret: Dict[str,Tuple[str,str]] = {kid:(str(meta["secret_hex"]), str(meta.get("algo","sha256")).lower()) for kid,meta in out.items()}
    now = time.time()
    for stmt in chain:
        parent = str(stmt.get("parent")); child = str(stmt.get("child")); exp = float(stmt.get("exp", 0))
        if not parent or not child: raise DelegationError("invalid delegation (missing ids)")
        if exp and now > exp: raise DelegationError(f"delegation expired for child {child}")
        if parent not in parent_secret: raise DelegationError(f"unknown parent {parent} in chain")
        phex, algo = parent_secret[parent]
        if not verify_delegation(stmt, phex): raise DelegationError(f"bad signature for child {child}")
        salt_hex = str(stmt.get("salt_hex"))
        child_secret = derive_child_secret_hex(phex, child_kid=child, salt_hex=salt_hex, algo=algo)
        out[child] = {"secret_hex": child_secret, "algo": algo}
        parent_secret[child] = (child_secret, algo)
    return out

def find_stmt_for_kid(chain: List[Dict[str,Any]], kid: str) -> Dict[str,Any] | None:
    for stmt in chain:
        if str(stmt.get("child")) == kid:
            return stmt
    return None

def enforce_scope_for_kid(chain: List[Dict[str,Any]], kid: str, expected_scope: str) -> None:
    stmt = find_stmt_for_kid(chain, kid)
    if stmt is None:
        raise DelegationError(f"no delegation found for kid '{kid}'")
    scopes = {s.lower() for s in (stmt.get("scopes") or [])}
    if expected_scope.lower() not in scopes:
        raise DelegationError(f"kid '{kid}' lacks required scope '{expected_scope}' (has: {sorted(scopes)})")
    exp = float(stmt.get("exp", 0))
    if exp and time.time() > exp:
        raise DelegationError(f"delegation for kid '{kid}' expired")