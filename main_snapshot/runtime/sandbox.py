# runtime/sandbox.py
import os, socket, functools, errno
from typing import Callable
from .errors import PolicyViolation
from policy.model import UserPolicy, check_host_allowed

# קבצים
def enforce_file_access(policy: UserPolicy, path: str, write: bool=False, size_mb: int=0):
    ap = policy.files
    abs_p = os.path.abspath(path)
    for d in ap.deny_paths:
        if abs_p.startswith(os.path.abspath(d)):
            raise PolicyViolation(f"file path denied: {path}")
    if write and ap.read_only:
        raise PolicyViolation("file writes disabled by policy")
    if ap.allow_paths:
        if not any(abs_p.startswith(os.path.abspath(p)) for p in ap.allow_paths):
            raise PolicyViolation(f"path not in allowlist: {path}")
    if size_mb and size_mb > ap.max_file_mb:
        raise PolicyViolation(f"file too large: {size_mb}MB > {ap.max_file_mb}MB")

# רשת
def enforce_connect(policy: UserPolicy, host: str, port: int):
    # DNS לפורמט host בלבד
    try:
        ip = socket.gethostbyname(host)
    except Exception:
        raise PolicyViolation(f"DNS failed: {host}")
    if not check_host_allowed(ip, policy.net) and not check_host_allowed(host, policy.net):
        raise PolicyViolation(f"network host denied: {host}")

# דקורטור להגבלת רשת
def net_guard(policy: UserPolicy):
    def wrapper(fn: Callable):
        @functools.wraps(fn)
        def inner(host, *a, **kw):
            enforce_connect(policy, host, kw.get("port",80))
            return fn(host, *a, **kw)
        return inner
    return wrapper

class PolicyViolation(Exception):
    pass
