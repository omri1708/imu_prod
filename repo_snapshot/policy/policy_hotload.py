# policy/policy_hotload.py
# Hot-reload למדיניות רשת/קבצים מקובץ YAML — משקף אל security/network_policies.py ו־security/filesystem_policies.py
from __future__ import annotations
import os, time, threading, yaml
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from security.network_policies import POLICY_DB, UserNetPolicy, NetRule
from security.filesystem_policies import FS_DB, UserFsPolicy, PathRule

@dataclass
class HotloadState:
    path: str
    interval_s: float = 2.0
    stop: bool = False
    _last_mtime: float = 0.0
    _thread: Optional[threading.Thread] = None

STATE: Optional[HotloadState] = None

def _to_net_rules(items: List[Dict[str,Any]]) -> List[NetRule]:
    rules=[]
    for it in (items or []):
        rules.append(NetRule(
            host=it.get("host","*"),
            ports=list(map(int, it.get("ports",[443]))),
            tls_only=bool(it.get("tls_only", False))
        ))
    return rules

def _to_fs_rules(items: List[Dict[str,Any]]) -> List[PathRule]:
    rules=[]
    for it in (items or []):
        rules.append(PathRule(
            path=os.path.expanduser(it.get("path","./")),
            mode=it.get("mode","ro"),
            ttl_seconds=int(it.get("ttl_seconds", 0))
        ))
    return rules

def _apply_cfg(cfg: Dict[str,Any]):
    users = (cfg or {}).get("user_policies", {})
    for uid, spec in users.items():
        # Net
        np = UserNetPolicy(
            user_id=uid,
            default_deny=bool(spec.get("default_net","deny")=="deny"),
            rules=_to_net_rules(spec.get("net_allow", [])),
            max_outbound_qps=int(spec.get("net_max_qps", 10)),
            max_concurrent=int(spec.get("net_max_concurrent", 20)),
        )
        POLICY_DB.put(np)
        # FS
        fp = UserFsPolicy(
            user_id=uid,
            default_deny=bool(spec.get("default_fs","deny")=="deny"),
            rules=_to_fs_rules(spec.get("fs_allow", [])),
            max_bytes=int(spec.get("fs_max_bytes", 512*1024*1024))
        )
        FS_DB.put(fp)

def _loop():
    global STATE
    st = STATE
    if not st: return
    while not st.stop:
        try:
            if os.path.exists(st.path):
                m = os.path.getmtime(st.path)
                if m > st._last_mtime:
                    with open(st.path,"r",encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    _apply_cfg(cfg or {})
                    st._last_mtime = m
        except Exception as e:
            # לא מפיל את ה-thread על טעויות YAML/קובץ; אפשר לרשום לוג
            pass
        time.sleep(st.interval_s)

def start_watcher(path: str, interval_s: float = 2.0):
    """הפעלת watcher — יש לקרוא פעם אחת בעת האיתחול."""
    global STATE
    STATE = HotloadState(path=path, interval_s=interval_s)
    t = threading.Thread(target=_loop, name="policy-hotload", daemon=True)
    STATE._thread = t
    t.start()

def stop_watcher():
    global STATE
    if STATE:
        STATE.stop = True
        t = STATE._thread
        STATE = None
        if t:
            t.join(timeout=1.0)