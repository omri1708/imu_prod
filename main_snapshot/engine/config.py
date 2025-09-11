# imu_repo/engine/config.py
from __future__ import annotations
import os, json, shutil, time
from typing import Any, Dict

ROOT = "/mnt/data/imu_repo"
CFG_DIR = os.path.join(ROOT, "config")
CFG_FILE = os.path.join(CFG_DIR, "runtime.json")
SNAP_DIR = os.path.join(ROOT, "snapshots")

_DEFAULT = {
    "ws": {
        "chunk_size": 64000,
        "permessage_deflate": True,
        "max_pending_msgs": 1024
    },
    "guard": {
        "min_trust": 0.7,
        "max_age_s": 3600
    },
    "evidence": {
        "required": True
    }
}

def ensure_dirs()->None:
    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(SNAP_DIR, exist_ok=True)

def load_config()->Dict[str,Any]:
    ensure_dirs()
    if not os.path.exists(CFG_FILE):
        save_config(_DEFAULT)
    try:
        return json.load(open(CFG_FILE, "r", encoding="utf-8"))
    except Exception:
        save_config(_DEFAULT); return dict(_DEFAULT)

def save_config(cfg: Dict[str,Any])->None:
    ensure_dirs()
    tmp = CFG_FILE + ".tmp"
    open(tmp, "w", encoding="utf-8").write(json.dumps(cfg, ensure_ascii=False, indent=2))
    os.replace(tmp, CFG_FILE)

def snapshot(tag: str|None=None)->str:
    """
    מעתיק config + logs לסנאפשוט חתום בזמן. מחזיר נתיב הסנאפשוט.
    """
    ensure_dirs()
    ts = int(time.time()*1000)
    name = f"{ts}_{tag or 'snapshot'}"
    out = os.path.join(SNAP_DIR, name)
    os.makedirs(out, exist_ok=True)
    # קונפיג
    if os.path.exists(CFG_FILE):
        shutil.copy2(CFG_FILE, os.path.join(out, "runtime.json"))
    # לוגים
    LOGS = os.path.join(ROOT, "logs")
    if os.path.isdir(LOGS):
        for fn in ("metrics.jsonl", "alerts.jsonl"):
            p = os.path.join(LOGS, fn)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(out, fn))
    return out