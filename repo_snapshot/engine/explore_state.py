# imu_repo/engine/explore_state.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any

STATE_DIR = os.getenv("IMU_STATE_DIR")
if not STATE_DIR:
    if os.path.exists("/mnt/data"):
        STATE_DIR = "/mnt/data/imu_repo/.state/explore"
    else:
        STATE_DIR = os.path.join(os.getcwd(), ".state", "explore")

os.makedirs(STATE_DIR, exist_ok=True)

def _path(key: str) -> str:
    return os.path.join(STATE_DIR, f"{key}.json")

def load_state(key: str) -> Dict[str, Any]:
    p = _path(key)
    if not os.path.exists(p):
        return {"last_explore_ts": 0.0, "last_regression_ts": 0.0, "cooldown_until": 0.0, "recent_fail_count": 0}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(key: str, st: Dict[str,Any]) -> None:
    with open(_path(key), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)

def mark_explore(key: str, ts: float | None = None) -> None:
    st = load_state(key)
    now = float(ts or time.time())
    st["last_explore_ts"] = now
    save_state(key, st)

def mark_regression(key: str, cooldown_s: float, ts: float | None = None) -> None:
    st = load_state(key)
    now = float(ts or time.time())
    st["last_regression_ts"] = now
    st["recent_fail_count"] = int(st.get("recent_fail_count", 0)) + 1
    st["cooldown_until"] = now + float(max(0.0, cooldown_s))
    save_state(key, st)

def clear_regression(key: str) -> None:
    st = load_state(key)
    st["recent_fail_count"] = 0
    st["cooldown_until"] = 0.0
    save_state(key, st)

def in_cooldown(key: str, now: float | None = None) -> bool:
    st = load_state(key)
    return float(now or time.time()) < float(st.get("cooldown_until", 0.0))
