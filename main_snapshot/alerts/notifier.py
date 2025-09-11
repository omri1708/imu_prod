# imu_repo/alerts/notifier.py
from __future__ import annotations
import os, json, time, threading
from typing import Dict, Any, Tuple

# ניתן לכוון מיקום לוגים עם ENV: IMU_LOG_DIR (#TODO )
ROOT = os.getenv("IMU_LOG_DIR", "/mnt/data/imu_repo/logs")
os.makedirs(ROOT, exist_ok=True)
_alert_f = os.path.join(ROOT, "alerts.jsonl")
_metrics_f = os.path.join(ROOT, "metrics.jsonl")
_lock = threading.RLock()


def _split_meta(meta: Dict[str,Any] | None) -> Tuple[str, Dict[str,Any]]:
    """
    מוציא bucket מתוך meta (אם קיים) או מ-ENV, ומחזיר (bucket, meta_ללא_bucket).
    מאפשר לשים bucket גם כשדה עליון וגם בתוך meta לשמירת תאימות.
    """
    m = dict(meta or {})
    b = m.pop("bucket", os.getenv("IMU_BUCKET", "default"))
    return str(b), m

def _w(path: str, obj: Dict[str,Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def alert(event: str, *, severity: str="info", meta: Dict[str,Any] | None=None) -> None:
    bucket, meta_ = _split_meta(meta)
    rec = {
        "ts": int(time.time()*1000),
        "bucket": bucket,
        "event": event,
        "severity": severity,
        "meta": {**meta_, "bucket": bucket},  # השאר גם בתוך meta אם חשוב לך
    }
    _w(_alert_f, rec)

def metrics_log(name: str, meta: Dict[str,Any] | None=None) -> None:
    bucket, meta_ = _split_meta(meta)
    rec = {
        "ts": int(time.time()*1000),
        "bucket": bucket,
        "name": name,
        "meta": {**meta_, "bucket": bucket},
    }
    _w(_metrics_f, rec)