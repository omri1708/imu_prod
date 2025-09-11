# engine/telemetry/audit.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os, json, time

_DEFAULT_DIR = os.getenv("IMU_AUDIT_DIR", ".imu/audit")
_DEFAULT_PATH = Path(_DEFAULT_DIR) / "pipeline.jsonl"

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def get_audit_path(ctx: Dict[str, Any] | None = None) -> Path:
    if ctx and isinstance(ctx.get("audit_path"), (str, os.PathLike)):
        p = Path(ctx["audit_path"])
    else:
        p = _DEFAULT_PATH
    _ensure_parent(p)
    return p

def emit_event(ctx: Dict[str, Any] | None, topic: str, **fields: Any) -> None:
    p = get_audit_path(ctx)
    rec = {"ts": time.time(), "topic": topic, "run_id": (ctx or {}).get("run_id")}
    rec.update(fields or {})
    _ensure_parent(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
