# imu_repo/verifiers/official_registry.py
from __future__ import annotations
import os, json, hmac, hashlib
from typing import Dict, Any, Optional

STATE_DIR = "/mnt/data/imu_repo/.state"
DB = os.path.join(STATE_DIR, "official_sources.json")
os.makedirs(STATE_DIR, exist_ok=True)

def _load() -> Dict[str, Any]:
    if not os.path.exists(DB):
        return {"sources": {}}
    with open(DB, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(db: Dict[str, Any]) -> None:
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def register_official_source(source_id: str, *, shared_secret: str, trust: float = 0.98, url_prefix: str = "official://") -> None:
    db = _load()
    db["sources"][source_id] = {
        "shared_secret": str(shared_secret),
        "trust": float(trust),
        "url_prefix": str(url_prefix),
    }
    _save(db)

def set_official_trust(source_id: str, trust: float) -> None:
    db = _load()
    if source_id not in db["sources"]:
        raise KeyError(f"unknown official source {source_id}")
    db["sources"][source_id]["trust"] = float(trust)
    _save(db)

def get_official(source_id: str) -> Optional[Dict[str, Any]]:
    db = _load()
    return db["sources"].get(source_id)

def hmac_sha256(secret: str, data_bytes: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), data_bytes, hashlib.sha256).hexdigest()

def sign_for_source(source_id: str, data_obj: Any) -> str:
    """נוח לטסטים: יוצר חתימה HMAC-SHA256 עבור data_obj."""
    rec = get_official(source_id)
    if not rec:
        raise KeyError(f"unknown official source {source_id}")
    blob = json.dumps(data_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hmac_sha256(rec["shared_secret"], blob)