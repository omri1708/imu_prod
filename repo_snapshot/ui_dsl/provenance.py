from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any, List, Tuple
from cas.store import put_json, put_bytes

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _walk_components(node: Dict[str,Any], out_assets: List[Dict[str,Any]]) -> None:
    # אוסף קישורים ל־assets (icons, images, scripts) אם קיימים במפרט
    if not isinstance(node, dict): return
    for k, v in list(node.items()):
        if k in ("icon","img","asset") and isinstance(v, dict) and "bytes" in v:
            meta = put_bytes(v["bytes"], media_type=v.get("media_type","application/octet-stream"))
            out_assets.append({"kind":"asset", "sha256": meta["sha256"], "media_type": meta["media_type"], "size": meta["size"]})
        elif isinstance(v, dict):
            _walk_components(v, out_assets)
        elif isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    _walk_components(it, out_assets)

def build_ui_provenance(*, ui_spec: Dict[str,Any], sources: List[Dict[str,Any]], policy: Dict[str,Any]) -> Dict[str,Any]:
    """
    יוצר מניפסט Provenance ל־UI:
      - החתמה של מבנה ה־UI (hash)
      - רשימת מקורות/ראיות (sources) כפי שנדרשים ב־Grounding
      - רשימת assets עם sha256 מה־CAS
    """
    now = time.time()
    assets: List[Dict[str,Any]] = []
    _walk_components(ui_spec, assets)
    spec_bytes = json.dumps(ui_spec, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    spec_hash = _sha(spec_bytes)
    manifest = {
        "type": "ui_provenance",
        "spec_sha256": spec_hash,
        "assets": assets,
        "sources": sources,
        "policy_fingerprint": _sha(json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")),
        "ts": now
    }
    saved = put_json(manifest)
    return {"ok": True, "manifest_sha256": saved["sha256"], "manifest": manifest}