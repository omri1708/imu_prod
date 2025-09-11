# imu_repo/provenance/runtime_lineage.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, Optional
from provenance.ca_store import put_bytes, put_json, sha256_hex, index_append, _dir

def _lineage_dir() -> str:
    return _dir("lineage")

def _url_key(url: str) -> str:
    return url.replace("://","__").replace("/","_")

def record_sample(url: str, raw: bytes, meta: Dict[str,Any]) -> Dict[str,Any]:
    """
    שומר דגימת Runtime:
      • תוכן ל-CAS (sha256:HASH)
      • רשומת lineage: {url, hash, meta, ts}
      • עדכון 'last' פר-URL (קובץ קטן)
    """
    huri = put_bytes(raw)
    h = huri.split(":",1)[1]
    rec = {"url": url, "hash": h, "meta": meta, "ts": time.time()}
    index_append("runtime_lineage", rec)
    # עדכון "last"
    lastp = os.path.join(_lineage_dir(), f"{_url_key(url)}.last.json")
    with open(lastp, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False))
    return {"uri": huri, "hash": h, "record": rec}

def get_last(url: str) -> Optional[Dict[str,Any]]:
    lastp = os.path.join(_lineage_dir(), f"{_url_key(url)}.last.json")
    if not os.path.exists(lastp): return None
    try:
        with open(lastp, "r", encoding="utf-8") as f:
            return json.loads(f.read() or "{}")
    except Exception:
        return None