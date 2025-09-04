# imu_repo/engine/runtime_guard.py
from __future__ import annotations
from typing import Any, Dict, List
from grounded.runtime_sample import (
    fetch_sample_with_raw, RuntimeFetchError, RuntimePolicyError
)
from grounded.value_checks import check_required_and_types, check_filters, check_sort, RuntimeRowError
from provenance.runtime_lineage import record_sample, get_last
from provenance.ca_store import sha256_hex
from engine.audit_log import record_event

class RuntimeBlocked(Exception): ...

def check_runtime_table(
    table_spec: Dict[str,Any],
    *,
    policy: Dict[str,Any],
    fetcher=None
) -> Dict[str,Any]:
    url = table_spec.get("binding_url") or ""
    if not url:
        return {"ok": True, "sampled": 0}

    if not bool(policy.get("runtime_check_enabled", True)):
        return {"ok": True, "sampled": 0, "skipped": True}

    try:
        rows, raw = fetch_sample_with_raw(
            url,
            timeout_s=float(policy.get("runtime_timeout_s", 3.0)),
            max_bytes=int(policy.get("runtime_max_bytes", 1_000_000)),
            sample_limit=int(policy.get("runtime_sample_limit", 200)),
            fetcher=fetcher
        )
    except (RuntimeFetchError, RuntimePolicyError) as e:
        raise RuntimeBlocked(f"runtime_fetch[url={url}]: {e}")
    # בדיקות ערכים
    columns = table_spec.get("columns") or []
    filters = table_spec.get("filters")
    sort    = table_spec.get("sort")

    checked = 0
    for r in rows:
        try:
            check_required_and_types(r, columns)
            check_filters(r, columns, filters)
            checked += 1
        except RuntimeRowError as re:
            raise RuntimeBlocked(f"runtime_row[url={url}]: {re}")

    try:
        check_sort(rows, sort)
    except RuntimeRowError as re:
        raise RuntimeBlocked(f"runtime_sort[url={url}]: {re}")

    # Lineage + Drift
    meta = {
        "table_path": table_spec.get("path"),
        "columns": columns,
        "filters": filters,
        "sort": sort,
        "sampled": len(rows)
    }
    rec = record_sample(url, raw, meta)  # שומר CAS+אינדוקס
    last = get_last(url)  
    prev = None
    try:
        # engine.audit_log אולי כותב לאחרים; אם אין, נתעלם. (avoid hard dep)
        pass
    except Exception:
        prev = None

    # קביעת prev_hash: מדיניות גוברת; אחרת קובץ מקומי (אופציונלי)
    prev_hash = policy.get("prev_content_hash")
    if not prev_hash:
        state_dir = policy.get("runtime_state_dir")  # למשל: "runs/runtime_state"
        if state_dir:
            import os, json, hashlib
            os.makedirs(state_dir, exist_ok=True)
            safe = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
            prev_f = os.path.join(state_dir, f"{safe}.json")
            try:
                if os.path.exists(prev_f):
                    with open(prev_f, "r", encoding="utf-8") as fh:
                        prev_hash = (json.load(fh) or {}).get("hash")
            except Exception:
                prev_hash = None
    changed = bool(prev_hash) and (prev_hash != rec["hash"])

    if not prev_hash:
        # fallback: אל תחסום רק בגלל שאין baseline ידוע; כן תרשום אירוע
        record_event("runtime_content_hash", {"hash": rec["hash"], "url": url}, severity="info")
    else:
        if changed:
            record_event("runtime_drift_detected",
                         {"url": url, "prev_hash": prev_hash, "new_hash": rec["hash"]},
                         severity="warn")
            if bool(policy.get("block_on_drift", False)):
                raise RuntimeBlocked(
                    f"runtime_drift[url={url}]: content_hash_changed {prev_hash} -> {rec['hash']}"
                )
        else:
            record_event("runtime_no_drift", {"url": url, "hash": rec["hash"]}, severity="info")

    return {"ok": True, "sampled": len(rows), "checked": checked, "hash": rec["hash"]}

# TODO- הערה: כדי לחסום Drift
#  כבר מהריצה השנייה ללא תלות חיצונית, אפשר לשמור לפני כתיבת הדגימה החדשה את “האחרון” ולהשוות — אבל בלי לשנות סמנטיקה, 
# תמכנו כאן ב־policy.prev_content_hash (ב־CI/קנרי אפשר להזין את הבייסליין).
#  אם תרצה — אעדכן לוגיקה ששומרת “previous.json” ומחזירה השוואה בתוך הפונקציה עצמה.