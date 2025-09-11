# imu_repo/engine/policy_compiler.py
from __future__ import annotations
import json
from typing import Any, Dict

class PolicyCompileError(Exception): ...

DEFAULTS = {
    "require_claims_for_all_responses": True,
    "min_distinct_sources": 2,
    "min_total_trust": 4,
    "default_number_tolerance": 0.05,
    "require_consistency_groups": True,
    "min_provenance_level": 1,  # L1
    "http_timeout_sec": 5.0,
    "http_download_for_hash": False,
    "max_http_age_days": 365,
    "max_points_per_source": 5,
    "p95_window": 500,
    # quarantine
    "quarantine_min_calls": 20,
    "quarantine_error_rate_threshold": 0.2,
    "quarantine_violation_rate_threshold": 0.05,
    "quarantine_backoff_base_sec": 30.0,
    "quarantine_backoff_max_sec": 3600.0
}

def compile_policy(domain_json: str) -> Dict[str,Any]:
    """
    קלט JSON (מחרוזת). דוגמה:
    {
      "trust_domains": {"example.com":3, "acme.org":2},
      "min_distinct_sources": 2,
      "min_total_trust": 5,
      "signing_keys": {"k1":{"secret_hex":"aabb...", "algo":"sha256"}},
      "signature_fresh_window_sec": 900,
      "min_provenance_level": 2,
      "min_provenance_by_type": {"latency":3},
      "p95_limits": {"plan":50, "rollout":200},
      "quarantine": {"min_calls":30, "err_rate":0.3, "vio_rate":0.1},
      "trusted_domains": ["example.com","acme.org"],
      "inline_trust": 1
    }
    מפיק policy dict לשימוש בכל הגייטים.
    """
    try:
        src = json.loads(domain_json or "{}")
    except Exception as e:
        raise PolicyCompileError(f"bad json: {e}")

    pol = dict(DEFAULTS)
    # Trust tiers
    td = src.get("trust_domains") or {}
    if not isinstance(td, dict):
        raise PolicyCompileError("trust_domains must be object of suffix->tier")
    pol["trust_domains"] = {str(k).lower(): int(v) for k,v in td.items()}
    # רשימת דומיינים "מאושרים" ל־HTTP
    allow = src.get("trusted_domains")
    if allow is not None:
        if not isinstance(allow, list):
            raise PolicyCompileError("trusted_domains must be list")
        pol["trusted_domains"] = [str(x).lower() for x in allow]
    # ספי Trust/מקורות
    for k in ("min_distinct_sources","min_total_trust","inline_trust","default_http_trust","max_points_per_source"):
        if k in src:
            pol[k] = int(src[k])
    # חתימות
    keys = src.get("signing_keys") or {}
    if not isinstance(keys, dict):
        raise PolicyCompileError("signing_keys must be object")
    pol["signing_keys"] = {}
    for kid, meta in keys.items():
        if not isinstance(meta, dict) or "secret_hex" not in meta:
            raise PolicyCompileError(f"bad signing key {kid}")
        pol["signing_keys"][str(kid)] = {"secret_hex": str(meta["secret_hex"]), "algo": str(meta.get("algo","sha256"))}
    if "signature_fresh_window_sec" in src:
        pol["signature_fresh_window_sec"] = float(src["signature_fresh_window_sec"])
    # Provenance
    if "min_provenance_level" in src:
        pol["min_provenance_level"] = int(src["min_provenance_level"])
    if "min_provenance_by_type" in src:
        by = src["min_provenance_by_type"]
        if not isinstance(by, dict):
            raise PolicyCompileError("min_provenance_by_type must be object")
        pol["min_provenance_by_type"] = {str(k): int(v) for k,v in by.items()}
    # Consistency
    if "default_number_tolerance" in src:
        pol["default_number_tolerance"] = float(src["default_number_tolerance"])
    pol["require_consistency_groups"] = bool(src.get("require_consistency_groups", pol["require_consistency_groups"]))
    # HTTP
    for k in ("http_timeout_sec","http_download_for_hash","max_http_age_days"):
        if k in src:
            pol[k] = src[k]
    # p95 limits
    p95 = src.get("p95_limits") or {}
    if not isinstance(p95, dict):
        raise PolicyCompileError("p95_limits must be object")
    for step, lim in p95.items():
        pol[f"{step}_p95_ms"] = float(lim)
    if "p95_window" in src:
        pol["p95_window"] = int(src["p95_window"])
    # quarantine
    q = src.get("quarantine") or {}
    if not isinstance(q, dict):
        raise PolicyCompileError("quarantine must be object")
    if "min_calls" in q: pol["quarantine_min_calls"] = int(q["min_calls"])
    if "err_rate"  in q: pol["quarantine_error_rate_threshold"] = float(q["err_rate"])
    if "vio_rate"  in q: pol["quarantine_violation_rate_threshold"] = float(q["vio_rate"])

    return pol


def _merge(a: Dict[str,Any], b: Dict[str,Any]) -> Dict[str,Any]:
    out = dict(a)
    for k,v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def compile_with_profiles(base_json: str) -> Dict[str,Dict[str,Any]]:
    """
    מפיק שלושה פרופילים: dev, stage, prod מתוך בסיס אחד.
    חוקיות:
      - dev: ספי trust/provenance רכים, p95 רחב, quarantine כבוי.
      - stage: ביניים, p95 בינוני, quarantine מופעל.
      - prod: ספים קשיחים (trust גבוה, provenance ≥ L2/L3), p95 מחמיר, quarantine מלא.
    """
    base = compile_policy(base_json)

    dev = _merge(base, {
        "min_distinct_sources": max(1, int(base.get("min_distinct_sources",1)) - 1),
        "min_total_trust": max(1, int(base.get("min_total_trust",1)) - 1),
        "min_provenance_level": max(0, int(base.get("min_provenance_level",1)) - 1),
        "require_consistency_groups": False,
        "p95_window": 200,  # חלון קצר לבדיקות
        "quarantine_min_calls": 10,
        "quarantine_error_rate_threshold": 1.0,   # למעשה כבוי
        "quarantine_violation_rate_threshold": 1.0
    })

    stage = _merge(base, {
        "min_distinct_sources": max(2, int(base.get("min_distinct_sources",2))),
        "min_total_trust": max(3, int(base.get("min_total_trust",3))),
        "min_provenance_level": max(1, int(base.get("min_provenance_level",1))),
        "require_consistency_groups": True,
        "p95_window": int(base.get("p95_window", 500)),
        "quarantine_error_rate_threshold": 0.3,
        "quarantine_violation_rate_threshold": 0.1
    })

    prod = _merge(base, {
        "min_distinct_sources": max(3, int(base.get("min_distinct_sources",2))+1),
        "min_total_trust": max(5, int(base.get("min_total_trust",4))+1),
        "min_provenance_level": max(2, int(base.get("min_provenance_level",1))+1),
        "require_consistency_groups": True,
        "p95_window": max(1000, int(base.get("p95_window", 500))*2),
        "quarantine_error_rate_threshold": 0.2,
        "quarantine_violation_rate_threshold": 0.05
    })

    return {"dev": dev, "stage": stage, "prod": prod}


def policy_passes(pol: Dict[str,Any]) -> Dict[str,Any]:
    """
    'passes' פשוטים שמעשירים מדיניות קיימת:
      - דרוג אוטומטי ל־latency/error claims.
      - קיבוע max_points_per_source אם חסר.
    """
    out = dict(pol)
    if "max_points_per_source" not in out:
        out["max_points_per_source"] = 5
    # אם אין min_provenance_by_type — נקבע לטענות latency≥L2
    mbt = out.get("min_provenance_by_type") or {}
    if "latency" not in mbt:
        mbt["latency"] = max(2, int(out.get("min_provenance_level",1)))
    out["min_provenance_by_type"] = mbt
    return out

STRICT_BUMPS = {
    "min_distinct_sources": 4,
    "min_total_trust": 8,
    "min_provenance_level": 3,  # דורש רעננות L3
    "require_consistency_groups": True,
    "p95_window": 2000,
    "quarantine_error_rate_threshold": 0.1,
    "quarantine_violation_rate_threshold": 0.02,
    "require_claims_for_all_responses": True,
    "default_number_tolerance": 0.005
}

def strict_prod_from(base_json: str) -> Dict[str,Any]:
    base = compile_policy(base_json)
    # טריקים של hardening
    base.setdefault("min_points_per_claim", 1.0)
    base.setdefault("max_points_per_source", 5)
    out = dict(base)
    for k,v in STRICT_BUMPS.items():
        out[k] = v
    # דוגמה ל־per-type:
    mbt = out.get("min_provenance_by_type") or {}
    mbt["latency"] = max(3, int(out["min_provenance_level"]))
    mbt["kpi"] = max(3, int(out["min_provenance_level"]))
    out["min_provenance_by_type"] = mbt
    return out

def keyring_from_policy(pol: Dict[str,Any]) -> Dict[str,Dict[str,str]]:
    """
    חילוץ keyring פשוט מהמדיניות עבור ה־verifier (צד הצרכן).
    """
    kr = {}
    for kid, meta in (pol.get("signing_keys") or {}).items():
        kr[kid] = {"secret_hex": str(meta["secret_hex"]), "algo": str(meta.get("algo","sha256"))}
    return kr

