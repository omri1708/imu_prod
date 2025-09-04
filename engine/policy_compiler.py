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