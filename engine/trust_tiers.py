# imu_repo/engine/trust_tiers.py  (גרסה מעודכנת)
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
from urllib.parse import urlparse

class TrustPolicyError(Exception): ...

def _norm_host(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        return h.lower()
    except Exception:
        return ""

def _best_suffix_tier(host: str, tiers: Dict[str,int]) -> Optional[int]:
    if not host or not tiers:
        return None
    best: Optional[int] = None
    for suf, tier in tiers.items():
        suf = suf.lower().strip()
        if not suf: 
            continue
        if host == suf or host.endswith("." + suf):
            if best is None or tier > best:
                best = tier
    return best

def trust_for_evidence(e: Dict[str,Any], policy: Dict[str,Any]) -> Tuple[int, str]:
    kind = e.get("kind")
    if kind == "inline":
        pts = int(policy.get("inline_trust", 1))
        return (pts, "inline")
    elif kind == "http":
        url = e.get("url")
        if not isinstance(url, str):
            return (0, "http:unknown")
        host = _norm_host(url)
        tiers = policy.get("trust_domains") or {}
        tier = _best_suffix_tier(host, tiers)
        if tier is None:
            tier = int(policy.get("default_http_trust", 0))
        return (int(tier), host or "http:unknown")
    else:
        return (0, f"unknown:{kind}")

def vet_sources_and_trust_for_claim(
    claim: Dict[str,Any],
    policy: Dict[str,Any]
) -> Tuple[int, int, List[str]]:
    evs = claim.get("evidence") or []
    source_to_pts: Dict[str,int] = {}
    rep = policy.get("reputation")  # אופציונלי: אובייקט עם factor(source_id)->float
    cap_per_src = int(policy.get("max_points_per_source", 5))
    for ev in evs:
        pts, src = trust_for_evidence(ev, policy)
        # reputation factor ∈ [1-alpha .. 1+alpha] (ראו engine/reputation.py)
        if rep is not None and hasattr(rep, "factor"):
            try:
                f = float(rep.factor(src))
                pts = int(round(max(0.0, pts * max(0.0, f))))
            except Exception:
                pass
        acc = source_to_pts.get(src, 0)
        source_to_pts[src] = min(cap_per_src, acc + max(0, pts))
    total = sum(source_to_pts.values())
    return total, len(source_to_pts), list(source_to_pts.keys())

def enforce_trust_requirements(
    claim: Dict[str,Any],
    policy: Dict[str,Any]
) -> None:
    min_src = int(policy.get("min_distinct_sources", 1))
    min_trust = int(policy.get("min_total_trust", 1))
    total, n, srcs = vet_sources_and_trust_for_claim(claim, policy)
    if n < min_src:
        raise TrustPolicyError(f"trust_fail: need >={min_src} distinct sources, got {n}: {srcs}")
    if total < min_trust:
        raise TrustPolicyError(f"trust_fail: need total_trust>={min_trust}, got {total} from {srcs}")