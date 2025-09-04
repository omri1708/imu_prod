# imu_repo/engine/trust_tiers.py
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
    """
    מחזיר את ה-tier הטוב ביותר (הגבוה) עבור host בהתאם למפת suffix->tier.
    התאמה: suffix מלא או סאב-דומיין.
    """
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
    """
    מחזיר (trust_points, source_id) עבור פריט evidence.
    - kind=inline: trust לפי policy['inline_trust'] (ברירת מחדל 1); source_id="inline"
    - kind=http: trust לפי trust_domains suffix map; אם לא נמצא → default_http_trust (ברירת מחדל 0)
      source_id = hostname (משמש לייחוד מקורות).
    """
    kind = e.get("kind")
    if kind == "inline":
        pts = int(policy.get("inline_trust", 1))
        return (pts, "inline")
    elif kind == "http":
        url = e.get("url")
        if not isinstance(url, str):
            # ייתכן שבשלב האריזה כבר אין url אלא meta_hash; במקרה זה אין לנו מקור גלוי → 0
            return (0, "http:unknown")
        host = _norm_host(url)
        tiers = policy.get("trust_domains") or {}
        tier = _best_suffix_tier(host, tiers)
        if tier is None:
            tier = int(policy.get("default_http_trust", 0))
        return (int(tier), host or "http:unknown")
    else:
        # סוג לא מוכר → לא אמין
        return (0, f"unknown:{kind}")

def vet_sources_and_trust_for_claim(
    claim: Dict[str,Any],
    policy: Dict[str,Any]
) -> Tuple[int, int, List[str]]:
    """
    סוכם trust על פני מקורות שונים (distinct source_id) ומחזיר:
      (total_trust_points, distinct_sources_count, sources_list)
    """
    evs = claim.get("evidence") or []
    source_to_pts: Dict[str,int] = {}
    for ev in evs:
        pts, src = trust_for_evidence(ev, policy)
        # מקסימום תרומה פר-מקור (מונע “פאמפינג” של אותו דומיין)
        cap = int(policy.get("max_points_per_source", 5))
        acc = source_to_pts.get(src, 0)
        source_to_pts[src] = min(cap, acc + max(0, pts))
    total = sum(source_to_pts.values())
    return total, len(source_to_pts), list(source_to_pts.keys())

def enforce_trust_requirements(
    claim: Dict[str,Any],
    policy: Dict[str,Any]
) -> None:
    """
    אוכף ספים:
      - min_distinct_sources: מספר מקורות ייחודיים מינימלי
      - min_total_trust: סכום trust מינימלי
    """
    min_src = int(policy.get("min_distinct_sources", 1))
    min_trust = int(policy.get("min_total_trust", 1))
    total, n, srcs = vet_sources_and_trust_for_claim(claim, policy)
    if n < min_src:
        raise TrustPolicyError(f"trust_fail: need >={min_src} distinct sources, got {n}: {srcs}")
    if total < min_trust:
        raise TrustPolicyError(f"trust_fail: need total_trust>={min_trust}, got {total} from {srcs}")