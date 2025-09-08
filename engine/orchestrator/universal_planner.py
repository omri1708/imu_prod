# Path: imu_repo/engine/universal_planner.py
# -*- coding: utf-8 -*-
"""
UniversalPlanner — iterative discovery → global market scan → claims vs reality →
coverage-tightened blueprint with proof of superiority.

Legal/Use Anchor (user-requested):
- No violation of GPT/LLM ToS; provided for lawful, permitted use only.
- You (the user) control exposure of data/sources. This module enforces grounding when requested.
- This file is supplied as part of a legitimate service workflow; adjust policies to your jurisdiction.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ------------------------------- Exceptions ---------------------------------
class ValidationFailed(Exception):
    """Raised when the planner cannot produce a minimally valid spec."""


# ------------------------------- Schema Hints --------------------------------
# NOTE: These are "shape hints" for an LLM, not strict JSON Schema.
ARCH_SCHEMA: Dict[str, Any] = {
    "title": "string",
    "components": [{}],              # UI/Services/Adapters/etc.
    "nav": {},                       # routing
    "screens": [{}],                 # UX screens
    "data": {                        # data model & sources
        "models": [{}],
        "sources": [{}]
    },
    "flows": [{}],                   # user/system flows
    "differentiation": ["string"],  # value props vs market
    "open_questions": ["string"],
    "non_functional": {
        "security": ["string"],
        "privacy": ["string"],
        "performance": ["string"],
        "accessibility": ["string"],
        "observability": ["string"]
    },
    "test_plan": {"levels": ["unit", "integration", "e2e"], "notes": "string"},
    "acceptance_criteria": ["string"],

    # Market-scan & superiority (extensions)
    "competitors": [{"name": "string", "links": ["string"], "regions": ["string"], "platforms": ["string"]}],
    "features_matrix": [{"feature": "string", "us": "string", "them": ["string"]}],
    "pricing_matrix": [{"tier": "string", "us": "string"}],
    "citations": [{"claim": "string", "sources": ["string"]}],

    # Claims vs Reality
    "competitor_claims": [{"competitor": "string", "feature": "string", "claim": "string", "evidence": ["string"], "last_seen": "string"}],
    "claim_checks": [{"competitor": "string", "feature": "string", "check_method": "string", "metric": "string", "result": {"status": "string"}, "citations": ["string"]}],
    "misalign_summary": [{"feature": "string", "competitor": "string", "score": 0.0, "notes": "string"}],

    # Proof-of-superiority harness
    "proof_of_superiority": [{
        "topic": "string",
        "our_target": {},
        "benchmark_harness": "string",
        "acceptance_criteria": ["string"],
        "success_thresholds": {"regions": 0, "languages": 0}
    }],
    "evidence_policy": {"min_sources": 2, "freshness_months": 12}
}

DISCOVERY_SCHEMA: Dict[str, Any] = {
    "questions": [
        {
            "id": "string",
            "question": "string",
            "slot": "string",       # semantic path inside SPEC
            "critical": True,
            "options": ["string"],   # suggested choices
        }
    ]
}

MARKET_SCHEMA: Dict[str, Any] = {
    "competitors": [{"name": "string", "links": ["string"], "regions": ["string"], "platforms": ["string"]}],
    "features_matrix": [{"feature": "string", "us": "string", "them": ["string"]}],
    "pricing_matrix": [{"tier": "string", "us": "string", "others": [{"name": "string", "value": "string"}]}],
    "gaps": ["string"],
    "differentiation": ["string"],
    "citations": [{"claim": "string", "sources": ["string"]}],
    "coverage": {"regions": 0, "languages": 0, "freshness_months": 12},
    "evidence_score": 0.0
}

CLAIMS_SCHEMA: Dict[str, Any] = {
    "competitor_claims": [
        {
            "competitor": "string",
            "feature": "string",
            "claim": "string",
            "regions": ["string"],
            "evidence": ["string"],
            "last_seen": "string"
        }
    ]
}

CLAIM_CHECKS_SCHEMA: Dict[str, Any] = {
    "claim_checks": [
        {
            "competitor": "string",
            "feature": "string",
            "check_method": "string",
            "metric": "string",
            "result": {"status": "string", "perf_gap": 0.0, "coverage_gap": 0.0, "staleness": 0.0},
            "details": "string",
            "citations": ["string"]
        }
    ]
}

COVERAGE_TARGETS: List[str] = [
    "ui_screens",        # at least 1 screen
    "nav",               # at least 1 route
    "data_models",       # at least 1 model
    "flows",             # at least 1 flow
    "backend_or_service",# at least 1 non-UI component
    "observability",     # monitoring/logging/metrics
    "security",          # auth/policies or similar
    "test_plan",         # testing strategy
    "acceptance_criteria"# explicit acceptance bullets
]

DEFAULT_EVIDENCE_POLICY = {"min_sources": 2, "freshness_months": 12}


# --------------------------------- Helpers ----------------------------------
def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _dedup_by_json(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for it in items:
        key = _dumps(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


# ----------------------------- UniversalPlanner ------------------------------
class UniversalPlanner:
    """
    Iterative discovery → global market scan → claims vs reality →
    coverage-tightened blueprint with proof of superiority.

    Flow:
      1) Discovery → questions (with slots & criticality)
      2) Auto-resolve answers from context/prior/known_tools; collect unresolved
      3) Optional ask_back if blockers exist
      4) Market scan (global, grounded) → differentiation+citations
      5) Initial spec synthesis
      6) Claims scan + checks → misalignment summary
      7) Inject superiority plan (bench harness + acceptance criteria)
      8) Coverage loop (refine for gaps) up to max_rounds
    """

    def __init__(
        self,
        gateway, 
        *,
        max_rounds: int = 3,
        regions: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        freshness_months: int = 12,
    ) -> None:
        if gateway is not None:
            self.gw = gateway
        else:
            from engine.llm_gateway import LLMGateway  # import עצל למניעת מעגליות
            self.gw = LLMGateway()
        self.max_rounds = max(1, int(max_rounds))
        self.regions = regions or ["US", "EU", "LATAM", "MENA", "APAC", "JP", "KR", "IN"]
        self.languages = languages or ["en", "es", "pt", "fr", "de", "it", "zh", "ja", "ko", "ar", "hi", "id", "tr", "ru", "he"]
        self.freshness_months = max(1, int(freshness_months))

    # ------------------------------ Entrypoints ------------------------------
    def analyze_request(
        self,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        prior: Optional[Dict[str, Any]] = None,
        known_tools: Optional[Dict[str, bool]] = None,
        *,
        ask_back: bool = False,
    ) -> Dict[str, Any]:
        return self.plan_blueprint(
            user_id=user_id,
            text=text,
            context=context,
            prior=prior or {},
            known_tools=known_tools or {},
            ask_back=ask_back,
        )

    def plan_blueprint(
        self,
        *,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        prior: Dict[str, Any],
        known_tools: Dict[str, bool],
        ask_back: bool = False,
    ) -> Dict[str, Any]:
        # 1) Discovery
        questions = self._discover_questions(user_id, text, context)
        answers, unresolved = self._auto_resolve(questions, text, context, prior, known_tools)

        if ask_back and self._has_blockers(unresolved):
            return {
                "next_action": "ask_user",
                "title": text[:80],
                "open_questions": [q.get("question", "?") for q in unresolved],
                "unresolved": unresolved,
                "ts": int(time.time() * 1000),
            }

        # 2) Global market scan (grounded)
        market = self._market_scan(user_id, text, context, known_tools)
        if isinstance(market, dict) and "__error__" in market:
          return {
              "next_action": "ask_user",
              "error": market.get("__error__"),
              "details": market.get("details"),
              "open_questions": [q.get("question", "?") for q in unresolved],
              "ts": int(time.time() * 1000),
          }
        # 3) Initial spec (with discovery answers + market inputs)
        spec = self._synthesize_spec(
            user_id=user_id,
            text=text,
            context=context,
            prior=prior,
            known_tools=known_tools,
            answers=answers,
            open_questions=[q.get("question", "?") for q in unresolved],
            market=market,
        )
        self._validate_minimal(spec, text)

        # 4) Claims scan + checks → misalignment summary
        claims = self._claims_scan(user_id, text, context, market)
        if isinstance(claims, dict) and "__error__" in claims:
          return {
              "next_action": "ask_user",
              "error": claims.get("__error__"),
              "details": claims.get("details"),
              "open_questions": [q.get("question", "?") for q in unresolved],
              "ts": int(time.time() * 1000),
          }
        checks = self._claim_checks(user_id, text, context, claims)
        misalign = self._claims_misalignment(claims, checks)
        spec["competitor_claims"] = claims
        spec["claim_checks"] = checks
        spec["misalign_summary"] = misalign

        # 5) Inject superiority plan from misalignments + market gaps
        spec = self._inject_market_and_superiority(spec, market, misalign)

        # 6) Coverage tighten loop
        for _round in range(self.max_rounds):
            gaps = self._compute_coverage_gaps(spec)
            if not gaps:
                break
            spec = self._refine_for_gaps(
                user_id=user_id,
                text=text,
                context=context,
                prior=prior,
                known_tools=known_tools,
                base_spec=spec,
                gaps=gaps,
            )
            self._validate_minimal(spec, text)

        # Final metadata
        spec.setdefault("title", text[:80])
        spec.setdefault("ts", int(time.time() * 1000))
        return spec

    # ----------------------- Discovery + auto-resolution ---------------------
    def _discover_questions(self, user_id: str, text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="ux_discovery",
            content={"prompt": text, "context": context},
            schema_hint=_dumps(DISCOVERY_SCHEMA),
            temperature=0.0,
        ) or {}
        if not res.get("ok", True):
          return {"__error__": res.get("error"), "details": res.get("details")}
        j = res.get("json") or {}
        qs = j.get("questions") or []
        return qs if isinstance(qs, list) else []

    def _auto_resolve(
        self,
        questions: List[Dict[str, Any]],
        text: str,
        context: Dict[str, Any],
        prior: Dict[str, Any],
        known_tools: Dict[str, bool],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        answers: List[Dict[str, Any]] = []
        unresolved: List[Dict[str, Any]] = []

        for q in _ensure_list(questions):
            slot = q.get("slot") or ""
            opts = _ensure_list(q.get("options"))
            critical = bool(q.get("critical"))

            # 1) single-option non-critical
            if len(opts) == 1 and not critical:
                answers.append({"id": q.get("id"), "slot": slot, "value": opts[0], "source": "default_option"})
                continue

            # 2) from context/prior (shallow dot path)
            v = self._lookup(context, slot)
            if v is None:
                v = self._lookup(prior, slot)
            if v is not None:
                answers.append({"id": q.get("id"), "slot": slot, "value": v, "source": "memory"})
                continue

            # 3) known_tools hints
            kt = self._guess_from_known_tools(slot, known_tools)
            if kt is not None:
                answers.append({"id": q.get("id"), "slot": slot, "value": kt, "source": "known_tools"})
                continue

            # no auto resolution
            unresolved.append(q)

        return answers, unresolved

    def _lookup(self, obj: Dict[str, Any], slot: str) -> Any:
        if not slot or not isinstance(obj, dict):
            return None
        cur: Any = obj
        for part in slot.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _guess_from_known_tools(self, slot: str, known_tools: Dict[str, bool]) -> Optional[Any]:
        if not slot:
            return None
        if slot.endswith("hosting"):
            if known_tools.get("vercel"):
                return "vercel"
            if known_tools.get("fly_io"):
                return "fly.io"
        if slot.endswith("db") or slot.endswith("database"):
            if known_tools.get("supabase"):
                return "supabase"
            if known_tools.get("firebase"):
                return "firebase"
        if slot.endswith("auth"):
            if known_tools.get("clerk"):
                return "clerk"
            if known_tools.get("auth0"):
                return "auth0"
        return None

    def _has_blockers(self, unresolved: List[Dict[str, Any]]) -> bool:
        return any(bool(q.get("critical")) for q in unresolved)

    # --------------------- Market scan (global, grounded) --------------------
    def _market_scan(self, user_id: str, text: str, context: Dict[str, Any], known_tools: Dict[str, bool]) -> Dict[str, Any]:
        payload = {
            "prompt": (
                "בצע מחקר שוק עולמי רב-לשוני. החזר JSON בלבד לפי הסכמה. "
                "חייב ציטוטים (sources) לכל טענה קשיחה."
            ) + f"\n[text]={text}",
            "context": {"memory": context, "known_tools": known_tools},
            "regions": self.regions,
            "languages": self.languages,
            "freshness_months": self.freshness_months,
            "evidence_policy": DEFAULT_EVIDENCE_POLICY,
        }
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="market_scan",
            prompt=_dumps(payload),
            schema_hint=_dumps(MARKET_SCHEMA),
            temperature=0.0,
            require_grounding=True,  # enforce FactGate in gateway if available
        ) or {}
        j = res.get("json") or {}
        return j if isinstance(j, dict) else {}

    # ---------------- Claims scan, checks, and misalignment ------------------
    def _claims_scan(self, user_id: str, text: str, context: Dict[str, Any], market: Dict[str, Any]) -> List[Dict[str, Any]] | Dict[str, Any]:
        payload = {
            "prompt": (
                "חלץ טענות פיצ'ר רשמיות של מתחרים (דפי מוצר/חנויות/דוקו/סקירות). "
                "החזר JSON בלבד, עם evidence."
            ) + f"\n[text]={text}",
            "context": {"memory": context},
            "competitors": market.get("competitors", []),
            "evidence_policy": DEFAULT_EVIDENCE_POLICY,
        }
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="claims_scan",
            prompt=_dumps(payload),
            schema_hint=_dumps(CLAIMS_SCHEMA),
            temperature=0.0,
            require_grounding=True,
        ) or {}
        if not res.get("ok", True):
          return {"__error__": res.get("error"), "details": res.get("details")}
        j = res.get("json") or {}
        items = j.get("competitor_claims") if isinstance(j, dict) else None
        return items if isinstance(items, list) else []

    def _claim_checks(self, user_id: str, text: str, context: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = {
            "prompt": (
                "בנה בדיקות/Benchmarks לאימות/הפרכת טענות מתחרים. החזר JSON בלבד."
            ) + f"\n[text]={text}",
            "context": {"memory": context},
            "claims": claims,
        }
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="claim_checks",
            prompt=_dumps(payload),
            schema_hint=_dumps(CLAIM_CHECKS_SCHEMA),
            temperature=0.0,
        ) or {}
        j = res.get("json") or {}
        items = j.get("claim_checks") if isinstance(j, dict) else None
        return items if isinstance(items, list) else []

    def _claims_misalignment(self, claims: List[Dict[str, Any]], checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        idx = {(c.get("competitor"), c.get("feature")): c for c in _ensure_list(claims)}
        for chk in _ensure_list(checks):
            key = (chk.get("competitor"), chk.get("feature"))
            claim = idx.get(key, {})
            contradiction = 1.0 if (chk.get("result", {}).get("status") == "contradicted") else 0.0
            perf_gap = float(chk.get("result", {}).get("perf_gap", 0.0))  # 0..1
            coverage_gap = float(chk.get("result", {}).get("coverage_gap", 0.0))
            freshness_penalty = float(chk.get("result", {}).get("staleness", 0.0))
            score = 0.5 * contradiction + 0.3 * perf_gap + 0.2 * coverage_gap - 0.1 * freshness_penalty
            out.append({
                "feature": chk.get("feature"),
                "competitor": chk.get("competitor"),
                "score": round(max(0.0, min(1.0, score)), 2),
                "notes": chk.get("details") or claim.get("claim", "")
            })
        return sorted(out, key=lambda x: x.get("score", 0.0), reverse=True)

    # -------------------------- Spec synthesis & refine ----------------------
    def _synthesize_spec(
        self,
        *,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        prior: Dict[str, Any],
        known_tools: Dict[str, bool],
        answers: List[Dict[str, Any]],
        open_questions: List[str],
        market: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "prompt": (
                "המר את הבקשה לארכיטקטורה מלאה לפי הסכמה (JSON בלבד). "
                "שלב open_questions, נתוני שוק ודיפרנציאציה עם ציטוטים. "
                "השתמש ב-known_tools והתאם ל-prior."
            ) + f"\n[text]={text}",
            "context": {"memory": context, "prior": prior, "known_tools": known_tools},
            "discovery": {"answers": answers, "open_questions": open_questions},
            "market": market,
            "target_coverage": COVERAGE_TARGETS,
            "evidence_policy": DEFAULT_EVIDENCE_POLICY,
        }
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="spec_refine",
            prompt=_dumps(payload),
            schema_hint=_dumps(ARCH_SCHEMA),
            temperature=0.0,
        ) or {}
        spec = res.get("json") or {}
        return spec if isinstance(spec, dict) else {}

    def _compute_coverage_gaps(self, spec: Dict[str, Any]) -> List[str]:
        gaps: List[str] = []
        # ui_screens
        if not _ensure_list(spec.get("screens")):
            gaps.append("ui_screens")
        # nav
        if not (isinstance(spec.get("nav"), dict) and spec.get("nav")):
            gaps.append("nav")
        # data models
        data = spec.get("data") or {}
        models = _ensure_list((data or {}).get("models"))
        if not models:
            gaps.append("data_models")
        # flows
        if not _ensure_list(spec.get("flows")):
            gaps.append("flows")
        # backend_or_service
        comps = _ensure_list(spec.get("components"))
        if not any(isinstance(c, dict) and c.get("type") in {"backend", "service", "api", "worker"} for c in comps):
            gaps.append("backend_or_service")
        # observability & security
        nf = spec.get("non_functional") or {}
        if not _ensure_list(nf.get("observability")):
            gaps.append("observability")
        if not _ensure_list(nf.get("security")):
            gaps.append("security")
        # test plan
        if not (isinstance(spec.get("test_plan"), dict) and spec["test_plan"].get("levels")):
            gaps.append("test_plan")
        # acceptance criteria
        if not _ensure_list(spec.get("acceptance_criteria")):
            gaps.append("acceptance_criteria")
        return gaps

    def _refine_for_gaps(
        self,
        *,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        prior: Dict[str, Any],
        known_tools: Dict[str, bool],
        base_spec: Dict[str, Any],
        gaps: List[str],
    ) -> Dict[str, Any]:
        payload = {
            "prompt": (
                "שפר את ה-SPEC כדי לסגור פערי כיסוי ספציפיים בלבד (JSON בלבד). "
                "היצמד ל-ARCH_SCHEMA."
            ) + f"\n[text]={text}",
            "context": {"memory": context, "prior": prior, "known_tools": known_tools},
            "current_spec": base_spec,
            "gaps": gaps,
            "target_coverage": COVERAGE_TARGETS,
        }
        res = self._gw_structured(
            user_id=user_id,
            task="orchestrate",
            intent="arch_expand",
            prompt=_dumps(payload),
            schema_hint=_dumps(ARCH_SCHEMA),
            temperature=0.0,
        ) or {}
        refined = res.get("json") or {}
        return self._merge_specs(base_spec, refined)

    # -------------------------- Merge & validation ---------------------------
    def _validate_minimal(self, spec: Dict[str, Any], text: str) -> None:
        if not isinstance(spec, dict) or not _ensure_list(spec.get("components")):
            raise ValidationFailed("planner_empty_spec")
        spec.setdefault("title", text[:80])
        spec.setdefault("ts", int(time.time() * 1000))
        spec.setdefault("differentiation", [])
        spec.setdefault("open_questions", [])
        spec.setdefault("evidence_policy", dict(DEFAULT_EVIDENCE_POLICY))

    def _merge_specs(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(patch, dict):
            return base
        out = dict(base)
        for k, v in patch.items():
            if v is None:
                continue
            if k in {"components", "screens", "flows", "acceptance_criteria", "differentiation"}:
                out[k] = _dedup_by_json(_ensure_list(base.get(k)) + _ensure_list(v))
            elif k == "nav":
                nv = dict(base.get("nav") or {})
                nv.update(v if isinstance(v, dict) else {})
                out[k] = nv
            elif k == "data":
                bv = base.get("data") or {}
                nv = dict(bv)
                for sub in ("models", "sources"):
                    nv[sub] = _dedup_by_json(_ensure_list(bv.get(sub)) + _ensure_list((v or {}).get(sub)))
                for subk, subv in (v or {}).items():
                    if subk not in {"models", "sources"}:
                        nv[subk] = subv
                out[k] = nv
            elif k == "non_functional":
                bv = base.get("non_functional") or {}
                nv = dict(bv)
                for sub in ("security", "privacy", "performance", "accessibility", "observability"):
                    nv[sub] = _dedup_by_json(_ensure_list(bv.get(sub)) + _ensure_list((v or {}).get(sub)))
                out[k] = nv
            elif k == "test_plan":
                bv = base.get("test_plan") or {}
                nv = dict(bv)
                if isinstance((v or {}).get("levels"), list):
                    nv["levels"] = _dedup_by_json(_ensure_list(bv.get("levels")) + _ensure_list(v.get("levels")))
                if (v or {}).get("notes"):
                    nv["notes"] = v["notes"]
                out[k] = nv
            else:
                out[k] = v
        return out

    # ---------------- Superiority injection from market & misalign -----------
    def _inject_market_and_superiority(self, spec: Dict[str, Any], market: Dict[str, Any], misalign: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Merge market basics
        for field in ("competitors", "features_matrix", "pricing_matrix", "gaps", "differentiation", "citations"):
            if market.get(field):
                if field in {"competitors", "features_matrix", "pricing_matrix", "citations"}:
                    spec[field] = _dedup_by_json(_ensure_list(spec.get(field)) + _ensure_list(market.get(field)))
                else:
                    spec[field] = _dedup_by_json(_ensure_list(spec.get(field)) + _ensure_list(market.get(field)))

        # Build proof_of_superiority from top misalignments + market gaps
        top = [m for m in _ensure_list(misalign) if m.get("score", 0.0) >= 0.6][:5]
        poc = spec.setdefault("proof_of_superiority", [])
        for m in top:
            poc.append({
                "topic": m.get("feature"),
                "our_target": {"better_than": m.get("competitor"), "delta": 0.2},
                "benchmark_harness": f"bench_{(m.get('feature') or 'feature').replace(' ', '_').lower()}",
                "acceptance_criteria": [
                    f"Outperform {m.get('competitor')} on {m.get('feature')} by ≥20% median across ≥3 regions"
                ],
                "success_thresholds": {"regions": 3, "languages": 0}
            })

        # Ensure acceptance criteria exist
        spec.setdefault("acceptance_criteria", [])
        if market.get("gaps"):
            for g in _ensure_list(market.get("gaps"))[:5]:
                spec["acceptance_criteria"].append(f"Address market gap: {g}")
        spec["acceptance_criteria"] = _dedup_by_json(spec["acceptance_criteria"])  # de-dup strings
        return spec

    # -------------------------- Gateway safe wrapper -------------------------
    def _gw_structured(self, *, require_grounding: bool = False, **kwargs) -> Dict[str, Any]:
        """Call gateway.structured with graceful fallback if signature differs."""
        # Try with require_grounding (for FactGate), otherwise fall back.
        try:
            return self.gw.structured(require_grounding=require_grounding, **kwargs) or {}
        except TypeError:
            return self.gw.structured(**kwargs) or {}
