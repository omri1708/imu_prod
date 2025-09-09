# PATH: engine/llm_gateway.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
from datetime import datetime, timedelta

from engine.prompt_builder import PromptBuilder
from user_model.model import UserStore
from engine.llm.cache_integrations import call_llm_with_cache
from engine.llm.cache import default_cache

"""
LLMGateway (hybrid, hardened)
- Combines the production-ready gateway (OpenAI/Anthropic + JSON-mode + provenance)
  with SubjectEngine persona enrichment and explicit FactGate grounding enforcement.
- If require_grounding=True → we *explicitly* call FactGate.require_sources(sources)
  and fail fast when sources are missing/invalid.

Anchors (per user's request):
- No violation of provider ToS; lawful, permitted use.
- Provided as part of a legitimate paid service; user controls usage & data.
- Any limitation should be explicit and documented.

This file is self-contained and ready for VS Code.
"""

# Optional SubjectEngine (for richer persona)
try:  # pragma: no cover - optional dependency
    from user_model.subject import SubjectEngine  # type: ignore
except Exception:  # pragma: no cover - keep gateway working without it
    SubjectEngine = None  # type: ignore

# --- Provenance & policy fallbacks -----------------------------------------------------------
try:  # pragma: no cover - prefer new API
    from grounded.provenance import ProvenanceStore as _ProvStore  # type: ignore
except Exception:  # pragma: no cover - older store name
    try:
        from grounded.provenance_store import EvidenceStore as _ProvStore  # type: ignore
    except Exception:
        _ProvStore = None  # type: ignore

try:  # pragma: no cover - policy may be a singleton or dict
    from grounded.source_policy import policy_singleton as _SP  # type: ignore
except Exception:  # sensible defaults if policy not present
    _SP = None

# FactGate (strict grounding)
try:  # pragma: no cover - library optional
    from grounded.fact_gate import (
        FactGate, RefusedNotGrounded,
        SchemaRule, UnitRule, FreshnessRule, EvidenceIndex  # עשויים לא להיות בשימוש, נופלים לגרייספול
    )  # type: ignore
except Exception:  # minimal stub for type continuity
    FactGate = None  # type: ignore

    class RefusedNotGrounded(Exception):  # type: ignore
        ...

# --- Providers detection ---------------------------------------------------------------------
_HAS_OPENAI = False
_HAS_ANTHROPIC = False
_openai_client = None
_anthropic_client = None

if os.environ.get("OPENAI_API_KEY"):
    try:  # pragma: no cover - runtime import
        from openai import OpenAI  # openai>=1.x

        _openai_client = OpenAI()
        _HAS_OPENAI = True
    except Exception:
        _openai_client = None
        _HAS_OPENAI = False

if os.environ.get("ANTHROPIC_API_KEY"):
    try:  # pragma: no cover - runtime import
        import anthropic  # type: ignore

        _anthropic_client = anthropic.Anthropic()
        _HAS_ANTHROPIC = True
    except Exception:
        _anthropic_client = None
        _HAS_ANTHROPIC = False


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else ([] if x is None else [x])

class LLMGateway:
    """Unified LLM gateway with persona, provenance & strict grounding.

    - Persona: combines UserStore profile and (optionally) SubjectEngine.persona(uid).
    - Grounding: when require_grounding=True → enforce via FactGate.require_sources(...).
    - Providers: OpenAI/Anthropic if keys exist; otherwise deterministic local fallbacks.
    """

    def __init__(
        self,
        templates_path: str = "config/prompts.jsonl",
        stats_path: str = "assurance_store/llm_stats.json",
        user_store_path: str = "./assurance_store_users",
        prov_root: Optional[str] = None,
        
    ) -> None:
        # Prompt templates & stats
        try:
            self.pb = PromptBuilder(templates_path, stats_path)
        except TypeError:  # backward-compat: older PromptBuilder()
            self.pb = PromptBuilder()  # type: ignore[misc]

        # User stores & subject engine (optional)
        self.users = UserStore(user_store_path)
        self.subject = SubjectEngine(self.users) if SubjectEngine else None

        # Provenance index for citations (optional)
        prov_root = prov_root or os.environ.get("IMU_PROV_ROOT") or "./assurance_store_text"
        self.prov = _ProvStore(prov_root) if _ProvStore else None

        # FactGate (optional but used for strict grounding)
        self.fact_gate = None
        if FactGate:
            try:
                # ננסה לבנות EvidenceIndex וכללי ברירת מחדל; אם לא מסתדר → fallback
                rules = [SchemaRule(), UnitRule(), FreshnessRule(max_age_seconds=86400)]
                idx = EvidenceIndex(self.prov) if self.prov else None  # עשוי להיות None אם אין פרובננס
                self.fact_gate = FactGate(idx=idx, rules=rules)  # type: ignore[arg-type]
            except Exception:
                self.fact_gate = FactGate()  # type: ignore[call-arg]

        # Model selection (overridable by ENV)
        self.oa_chat_model = _env("IMU_OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.oa_json_model = _env("IMU_OPENAI_JSON_MODEL", "gpt-4o-mini")
        self.claude_model = _env("IMU_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self._cache = default_cache()
    # -------------------------------- Persona -----------------------------------------------
    def _persona(self, user_id: str) -> Dict[str, Any]:
        """Return persona dict using both SubjectEngine and UserStore (robust).

        Resolution order:
        1) SubjectEngine.persona(uid) if available & returns dict.
        2) UserStore common APIs: get/load/read/fetch/get_user/get_profile.
        3) Fallback to JSON files in user_store_path.
        4) Minimal default persona.
        """
        # 1) SubjectEngine (if present)
        if self.subject is not None:
            try:
                p = self.subject.persona(user_id)  # type: ignore[attr-defined]
                if isinstance(p, dict) and p:
                    return p
            except Exception:
                pass

        # 2) Try several common method names on UserStore
        rec: Optional[Dict[str, Any]] = None
        for attr in ("get", "load", "read", "fetch", "get_user", "get_profile"):
            fn = getattr(self.users, attr, None)
            if callable(fn):
                try:
                    val = fn(user_id)
                    if isinstance(val, dict) and val:
                        rec = val
                        break
                except Exception:
                    continue

        # 3) Fallback: disk files under user_store_path
        if not rec:
            base = getattr(self.users, "root", "./assurance_store_users")
            for fname in (f"{user_id}.json", f"{user_id}.mem.json"):
                path = os.path.join(str(base), fname)
                if os.path.exists(path):
                    try:
                        rec = json.loads(open(path, "r", encoding="utf-8").read())
                        break
                    except Exception:
                        pass

        persona = (rec or {}).get("persona") if isinstance(rec, dict) else None
        if isinstance(persona, dict):
            return persona
        return {"uid": user_id, "tone": {"style": "concise", "temperature": 0.0}}

    # ------------------------------- Provider calls -----------------------------------------
    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        assert _HAS_OPENAI and _openai_client is not None, "OpenAI client not initialized"
        model = model or (self.oa_json_model if json_mode else self.oa_chat_model)
        params: Dict[str, Any] = {"model": model, "messages": messages, "temperature": float(temperature)}
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        resp = _openai_client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        json_mode: bool = False,  # Anthropics JSON mode varies; we keep manual discipline
        temperature: float = 0.0,
    ) -> str:
        assert _HAS_ANTHROPIC and _anthropic_client is not None, "Anthropic client not initialized"
        model = model or self.claude_model
        system_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
        user_txt = "\n".join([m["content"] for m in messages if m["role"] == "user"]).strip()
        resp = _anthropic_client.messages.create(  # type: ignore[call-arg]
            model=model,
            system=system_txt or None,
            messages=[{"role": "user", "content": user_txt}],
            temperature=float(temperature),
            max_tokens=4096,
        )
        parts = getattr(resp, "content", [])
        for p in parts:
            # SDKs return Attrs or dicts; support both
            t = getattr(p, "type", None) or (p.get("type") if isinstance(p, dict) else None)
            if t in (None, "text"):
                return getattr(p, "text", p.get("text", "")) if isinstance(p, dict) else (getattr(p, "text", "") or "")
        return ""

    # -------------------------------- Public API --------------------------------------------
    def chat(
        self,
        user_id: str,
        task: str,
        intent: str,
        content: Dict[str, Any],
        *,
        require_grounding: bool = False,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Chat answer with optional strict grounding and recorded citations.

        - If require_grounding=True, we *enforce* FactGate.require_sources(sources)
          *before* calling a model. If the gate fails → return ok=False.
        - Citations (urls) are always recorded to provenance if available.
        """
        persona = self._persona(user_id)
        msgs = self.pb.compose(user_id=user_id, task=task, intent=intent, persona=persona, content=content, json_only=False)

        sources: List[str] = list(content.get("sources") or [])
        citations: List[Dict[str, Any]] = []

        # Record to provenance (even if not strictly required)
        if self.prov and sources:
            for u in sources:
                try:
                    digest = self.prov.put(json.dumps({"url": u}, ensure_ascii=False).encode("utf-8"), {"url": u, "kind": "url"})
                    citations.append({"url": u, "digest": digest})
                except Exception:
                    citations.append({"url": u, "digest": None})
        else:
            citations = [{"url": u, "digest": None} for u in sources]

        # === Strict grounding enforcement ===
        if require_grounding:
            if not sources:
                return {"ok": False, "error": "not_grounded", "details": "sources required but missing"}
            if not self.fact_gate:
                return {"ok": False, "error": "not_grounded", "details": "FactGate unavailable"}
            try:
                # Explicit hard check (from v1 semantics)
                _ = self.fact_gate.require_sources(sources)  # type: ignore[union-attr]
            except RefusedNotGrounded as e:  # explicit semantic failure
                return {"ok": False, "error": "not_grounded", "details": str(e)}
            except Exception as e:  # unexpected gate failure
                return {"ok": False, "error": "gate_error", "details": str(e)}
        
        meta = {
            "model": self.oa_chat_model if _HAS_OPENAI else (self.claude_model if _HAS_ANTHROPIC else "local"),
            "system_v": "1", "template_v": "1",
            "tools": [], "ctx_ids": [], "persona_v": "1", "policy_v": "1",
            "ttl_s": 3600, "allow_near_hit": True
        }
        prompt = {"user_text": msgs[-1]["content"], "meta": meta}
        def _llm_fn(p):  # עוטף את הקריאה האמיתית
            if _HAS_OPENAI:
                return {"content": self._openai_chat(msgs, json_mode=False, temperature=temperature)}
            elif _HAS_ANTHROPIC:
                return {"content": self._anthropic_chat(msgs, json_mode=False, temperature=temperature)}
            else:
                return {"content": self._local_answer(msgs, sources)}
        cached = call_llm_with_cache(_llm_fn, prompt, ctx={"user_id": user_id}, cache=self._cache)
        text = cached["content"]

        # === Model call (real if keys exist; else local fallback) ===
        if _HAS_OPENAI:
            text = self._openai_chat(msgs, json_mode=False, temperature=temperature)
        elif _HAS_ANTHROPIC:
            text = self._anthropic_chat(msgs, json_mode=False, temperature=temperature)
        else:
            text = self._local_answer(msgs, sources)

        payload = {
            "text": text,
            "citations": citations,
            "root": (citations[0].get("digest") if citations else None) or "prov",
        }
        return {"ok": True, "payload": payload}

    def structured(
        self,
        user_id: str,
        task: str,
        intent: str,
        schema_hint: Optional[str] = None,
        prompt: Optional[str] = None,
        *,
        content: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        require_grounding: bool = False,
    ) -> Dict[str, Any]:
        """Structured JSON response with optional strict grounding & freshness.

        When ``require_grounding`` is True, we enforce:
          - at least ``min_sources`` citations (deduped by URL)
          - if ``freshness_months`` > 0, at least ``min_sources`` citations carry a
            recent date (<= freshness window). Dates are taken from common fields
            like: ``last_seen`` / ``date`` in citations or competitor_claims / claim_checks.
          - provenance recording and FactGate.require_sources(urls) if available
        """
        persona = self._persona(user_id)
        if content is None:
            content = {"prompt": prompt or "", "schema_hint": schema_hint or "{}", "context": {}}
        msgs = self.pb.compose(user_id=user_id, task=task, intent=intent, persona=persona, content=content, json_only=True)

        # --- real model calls or local fallback ---
        raw = ""
        if _HAS_OPENAI:
            raw = self._openai_chat(msgs, json_mode=True, temperature=temperature)
        elif _HAS_ANTHROPIC:
            # ask Claude to return raw JSON text
            msgs[-1]["content"] += "החזר JSON תקין בלבד, ללא טקסט נוסף."
            raw = self._anthropic_chat(msgs, json_mode=False, temperature=temperature)
        else:
            # If grounding is required but we have no external model, fail fast
            if require_grounding:
                return {"ok": False, "error": "not_grounded", "details": "llm_unavailable_for_grounded_call"}
            return self._structured_local(intent, str(content.get("prompt", "")))

        # --- normalize to dict ---
        try:
            data = json.loads(raw)
        except Exception:
            import re as _re
            m = _re.search(r"\{.*\}", raw, _re.S)
            data = json.loads(m.group(0)) if m else {}

        # --- strict grounding & freshness enforcement (optional) ---
        if require_grounding:
            # extract policy (allow override by ENV)
            pol = data.get("evidence_policy") if isinstance(data, dict) else None
            def _env_int(name: str, default: int) -> int:
                try:
                    return int(os.environ.get(name, default))
                except Exception:
                    return default
            min_sources = int((pol or {}).get("min_sources", _env_int("IMU_EVIDENCE_MIN_SOURCES", 2)))
            freshness_months = int((pol or {}).get("freshness_months", _env_int("IMU_EVIDENCE_FRESHNESS_MONTHS", 12)))

            # helpers (local to this method)
            def _parse_dt(s: str) -> Optional[datetime]:
                if not s or not isinstance(s, str):
                    return None
                s2 = s.strip().replace("Z", "")
                # try ISO / YYYY-MM-DD
                for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return datetime.fromisoformat(s2) if fmt is None else datetime.strptime(s2, fmt)
                    except Exception:
                        pass
                return None

            def _collect_urls_and_dates(d: Dict[str, Any]) -> List[Tuple[str, Optional[datetime]]]:
                pairs: List[Tuple[str, Optional[datetime]]] = []
                # 1) top-level citations: can be [str] or [{sources:[..], last_seen:..}]
                for c in _ensure_list(d.get("citations")):
                    if isinstance(c, str):
                        pairs.append((c, None))
                    elif isinstance(c, dict):
                        dt = _parse_dt(str(c.get("last_seen") or c.get("date") or ""))
                        for u in _ensure_list(c.get("sources")):
                            if isinstance(u, str):
                                pairs.append((u, dt))
                # 2) competitor_claims: evidence[] + last_seen
                for cl in _ensure_list(d.get("competitor_claims")):
                    dt = _parse_dt(str(cl.get("last_seen") or ""))
                    for u in _ensure_list(cl.get("evidence")):
                        if isinstance(u, str):
                            pairs.append((u, dt))
                # 3) claim_checks: citations[] (date may be in details/result; ignore unless explicit)
                for chk in _ensure_list(d.get("claim_checks")):
                    dt = _parse_dt(str(chk.get("last_seen") or ""))
                    for u in _ensure_list(chk.get("citations")):
                        if isinstance(u, str):
                            pairs.append((u, dt))
                return pairs

            pairs = _collect_urls_and_dates(data if isinstance(data, dict) else {})
            # dedup by URL, keep the *freshest known date* per URL
            url_to_dt: Dict[str, Optional[datetime]] = {}
            for u, dt in pairs:
                if u not in url_to_dt:
                    url_to_dt[u] = dt
                else:
                    # keep newer
                    if dt and (url_to_dt[u] is None or (dt > url_to_dt[u])):
                        url_to_dt[u] = dt

            urls = list(url_to_dt.keys())
            if len(urls) < min_sources:
                return {"ok": False, "error": "not_grounded", "details": f"insufficient_citations: {len(urls)}/{min_sources}"}

            # freshness check: require at least min_sources URLs with known, recent dates
            if freshness_months > 0:
                cutoff = datetime.utcnow() - timedelta(days=30 * freshness_months)
                fresh_count = sum(1 for dt in url_to_dt.values() if dt and dt >= cutoff)
                if fresh_count < min_sources:
                    return {
                        "ok": False,
                        "error": "not_fresh",
                        "details": f"fresh_citations {fresh_count} < required {min_sources} within {freshness_months}m",
                    }

            # record to provenance (best-effort)
            citations: List[Dict[str, Any]] = []
            if self.prov:
                for u in urls:
                    try:
                        dgt = self.prov.put(json.dumps({"url": u}, ensure_ascii=False).encode("utf-8"), {"url": u, "kind": "url"})
                        citations.append({"url": u, "digest": dgt})
                    except Exception:
                        citations.append({"url": u, "digest": None})

            # FactGate hard-check
            if self.fact_gate:
                try:
                    self.fact_gate.require_sources(urls)  # may enforce trust/freshness by policy rules
                except Exception as e:
                    return {"ok": False, "error": "not_grounded", "details": str(e)}

            # attach normalized citations back for callers
            if isinstance(data, dict) and citations:
                data.setdefault("citations", citations)

        return {"ok": True, "json": data}

    # -------------------------------- Fallbacks ----------------------------------------------
    def _local_answer(self, messages: List[Dict[str, str]], sources: List[str]) -> str:
        usr = next((m["content"] for m in messages if m["role"] == "user"), "")
        base = usr.split("\n")[-1].strip() or "תשובה קצרה"
        cite = f"\n(מקורות: {', '.join(sources)})" if sources else ""
        return base + cite

    def _structured_local(self, intent: str, prompt: str) -> Dict[str, Any]:
        if intent in ("summarize", "memory"):
            lines = [ln.strip() for ln in str(prompt).splitlines() if ln.strip()]
            return {"ok": True, "json": {"summary": " ".join(lines[-6:])[:800], "facts": []}}
        title = str(prompt)[:64] or "App"
        spec = {
            "title": title,
            "summary": title,
            "components": [{"name": "api", "type": "api", "tech": ["python", "fastapi"], "requires": []}],
            "tools": [{"category": "nlp.llm", "preferred": ["provider_local"], "acceptable": []}],
            "dataflows": [{"from": "web", "to": "api", "via": "http"}],
            "deployment": {"targets": ["docker"], "env": ["dev"]},
            "privacy": {"consent": [], "retention_days": 0},
            "tests": [{"name": "unit", "kind": "unit", "target": "api"}],
        }
        return {"ok": True, "json": spec}


__all__ = ["LLMGateway"]


