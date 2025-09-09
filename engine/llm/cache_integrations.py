# engine/llm/cache_integrations.py
from __future__ import annotations
import time
from typing import Any, Dict, Callable
from engine.llm.cache import LLMCache

# Wrapper: קריאה ל‑LLM עם Cache + Audit ל‑hit/miss

def _normalize(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()[:4000]


def call_llm_with_cache(llm_fn: Callable[[Dict[str,Any]], Dict[str,Any]], prompt: Dict[str,Any], *, ctx: Dict[str,Any], cache: LLMCache) -> Dict[str,Any]:
    meta = prompt.get("meta", {})
    model = meta.get("model", "unknown")
    key = cache.make_key(
        model=model,
        system_v=str(meta.get("system_v","0")),
        template_v=str(meta.get("template_v","0")),
        tools_set=",".join(sorted(meta.get("tools",[]))),
        user_text_norm=_normalize(prompt.get("user_text","")),
        ctx_ids=",".join(sorted(meta.get("ctx_ids",[]))),
        persona_v=str(meta.get("persona_v","0")),
        policy_v=str(meta.get("policy_v","0")),
    )

    want_fresh = bool(meta.get("fresh_only", False))
    ttl_s = int(meta.get("ttl_s", 3600))

    if not want_fresh:
        ok, entry = cache.get(key)
        if ok and entry:
            return {"ok": True, "cached": True, "model": model, "content": entry.payload.get("content"), "meta": entry.payload.get("meta",{}), "cache_key": key}

    # near-hit אופציונלי
    if meta.get("allow_near_hit", False) and not want_fresh:
        nh = cache.near_hit(query=_normalize(prompt.get("user_text","")), model=model)
        if nh:
            e = nh[0]
            return {"ok": True, "cached": "near", "model": model, "content": e.payload.get("content"), "meta": e.payload.get("meta",{}), "cache_key": e.key}

    # קריאה אמיתית למודל
    out = llm_fn(prompt)
    payload = {"content": out.get("content"), "meta": out.get("meta",{}), "_user_text_norm": _normalize(prompt.get("user_text",""))}
    cache.put(key, model=model, payload=payload, ttl_s=ttl_s)
    return {"ok": True, "cached": False, "model": model, "content": out.get("content"), "meta": out.get("meta",{}), "cache_key": key}