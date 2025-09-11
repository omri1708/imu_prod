# PATH: engine/blueprints/market_scan.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import json
from datetime import datetime

"""
- Uses LLMGateway with require_grounding=True (citations required).
"""

try:
    from engine.llm_gateway import LLMGateway
except Exception:
    # fallback
    class LLMGateway:
        def chat(self, **kw): return {"ok": False, "error": "gateway-missing"}

def _md_header(title: str) -> str:
    return f"# {title}\n\n_Generated: {datetime.utcnow().isoformat()}Z_\n\n"

def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    topic = (spec.get("title") or spec.get("summary") or "Product Market Scan").strip()
    ctx   = spec.get("context") or {}
    uid   = (ctx.get("user_id") or "user")

    # אם יש URLs שהוגדרו מראש בספק (או הועברו), נשתמש בהם; אחרת – נבקש מהמשתמש בצ'אט.
    sources = list((spec.get("analysis") or {}).get("sources") or [])
    gw = LLMGateway()

    prompt = (
        "Conduct a grounded market scan for the requested product.\n"
        "Return a JSON with fields:\n"
        "{ competitors: [ {name, url, features:[...], pricing, strengths, weaknesses} ],\n"
        "  gaps: [string],\n"
        "  recommendations: [string],\n"
        "  backlog: [ {title, rationale, metric} ] }\n"
        "Use the provided sources only; if too few, say 'insufficient_sources'."
    )

    res = gw.chat(
        user_id=uid, task="market_scan", intent="scan",
        content={"prompt": prompt, "sources": sources, "context": ctx},
        require_grounding=True, temperature=0.0
    )

    payload = res.get("payload") or {}
    citations = payload.get("citations") or []
    text = payload.get("text") or ""
    try:
        data = json.loads(text)
    except Exception:
        data = {"status": "insufficient_sources", "note": "Please provide URLs to competitors, reviews, docs."}

    # Markdown summary
    md = _md_header(f"Market Scan — {topic}")
    if data.get("status") == "insufficient_sources":
        md += "⚠️ **Insufficient sources** — please provide competitor URLs / reviews / docs to proceed.\n"
    else:
        md += "## Competitors\n"
        for c in data.get("competitors", []):
            md += f"- **{c.get('name')}** — {c.get('url')}\n"
            if c.get("features"):
                md += "  - features: " + ", ".join(c["features"]) + "\n"
            if c.get("pricing"):
                md += f"  - pricing: {c.get('pricing')}\n"
            if c.get("strengths"):
                md += "  - strengths: " + "; ".join(c["strengths"]) + "\n"
            if c.get("weaknesses"):
                md += "  - weaknesses: " + "; ".join(c["weaknesses"]) + "\n"
        md += "\n## Gaps\n" + "\n".join(f"- {g}" for g in data.get("gaps", [])) + "\n"
        md += "\n## Recommendations\n" + "\n".join(f"- {r}" for r in data.get("recommendations", [])) + "\n"
        md += "\n## Backlog\n" + "\n".join(f"- **{b.get('title')}** — {b.get('rationale')} (metric: {b.get('metric')})" for b in data.get("backlog", [])) + "\n"
        if citations:
            md += "\n## Citations\n" + "\n".join(f"- {c}" for c in citations) + "\n"

    return {
        "docs/market_scan.json": json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        "docs/market_scan.md": md.encode("utf-8"),
    }
