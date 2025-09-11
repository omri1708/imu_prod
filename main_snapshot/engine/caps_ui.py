# imu_repo/engine/caps_ui.py
from __future__ import annotations
from typing import Dict, Any, List
from grounded.claims import current
from engine.capability_wrap import text_capability_for_user
from engine.policy_ctx import get_user
from ui.dsl import Page, Component, validate_page, DSLValidationError
from ui.render import render_html

async def _ui_render_impl(payload: Dict[str, Any]) -> str:
    """
    payload:
      title: str
      components: [{kind, id, props?, children?}]
      theme?: {}
      nonce?: str
    """
    uid = get_user() or "anon"
    comps_json: List[Dict[str,Any]] = payload.get("components", [])
    def comp_from(d: Dict[str,Any]) -> Component:
        children = [comp_from(x) for x in d.get("children", [])]
        return Component(kind=d["kind"], id=d["id"], props=d.get("props", {}), children=children)
    components = [comp_from(c) for c in comps_json]
    page = Page(title=str(payload.get("title","Untitled")), components=components, theme=payload.get("theme", {}))
    try:
        html = render_html(page, nonce=str(payload.get("nonce", "IMU_NONCE")))
        current().add_evidence("ui_render_ok", {
            "source_url":"imu://ui/sandbox","trust":0.97,"ttl_s":600,
            "payload":{"user": uid, "title": page.title}
        })
        return html
    except DSLValidationError as e:
        current().add_evidence("ui_render_reject", {
            "source_url":"imu://ui/sandbox","trust":0.6,"ttl_s":600,
            "payload":{"error": str(e)}
        })
        return f"[FALLBACK] ui_render_rejected: {e}"

def ui_render_capability(user_id: str):
    # עלות קטנה — רינדור מקומי בלבד
    return text_capability_for_user(_ui_render_impl, user_id=user_id, capability_name="ui.render", cost=1.0)