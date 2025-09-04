# imu_repo/tests/test_stage92_ui_sandbox.py
from __future__ import annotations
import re
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.caps_ui import ui_render_capability

def _cfg():
    cfg = load_config()
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    cfg["phi"] = {"max_allowed": 200.0, "per_capability_cost": {"ui.render": 1.0}}
    save_config(cfg)

def _has_csp(html: str) -> bool:
    return 'http-equiv="Content-Security-Policy"' in html and "default-src 'none'" in html

def _has_no_remote_refs(html: str) -> bool:
    # אין http/https ב-src/href
    return not re.search(r'\s(?:src|href)\s*=\s*"(https?:)?//', html, flags=re.I)

def _no_inline_script_without_nonce(html: str) -> bool:
    for m in re.finditer(r"<script([^>]*)>", html, flags=re.I):
        attrs = m.group(1)
        if "nonce=" not in attrs:
            return False
    return True

def test_ui_render_safety_and_evidence():
    _cfg()
    current().reset()
    with user_scope("aria"):
        cap = ui_render_capability("aria")
        payload = {
            "title":"Hello UI",
            "components":[
                {"kind":"text","id":"t1","props":{"text":"Welcome!"}},
                {"kind":"input","id":"q","props":{"type":"search","placeholder":"Type here"}},
                {"kind":"button","id":"go","props":{"label":"Go","action":"search"}},
                {"kind":"list","id":"lst","props":{"items":["a","b","c"]}},
                {"kind":"image","id":"logo","props":{"src":"data:image/png;base64,iVBORw0KGgo=","alt":"L"}},
                {"kind":"spacer","id":"sp1","props":{"h":24}},
                {"kind":"container","id":"box","children":[
                    {"kind":"text","id":"inner","props":{"text":"inside"}}
                ]}
            ]
        }
        out = cap.sync(payload)
        html = out["text"]
        assert html.startswith("<!DOCTYPE html>"), "not an html doc"
        assert _has_csp(html), "missing CSP"
        assert _has_no_remote_refs(html), "remote ref detected"
        assert _no_inline_script_without_nonce(html), "inline script without nonce"
        evs = current().snapshot()
        kinds = {e["kind"] for e in evs}
        assert "ui_render" in kinds and "ui_render_ok" in kinds, kinds

def test_ui_rejects_unsafe_image():
    _cfg()
    current().reset()
    with user_scope("aria"):
        cap = ui_render_capability("aria")
        bad = {
            "title":"Bad",
            "components":[
                {"kind":"image","id":"im","props":{"src":"https://evil.example/x.png","alt":"x"}}
            ]
        }
        out = cap.sync(bad)
        assert "ui_render_rejected" in out["text"], out

def run():
    test_ui_render_safety_and_evidence()
    test_ui_rejects_unsafe_image()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())