# imu_repo/tests/test_stage92c_ui_extended.py
from __future__ import annotations
import re, json
from grounded.claims import current
from engine.user_scope import user_scope
from engine.config import load_config, save_config
from engine.caps_ui import ui_render_capability

def _cfg():
    cfg = load_config()
    cfg["guard"] = {"min_trust": 0.0, "max_age_s": 3600.0, "min_count": 0, "required_kinds": []}
    cfg["phi"] = {"max_allowed": 200.0, "per_capability_cost": {"ui.render": 1.0}}
    save_config(cfg)

def _has_perm_meta(html: str) -> bool:
    return 'http-equiv="Permissions-Policy"' in html

def test_ui_dsl_extended_form_schema_and_permissions():
    _cfg()
    current().reset()
    with user_scope("noa"):
        cap = ui_render_capability("noa")
        payload = {
            "title":"FormX",
            "permissions":{"geolocation": True, "microphone": False, "camera": False},
            "components":[
                {"kind":"form","id":"f1","props":{"submit_label":"Send","schema":{
                    "rules":[
                        {"field":"email","type":"string","required":True,"pattern":"^[^@]+@[^@]+\\.[^@]+$"},
                        {"field":"age","type":"number","minimum":18,"maximum":99},
                        {"field":"agree","type":"boolean","required":True}
                    ]}},
                 "children":[
                    {"kind":"input","id":"email","props":{"type":"email","name":"email","placeholder":"you@example.com"}},
                    {"kind":"input","id":"age","props":{"type":"number","name":"age","placeholder":"18"}},
                    {"kind":"checkbox","id":"agree","props":{"label":"I agree","name":"agree"}}
                 ]},
                {"kind":"button","id":"askgeo","props":{"label":"Enable Location","action":"perm:geolocation"}},
                {"kind":"button","id":"readgeo","props":{"label":"Get Location","action":"sensor:geo"}},
                {"kind":"table","id":"t","props":{"columns":["A","B"],"rows":[["1","2"],["3","4"]]}},
            ]
        }
        out = cap.sync(payload)
        html = out["text"]
        assert html.startswith("<!DOCTYPE html>")
        assert 'data-imu-form="1"' in html, "form marker missing"
        assert "window.__IMU_FORM_VALIDATORS__" in html, "validators bundle missing"
        assert _has_perm_meta(html), "permissions-policy meta missing"
        # אין רפרנסים חיצוניים
        assert not re.search(r'\s(?:src|href)\s*=\s*"(https?:)?//', html, flags=re.I)

        evs = current().snapshot()
        kinds = {e["kind"] for e in evs}
        assert "ui_render" in kinds and "ui_render_ok" in kinds

def run():
    test_ui_dsl_extended_form_schema_and_permissions()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())