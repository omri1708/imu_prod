# imu_repo/tests/test_stage93a_ui_grid_filter_sort.py
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

def test_grid_and_table_features():
    _cfg()
    current().reset()
    with user_scope("dev"):
        cap = ui_render_capability("dev")
        payload = {
            "title":"Grid + Table",
            "permissions":{},
            "components":[
                {"kind":"grid","id":"g","props":{"cols":12,"gap":16,"breakpoints":{"sm":480,"md":768,"lg":1024,"xl":1280}}},
                {"kind":"col","id":"c1","props":{"span":{"xs":12,"md":8}},"children":[
                    {"kind":"text","id":"t1","props":{"text":"Area A"}}
                ]},
                {"kind":"col","id":"c2","props":{"span":{"xs":12,"md":4}},"children":[
                    {"kind":"text","id":"t2","props":{"text":"Area B"}}
                ]},
                {"kind":"table","id":"tbl","props":{
                    "columns":["Name","Score"],
                    "rows":[["A", "10"],["B","2"],["C","30"]],
                    "filter": True, "sortable": True
                }}
            ]
        }
        out = cap.sync(payload)
        html = out["text"]
        assert '<div id="g" class="imu-grid">' in html
        assert 'id="c1"' in html and 'id="c2"' in html
        # CSS per-col spans
        assert '#c1{grid-column:span' in html and '#c2{grid-column:span' in html
        # table controls (filter enabled)
        assert 'data-imu-table=' in html
        assert 'imu-table-controls' in html or 'Filter...' in html
        kinds = {e["kind"] for e in current().snapshot()}
        assert "ui_render" in kinds and "ui_table_render" in kinds

def run():
    test_grid_and_table_features()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())