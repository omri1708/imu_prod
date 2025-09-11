# imu_repo/tests/test_stage94b_ui_advanced_grid_table.py
from __future__ import annotations
from ui.dsl import Page, Component, validate_page
from ui.render import render_html

def test_named_areas_and_nested_grids_and_freeze():
    page = Page(
        title="Advanced Grid",
        components=[
            Component(kind="grid", id="g", props={
                "cols": 12,
                "areas": [
                    "header header header header header header header header header header header header",
                    "nav nav nav content content content content content content content ads ads",
                    "footer footer footer footer footer footer footer footer footer footer footer footer"
                ],
                "gap": 16
            }, children=[
                Component(kind="col", id="hdr", props={"area":"header"}, children=[
                    Component(kind="text", id="t1", props={"text":"Header"})
                ]),
                Component(kind="col", id="nv", props={"area":"nav"}, children=[
                    Component(kind="grid", id="g2", props={"cols": 2}, children=[
                        Component(kind="col", id="nv1", props={"span":1}, children=[Component(kind="text", id="tt1", props={"text":"Item 1"})]),
                        Component(kind="col", id="nv2", props={"span":1}, children=[Component(kind="text", id="tt2", props={"text":"Item 2"})])
                    ])
                ]),
                Component(kind="col", id="ct", props={"area":"content"}, children=[
                    Component(kind="table", id="tbl", props={
                        "columns": ["Name","Team","Score","Country","Date"],
                        "rows": [["Ana","A",10,"PT","2025-08-01"],["Ben","B",2,"IL","2025-07-01"],["Chen","C",30,"CN","2025-06-01"]],
                        "filter": True, "sortable": True,
                        "freeze_left": 1, "freeze_right": 1, "sticky_header": True
                    })
                ]),
                Component(kind="col", id="ads", props={"area":"ads"}, children=[Component(kind="text", id="ad", props={"text":"Ads"})]),
                Component(kind="col", id="ftr", props={"area":"footer"}, children=[Component(kind="text", id="ft", props={"text":"Footer"})]),
            ])
        ]
    )
    validate_page(page)
    html = render_html(page, nonce="X")
    assert "grid-template-areas:" in html
    assert "#hdr{grid-area:header}" in html or "grid-area:header" in html
    assert '<table id="tbl"' in html
    assert 'data-imu-table=' in html
    # freeze meta exist
    assert '"freeze_left": 1' in html and '"freeze_right": 1' in html
    # nested grid present
    assert 'id="g2"' in html

def run():
    test_named_areas_and_nested_grids_and_freeze()
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())