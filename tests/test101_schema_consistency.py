# imu_repo/tests/test_schema_consistency.py
from __future__ import annotations
from grounded.claims import current
from ui.dsl import Page, Component
from engine.rollout_guard import run_negative_suite, RolloutBlocked

def setup_module(module=None):
    try: current().clear()
    except Exception: pass

def _page_with_table():
    return Page(
        title="Orders",
        components=[
            Component(kind="table", id="orders",
                      props={
                        "endpoint": "https://api.example.com/orders",
                        "columns": [
                            {"name":"order_id", "type":"string", "required": True},
                            {"name":"amount", "type":"number", "unit":"USD", "required": True},
                            {"name":"created_at", "type":"date"}
                        ],
                        "filters": {"amount": {"op":">", "value":100}},
                        "sort": {"by":"created_at", "dir":"desc"}
                      })
        ]
    )

def test_schema_block_when_missing_column():
    page = _page_with_table()
    # ראייה אחת עם סכימה שחסרה 'amount'
    current().clear()
    current().add_evidence("sch1", {
        "kind":"schema",
        "source_url":"https://api.example.com/orders",
        "ttl_s": 86400,
        "payload": {
            "columns":[
                {"name":"order_id", "type":"string"},
                {"name":"created_at", "type":"datetime"}
            ]
        },
        "trust": 0.9
    })
    try:
        run_negative_suite(page, current().snapshot(),
                           policy={"min_trust":0.7,"min_sources":1,"min_schema_sources":1,"min_schema_trust":0.7})
        assert False, "expected RolloutBlocked for missing column"
    except RolloutBlocked as e:
        assert "schema_error" in str(e) and "column 'amount'" in str(e)

def test_schema_pass_with_multi_sources_and_compat_date_datetime():
    page = _page_with_table()
    current().clear()
    # שתי ראיות: אחת orders, אחת prefix API root; טיפוס datetime נסבל מול UI date
    current().add_evidence("sch1", {
        "kind":"schema","source_url":"https://api.example.com/orders",
        "ttl_s": 86400, "trust":0.82,
        "payload":{"columns":[
            {"name":"order_id","type":"string"},
            {"name":"amount","type":"number","unit":"USD"},
            {"name":"created_at","type":"datetime"}  # תואם ל-ui 'date'
        ]}
    })
    current().add_evidence("sch2", {
        "kind":"docs","source_url":"https://api.example.com",
        "ttl_s": 86400, "trust":0.78,
        "payload":{"schema":{"columns":[
            {"name":"amount","type":"number","unit":"USD"}
        ]}}
    })
    res = run_negative_suite(page, current().snapshot(),
                             policy={"min_trust":0.7,"min_sources":1,"min_schema_sources":2,"min_schema_trust":0.7})
    assert res["ok"] and res["tables_checked"] >= 1