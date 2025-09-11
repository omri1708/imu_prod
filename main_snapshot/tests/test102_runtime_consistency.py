# imu_repo/tests/test_runtime_consistency.py
from __future__ import annotations
import json
from ui.dsl import Page, Component
from engine.runtime_guard import check_runtime_table, RuntimeBlocked

def _page_orders():
    return Page(
        title="Orders RT",
        components=[
            Component(kind="table", id="orders",
                props={
                    "endpoint": "https://api.example.com/orders",
                    "columns": [
                        {"name":"order_id","type":"string","required":True},
                        {"name":"amount","type":"number","required":True},
                        {"name":"created_at","type":"date"}
                    ],
                    "filters": {"amount":{"op":">=","value":100}},
                    "sort": {"by":"created_at","dir":"asc"}
                })
        ]
    )

def _mk_fetcher_ok(data_rows):
    payload = json.dumps({"items": data_rows}).encode("utf-8")
    def fetcher(url: str) -> bytes:
        assert url.startswith("https://api.example.com")
        return payload
    return fetcher

def test_runtime_pass_sorted_and_filtered():
    page = _page_orders()
    table_spec = {
        "path":"page.components[0]",
        "binding_url":"https://api.example.com/orders",
        "columns":[
            {"name":"order_id","type":"string","required":True},
            {"name":"amount","type":"number","required":True},
            {"name":"created_at","type":"date"},
        ],
        "filters":{"amount":{"op":">=","value":100}},
        "sort":{"by":"created_at","dir":"asc"}
    }
    rows = [
        {"order_id":"A1","amount":100,"created_at":"2024-01-01"},
        {"order_id":"A2","amount":150,"created_at":"2024-01-02"},
        {"order_id":"A3","amount":999.5,"created_at":"2024-01-03"},
    ]
    res = check_runtime_table(
        table_spec,
        policy={"runtime_check_enabled":True, "runtime_sample_limit":50},
        fetcher=_mk_fetcher_ok(rows)
    )
    assert res["ok"] and res["sampled"] == 3 and res["checked"] == 3

def test_runtime_block_missing_required():
    table_spec = {
        "path":"page.components[0]",
        "binding_url":"https://api.example.com/orders",
        "columns":[
            {"name":"order_id","type":"string","required":True},
            {"name":"amount","type":"number","required":True}
        ],
        "filters":None,"sort":None
    }
    rows = [{"order_id":"X","created_at":"2024-01-01"}]  # חסר amount
    try:
        check_runtime_table(table_spec, policy={"runtime_check_enabled":True}, fetcher=_mk_fetcher_ok(rows))
        assert False, "expected RuntimeBlocked"
    except RuntimeBlocked as e:
        assert "missing required column 'amount'" in str(e)

def test_runtime_block_unsorted():
    table_spec = {
        "path":"page.components[0]",
        "binding_url":"https://api.example.com/orders",
        "columns":[
            {"name":"order_id","type":"string"},
            {"name":"created_at","type":"date"}
        ],
        "filters":None,
        "sort":{"by":"created_at","dir":"asc"}
    }
    rows = [
        {"order_id":"A2","created_at":"2024-01-02"},
        {"order_id":"A1","created_at":"2024-01-01"},
    ]
    try:
        check_runtime_table(table_spec, policy={"runtime_check_enabled":True}, fetcher=_mk_fetcher_ok(rows))
        assert False, "expected RuntimeBlocked"
    except RuntimeBlocked as e:
        assert "runtime_sort" in str(e)