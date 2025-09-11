from __future__ import annotations
import pytest
from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

def test_runtime_remove_filter_then_pass(policy_base, fetcher_rows):
    spec = {
        "path": "page.components[0]",
        "binding_url": "https://api.example.com/orders",
        "columns": [
            {"name": "order_id", "type": "string", "required": True},
            {"name": "amount",   "type": "number", "required": True},
        ],
        "filters": {"amount": {"op": ">=", "value": 100}},
        "sort": None,
    }
    rows = [{"order_id": "A", "amount": 50}, {"order_id": "B", "amount": 75}]

    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy_base, fetcher=fetcher_rows(rows))

    diags = diagnose(blk.value)
    rems  = propose_remedies(diags, policy=policy_base, table_specs=[spec])
    assert rems, "ציפינו לרמדיז להסרת פילטר"
    apply_remedies(rems, policy=policy_base, table_specs=[spec])

    out = check_runtime_table(spec, policy=policy_base, fetcher=fetcher_rows(rows))
    assert out["ok"] and out["checked"] == len(rows)
    assert spec.get("filters") in (None, {})

def test_runtime_relax_required_then_pass(policy_base, fetcher_rows):
    spec = {
        "path": "page.components[0]",
        "binding_url": "https://api.example.com/orders",
        "columns": [
            {"name": "order_id", "type": "string", "required": True},
            {"name": "amount",   "type": "number", "required": True},
        ],
        "filters": None, "sort": None,
    }
    rows = [{"order_id": "A"}, {"order_id": "B"}]  # אין amount

    # נבקש במדיניות לאפשר ריכוך required
    policy = dict(policy_base)
    policy["allow_relax_required_if_missing"] = True

    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy, fetcher=fetcher_rows(rows))

    diags = diagnose(blk.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[spec])
    assert rems, "ציפינו לרמדיז ל-required"
    apply_remedies(rems, policy=policy, table_specs=[spec])

    out = check_runtime_table(spec, policy=policy, fetcher=fetcher_rows(rows))
    assert out["ok"]
    req = {c["name"]: c.get("required", False) for c in spec["columns"]}
    assert req["amount"] is False
