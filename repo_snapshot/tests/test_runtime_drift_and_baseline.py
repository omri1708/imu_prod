from __future__ import annotations
from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

def test_runtime_drift_map_then_accept(policy_base, fetcher_rows):
    url = "https://api.example.com/users"
    spec = {
        "path": "page.components[0]",
        "binding_url": url,
        "columns": [{"name": "id", "type": "string", "required": True}],
        "filters": None, "sort": None,
    }

    rows_v1 = [{"id": "u1"}, {"id": "u2"}]
    rows_v2 = [{"id": "u1"}, {"id": "u2"}, {"id": "u3"}]

    # ריצה ראשונה – מייצרת previous.json
    out1 = check_runtime_table(spec, policy=policy_base, fetcher=fetcher_rows(rows_v1))
    assert out1["ok"] and "hash" in out1

    # שינוי תוכן → drift → חסימה
    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy_base, fetcher=fetcher_rows(rows_v2))

    # remedy: עדכון runtime_prev_hash_map[url] ל-hash חדש
    diags = diagnose(blk.value)
    rems  = propose_remedies(diags, policy=policy_base, table_specs=[spec])
    assert rems, "ציפינו לרמדיז drift"
    apply_remedies(rems, policy=policy_base, table_specs=[spec])

    new_hash = policy_base["runtime_prev_hash_map"].get(url)
    assert new_hash, "baseline לא התעדכן במפה"

    eff = dict(policy_base); eff["prev_content_hash"] = new_hash
    out2 = check_runtime_table(spec, policy=eff, fetcher=fetcher_rows(rows_v2))
    assert out2["ok"] and out2["hash"] == new_hash
