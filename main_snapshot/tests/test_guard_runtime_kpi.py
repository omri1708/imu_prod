# imu_repo/tests/test_guard_runtime_kpi.py
from __future__ import annotations
import json
from pathlib import Path
import pytest

from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies
from engine.kpi_regression import gate_from_files, KPIRegressionBlocked

# ---------- helpers ----------
def _fetcher_rows(rows):
    """מייצר fetcher שמחזיר {"items": rows} כ-bytes כדי ש-fetch_sample_with_raw ישאב ממנו."""
    payload = json.dumps({"items": rows}).encode("utf-8")
    def f(url: str) -> bytes:
        assert url.startswith("https://api.example.com")
        return payload
    return f

# ---------- 1) Runtime: פילטר חוסם -> רמדיז -> עובר ----------
def test_runtime_auto_remove_filter_then_pass():
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
    policy = {
        "runtime_check_enabled": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "allow_remove_filter_if_blocked": True,
    }
    rows = [{"order_id": "A", "amount": 50}, {"order_id": "B", "amount": 75}]

    # חסימה ראשונה בגלל פילטר
    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows))

    # רמדיז: הסרת פילטר
    diags = diagnose(blk.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[spec])
    assert rems, "ציפינו לרמדיז להסרת פילטר"
    apply_remedies(rems, policy=policy, table_specs=[spec])

    # עכשיו עובר
    out = check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows))
    assert out["ok"] and out["checked"] == len(rows)
    assert spec.get("filters") in (None, {})  # הפילטר הוסר

# ---------- 2) Runtime: Drift עם עדכון baseline במפה ----------
def test_runtime_drift_then_update_map_and_pass(tmp_path: Path):
    url = "https://api.example.com/users"
    spec = {
        "path": "page.components[0]",
        "binding_url": url,
        "columns": [{"name": "id", "type": "string", "required": True}],
        "filters": None, "sort": None,
    }
    rows_v1 = [{"id": "u1"}, {"id": "u2"}]
    rows_v2 = [{"id": "u1"}, {"id": "u2"}, {"id": "u3"}]

    policy = {
        "runtime_check_enabled": True,
        "block_on_drift": True,
        "runtime_state_dir": str(tmp_path / "rt_state"),  # fallback ל-local previous.json
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "allow_update_prev_hash_on_schema_ok": True,
        "runtime_prev_hash_map": {}  # מפה פר-טבלה (CI יכול להזין כאן baseline)
    }

    # ריצה 1: קובעת previous.json
    out1 = check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows_v1))
    assert out1["ok"] and "hash" in out1

    # ריצה 2: שינוי תוכן -> drift -> חסימה
    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows_v2))

    # רמדיז drift: לעדכן runtime_prev_hash_map[url] ל-hash החדש
    diags = diagnose(blk.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[spec])
    assert rems, "ציפינו לרמדיז drift"
    apply_remedies(rems, policy=policy, table_specs=[spec])

    # מוודאים שה-baseline עודכן במפה
    new_hash = policy["runtime_prev_hash_map"].get(url)
    assert new_hash, "runtime_prev_hash_map לא התעדכן"

    # מריצה 3: מאשרים את ה-hash החדש כ-prev ומצפים לעבור
    eff = dict(policy); eff["prev_content_hash"] = new_hash
    out2 = check_runtime_table(spec, policy=eff, fetcher=_fetcher_rows(rows_v2))
    assert out2["ok"] and out2["hash"] == new_hash

# ---------- 3) KPI: העלאת סף p95 לפי auto_raise_limits ----------
def test_kpi_auto_raise_p95_then_pass(tmp_path: Path):
    base_p = tmp_path / "base.jsonl"
    cand_p = tmp_path / "cand.jsonl"

    base_p.write_text("\n".join([
        json.dumps({"ok": True, "latency_ms": 60}),
        json.dumps({"ok": True, "latency_ms": 70}),
        json.dumps({"ok": True, "latency_ms": 80}),
    ]), encoding="utf-8")

    cand_p.write_text("\n".join([
        json.dumps({"ok": True, "latency_ms": 110}),
        json.dumps({"ok": True, "latency_ms": 115}),
        json.dumps({"ok": True, "latency_ms": 100}),
    ]), encoding="utf-8")

    policy = {
        "kpi_baseline_path": str(base_p),
        "kpi_candidate_path": str(cand_p),
        "max_p95_increase_ms": 10.0,      # קטן מדי בהתחלה
        "max_error_rate_increase": 0.0,
        "block_on_schema_regression": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "auto_raise_limits": {"p95_ms": 40.0},  # מותר להעלות עד 40ms
    }

    # ניסיון 1: חסימה
    with pytest.raises(KPIRegressionBlocked) as kb:
        gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)

    # רמדיז: העלאת p95 עד לגבול המותר → ניסיון 2: עובר
    diags = diagnose(kb.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[])
    assert rems, "ציפינו לרמדיז להעלאת p95"
    apply_remedies(rems, policy=policy, table_specs=[])
    out = gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)
    assert out["ok"]
    assert policy["max_p95_increase_ms"] >= 35.0
