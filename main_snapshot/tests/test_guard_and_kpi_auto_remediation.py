# imu_repo/tests/test_guard_and_kpi_auto_remediation.py
from __future__ import annotations
import json, time
from pathlib import Path

import pytest

from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies
from engine.kpi_regression import gate_from_files, KPIRegressionBlocked

# ---------- helpers ----------

def _fetcher_rows(rows):
    """
    מחזיר fetcher שמחזיר bytes של JSON עם {"items": rows}
    כדי ש-fetch_sample_with_raw יוכל לדגום.
    """
    payload = json.dumps({"items": rows}).encode("utf-8")
    def f(url: str) -> bytes:
        assert url.startswith("https://api.example.com")
        return payload
    return f


# ---------- 1) Runtime: פילטר חוסם → רמדיז → עובר ----------

def test_runtime_auto_remove_filter_then_pass():
    spec = {
        "path": "page.components[0]",
        "binding_url": "https://api.example.com/orders",
        "columns": [
            {"name": "order_id", "type": "string", "required": True},
            {"name": "amount",   "type": "number", "required": True},
        ],
        # הפילטר חוסם (amount>=100) אבל הנתונים יכילו ערכים קטנים
        "filters": {"amount": {"op": ">=", "value": 100}},
        "sort": None,
    }
    policy = {
        "runtime_check_enabled": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "allow_remove_filter_if_blocked": True,
    }
    rows = [{"order_id": "A", "amount": 50}, {"order_id": "B", "amount": 75}]

    # ניסיון ראשון: יחסם בגלל הפילטר
    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows))

    # דיאגנוזה + רמדיז
    diags = diagnose(blk.value)
    rems = propose_remedies(diags, policy=policy, table_specs=[spec])
    assert rems, "ציפינו לרמדיז שיסירו פילטר חוסם"
    apply_remedies(rems, policy=policy, table_specs=[spec])

    # עכשיו אמור לעבור
    out = check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows))
    assert out["ok"] and out["checked"] == len(rows)
    # הפילטר הוסר או נוקה
    assert spec.get("filters") in (None, {})


# ---------- 2) Runtime: Drift עם מפה פר-טבלה → עדכון baseline במפה → עובר ----------

def test_runtime_drift_map_then_accept_baseline_update(tmp_path: Path):
    url = "https://api.example.com/users"
    spec = {
        "path": "page.components[0]",
        "binding_url": url,
        "columns": [{"name": "id", "type": "string", "required": True}],
        "filters": None,
        "sort": None,
    }

    # payload1 ו-payload2 שונים → hash שונה → drift
    rows_v1 = [{"id": "u1"}, {"id": "u2"}]
    rows_v2 = [{"id": "u1"}, {"id": "u2"}, {"id": "u3"}]

    policy = {
        "runtime_check_enabled": True,
        "block_on_drift": True,
        # נשתמש גם ב-local state לאוטומציה (fallback) וגם במפה כשצריך שליטה:
        "runtime_state_dir": str(tmp_path / "rt_state"),
        # מפה פר-טבלה (תחילה ריקה)
        "runtime_prev_hash_map": {},
        # התירו קבלת hash חדש כאשר הסכימה תקינה (רמדיז)
        "allow_update_prev_hash_on_schema_ok": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
    }

    # ריצה 1: קובעת previous.json (לא נחסם, אין baseline במפה/מדיניות)
    out1 = check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows_v1))
    assert out1["ok"] and "hash" in out1

    # ריצה 2: תוכן השתנה → drift → יחסם כי block_on_drift=True
    with pytest.raises(RuntimeBlocked) as blk:
        check_runtime_table(spec, policy=policy, fetcher=_fetcher_rows(rows_v2))

    # רמדיז drift: יעדכן runtime_prev_hash_map[url] = new_hash (דרך propose/apply)
    diags = diagnose(blk.value)
    rems = propose_remedies(diags, policy=policy, table_specs=[spec])
    assert rems, "ציפינו לרמדיז drift"
    apply_remedies(rems, policy=policy, table_specs=[spec])

    # חשוב: ב-rollout_guard אנחנו מזריקים map→prev_content_hash לכל טבלה.
    # המדמה כאן: נעטוף מדיניות אפקטיבית עם prev_content_hash מה-Map:
    eff = dict(policy)
    new_hash = policy["runtime_prev_hash_map"].get(url)
    assert new_hash, "הרמדיז אמור היה לעדכן את המפה עם ה-hash החדש"
    eff["prev_content_hash"] = new_hash

    # ריצה 3: אמור לעבור כי baseline עודכן במפה
    out2 = check_runtime_table(spec, policy=eff, fetcher=_fetcher_rows(rows_v2))
    assert out2["ok"] and out2["hash"] == new_hash


# ---------- 3) KPI: העלאת סף p95 אוטומטית לפי policy.auto_raise_limits ----------

def test_kpi_auto_raise_p95_then_pass(tmp_path: Path):
    base_p = tmp_path / "base.jsonl"
    cand_p = tmp_path / "cand.jsonl"

    # baseline ~ p95≈80
    base_p.write_text("\n".join([
        json.dumps({"ok": True, "latency_ms": 60}),
        json.dumps({"ok": True, "latency_ms": 70}),
        json.dumps({"ok": True, "latency_ms": 80}),
    ]), encoding="utf-8")

    # candidate ~ p95≈115
    cand_p.write_text("\n".join([
        json.dumps({"ok": True, "latency_ms": 110}),
        json.dumps({"ok": True, "latency_ms": 115}),
        json.dumps({"ok": True, "latency_ms": 100}),
    ]), encoding="utf-8")

    policy = {
        "kpi_baseline_path": str(base_p),
        "kpi_candidate_path": str(cand_p),
        "max_p95_increase_ms": 10.0,      # קטן מדי בהתחלה → ייחסם
        "max_error_rate_increase": 0.0,
        "block_on_schema_regression": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "auto_raise_limits": {"p95_ms": 40.0},  # מותר להעלות עד 40ms
    }

    # ניסיון ראשון: חסימה
    with pytest.raises(KPIRegressionBlocked) as kb:
        gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)

    # רמדיז: העלאת סף p95 עד לגבול המותר → ואז אמור לעבור
    diags = diagnose(kb.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[])
    assert rems, "ציפינו לרמדיז להעלאת p95"
    apply_remedies(rems, policy=policy, table_specs=[])

    # ניסיון שני: עובר
    out = gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)
    assert out["ok"]
    assert policy["max_p95_increase_ms"] >= 35.0  # עלה מספיק
