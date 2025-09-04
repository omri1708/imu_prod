from __future__ import annotations
import json
from pathlib import Path
import pytest

from engine.kpi_regression import gate_from_files, KPIRegressionBlocked, KPIDataError
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

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
        "max_p95_increase_ms": 10.0,
        "max_error_rate_increase": 0.0,
        "block_on_schema_regression": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "auto_raise_limits": {"p95_ms": 40.0},
    }

    with pytest.raises(KPIRegressionBlocked) as kb:
        gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)

    diags = diagnose(kb.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[])
    assert rems
    apply_remedies(rems, policy=policy, table_specs=[])

    out = gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)
    assert out["ok"]
    assert policy["max_p95_increase_ms"] >= 35.0

def test_kpi_invalid_jsonl_raises(tmp_path: Path):
    base_p = tmp_path / "b.jsonl"
    cand_p = tmp_path / "c.jsonl"
    base_p.write_text('{"ok":true,"latency_ms":80}\n', encoding="utf-8")
    # שורה פגומה (לא JSON) → KPIDataError
    cand_p.write_text('not_json_line\n', encoding="utf-8")

    policy = {
        "kpi_baseline_path": str(base_p),
        "kpi_candidate_path": str(cand_p),
        "max_p95_increase_ms": 0.0,
        "max_error_rate_increase": 0.0,
        "block_on_schema_regression": True,
    }

    with pytest.raises(KPIDataError):
        gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)
