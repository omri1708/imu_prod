from __future__ import annotations
import json
from pathlib import Path
import pytest

from engine.kpi_regression import gate_from_files, KPIRegressionBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

def test_generated_kpi_cases(tmp_path: Path):
    src = Path("imu_repo/tests/generated/kpi_cases")
    base = tmp_path / "b.jsonl"
    cand = tmp_path / "c.jsonl"
    base.write_text((src / "baseline.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    cand.write_text((src / "candidate.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    policy = {
        "kpi_baseline_path": str(base),
        "kpi_candidate_path": str(cand),
        "max_p95_increase_ms": 5.0,  # בכוונה נמוך
        "max_error_rate_increase": 0.0,
        "block_on_schema_regression": True,
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "auto_raise_limits": {"p95_ms": 50.0},
    }

    with pytest.raises(KPIRegressionBlocked) as kb:
        gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)

    diags = diagnose(kb.value)
    rems  = propose_remedies(diags, policy=policy, table_specs=[])
    assert rems
    apply_remedies(rems, policy=policy, table_specs=[])
    out = gate_from_files(policy["kpi_baseline_path"], policy["kpi_candidate_path"], policy=policy)
    assert out["ok"]
    assert policy["max_p95_increase_ms"] >= 25.0
