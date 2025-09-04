# imu_repo/tests/test_kpi_regression.py
from __future__ import annotations
import json, tempfile, os, shutil
from engine.kpi_regression import load_and_summarize, compare_kpis, gate_from_files, KPIRegressionBlocked

def _write_jsonl(p: str, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def test_kpi_regression_pass():
    tmp = tempfile.mkdtemp(prefix="imu_kpi_")
    try:
        base_p = os.path.join(tmp, "base.jsonl")
        cand_p = os.path.join(tmp, "cand.jsonl")
        # baseline: p95~90ms, 2% errors
        _write_jsonl(base_p, [
            {"ok":True,"latency_ms":80},
            {"ok":True,"latency_ms":90},
            {"ok":False,"latency_ms":120,"schema_errors":0},
            {"ok":True,"latency_ms":70}
        ])
        # candidate: p95~120ms (delta 30ms), 2.5% errors (delta 0.5%)
        _write_jsonl(cand_p, [
            {"ok":True,"latency_ms":95},
            {"ok":True,"latency_ms":110},
            {"ok":False,"latency_ms":130,"schema_errors":0},
            {"ok":True,"latency_ms":85}
        ])
        base = load_and_summarize(base_p)
        cand = load_and_summarize(cand_p)
        res  = compare_kpis(base, cand, policy={
            "max_p95_increase_ms": 50.0,
            "max_error_rate_increase": 0.01,  # 1%
            "block_on_schema_regression": True
        })
        assert res["ok"]
        # גם gate מהקבצים אמור לעבוד
        res2 = gate_from_files(base_p, cand_p, policy={
            "max_p95_increase_ms": 50.0,
            "max_error_rate_increase": 0.01,
            "block_on_schema_regression": True
        })
        assert res2["ok"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_kpi_regression_block_on_p95_or_errors():
    tmp = tempfile.mkdtemp(prefix="imu_kpi_")
    try:
        base_p = os.path.join(tmp, "base.jsonl")
        cand_p = os.path.join(tmp, "cand.jsonl")
        _write_jsonl(base_p, [
            {"ok":True,"latency_ms":50},
            {"ok":True,"latency_ms":60},
            {"ok":True,"latency_ms":70},
            {"ok":False,"latency_ms":80}
        ])
        # החמרה גדולה: גם p95 וגם error rate
        _write_jsonl(cand_p, [
            {"ok":True,"latency_ms":150},
            {"ok":False,"latency_ms":200},
            {"ok":True,"latency_ms":160},
            {"ok":False,"latency_ms":300,"schema_errors":1}
        ])
        base = load_and_summarize(base_p)
        cand = load_and_summarize(cand_p)
        try:
            compare_kpis(base, cand, policy={
                "max_p95_increase_ms": 50.0,
                "max_error_rate_increase": 0.01,
                "block_on_schema_regression": True
            })
            assert False, "expected KPIRegressionBlocked"
        except KPIRegressionBlocked as e:
            msg = str(e)
            assert "p95 regression" in msg or "error-rate regression" in msg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)