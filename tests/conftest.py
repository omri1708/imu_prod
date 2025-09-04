from __future__ import annotations
import os, json
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def _env_isolation(tmp_path, monkeypatch):
    """
    • מפנה ספריות Audit/State לקבצים זמניים
    • שקט – לא מלכלך את העץ שלך
    """
    audit_dir = tmp_path / ".imu_audit"
    monkeypatch.setenv("IMU_AUDIT_DIR", str(audit_dir))
    # מפתח חתימה קבוע כדי לקבל fingerprints דטרמיניסטיים ב-Audit
    monkeypatch.setenv("IMU_AUDIT_KEY", "test_hmac_key")
    yield

@pytest.fixture
def fetcher_rows():
    """יוצר fetcher שמחזיר JSON bytes {"items": rows} ל-runtime_guard."""
    def _mk(rows):
        payload = json.dumps({"items": rows}).encode("utf-8")
        def f(url: str) -> bytes:
            assert url.startswith("https://api.example.com")
            return payload
        return f
    return _mk

@pytest.fixture
def policy_base(tmp_path):
    """מדיניות בסיס להרצת טסטים מקומיים (drift + auto-remediation)."""
    return {
        "runtime_check_enabled": True,
        "block_on_drift": True,
        "runtime_state_dir": str(tmp_path / "rt_state"),  # previous.json לוקאלי
        "auto_remediation": {"enabled": True, "apply_levels": ["conservative"], "max_rounds": 1},
        "allow_update_prev_hash_on_schema_ok": True,
        "runtime_prev_hash_map": {}  # מפה פר-טבלה (CI יכול להזין)
    }
