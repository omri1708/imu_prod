# tests/test_ci_metrics_helper.py
import os
def test_ci_metrics_script_exists():
    assert os.path.exists("scripts/ci_metrics.sh")