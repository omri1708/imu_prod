# tests/test_kind_ci_workflow_and_scripts.py

import os, stat

def test_kind_ci_workflow_and_scripts_exist():
    assert os.path.exists(".github/workflows/kind-smoke.yml")
    for p in ("scripts/smoke_kind_ci.sh",):
        assert os.path.exists(p)
        assert os.stat(p).st_mode & stat.S_IXUS