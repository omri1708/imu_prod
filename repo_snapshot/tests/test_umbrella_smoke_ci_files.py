# tests/test_umbrella_smoke_ci_files.py
import os, stat

def test_umbrella_smoke_ci_files_exist():
    assert os.path.exists("scripts/umbrella_smoke_kind_ci.sh")
    assert os.path.exists(".github/workflows/umbrella-kind-smoke.yml")
    assert os.stat("scripts/umbrella_smoke_kind_ci.sh").st_mode & stat.S_IXUSR