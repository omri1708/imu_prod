# tests/test_kind_smoke_scripts.py
import os, stat
def test_kind_scripts_exist():
    for p in ("scripts/kind_setup.sh","scripts/smoke_kind.sh"):
        assert os.path.exists(p)
        assert os.stat(p).st_mode & stat.S_IXUSR