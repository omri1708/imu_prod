# tests/test_smoke_all_script.py

import os, stat
def test_smoke_all_script_exists():
    p="scripts/smoke_all.sh"
    assert os.path.exists(p)
    assert os.stat(p).st_mode & stat.S_IXUSR