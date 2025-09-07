# tests/test_quick_deploy_script.py
import os
def test_quick_deploy_script_exists_and_executable():
    p="scripts/quick_deploy.sh"
    assert os.path.exists(p)
    st=os.stat(p)
    assert (st.st_mode & 0o111) != 0  # executable bit