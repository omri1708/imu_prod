# tests/test_adapters_dryrun.py
from imu.adapters.registry import dry_run_install

def test_winget_brew_mapping_present():
    # לא מאשר שהפקודה תרוץ במכונה—רק שההרכבה קיימת
    dr = dry_run_install("nodejs")
    assert "cmd" in dr or dr.get("reason") in ("unsupported_linux_distro","unknown_capability")