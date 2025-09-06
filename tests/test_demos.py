# tests/test_demos.py
from security.policy import UserPolicy, check_network, check_fs, Denied
from adapters.installer import have
import os

def test_policy_denies_outbound():
    pol=UserPolicy(user_id="u", default_net="deny")
    try:
        check_network(pol,"connect","example.com",443)
    except Denied:
        pass
    else:
        raise AssertionError("expected network denied")

def test_fs_denies_exec_by_default(tmp_path):
    pol=UserPolicy(user_id="u", default_fs="deny")
    f = tmp_path/"a.sh"; f.write_text("#!/bin/sh\necho hi\n")
    try:
        check_fs(pol,"exec", str(f), require_exec=True)
    except Denied:
        pass
    else:
        raise AssertionError("expected exec denied")

def test_tools_mapping_presence():
    # לא מחייב התקנה בסביבת CI, רק בודק שאין קריסה כשבודקים זמינות
    _ = have("git")