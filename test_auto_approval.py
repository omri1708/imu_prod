# tests/test_auto_approval.py
from __future__ import annotations
from policy.auto_approval import auto_approve

def test_admin_override_ok():
    gates={"ok":False,"reasons":["p95:exceeded"]}
    res=auto_approve(gates,"demo-user")  # demo-user has 'admin' by default in RBAC setup
    assert res["approve"] is True

def test_viewer_denied():
    gates={"ok":True}
    res=auto_approve(gates,"viewer-no-role")
    assert res["approve"] is False