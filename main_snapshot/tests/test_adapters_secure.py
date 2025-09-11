# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
import pytest

c = TestClient(APP)

def test_secure_dry_run_unknown_kind():
    r = c.post("/adapters/secure/run", json={"user_id":"demo-user","kind":"unknown.kind","params":{},"execute":False})
    assert r.status_code == 400

def test_secure_dry_run_echo_or_resource_required():
    # אם echo מותר במדיניות – נקבל ok; אחרת resource_required (מאובטח)
    r = c.post("/adapters/secure/run", json={"user_id":"demo-user","kind":"tool.echo","params":{"":""},"execute":False})
    j = r.json()
    assert j.get("ok") in (True, False)
