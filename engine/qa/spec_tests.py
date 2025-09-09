# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def gen_acceptance_tests(spec: Dict[str, Any]) -> bytes:
    behavior = (spec.get("behavior") or {})
    compute_name = behavior.get("name")

    T = []
    T.append("from fastapi.testclient import TestClient")
    T.append("import services.api.app as appmod")
    T.append("c = TestClient(appmod.app)\n")
    T.append("def test_healthz():")
    T.append("    r = c.get('/healthz'); assert r.status_code==200 and r.json().get('ok') is True\n")
    T.append("def test_root_lists_entities():")
    T.append("    r = c.get('/'); assert r.status_code==200 and isinstance(r.json().get('entities'), list)\n")
    if compute_name:
        T.append(f"def test_compute_{compute_name}():")
        T.append(f"    r = c.post('/compute/{compute_name}', json={{}}); assert r.status_code==200 and r.json().get('ok') is True\n")
    return ("\n".join(T) + "\n").encode("utf-8")
