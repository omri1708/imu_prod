# tests/test_unity_k8s_e2e.py (בדיקת קצה־לקצה—תריץ רק אם Unity/K8s זמינים)
# -*- coding: utf-8 -*-
import os, shutil, subprocess, pytest, time, requests

def have(cmd:str)->bool:
    from shutil import which
    return which(cmd) is not None

@pytest.mark.integration
def test_unity_to_k8s_smoke():
    # בדיקה רצה רק אם יש Unity ו-kubectl זמינים
    if not have("Unity") and not have("unity"):
        pytest.skip("Unity CLI not present")
    if not have("kubectl"):
        pytest.skip("kubectl not present")

    # נדרש שרץ artifact_server ב-:8089 ו-api/http_api (uvicorn) ב-:8000
    try:
        requests.get("http://localhost:8089/upload", timeout=1)
    except Exception:
        pytest.skip("artifact server not running")

    project = os.environ.get("UNITY_PROJECT_PATH")
    if not project or not os.path.isdir(project):
        pytest.skip("UNITY_PROJECT_PATH not set to a valid path")

    body = {
        "kind": "unity_k8s",
        "project_path": project,
        "k8s_image": "alpine:3.19",
        "artifact_server_url": "http://localhost:8089",
        "claims": {"claims": []}  # אם נדרשת אכיפת Evidence קשיחה—הכנס HASH-ים תקפים
    }
    r = requests.post("http://localhost:8000/run_adapter?user=default&kind=unity_k8s", json=body, timeout=5)
    assert r.status_code in (200, 428), r.text