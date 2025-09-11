# tests/test_e2e_unity_k8s.py
import os, json, time, urllib.request
from contextlib import closing
from server.boot import boot_http
from engine.errors import ResourceRequired

def post(path, obj, user="anon"):
    b=json.dumps(obj).encode()
    req=urllib.request.Request(f"http://127.0.0.1:8088{path}", data=b, method="POST",
                               headers={"Content-Type":"application/json", "X-IMU-User":user})
    with closing(urllib.request.urlopen(req)) as r: 
        return json.loads(r.read().decode())

def test_unity_then_k8s_flow():
    srv=boot_http(); time.sleep(0.2)
    try:
        # Unity build
        try:
            r1=post("/run_adapter?name=unity_build", {"project_path":"./unity_project","build_target":"StandaloneLinux64"})
        except ResourceRequired as e:
            # מותר – אין Unity. הוכחה שהמנגון פועל
            assert "Unity Editor CLI not found" in str(e) or "resource_required" in str(e)
            return
        assert r1["status"]=="started"
        # (בפרקטיקה נחכה לאירוע 'done' בסטרים – כאן נמשיך לק8ס)
        try:
            r2=post("/run_adapter?name=k8s_deploy", {"name":"imu-app","image":"nginx:alpine","replicas":1})
        except ResourceRequired as e:
            assert "kubectl" in e.capability
            return
        assert r2["status"]=="started"
    finally:
        # אין stop נוח ל-ThreadingHTTPServer; ב־CI פשוט נגמר הפרוסס
        pass