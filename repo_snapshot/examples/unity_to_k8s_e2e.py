# examples/unity_to_k8s_e2e.py

# -*- coding: utf-8 -*-
import requests, time, json, threading
from attic.engine.http_api import serve

def _serve():
    serve("127.0.0.1", 8099)

if __name__=="__main__":
    t=threading.Thread(target=_serve, daemon=True); t.start()
    time.sleep(0.5)

    # 1) Unity build (דורש Unity מותקן; אחרת נקבל ResourceRequired 428)
    r = requests.post("http://127.0.0.1:8099/run_adapter",
                      json={"adapter":"unity_build","args":{"project_dir":"./UnityProject","target":"Linux64"}})
    print("unity_build:", r.status_code, r.text)
    if r.status_code!=200:
        print("Cannot continue to k8s without artifact.")
        exit(0)
    digest = r.json()["result"]["artifact_digest"]

    # 2) K8s deploy (דורש kubectl; אחרת ResourceRequired)
    manifest = "./k8s/unity-artifact.yaml"
    r2 = requests.post("http://127.0.0.1:8099/run_adapter",
                       json={"adapter":"k8s_deploy","args":{"manifest":manifest}})
    print("k8s_deploy:", r2.status_code, r2.text)