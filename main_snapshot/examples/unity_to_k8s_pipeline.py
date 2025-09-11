# examples/unity_to_k8s_pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time, requests

API = "http://127.0.0.1:8088"

def verify_sources():
    # טוענים claim אמיתי כלשהו (דוגמה בלבד)
    return [{"type":"http_json","url":"https://api.github.com","path":"current_user_url","expected_eq":"https://api.github.com/user","fresh_seconds":86400}]

def run_unity_build():
    req = {
        "user_id":"u1",
        "adapter":"unity.build",
        "run_id": f"unity_{int(time.time())}",
        "claims": verify_sources(),
        "args": {
            "project_path": "/path/to/unity/project",     # לשנות אצלך
            "out_path": "./Build/Standalone"
        }
    }
    r = requests.post(f"{API}/run_adapter", json=req, timeout=60*60)
    return r.status_code, r.json()

def run_k8s_deploy(image:str):
    manifest = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: imu-unity-job
spec:
  template:
    spec:
      containers:
      - name: unity-job
        image: {image}
        command: ["bash","-lc","echo Hello from Unity artifact && sleep 10"]
      restartPolicy: Never
"""
    req = {
        "user_id":"u1",
        "adapter":"k8s.deploy",
        "run_id": f"k8s_{int(time.time())}",
        "claims": verify_sources(),
        "args": {"manifest_yaml": manifest}
    }
    r = requests.post(f"{API}/run_adapter", json=req, timeout=60)
    return r.status_code, r.json()

if __name__ == "__main__":
    print("Build Unity...")
    print(run_unity_build())
    print("Deploy K8s...")
    print(run_k8s_deploy("alpine:3.19"))