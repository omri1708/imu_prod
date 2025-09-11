# demos/docker_cosign_demo.py
"""
Sign a docker image with cosign if available; otherwise request capability command.
"""
from __future__ import annotations
import os, json, urllib.request, shutil
from adapters.docker_sign import sign_with_cosign

def _post(api: str, path: str, data: dict) -> dict:
    req=urllib.request.Request(api+path, method="POST", data=json.dumps(data).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r: return json.loads(r.read().decode())

def main(image: str):
    api=os.environ.get("IMU_API","http://127.0.0.1:8000")
    if shutil.which("cosign"):
        res = sign_with_cosign(image)
        print(res)
    else:
        cmd = _post(api, "/capabilities/request", {"user_id":"demo-user","capability":"cosign"})
        print("install:", cmd.get("command"))

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: docker_cosign_demo.py <image>")
        exit(2)
    main(sys.argv[1])