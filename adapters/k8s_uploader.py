# adapters/k8s_uploader.py (העלאה ל-Artifact-Server + יצירת/עדכון Job ב-K8s דרך kubectl)
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, subprocess, json, tempfile, textwrap, shlex, pathlib
from typing import Dict, Any

class ActionRequired(Exception):
    def __init__(self, what: str, how: str): super().__init__(what); self.what=what; self.how=how

def _which(cmd: str)->bool:
    from shutil import which
    return which(cmd) is not None

def upload_dir_with_tar(artifact_server_url: str, dir_path: str) -> Dict[str, Any]:
    # אורזים תיקייה ל-tar.gz ושולחים ל-Artifact-Server
    import tarfile, io, requests
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        p = pathlib.Path(dir_path)
        for f in p.rglob("*"):
            if f.is_file():
                tf.add(f, arcname=str(f.relative_to(p)))
    buf.seek(0)
    files = {"file": ("artifact.tar.gz", buf.getvalue(), "application/gzip")}
    data = {"source":"unity_build","ttl":str(30*24*3600),"trust":str(0.85)}
    r = requests.post(artifact_server_url.rstrip("/")+"/upload", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

def deploy_k8s_job(job_name: str, image: str, env: Dict[str,str], namespace: str="default"):
    if not _which("kubectl"):
        raise ActionRequired("kubectl not found","Install kubectl and configure cluster context.")
    job_yaml = textwrap.dedent(f"""
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: {job_name}
      namespace: {namespace}
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: worker
              image: {image}
              env:
    """)
    for k,v in env.items():
        job_yaml += f"                - name: {k}\n                  value: \"{v}\"\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(job_yaml)
        tmp = f.name
    subprocess.check_call(["kubectl","apply","-f", tmp])
    return {"ok": True, "job": job_name, "namespace": namespace}