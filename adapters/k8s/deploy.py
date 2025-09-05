# adapters/k8s/deploy.py
# -*- coding: utf-8 -*-
import os, json, tempfile
from ..contracts import ensure_tool, run, record_provenance

BASIC_DEPLOY_YAML = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels: {{ app: {name} }}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: {port}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
  - protocol: TCP
    port: {port}
    targetPort: {port}
"""

def deploy(name: str, image: str, port: int = 80, replicas: int = 1, kubeconfig: str = None) -> dict:
    ensure_tool("kubectl", "Install kubectl and configure KUBECONFIG")
    y = BASIC_DEPLOY_YAML.format(name=name, image=image, port=port, replicas=replicas)
    td = tempfile.mkdtemp(prefix="imu_k8s_")
    manifest = os.path.join(td, "deploy.yaml")
    with open(manifest, "w") as f: f.write(y)
    cmd = ["kubectl", "apply", "-f", manifest]
    if kubeconfig: os.environ["KUBECONFIG"] = kubeconfig
    out = run(cmd)
    prov = record_provenance("k8s_apply", {"name": name, "image": image}, manifest)
    return {"manifest": manifest, "provenance": prov.__dict__, "log": out}