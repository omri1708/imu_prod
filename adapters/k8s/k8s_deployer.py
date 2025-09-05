# adapters/k8s_deployer.py
import os, shutil, subprocess, json, time
from engine.errors import ResourceRequired

def _kubectl():
    return shutil.which("kubectl")

def deploy(payload:dict)->dict:
    kc = _kubectl()
    if not kc:
        raise ResourceRequired("kubectl",
            "Install kubectl and configure KUBECONFIG. https://kubernetes.io/docs/tasks/tools/",
            requires_consent=True)
    name = payload.get("name","imu-app")
    image = payload.get("image","nginx:alpine")
    replicas = int(payload.get("replicas",1))
    ns = payload.get("namespace","default")
    yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{ name: {name}, namespace: {ns} }}
spec:
  replicas: {replicas}
  selector: {{ matchLabels: {{ app: {name} }} }}
  template:
    metadata: {{ labels: {{ app: {name} }} }}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports: [{{containerPort: 80}}]
---
apiVersion: v1
kind: Service
metadata: {{ name: {name}, namespace: {ns} }}
spec:
  selector: {{ app: {name} }}
  ports: [{{port: 80, targetPort: 80}}]
"""
    p = subprocess.run([kc, "apply", "-f", "-"], input=yaml.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode!=0: raise RuntimeError(f"kubectl_apply_failed: {p.stdout.decode()[-4000:]}")
    # rollout
    r = subprocess.run([kc,"rollout","status","deploy/"+name,"-n",ns], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return {"ok": p.returncode==0, "apply": p.stdout.decode(), "rollout": r.stdout}