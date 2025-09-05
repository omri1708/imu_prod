# adapters/k8s/rollout.py
# -*- coding: utf-8 -*-
import shutil, subprocess, time, json
from ..contracts import ResourceRequired

def _kubectl():
    kb = shutil.which("kubectl")
    if not kb:
        raise ResourceRequired("kubectl", "Install kubectl and configure kubeconfig")
    return kb

def rollout_deploy(name: str, image: str, port: int, replicas: int = 2, readiness_seconds: int = 30):
    kb = _kubectl()
    # יצירת Deployment + Service אם לא קיימים
    subprocess.run([kb, "apply", "-f", "-"], input=f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}}}
spec:
  replicas: {replicas}
  selector: {{ matchLabels: {{ app: {name} }} }}
  template:
    metadata: {{ labels: {{ app: {name} }} }}
    spec:
      containers:
      - name: app
        image: {image}
        ports: [{{containerPort: {port}}}]
        readinessProbe:
          httpGet: {{ path: /, port: {port} }}
          initialDelaySeconds: 2
          periodSeconds: 2
---
apiVersion: v1
kind: Service
metadata: {{ name: {name} }}
spec:
  selector: {{ app: {name} }}
  ports: [{{ port: {port}, targetPort: {port} }}]
""".encode("utf-8"), check=True)
    # health gate: מחכים שכל הפודים READY בפרק זמן מוקצב
    dead = time.time() + readiness_seconds
    while time.time() < dead:
        out = subprocess.run([kb, "get", "deploy", name, "-o", "json"], check=True, stdout=subprocess.PIPE, text=True).stdout
        j = json.loads(out)
        desired = j["spec"]["replicas"]
        ready = j["status"].get("readyReplicas", 0)
        if ready >= desired:
            return {"ok": True, "ready": ready, "desired": desired}
        time.sleep(2)
    return {"ok": False, "reason": "readiness_timeout"}