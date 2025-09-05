# adapters/k8s/deploy.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, tempfile, yaml
from ..contracts import ensure_tool, run, record_provenance
from adapters.contracts import ResourceRequired
import subprocess, shlex, time
from contracts.adapters import k8s_env

def kubectl(cmd: str):
    k8s_env()
    subprocess.check_call(f"kubectl {cmd}", shell=True)

def apply_safe(manifest: str):
    kubectl(f"apply -f {shlex.quote(manifest)}")

def rollout_status(kind: str, name: str, ns: str = "default", timeout: int = 300):
    kubectl(f"-n {shlex.quote(ns)} rollout status {kind}/{name} --timeout={timeout}s")

def canary_and_rollout(main_manifest: str, canary_manifest: str, *, kind="deployment", name="app", ns="default"):
    # שלב 1: Canary
    apply_safe(canary_manifest)
    rollout_status(kind, f"{name}-canary", ns)
    # (כאן רצוי מדדים/בריאות חיצוניים — מחוץ לסקופ הקובץ הזה)
    time.sleep(2)
    # שלב 2: Rollout מלא
    apply_safe(main_manifest)
    rollout_status(kind, name, ns)
    
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

def _kubectl():
    if not shutil.which("kubectl"):
        raise ResourceRequired("kubectl", "Install kubectl and configure context")
    return "kubectl"

def apply_manifest(manifest: dict, namespace: str = None):
    kc = _kubectl()
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        yaml.safe_dump(manifest, f)
        path = f.name
    cmd = [kc, "apply", "-f", path]
    if namespace: cmd += ["-n", namespace]
    subprocess.run(cmd, check=True)
    return {"ok": True, "applied": path}

def deploy_image(image: str, name: str = "imu-job", namespace: str = None, gpu: bool = False):
    # Job בסיסי; GPU אופציונלי (nodeSelector/tolerations בהתאם לסביבה שלך)
    m = {
      "apiVersion":"batch/v1",
      "kind":"Job",
      "metadata":{"name":name},
      "spec":{
        "template":{
          "spec":{
            "restartPolicy":"Never",
            "containers":[{
              "name":name,
              "image":image,
              "resources": {"limits":{"nvidia.com/gpu": 1}} if gpu else {}
            }]
          }
        }
      }
    }
    return apply_manifest(m, namespace=namespace)