# adapters/k8s/deploy.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, tempfile, yaml
from ..contracts import ensure_tool, run, record_provenance
from adapters.contracts import ResourceRequired

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