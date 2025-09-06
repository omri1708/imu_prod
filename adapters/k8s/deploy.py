# adapters/k8s/deploy.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess, tempfile, yaml
from typing import Dict, Any, Optional, List
from ..contracts import ensure_tool, run, record_provenance
from adapters.contracts import ResourceRequired
import shlex, time
from contracts.adapters import k8s_env
from provenance.audit import AuditLog


def _kubectl(*parts):
    return " ".join(["kubectl"] + [shlex.quote(p) for p in parts])


def run_k8s_deploy(cfg: Dict[str,Any], audit: AuditLog):
    man_dir = cfg["manifests_dir"]
    ns = cfg["namespace"]
    wait = cfg.get("wait", True)
    follow = cfg.get("follow_logs", True)
    selector = cfg.get("selector","app=imu")
    # ensure ns
    subprocess.call(_kubectl("create","ns",ns), shell=True)
    # apply
    cmd_apply = _kubectl("-n", ns, "apply", "-f", man_dir)
    audit.append("adapter.k8s","apply",{"cmd":cmd_apply})
    subprocess.check_call(cmd_apply, shell=True)
    if wait:
        cmd_wait = _kubectl("-n", ns, "rollout", "status", "deployment","-l", selector, "--timeout=120s")
        audit.append("adapter.k8s","wait",{"cmd":cmd_wait})
        subprocess.check_call(cmd_wait, shell=True)
    if follow:
        # follow first pod matching selector
        get_pod = _kubectl("-n", ns, "get","pods","-l", selector, "-o","jsonpath={.items[0].metadata.name}")
        pod = subprocess.check_output(get_pod, shell=True, text=True)
        cmd_logs = _kubectl("-n", ns, "logs", "-f", pod.strip())
        audit.append("adapter.k8s","logs",{"cmd":cmd_logs})
        # non-blocking stream tip: tail few lines
        subprocess.Popen(cmd_logs, shell=True)
    audit.append("adapter.k8s","success",{"namespace":ns})
    return {"ok": True, "namespace": ns}


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