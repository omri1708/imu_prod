import os, json, tempfile
from typing import Dict
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from common.exc import ResourceRequired
from storage import cas
from storage.provenance import record_provenance

class K8sAdapter:
    """בניית מניפסט K8s ו-rollout מדורג."""
    def build(self, job: Dict, user: str, workspace: str, policy, ev_index) -> AdapterResult:
        _need("kubectl", "Install kubectl: https://kubernetes.io/docs/tasks/tools/")
        manifest = job.get("manifest") or ""
        if not manifest:
            # ניצור מניפסט דמה אם ניתן (deployment nginx)
            manifest = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: imu-demo
spec:
  replicas: 1
  selector: { matchLabels: { app: imu-demo } }
  template:
    metadata: { labels: { app: imu-demo } }
    spec:
      containers:
      - name: web
        image: nginx:stable
"""
        # dry-run server
        code,out,err = run(["kubectl","apply","-f","-","--dry-run=server"], cwd=workspace, env=None)
        if code != 0:
            raise ResourceRequired("k8s_cluster", ["kube-context"], "kubectl must be configured (kubeconfig/context)")
        # נשמור את המניפסט כחפץ
        man_path = os.path.join(workspace, "k8s", "manifest.yaml")
        h = put_artifact_text(man_path, manifest)
        evidence = [evidence_from_text("k8s_manifest", manifest)]
        record_provenance(man_path, evidence, trust=0.85)
        claims = [{"kind":"k8s_deployable","hash":h,"user":user}]
        return AdapterResult(artifacts={man_path: h}, claims=claims, evidence=evidence)

    def rollout(self, manifest: str):
        _need("kubectl", "Install kubectl: https://kubernetes.io/docs/tasks/tools/")
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml")
        tmp.write(manifest)
        tmp.flush(); tmp.close()
        code,out,err = run(["kubectl","apply","-f", tmp.name])
        if code != 0:
            raise RuntimeError(f"kubectl apply failed: {err}")
        return True