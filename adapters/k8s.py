import os, json, tempfile
from typing import Dict, Any
from adapters.base import _need, run, put_artifact_text, evidence_from_text
from engine.adapter_types import AdapterResult
from common.exc import ResourceRequired
from storage import cas
from storage.provenance import record_provenance
import shutil, subprocess, json, tempfile, os
from .contracts import AdapterResult, require
from adapters.base import AdapterBase, PlanResult
from engine.policy import RequestContext
import subprocess, shlex, tempfile, os
from typing import Dict, Any, List, Tuple
from engine.provenance import Evidence
from engine.policy import UserSpacePolicy
import subprocess, json, shlex, os
from typing import Dict, Any, Optional
from runtime.sandbox import enforce_file_access
from policy.model import UserPolicy


def deploy_manifest(yaml_path:str, namespace:str="default") -> AdapterResult:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return AdapterResult(False, "kubectl not found", {})
    try:
        out = subprocess.run([kubectl, "apply", "-f", yaml_path, "-n", namespace], capture_output=True, text=True, timeout=600)
        ok = (out.returncode == 0)
        return AdapterResult(ok, out.stderr if not ok else "ok", {"log": out.stdout})
    except Exception as e:
        return AdapterResult(False, str(e), {})


def deploy_k8s_manifest(manifest_yaml: str, namespace: str="default") -> AdapterResult:
    kubectl = shutil.which("kubectl")
    if not kubectl:
        return require("kubectl", "Install kubectl and configure kubeconfig",
                       ["curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl",
                        "chmod +x kubectl && sudo mv kubectl /usr/local/bin/"])
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        f.write(manifest_yaml)
        path = f.name
    try:
        subprocess.run([kubectl, "apply", "-n", namespace, "-f", path], check=True)
        out = subprocess.check_output([kubectl, "get", "all", "-n", namespace, "-o", "json"])
        return AdapterResult(status="ok", message="K8s applied", outputs={"state": json.loads(out.decode())})
    except subprocess.CalledProcessError as e:
        return AdapterResult(status="error", message=f"kubectl failed: {e}", outputs={})
    finally:
        try: os.remove(path)
        except Exception: pass

class K8sAdapter(AdapterBase):
    """בניית מניפסט K8s ו-rollout מדורג."""
    name = "k8s"

    def build_command(self, args: Dict[str, Any], dry_run: bool, policy: UserSpacePolicy) -> List[str]:
        manifest = args.get("manifest_yaml")
        if not manifest:
            raise ValueError("missing manifest_yaml")
        # כותבים לקובץ זמני כדי לאפשר kubectl apply -f
        tmp = args.get("_tmp_path") or tempfile.mkstemp(prefix="imu_", suffix=".yaml")[1]
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(manifest)
        cmd = ["bash","-lc", f"kubectl apply -f {shlex.quote(tmp)}{' --dry-run=client' if args.get('dry_run',True) else ''}"]
        # שומרים היכן הכתיבה לצורך ניקוי עתידי (אם תרצה)
        args["_tmp_path"] = tmp
        return cmd

    def execute(self, cmd: List[str], policy: UserSpacePolicy) -> Tuple[bool,str,str]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=policy.p95_ms/1000)
        except subprocess.TimeoutExpired:
            proc.kill()
            return False, "", "timeout"
        return proc.returncode==0, out, err

    def produce_evidence(self, cmd: List[str], args: Dict[str, Any]):
        return [Evidence(claim="k8s.apply.plan", source="adapters.k8s", trust=0.75, extra={"cmd":cmd,"tmp":args.get('_tmp_path')})]

    def plan(self, spec: Dict[str, Any], ctx: RequestContext) -> PlanResult:
        ns = spec.get("namespace","default")
        file = spec.get("manifest","deploy.yaml")
        dry = "--dry-run=client" if spec.get("client_dry_run", True) else ""
        cmds = [f"kubectl apply -n {ns} -f {file} {dry}".strip()]
        return PlanResult(commands=cmds, env={}, notes="kubectl apply")

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
    

def dry_run(image: str, namespace: str="default") -> Dict[str, Any]:
    spec = {
        "apiVersion":"batch/v1",
        "kind":"Job",
        "metadata":{"name":"imu-job-dryrun","namespace":namespace},
        "spec":{"template":{"spec":{"restartPolicy":"Never","containers":[{"name":"job","image":image,"args":["echo","hello"]}]}}}
    }
    cmds = [
        "kubectl version --client",
        f"kubectl apply --dry-run=client -f - <<<'{json.dumps(spec)}'",
    ]
    return {"ok": True, "cmds": cmds, "needs": ["kubectl context", "RBAC to create Jobs"]}

def run(policy: UserPolicy, image: str, args=None, namespace: str="default") -> Dict[str, Any]:
    if args is None: args=[]
    manifest = {
        "apiVersion":"batch/v1",
        "kind":"Job",
        "metadata":{"name":"imu-job","namespace":namespace},
        "spec":{"template":{"spec":{"restartPolicy":"Never","containers":[{"name":"job","image":image,"args":args}]}}}
    }
    p = subprocess.run(["kubectl","apply","-f","-"], input=json.dumps(manifest),
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"ok": p.returncode==0, "log": p.stdout}