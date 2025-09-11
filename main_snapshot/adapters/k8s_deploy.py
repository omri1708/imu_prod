# adapters/k8s_deploy.py
from __future__ import annotations
from typing import Optional, List
from pathlib import Path
from contracts.base import ensure_tool, run_ok, Artifact, ResourceRequired
from provenance.store import ProvenanceStore

def _ensure_kubectl():
    ensure_tool("kubectl", "Install kubectl: https://kubernetes.io/docs/tasks/tools/")

def apply_manifests(manifests: List[str], namespace: Optional[str]=None, wait_deployments: bool=True, store: Optional[ProvenanceStore]=None) -> Artifact:
    """
    מיישם מניפסטים ל־K8s, וממתין ל־rollout של Deployment-ים.
    """
    _ensure_kubectl()
    for m in manifests:
        if not Path(m).exists():
            raise FileNotFoundError(f"manifest_missing: {m}")
        cmd = ["kubectl", "apply", "-f", m]
        if namespace:
            cmd += ["-n", namespace]
        run_ok(cmd)

    if wait_deployments:
        # נחפש Deployments בקבצים ונהמתין ל-rollout
        for m in manifests:
            text = Path(m).read_text()
            if "kind: Deployment" in text:
                # naive fetch of metadata.name
                name = None
                for line in text.splitlines():
                    if line.strip().startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                        break
                if name:
                    cmd = ["kubectl", "rollout", "status", f"deployment/{name}"]
                    if namespace:
                        cmd += ["-n", namespace]
                    run_ok(cmd)

    # נרשום אוסף מניפסטים כ-artifact לוגי
    pack = Artifact(path=str(Path(manifests[0]).resolve().parent), kind="k8s-release", metadata={"files": [str(Path(m).resolve()) for m in manifests]})
    if store:
        pack = store.add(pack, trust_level="applied", evidence={"tool": "kubectl"})
    return pack