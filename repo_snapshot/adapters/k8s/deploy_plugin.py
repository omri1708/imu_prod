# adapters/k8s/deploy_plugin.py
import subprocess, json, tempfile, os
from typing import Dict, Any, Optional
from contracts.base import AdapterResult, require, ResourceRequired
from provenance import cas

def apply_manifest(yaml_text: str, namespace: str="default") -> AdapterResult:
    require("kubectl")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text); f.flush()
        path = f.name
    try:
        proc = subprocess.run(["kubectl","apply","-n",namespace,"-f",path],
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as ex:
        return AdapterResult(False, logs=ex.stdout+"\n"+ex.stderr)
    cid = cas.put_file(path, {"type":"k8s_manifest","namespace":namespace})
    return AdapterResult(True, artifact_path=path, logs=proc.stdout, provenance_cid=cid)

def get_pods(namespace: str="default") -> AdapterResult:
    require("kubectl")
    try:
        proc = subprocess.run(["kubectl","get","pods","-n",namespace,"-o","json"],
                              capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as ex:
        return AdapterResult(False, logs=ex.stdout+"\n"+ex.stderr)
    data = json.loads(proc.stdout)
    metrics = {"pod_count": len(data.get("items",[]))}
    cid = cas.put_bytes(proc.stdout.encode("utf-8"), {"type":"k8s_pods","namespace":namespace})
    return AdapterResult(True, metrics=metrics, logs=proc.stdout, provenance_cid=cid)