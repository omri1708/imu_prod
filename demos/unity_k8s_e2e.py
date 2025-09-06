# demos/unity_k8s_e2e.py
from __future__ import annotations
import os, subprocess, shutil, tempfile, json
from typing import Optional
from adapters.installer import ensure
from common.provenance import CAS
from security.policy import UserPolicy, check_fs, check_network, record_audit
from common.ws_progress import WSProgress
import hashlib, time

def run(cmd:list, cwd:Optional[str]=None):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, cwd=cwd)

def build_unity(project_dir:str, build_target:str, out_dir:str):
    # דרוש Unity Hub/CLI; בדיקות גמישות: unity, Unity, unity-editor
    unity = shutil.which("unity") or shutil.which("Unity") or shutil.which("unity-editor")
    if not unity:
        raise RuntimeError("Unity CLI not found. Install Unity Hub/Editor and expose CLI.")
    os.makedirs(out_dir, exist_ok=True)
    # דוגמת build headless (תלוי גרסת Unity)
    run([unity,
         "-quit","-batchmode",
         "-projectPath", project_dir,
         "-buildTarget", build_target,
         "-executeMethod","BuildScript.CommandLineBuild",
         "-logFile", os.path.join(out_dir,"unity_build.log")])

def docker_build_push(image:str, context:str):
    run(["docker","build","-t",image,context])
    run(["docker","push",image])

def k8s_deploy(image:str, namespace:str, name:str):
    manifest=f"""
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {namespace}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports: [{{containerPort: 8080}}]
"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(manifest); path=f.name
    run(["kubectl","apply","-f",path])

def main(project_dir:str, build_target:str, image:str, namespace:str, name:str,
         user="default", ws_url="ws://localhost:8765"):
    policy = UserPolicy(user_id=user)
    # מדיניות קבצים/רשת (deny-all ברירת מחדל) – נבדוק נקודות קריטיות:
    check_fs(policy,"read", project_dir)
    check_network(policy,"connect","registry-1.docker.io",443)
    check_network(policy,"connect","localhost",6443)  # דוגמה ל־k8s local

    # כלים נדרשים
    ensure("docker","docker")
    ensure("kubectl","kubectl")

    ws = WSProgress(ws_url, topic=f"unity:{name}")
    cas=CAS("cas")

    # 1) Build
    ws.emit("progress", {"phase":"unity_build","pct":5})
    out_dir=os.path.join("artifacts","unity_build"); os.makedirs(out_dir, exist_ok=True)
    build_unity(project_dir, build_target, out_dir)
    ws.emit("progress", {"phase":"unity_build","pct":60})

    # 2) Package (Docker)
    # נבנה Dockerfile מינימלי אם לא קיים
    df=os.path.join(project_dir,"Dockerfile")
    if not os.path.exists(df):
        with open(df,"w") as f:
            f.write("FROM nginx:alpine\nCOPY ./Build /usr/share/nginx/html\n")
    docker_build_push(image, project_dir)
    ws.emit("progress", {"phase":"docker_push","pct":80})

    # 3) Deploy to K8s
    k8s_deploy(image, namespace, name)
    ws.emit("progress", {"phase":"k8s_deploy","pct":95})

    # 4) Evidence & CAS
    # נארוז ארטיפקט לדוגמה
    bundle=os.path.join(out_dir,"bundle.zip")
    shutil.make_archive(bundle[:-4], "zip", out_dir)
    ev=cas.put(bundle)
    record_audit("publish_artifact", user, {"sha256": ev.sha256, "path": ev.path, "image": image})

    ws.emit("done", {"phase":"complete","pct":100,"artifact_sha256": ev.sha256})

if __name__=="__main__":
    # דוגמה: python demos/unity_k8s_e2e.py /path/to/UnityProject WebGL myrepo/unity:latest default gameweb
    import sys
    if len(sys.argv)<6:
        print("usage: unity_k8s_e2e.py <project_dir> <build_target> <image> <namespace> <name>")
        sys.exit(2)
    main(*sys.argv[1:6])