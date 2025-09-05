# adapters/adapter_runner.py
# -*- coding: utf-8 -*-
import os, subprocess, json, tempfile, time, shutil
from typing import Dict, Any, Callable
from policy.user_policy import UserSubspacePolicy
from provenance.store import ProvenanceStore, ResourceRequired

def _which(cmd:str)->str:
    for p in os.environ.get("PATH","").split(os.pathsep):
        cand=os.path.join(p, cmd)
        if os.name=="nt":
            for s in ("", ".exe",".bat",".cmd"):
                if os.path.exists(cand+s): return cand+s
        else:
            if os.path.exists(cand) and os.access(cand, os.X_OK):
                return cand
    return ""

def run_adapter(adapter:str, args:Dict[str,Any], policy:UserSubspacePolicy,
                emit:Callable[[str,Dict[str,Any]],None],
                prov:ProvenanceStore)->Dict[str,Any]:
    """
    מפעיל מתאמים בעלי צד־שלישי (Unity/Android/iOS/K8s/CUDA).
    ללא “דמו”: אם הבינארי חסר, נזרקת ResourceRequired עם הוראות התקנה.
    """
    if adapter=="unity_build":
        unity_path=args.get("unity_path") or _which("Unity")
        if not unity_path:
            raise ResourceRequired("Unity CLI",
                "Install Unity Hub + Editor CLI. Ensure 'Unity' exists in PATH or pass unity_path.")
        project_dir=args["project_dir"]; target=args.get("target","Linux64")
        out_dir=args.get("out_dir",".imu_out/unity")
        os.makedirs(out_dir, exist_ok=True)
        emit("progress",{"phase":"unity:begin","target":target})
        cmd=[unity_path, "-quit","-batchmode","-projectPath",project_dir,"-buildTarget",target,
             "-executeMethod","BuildScript.PerformBuild","-logFile", os.path.join(out_dir,"unity.log")]
        t0=time.time()
        cp=subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode!=0:
            raise RuntimeError(f"unity_build_failed:\n{cp.stdout}\n{cp.stderr}")
        elapsed=time.time()-t0
        emit("progress",{"phase":"unity:done","ms":int(elapsed*1000)})
        # רושמים Artifact ל-Provenance:
        bundle_path=os.path.join(out_dir,"build.zip")
        if not os.path.exists(bundle_path): # בהנחה שב־BuildScript נוצר zip
            # אם אין – נייצר זמנית מכל מה שנבנה תחת out_dir
            import zipfile
            with zipfile.ZipFile(bundle_path,"w") as z:
                for root,_,files in os.walk(out_dir):
                    for fn in files:
                        fp=os.path.join(root,fn)
                        arc=os.path.relpath(fp,out_dir)
                        z.write(fp, arcname=arc)
        with open(bundle_path,"rb") as f:
            digest=prov.put(f.read(), source="unity_build", trust=80, ttl_s=policy.ttl_s_hard,
                            evidence={"adapter":"unity_build","target":target})
        return {"artifact_digest":digest,"out_dir":out_dir}

    if adapter=="k8s_deploy":
        kubectl=_which("kubectl")
        if not kubectl:
            raise ResourceRequired("kubectl", "Install kubectl and ensure it is in PATH and kubeconfig is set.")
        manifest=args["manifest"]  # קובץ yaml מלא
        emit("progress",{"phase":"k8s:apply","manifest":manifest})
        cp=subprocess.run([kubectl,"apply","-f",manifest], capture_output=True, text=True)
        if cp.returncode!=0:
            raise RuntimeError(f"k8s_apply_failed:\n{cp.stdout}\n{cp.stderr}")
        emit("progress",{"phase":"k8s:rollout-status"})
        cp2=subprocess.run([kubectl,"rollout","status","-f",manifest,"--timeout=120s"], capture_output=True, text=True)
        if cp2.returncode!=0:
            raise RuntimeError(f"k8s_rollout_failed:\n{cp2.stdout}\n{cp2.stderr}")
        return {"status":"deployed"}

    if adapter=="cuda_job":
        nvidia_smi=_which("nvidia-smi")
        if not nvidia_smi:
            raise ResourceRequired("CUDA/GPU", "Install NVIDIA driver + CUDA toolkit; ensure nvidia-smi available.")
        script=args["script"]
        emit("progress",{"phase":"cuda:run","script":script})
        cp=subprocess.run(["bash","-lc",script], capture_output=True, text=True)
        if cp.returncode!=0:
            raise RuntimeError(f"cuda_job_failed:\n{cp.stdout}\n{cp.stderr}")
        return {"status":"ok","stdout":cp.stdout}

    if adapter=="android_build":
        gradle=_which("gradle")
        if not gradle:
            raise ResourceRequired("Gradle/Android SDK",
                "Install JDK + Android SDK + Gradle; ensure 'gradle' available. Accept licenses via sdkmanager.")
        project_dir=args["project_dir"]; task=args.get("task","assembleRelease")
        emit("progress",{"phase":"android:gradle","task":task})
        cp=subprocess.run([gradle, "-p", project_dir, task], capture_output=True, text=True)
        if cp.returncode!=0:
            raise RuntimeError(f"android_build_failed:\n{cp.stdout}\n{cp.stderr}")
        return {"status":"ok"}

    if adapter=="ios_build":
        xcodebuild=_which("xcodebuild")
        if not xcodebuild:
            raise ResourceRequired("Xcode", "Install Xcode + CLT from App Store / xcode-select --install.")
        workspace=args["workspace"]; scheme=args["scheme"]
        emit("progress",{"phase":"ios:xcodebuild","scheme":scheme})
        cp=subprocess.run([xcodebuild,"-workspace",workspace,"-scheme",scheme,"-configuration","Release","build"],
                          capture_output=True, text=True)
        if cp.returncode!=0:
            raise RuntimeError(f"ios_build_failed:\n{cp.stdout}\n{cp.stderr}")
        return {"status":"ok"}

    raise RuntimeError(f"unknown_adapter:{adapter}")