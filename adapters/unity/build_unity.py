# adapters/unity/build_unity.py
import os, subprocess, shutil
from contracts.base import AdapterResult, require, ResourceRequired
from provenance import cas

def build(project_dir: str, method: str="IMU_CLI_Build.BuildLinux64", log_path: str="unity_build.log"):
    require("unity") if False else None  # Unity often installed as Hub; try `Unity` cli name below.
    unity_bins = ["unity-editor", "Unity", "/Applications/Unity/Hub/Editor/Unity.app/Contents/MacOS/Unity"]
    unity = next((b for b in unity_bins if shutil.which(os.path.basename(b)) or os.path.exists(b)), None)
    if not unity:
        raise ResourceRequired("unity_cli","Install Unity Editor and expose CLI (Hub).")
    cmd = [unity, "-quit","-batchmode","-projectPath", project_dir, "-executeMethod", method, "-logFile", log_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as ex:
        return AdapterResult(False, logs=ex.stdout+"\n"+ex.stderr)
    # Collect artifact path from log (simple heuristic)
    out_path = None
    if os.path.exists("Builds/Linux/IMUGame.x86_64"): out_path = "Builds/Linux/IMUGame.x86_64"
    cid = cas.put_file(out_path, {"type":"unity_build"}) if out_path else None
    return AdapterResult(bool(out_path), artifact_path=out_path, logs=proc.stdout, provenance_cid=cid)