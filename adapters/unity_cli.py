# adapters/unity_cli.py
from __future__ import annotations
import os, subprocess, shutil, time
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event
from engine.progress import EMITTER
from perf.measure import measure, BUILD_PERF
import shutil
import os, subprocess, asyncio, shlex
from typing import Dict, Any, AsyncIterator
import os, subprocess, shlex, tempfile, json, time, hashlib
from typing import Dict, List
from engine.errors import ResourceRequired

class ActionRequired(Exception):
    def __init__(self, what: str, how: str):
        super().__init__(what); self.what=what; self.how=how


class UnityBuildError(Exception): pass


def _find_unity():
    candidates = [
        "/Applications/Unity/Hub/Editor",                   # macOS Hub
        os.path.expanduser("~/Applications/Unity/Hub/Editor"),
        "C:\\Program Files\\Unity\\Hub\\Editor",            # Windows
        "/opt/Unity/Editor"                                 # Linux
    ]
    for base in candidates:
        if os.path.isdir(base):
            # קח גרסה ראשונה
            for v in os.listdir(base):
                exe_mac = os.path.join(base, v, "Unity.app/Contents/MacOS/Unity")
                exe_lin = os.path.join(base, v, "Editor/Unity")
                exe_win = os.path.join(base, v, "Editor/Unity.exe")
                for exe in (exe_mac, exe_lin, exe_win):
                    if os.path.exists(exe): return exe
    return shutil.which("Unity") or shutil.which("unity")


def run_unity_build(payload:dict)->dict:
    proj = payload.get("project_path","./unity_project")
    out  = payload.get("output_path","./builds/Standalone")
    target = payload.get("build_target","StandaloneWindows64")  # או Android/iOS וכו'
    unity = _find_unity()
    if not unity:
        raise ResourceRequired("unity_cli",
            "Unity Editor CLI not found. Install via Unity Hub and expose editor binary in PATH. "
            "macOS: /Applications/Unity/Hub/Editor/<ver>/Unity.app/Contents/MacOS/Unity",
            requires_consent=True)

    os.makedirs(out, exist_ok=True)
    cmd = [
        unity, "-batchmode", "-quit",
        "-projectPath", os.path.abspath(proj),
        "-executeMethod", "BuildScript.PerformBuild",
        "-buildTarget", target,
        "-logFile", os.path.abspath("./_logs/unity_build.log")
    ]
    os.makedirs("./_logs", exist_ok=True)
    t0=time.time()
    sp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt=int((time.time()-t0)*1000)
    if sp.returncode!=0:
        raise RuntimeError(f"unity_build_failed rc={sp.returncode} out={sp.stdout[-4000:]}")
    # נניח שנוצר ארטיפקט
    return {"ok":True,"ms":dt,"target":target,"output_path":os.path.abspath(out)}


def run_unity_headless(project_path: str, target: str, output_dir: str, unity_path: str="Unity"):
    """
    מריץ build של Unity במצב headless.
    target: Android|iOS|StandaloneWindows64|StandaloneOSX|WebGL וכו'
    """
    os.makedirs(output_dir, exist_ok=True)
    logf = os.path.join(output_dir, "unity_build.log")
    args = [
        unity_path,
        "-batchmode",
        "-quit",
        "-projectPath", project_path,
        "-buildTarget", target,
        "-logFile", logf,
        "-executeMethod", "BuildScript.PerformBuild"
    ]
    p = subprocess.run(args, cwd=project_path)
    if p.returncode != 0:
        raise UnityBuildError(f"unity_build_failed: see {logf}")
    # נניח שה־BuildScript שם artifact ב־output_dir
    # נאתר אותו:
    arts = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith((".apk",".aab",".ipa",".xapk",".zip",".app",".exe",".wasm",".data",".bundle")):
                path = os.path.join(root, f)
                arts.append(path)
    if not arts:
        raise UnityBuildError("no_artifacts_found")
    return arts


async def _run_unity_build(project_path: str, target: str="StandaloneLinux64") -> AsyncIterator[Dict[str,Any]]:
    """
    מריץ Unity CLI ב-batchmode. דורש התקנת Unity (Hub/Editor) ונתיב 'unity' או 'Unity' ב-PATH.
    מפיק אירועי progress/timeline בזמן אמת.
    """
    unity_cmds = ["Unity", "unity", "/Applications/Unity/Hub/Editor/Unity"]
    exe = None
    for c in unity_cmds:
        try:
            subprocess.check_output([c, "-version"], stderr=subprocess.STDOUT)
            exe = c; break
        except Exception:
            pass
    if not exe:
        raise ActionRequired(
            "Unity CLI not found",
            "Install Unity Editor (batchmode) and ensure 'Unity' is in PATH. See https://unity.com/download"
        )
    log_file = os.path.join(project_path, "Editor.log")
    args = f'{exe} -batchmode -quit -projectPath "{project_path}" -buildTarget {target} -logFile "{log_file}"'
    proc = await asyncio.create_subprocess_shell(
        args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    yield {"type":"timeline","event":f"unity:start target={target}"}
    while True:
        line = await proc.stdout.readline()
        if not line: break
        txt = line.decode(errors="ignore").strip()
        if "progress" in txt.lower():
            # ניסיון גס לחשוף התקדמות מהלוג
            yield {"type":"progress","value":1,"total":1,"detail":txt}
        yield {"type":"timeline","event":f"unity:log {txt[:200]}"}
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"Unity build failed with rc={rc}. See {log_file}")
    # איפה הבילד? (פשטות: נניח שהפרויקט מגדיר נתיב פלט)
    out_dir = os.path.join(project_path, "Builds", target)
    yield {"type":"timeline","event":f"unity:done out_dir={out_dir}"}
    yield {"type":"artifact","path":out_dir}


def _find_unity_bin():
    return shutil.which("unity") or shutil.which("Unity") or shutil.which("Unity.app/Contents/MacOS/Unity")


def build_unity(project_path: str, build_method: str, batchmode: bool=True, quit_on_finish: bool=True) -> BuildResult:
    EMITTER.emit("timeline", {"phase":"unity.prepare","project":project_path,"method":build_method})
    unity_bin=_find_unity_bin()
    if not unity_bin:
        raise ResourceRequired("Unity Editor CLI",
            "Install Unity Hub + Editor; add Unity binary to PATH (e.g. /Applications/Unity/Hub/Editor/<ver>/Unity.app/Contents/MacOS/Unity)",
            "Unity not found")
    cmd=[unity_bin,"-projectPath",project_path,"-executeMethod",build_method]
    if batchmode: cmd.append("-batchmode")
    if quit_on_finish: cmd.append("-quit")
    (out, dt)=measure(run, cmd, None, None, 7200)
    BUILD_PERF.add(dt)
    EMITTER.emit("metrics", {"kind":"unity.build","project":project_path,"secs":dt, **BUILD_PERF.snapshot()})
    candidates=[]
    for root,_,files in os.walk(project_path):
        for f in files:
            if f.endswith((".exe",".apk",".aab",".ipa",".xapk",".app",".wasm",".data",".bundle",".framework")):
                candidates.append(os.path.join(root,f))
    if not candidates: raise ProcessFailed(cmd,0,out,"Unity build produced no known artifacts")
    artifact=max(candidates,key=os.path.getmtime)
    digest=CAS_STORE.put_file(artifact)
    EMITTER.emit("timeline", {"phase":"unity.artifact","path":artifact,"sha256":digest})
    record_event("artifact.store", {"platform":"unity","path":artifact,"sha256":digest})
    return BuildResult(artifact=artifact, sha256=digest, meta={"method":build_method})