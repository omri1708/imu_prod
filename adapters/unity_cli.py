# adapters/unity_cli.py
from __future__ import annotations
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event
from engine.progress import EMITTER
from perf.measure import measure, BUILD_PERF
import os, shutil

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