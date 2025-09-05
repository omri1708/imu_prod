# adapters/ios_builder.py
from __future__ import annotations
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from adapters.contracts.base import record_event

from engine.progress import EMITTER
from perf.measure import measure, BUILD_PERF
import os, tempfile

def build_ios_xcode(project_path: str, scheme: str, sdk: str="iphoneos", configuration: str="Release") -> BuildResult:
    EMITTER.emit("timeline", {"phase":"ios.prepare","project":project_path,"scheme":scheme})
    require_binary("xcodebuild","Install Xcode + CLT via App Store / xcode-select --install","Xcode required")
    build_dir = tempfile.mkdtemp(prefix="xcode-build-")
    (out, dt) = measure(run, ["xcodebuild","-project",project_path,"-scheme",scheme,"-sdk",sdk,"-configuration",configuration,"BUILD_DIR="+build_dir,"build"], None, None, 7200)
    BUILD_PERF.add(dt)
    EMITTER.emit("metrics", {"kind":"ios.build","project":project_path,"scheme":scheme,"secs":dt, **BUILD_PERF.snapshot()})
    found=[]
    for root,_,files in os.walk(build_dir):
        for f in files:
            if f.endswith((".ipa",".app")): found.append(os.path.join(root,f))
    if not found: raise ProcessFailed(["xcodebuild"],0,out,"No .ipa/.app produced")
    artifact=max(found,key=os.path.getmtime)
    digest=CAS_STORE.put_file(artifact)
    EMITTER.emit("timeline", {"phase":"ios.artifact","path":artifact,"sha256":digest})
    record_event("artifact.store", {"platform":"ios","path":artifact,"sha256":digest})
    return BuildResult(artifact=artifact, sha256=digest, meta={"sdk":sdk,"configuration":configuration})