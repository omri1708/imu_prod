# adapters/ios/build_ios.py
import os, subprocess
from typing import Dict, Any
from contracts.base import AdapterResult, require, ResourceRequired
from provenance import cas

def build(archive_out: str, project_path: str, scheme: str, sdk: str="iphoneos") -> AdapterResult:
    require("xcodebuild")
    # Build archive (non-codesigned generic, suitable for CI artifact)
    cmd = ["xcodebuild","-scheme",scheme,"-sdk",sdk,"-project",project_path,"archive",
           "-archivePath", archive_out]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as ex:
        return AdapterResult(False, logs=ex.stdout+"\n"+ex.stderr)
    cid = cas.put_file(archive_out + ".xcarchive/Info.plist" if os.path.isdir(archive_out + ".xcarchive") else archive_out,
                       {"type":"ios_archive","scheme":scheme,"sdk":sdk})
    return AdapterResult(True, artifact_path=archive_out, metrics={}, logs=proc.stdout, provenance_cid=cid)
