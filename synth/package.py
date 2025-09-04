# synth/package.py
from __future__ import annotations
import os, tarfile, io, time
from typing import Dict, Any

def make_tarball(src_dir: str, out_path: str) -> str:
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))
    return out_path