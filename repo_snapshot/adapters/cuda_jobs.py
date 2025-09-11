# adapters/cuda_jobs.py
from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Optional
from contracts.base import ensure_tool, run_ok, Artifact, ResourceRequired
from provenance.store import ProvenanceStore

def _ensure_cuda():
    # נבדוק nvcc או nvidia-smi, אחת מהן מינימלית; להידור נדרשת nvcc.
    nvcc = shutil.which("nvcc")
    smi = shutil.which("nvidia-smi")
    if not nvcc and not smi:
        raise ResourceRequired("CUDA toolkit/driver", "Install NVIDIA drivers and CUDA Toolkit (nvcc / nvidia-smi).")
    return nvcc, smi

def compile_cuda_kernel(cuda_file: str, output_bin: Optional[str]=None, store: Optional[ProvenanceStore]=None) -> Artifact:
    nvcc, _ = _ensure_cuda()
    if not nvcc:
        raise ResourceRequired("nvcc", "Install CUDA Toolkit to get 'nvcc'.")
    src = Path(cuda_file).resolve()
    if not src.exists():
        raise FileNotFoundError(f"missing_cuda_source: {src}")
    out = Path(output_bin or (src.parent / (src.stem + ".out")))
    run_ok(["nvcc", str(src), "-o", str(out)])
    art = Artifact(path=str(out), kind="bin", metadata={"src": str(src)})
    if store:
        art = store.add(art, trust_level="built-local", evidence={"builder": "nvcc"})
    return art