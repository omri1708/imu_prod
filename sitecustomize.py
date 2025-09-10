# sitecustomize.py
import os
import  builtins

SRC_PREFIX = "/mnt/data/imu_repo"
DST_PREFIX = os.environ.get("IMU_DATA_ROOT") or "/tmp/imu_runs"
try:
    os.makedirs(DST_PREFIX, exist_ok=True)
except Exception:
    DST_PREFIX = "/tmp/imu_runs"
    os.makedirs(DST_PREFIX, exist_ok=True)

def _map_path(p: str) -> str:
    p = str(p)
    return p.replace(SRC_PREFIX, DST_PREFIX, 1) if p.startswith(SRC_PREFIX) else p

# patch os.makedirs / os.mkdir
_orig_makedirs = os.makedirs
def makedirs(path, *a, **kw): return _orig_makedirs(_map_path(path), *a, **kw)
os.makedirs = makedirs

_orig_mkdir = os.mkdir
def mkdir(path, *a, **kw): return _orig_mkdir(_map_path(path), *a, **kw)
os.mkdir = mkdir

# patch open
_orig_open = builtins.open
def open_patched(file, *a, **kw): return _orig_open(_map_path(file), *a, **kw)
builtins.open = open_patched
