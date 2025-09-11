# engine/orchestrator/run_store.py
import os, time, uuid, shutil, contextlib, fcntl

RUNS_DIR = os.getenv("IMU_RUNS_DIR", ".imu/runs")
TTL_SEC  = int(os.getenv("IMU_RUN_TTL_SEC", "86400"))  # יום

os.makedirs(RUNS_DIR, exist_ok=True)

@contextlib.contextmanager
def run_context(user: str | None = None):
    run_id = f"{int(time.time())}_{uuid.uuid4().hex}_{(user or 'anon')}"
    path   = os.path.join(RUNS_DIR, run_id)
    os.makedirs(path, exist_ok=True)
    lock_p = os.path.join(path, ".lock")

    with open(lock_p, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield type("Run", (), {"id": run_id, "path": path})
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def gc_old_runs():
    now = time.time()
    for name in os.listdir(RUNS_DIR):
        p = os.path.join(RUNS_DIR, name)
        try:
            if os.path.isdir(p) and (now - os.path.getmtime(p)) > TTL_SEC:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
