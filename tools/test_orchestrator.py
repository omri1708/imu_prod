from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

OUT = Path("/mnt/data/imu_repo/runs/_test_runs")
OUT.mkdir(parents=True, exist_ok=True)

def _write_jsonl(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run_pytest_and_log(marker: str | None = None) -> int:
    ts = int(time.time())
    rep = OUT / f"pytest_{ts}.jsonl"
    cmd = ["pytest", "-q", "--maxfail=1", "--disable-warnings"]
    if marker:
        cmd += ["-m", marker]
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start
    _write_jsonl(rep, {
        "ts": ts,
        "cmd": cmd,
        "rc": r.returncode,
        "duration_s": dur,
        "stdout": r.stdout[-4000:],  # tail להקטין נפח
        "stderr": r.stderr[-4000:]
    })
    print(r.stdout, end="")
    print(r.stderr, file=sys.stderr, end="")
    return r.returncode

if __name__ == "__main__":
    sys.exit(run_pytest_and_log(marker=None))
