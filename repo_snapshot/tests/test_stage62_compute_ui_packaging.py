# imu_repo/tests/test_stage62_compute_ui_packaging.py
from __future__ import annotations
import random, os, time, asyncio, subprocess, sys
from typing import List
from compute.ops import vec_add, matmul, conv1d
from compute.registry import REGISTRY
from ui.toolkits_bridge import ensure_static_ui, serve_static_ui
from packaging.packager import build_zipapp, run_zipapp

def assert_true(b, msg=""):
    if not b:
        print("ASSERT FAIL:", msg)
        raise SystemExit(1)

def test_compute_correctness():
    # vec_add
    a=[1,2,3,4]; b=[10,20,30,40]
    out = vec_add(a,b)
    assert_true(out==[11,22,33,44], "vec_add wrong")

    # matmul
    A = [[1,2],[3,4]]
    B = [[5,6],[7,8]]
    C = matmul(A,B)
    assert_true(C==[[19,22],[43,50]], f"matmul wrong:{C}")

    # conv1d
    x=[1,2,3,4]; w=[1,0,-1]
    y = conv1d(x,w,pad=1,stride=1)
    # pad=1 => z=[0,1,2,3,4,0]; conv => [1,2,1, -4]
    assert_true(y==[1.0,2.0,1.0,-4.0], f"conv1d wrong:{y}")

def test_autotune_learns():
    # בנה מטריצות גדולות יחסית כדי להעדיף gpu_sim
    n=32; k=32; m=32
    import random
    A = [[random.random() for _ in range(k)] for __ in range(n)]
    B = [[random.random() for _ in range(m)] for __ in range(k)]
    # ריצה כפולה – השנייה אמורה לבחור אוטומטית backend המתאים (נרשם ל-autotune.json)
    C1 = matmul(A,B)
    C2 = matmul(A,B)
    # בדיקת בלתי־ריקות
    assert_true(len(C1)==n and len(C1[0])==m)
    # יש קובץ autotune
    assert_true(os.path.exists("/mnt/data/imu_repo/autotune.json"))

def test_ui_bridge():
    d = ensure_static_ui()
    assert_true(os.path.exists(os.path.join(d,"index.html")))
    t = serve_static_ui()
    time.sleep(0.2)
    # לא מושכים מהדפדפן, רק בודקים שהשרת חי – אין חריגה
    # (כאן לא מפעילים WebSocket – נבדק בשלבים קודמים)

def test_packaging_zipapp():
    path = build_zipapp("/mnt/data/imu_app.pyz")
    assert_true(os.path.exists(path))
    # הרצה זריזה במצב console; יוצא אחרי שנייה (נבטל ע"י Popen/terminate)
    p = subprocess.Popen([sys.executable, path, "--mode","console"])
    time.sleep(1.0)
    p.terminate()
    try: p.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        p.kill()

def run():
    test_compute_correctness()
    test_autotune_learns()
    test_ui_bridge()
    test_packaging_zipapp()
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(run())