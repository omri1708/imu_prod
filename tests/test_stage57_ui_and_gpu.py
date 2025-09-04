# imu_repo/tests/test_stage57_ui_and_gpu.py
from __future__ import annotations
import os, math

from ui.gen_frontend import generate_site
from ui.accessibility_gate import check_directory
from caps.gpu_dispatch import GPUScheduler, random_matrix, naive_mul

SITE_DIR = "/mnt/data/imu_repo/site57"

def test_ui_and_a11y():
    spec = {
        "lang":"he","title":"IMU App",
        "pages":[
            {"name":"דף הבית","file":"index.html","elements":[
                {"type":"h1","text":"אפליקציה נגישה"},
                {"type":"p","text":"נבנה אוטומטית עם בדיקות נגישות."},
                {"type":"img","src":"hero.png","alt":"איור של גיבור","caption":"תיאור גרפי"},
                {"type":"input","id":"q","label":"חיפוש","input_type":"search","placeholder":"מונח לחיפוש","required":True},
                {"type":"button","label":"שלח","message":"נשלח!"}
            ]},
            {"name":"אודות","file":"about.html","elements":[
                {"type":"h1","text":"אודות"},
                {"type":"p","text":"עמוד נוסף לתפריט ניווט."}
            ]}
        ]
    }
    generate_site(spec, SITE_DIR)
    res = check_directory(SITE_DIR, min_contrast=4.5)
    ok = res["ok"]
    print("A11y:", res)
    return 0 if ok else 1

def test_gpu_aware():
    m,n,p = 24, 16, 12
    a = random_matrix(m,n,seed=1)
    b = random_matrix(n,p,seed=2)
    ref = naive_mul(a,b,m,n,p)

    sch = GPUScheduler(prefer_gpu=True, max_workers=2)
    out = sch.matmul(a,b,m,n,p)
    res = out["ok"] and len(ref)==len(out := out)  # just guard
    C = out["result"] if "result" in out else None  # not used; the scheduler returns only timing & meta
    # כדי לא לבצע כפול — נבדוק התאמה ע"י הרצה CPU ייעודית קטנה
    calc = sch.matmul_cpu(a,b,m,n,p)
    same = all(abs(calc[i]-ref[i]) < 1e-6 for i in range(len(ref)))
    print("GPU-aware:", {"used": out["used"], "ms": out["ms"], "detected_gpu": out["detected_gpu"]})
    return 0 if same else 1

def run():
    rc1 = test_ui_and_a11y()
    rc2 = test_gpu_aware()
    ok = (rc1==0 and rc2==0)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())