# imu_repo/self_improve/testgen.py
from __future__ import annotations
import os, time, textwrap, importlib.util, sys
from typing import Tuple

TEST_DIR = "/mnt/data/imu_repo/tests/auto"
os.makedirs(TEST_DIR, exist_ok=True)

def create_test(module_stub_name: str, code: str) -> Tuple[str,str]:
    """
    יוצר קובץ בדיקה tests/auto/<timestamp>_<name>.py ומחזיר (path, module_name)
    """
    ts = int(time.time()*1000)
    base = f"{ts}_{module_stub_name}.py"
    path = os.path.join(TEST_DIR, base)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(code))
    mod_name = f"tests.auto.{os.path.splitext(base)[0]}"
    # ודא שהנתיב קיים ב-sys.path
    root = "/mnt/data/imu_repo"
    if root not in sys.path:
        sys.path.append(root)
    return path, mod_name