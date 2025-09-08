# imu_repo/packaging/packager.py
from __future__ import annotations
from typing import Dict, Any
import os, shutil, tempfile, zipapp, textwrap, json, subprocess, sys

ENTRY = """\
# __main__.py – אריזת ריצה ל-IMU
from __future__ import annotations
import argparse, json, asyncio
from streaming.ws_server import WSServer
from ui.toolkits_bridge import serve_static_ui, console_render

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ui","console"], default="console")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8976)
    args = p.parse_args()
    if args.mode=="ui":
        serve_static_ui(host=args.host, port=8975)
    srv = WSServer(args.host, args.port, handler=lambda s: asyncio.sleep(0.0) or (f"echo:{s}"))
    await srv.start()
    console_render("IMU package running – press Ctrl+C to stop")
    try:
        while True: await asyncio.sleep(1)
    except KeyboardInterrupt:
        await srv.stop()

if __name__=="__main__":
    asyncio.run(main())
"""

def build_zipapp(target_path: str="/mnt/data/imu_app.pyz") -> str:
    """
    אורז תתי־ספריות חיוניות לתוך zipapp והרצה ב־python target.pyz
    """
    base = "/mnt/data/imu_repo"
    req = ["realtime","ui","compute","engine","packaging","dist"]
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "imu_pkg")
        os.makedirs(dst, exist_ok=True)
        # העתק מודולים נדרשים
        for r in req:
            src = os.path.join(base, r)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dst, r))
        # כתוב __main__.py
        with open(os.path.join(dst, "__main__.py"), "w", encoding="utf-8") as f:
            f.write(ENTRY)
        # בנה zipapp
        zipapp.create_archive(dst, target=target_path, interpreter="/usr/bin/env python3")
    return target_path

def run_zipapp(path: str, args: list[str] | None=None) -> int:
    cmd = [sys.executable, path] + (args or [])
    return subprocess.call(cmd)