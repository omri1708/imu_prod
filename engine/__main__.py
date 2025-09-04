# imu_repo/engine/__main__.py
from __future__ import annotations
import argparse, json, sys
from typing import Any, Dict, List, Tuple

from engine.pipeline import Engine
from ui.desktop import DesktopApp


def run_cli(program_path: str, payload_path: str) -> int:
    eng = Engine()
    with open(program_path, "r", encoding="utf-8") as f:
        program: List[Dict[str,Any]] = json.load(f)
    with open(payload_path, "r", encoding="utf-8") as f:
        payload: Dict[str,Any] = json.load(f)
    status, body = eng.run_program(program, payload)
    print(json.dumps({"status": status, "body": body}, ensure_ascii=False, indent=2))
    return 0 if 200 <= status < 400 else 1


def run_desktop() -> int:
    eng = Engine()
    
    def _runner(program, payload) -> Tuple[int, Dict[str,Any]]:
        return eng.run_program(program, payload)
    app = DesktopApp(_runner)
    app.start()
    return 0


def run_smoke() -> int:
    from tests.smoke import main as smoke_main
    smoke_main()
    return 0


def run_web(host: str, port: int) -> int:
    try:
        from ui.web import WebUI
    except Exception as e:
        print(f"[ERR] WebUI is unavailable: {e}", file=sys.stderr)
        return 2
    srv = WebUI()
    srv.serve(host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="IMU engine entrypoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cli = sub.add_parser("cli", help="Run a program with a payload (files)")
    p_cli.add_argument("--program", required=True, help="path to JSON list of ops")
    p_cli.add_argument("--payload", required=True, help="path to JSON object")

    sub.add_parser("desktop", help="Start Desktop UI (tkinter)")
    sub.add_parser("smoke", help="Run smoke test")

    p_web = sub.add_parser("web", help="Start Web UI (FastAPI)")
    p_web.add_argument("--host", default="0.0.0.0")
    p_web.add_argument("--port", default=8000, type=int)

    args = p.parse_args(argv)
    if args.cmd == "cli":
        return run_cli(args.program, args.payload)
    if args.cmd == "desktop":
        return run_desktop()
    if args.cmd == "smoke":
        return run_smoke()
    if args.cmd == "web":
        return run_web(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
