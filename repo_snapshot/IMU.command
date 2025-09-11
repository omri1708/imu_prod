#!/bin/bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "Python not found"; exit 1; fi
# מרימים את השרת ברקע ופותחים את הדפדפן
"$PY" -m uvicorn server.boot_strict:APP --host 127.0.0.1 --port 8000 >/tmp/imu.log 2>&1 &
sleep 1
open "http://127.0.0.1:8000/chat/"