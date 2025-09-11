#!/usr/bin/env bash
set -euo pipefail

python3 -c 'import sys; print(sys.version)' >/dev/null || { echo "Python3 missing"; exit 1; }
python3 -m pip install --quiet fastapi uvicorn pydantic pytest pyyaml requests cryptography >/dev/null || true

# תיקיות ברירת-מחדל
mkdir -p assurance_store assurance_store_text assurance_store_programs assurance_store_adapters logs adapters/generated run_sandboxes

# policy.yaml אם חסר
if [ ! -f executor/policy.yaml ]; then
  mkdir -p executor
  cat > executor/policy.yaml <<'YAML'
strict_fs: true
no_net_default: true
cpu_seconds: 30
mem_bytes: 268435456
wall_seconds: 45
open_files: 256
allow_env: ["PATH","LANG","LC_ALL","TERM"]
allowed_tools:
  - name: echo
    args_regex: ".*"
    allow_net: false
  - name: python
    args_regex: ".*"
    allow_net: false
  - name: bwrap
    args_regex: ".*"
    allow_net: false
YAML
fi

echo "Starting IMU Core on http://127.0.0.1:8000"
exec python3 -m uvicorn server.bootstrap:APP --host 0.0.0.0 --port 8000
