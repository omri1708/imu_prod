#!/usr/bin/env bash
set -euo pipefail

DATE="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT="$ROOT/runs/support_bundle_$DATE"
mkdir -p "$OUT"

# --- Repo / Git info ---
{
  echo "## GIT"
  echo -n "HEAD: "; git -C "$ROOT" rev-parse HEAD || true
  echo
  echo "### status --porcelain"
  git -C "$ROOT" status --porcelain=v1 || true
  echo
  echo "### remote -v"
  git -C "$ROOT" remote -v || true
  echo
  echo "### last 50 commits"
  git -C "$ROOT" log --oneline -n 50 || true
} > "$OUT/git.txt" || true

# --- System info ---
{
  echo "## SYSTEM"
  uname -a
  command -v sw_vers >/dev/null && sw_vers || true
  echo
  echo "### Tools"
  python3 --version 2>&1 || true
  pip3 --version 2>&1 || true
  uvicorn --version 2>/dev/null || true
  docker --version 2>/dev/null || true
  kubectl version --client --output=yaml 2>/dev/null || true
} > "$OUT/system.txt"

# --- Python deps ---
pip3 freeze > "$OUT/pip-freeze.txt" || true

# --- ENV (redacted) ---
if [[ -f "$ROOT/.env" ]]; then
  sed -E 's#(=).*#=\[REDACTED\]#' "$ROOT/.env" > "$OUT/env.redacted"
fi

# --- Key configs (copy if exist) ---
for f in requirements.txt pyproject.toml poetry.lock setup.cfg setup.py docker-compose.yml Makefile pytest.ini; do
  [[ -f "$ROOT/$f" ]] && cp "$ROOT/$f" "$OUT/" || true
done

# --- Project tree (without heavy dirs) ---
if command -v tree >/dev/null; then
  tree -a -I '.git|.venv|__pycache__|.mypy_cache|.pytest_cache|runs|logs' "$ROOT" > "$OUT/tree.txt" || true
fi

# --- Logs (tail only to keep small) ---
for d in runs logs; do
  if [[ -d "$ROOT/$d" ]]; then
    mkdir -p "$OUT/$d"
    # tail אחרון מקבצים נפוצים
    while IFS= read -r -d '' lf; do
      tail -n 1000 "$lf" > "$OUT/$d/$(basename "$lf").tail" || true
    done < <(find "$ROOT/$d" -maxdepth 1 -type f \( -name '*.log' -o -name '*.jsonl' -o -name '*.out' \) -print0 2>/dev/null)
  fi
done

# --- pytest cache (אם קיים) ---
for p in .pytest_cache pytest_cache; do
  [[ -d "$ROOT/$p" ]] && tar -czf "$OUT/$p.tgz" -C "$ROOT" "$p" || true
done

# --- Pack everything ---
TAR="$ROOT/runs/support_bundle_$DATE.tgz"
tar -czf "$TAR" -C "$OUT" .
echo "Bundle created: $TAR"
