#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs
OUT="runs/phase_audit_$(date -u +%Y%m%dT%H%M%SZ).md"
PH_TMP="$(mktemp)"

# אסוף את כל הענפים phase_* ממוינים מספרית
git for-each-ref --format='%(refname:short)' 'refs/heads/phase_*' \
  | awk -F_ '{printf "%s %s\n",$0,$2}' \
  | sort -k2,2n \
  | awk '{print $1}' > "$PH_TMP"

{
  echo "# Phase Audit"
  echo
  echo "- Generated: $(date -u '+%Y-%m-%d %H:%M:%SZ')"
  echo
} > "$OUT"

while IFS= read -r BR; do
  [[ -z "$BR" ]] && continue
  N="${BR#phase_}"

  # בסיס להשוואה: phase_{N-1} אם קיים, אחרת main, ואם גם הוא לא קיים—master
  if git show-ref --verify --quiet "refs/heads/phase_$((N-1))"; then
    BASE="phase_$((N-1))"
  elif git show-ref --verify --quiet "refs/heads/main"; then
    BASE="main"
  elif git show-ref --verify --quiet "refs/heads/master"; then
    BASE="master"
  else
    echo "## $BR" >> "$OUT"
    echo "_no suitable base branch found_" >> "$OUT"
    echo >> "$OUT"
    continue
  fi

  {
    echo "## $BR (base: $BASE)"
    DIFF_NSTAT="$(git diff --shortstat --find-renames "$BASE".."$BR" || true)"
    echo
    echo "- Diff: $DIFF_NSTAT"
    echo
    echo "| status | path | trace |"
    echo "|---|---|---|"
  } >> "$OUT"

  # לכל קובץ: בנה TRACE של כל ה-phases שנגעו בו (לפי הודעות קומיט)
  git diff --name-status --find-renames "$BASE".."$BR" \
  | while IFS=$'\t' read -r ST P1 P2; do
      [[ -z "$ST" ]] && continue
      if [[ "${ST:0:1}" == "R" ]]; then F="$P2"; else F="$P1"; fi
      TRACE="$(git log --reverse --pretty=%s -- "$F" 2>/dev/null \
        | grep -Eo 'phase_[0-9]+' \
        | awk '!seen[$0]++' \
        | paste -sd '-->' -)"
      [[ -n "$TRACE" ]] && TRACE="${TRACE}-->" || TRACE=""
      printf '| %s | `%s` | %s%s |\n' "$ST" "$F" "$TRACE" "$BR" >> "$OUT"
    done

  echo >> "$OUT"
done < "$PH_TMP"

rm -f "$PH_TMP"
echo "✅ Wrote $OUT"
