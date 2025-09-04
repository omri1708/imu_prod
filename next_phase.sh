cat > ~/.git-next-phase <<'SH'
#!/bin/sh
set -e

# ודא שאנחנו בתוך רפו
git rev-parse --git-dir >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }

CUR="$(git rev-parse --abbrev-ref HEAD)"
[ "$CUR" = "HEAD" ] && { echo "Detached HEAD — checkout a branch first"; exit 1; }

# ודא שיש origin ב-SSH
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' missing. Add: git remote add origin git@github.com:omri1708/IMU_repo.git"
  exit 1
fi
URL="$(git remote get-url origin)"
case "$URL" in git@github.com:*) ;; *) echo "Remote origin is not SSH: $URL"; exit 1 ;; esac

# שלוף את המספר מהשם phase_<num>
NUM="$(printf "%s" "$CUR" | sed -n 's/^phase_\([0-9][0-9]*\)$/\1/p')"
[ -z "$NUM" ] && { echo "Current branch must be 'phase_<number>' (got: $CUR)"; exit 1; }

NEXT=$((NUM + 1))
NEXT_BRANCH="phase_$NEXT"

# הודעת קומיט (ברירת מחדל: checkpoint)
if [ $# -gt 0 ]; then MSG="$*"; else MSG="checkpoint"; fi

git add -A
# קומיט גם אם אין שינויים (שומר רצף)
if ! git diff --quiet --staged || ! git diff --quiet; then
  git commit -m "$MSG"
else
  git commit --allow-empty -m "$MSG"
fi

git push -u origin "$CUR"

# יצירת/מעבר לענף הבא
if git show-ref --verify --quiet "refs/heads/$NEXT_BRANCH"; then
  git checkout "$NEXT_BRANCH"
else
  git checkout -b "$NEXT_BRANCH"
fi

echo "✅ pushed $CUR and moved to $NEXT_BRANCH (origin via SSH)"
SH

chmod +x ~/.git-next-phase
git config --global alias.next-phase '!~/.git-next-phase'
