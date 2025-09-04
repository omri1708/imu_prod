cat > ~/.git-next-phase <<'SH'
#!/bin/sh
set -e

git rev-parse --git-dir >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }
CUR="$(git rev-parse --abbrev-ref HEAD)"
[ "$CUR" = "HEAD" ] && { echo "Detached HEAD — checkout a branch first"; exit 1; }

case "$CUR" in
  phase_[0-9]*) ;;
  *) echo "Current branch must be phase_<number> (got: $CUR)"; exit 1;;
esac
NUM="${CUR#phase_}"
case "$NUM" in ''|*[!0-9]*) echo "Bad phase number: $NUM"; exit 1;; esac

NEXT=$((NUM + 1))
NEXT_BRANCH="phase_$NEXT"

# ודא origin עם SSH
URL="$(git remote get-url origin 2>/dev/null || true)"
[ -z "$URL" ] && { echo "Missing remote 'origin' (SSH). Add: git remote add origin git@github.com:omri1708/IMU_repo.git"; exit 1; }
case "$URL" in git@github.com:*) ;; *) echo "origin is not SSH: $URL"; exit 1;; esac

# הודעת קומיט: "phase_N" או "phase_N: ..." אם סופקת הודעה
if [ $# -gt 0 ]; then MSG="$CUR: $*"; else MSG="$CUR"; fi

git add -A
if ! git diff --quiet --staged || ! git diff --quiet; then
  git commit -m "$MSG"
else
  git commit --allow-empty -m "$MSG"
fi

git push -u origin "$CUR"

# פיצול לשלב הבא מתוך השלב הנוכחי (דלתא מול הקודם)
if git show-ref --verify --quiet "refs/heads/$NEXT_BRANCH"; then
  git switch "$NEXT_BRANCH" 2>/dev/null || git checkout "$NEXT_BRANCH"
else
  git switch -c "$NEXT_BRANCH" 2>/dev/null || git checkout -b "$NEXT_BRANCH"
fi

echo "✅ pushed $CUR (msg: '$MSG'); branched $NEXT_BRANCH off $CUR"
SH

chmod +x ~/.git-next-phase
git config --global alias.next-phase '!~/.git-next-phase'
