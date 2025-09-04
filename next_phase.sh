cat > ~/.git-next-phase <<'SH'
#!/bin/sh
set -e

git rev-parse --git-dir >/dev/null 2>&1 || { echo "Not a git repo"; exit 1; }
CUR="$(git rev-parse --abbrev-ref HEAD)"
[ "$CUR" = "HEAD" ] && { echo "Detached HEAD — checkout a branch first"; exit 1; }

case "$CUR" in
  phase_[0-9]*) ;; *) echo "Current branch must be phase_<number> (got: $CUR)"; exit 1;;
esac
NUM="${CUR#phase_}"
case "$NUM" in ''|*[!0-9]*) echo "Bad phase number: $NUM"; exit 1;; esac
PREV="phase_$((NUM-1))"
NEXT="phase_$((NUM+1))"

URL="$(git remote get-url origin 2>/dev/null || true)"
[ -z "$URL" ] && { echo "Missing remote 'origin' (SSH). Add: git remote add origin git@github.com:omri1708/IMU_repo.git"; exit 1; }
case "$URL" in git@github.com:*) ;; *) echo "origin is not SSH: $URL"; exit 1;; esac

# ודא שהשלב הקודם קיים ברמוֹט וש-HEAD צאצא שלו (שומר דלתא נקייה)
git fetch -q origin
if ! git ls-remote --exit-code --heads origin "$PREV" >/dev/null 2>&1; then
  [ "$PREV" = "phase_-1" ] || { echo "Remote branch '$PREV' missing. Push it first."; exit 1; }
fi
if git show-ref --verify --quiet "refs/remotes/origin/$PREV"; then
  git merge-base --is-ancestor "origin/$PREV" HEAD || { echo "Current branch is not based on origin/$PREV"; exit 1; }
fi

# הודעת קומיט: 'phase_N' או 'phase_N: ...'
MSG="$CUR"; [ $# -gt 0 ] && MSG="$CUR: $*"

git add -A
if ! (git diff --quiet --staged && git diff --quiet); then
  git commit -m "$MSG"
else
  git commit --allow-empty -m "$MSG"
fi

git push -u origin "$CUR"

# פיצול לשלב הבא מהשלב הנוכחי (דלתא מול הקודם)
if git show-ref --verify --quiet "refs/heads/$NEXT"; then
  git switch "$NEXT" 2>/dev/null || git checkout "$NEXT"
else
  git switch -c "$NEXT" 2>/dev/null || git checkout -b "$NEXT"
fi

echo "✅ pushed $CUR (msg: '$MSG'); branched $NEXT off $CUR"
SH

chmod +x ~/.git-next-phase
git config --global alias.next-phase '!~/.git-next-phase'
