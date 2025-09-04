cat > ~/.git-next-phase <<'SH'
#!/bin/sh
set -e
# ענף נוכחי חייב להיות phase_<n>
CUR="$(git rev-parse --abbrev-ref HEAD)"
case "$CUR" in phase_[0-9]*) ;; *) echo "be on phase_<n> branch (got: $CUR)"; exit 1;; esac
N="${CUR#phase_}"; NEXT="phase_$((N+1))"
MSG="$CUR"; [ $# -gt 0 ] && MSG="$CUR: $*"

# מקמֵט בדיוק את מה ששונה מאז הקומיט האחרון בענף (HEAD)
git add -A
if git diff --quiet --cached && git diff --quiet; then
  git commit --allow-empty -m "$MSG"
else
  git commit -m "$MSG"
fi

# דחיפה דרך origin (SSH או מה שמוגדר אצלך)
git push -u origin "$CUR"

# יצירת הענף הבא מהקומיט הזה
git switch -c "$NEXT" 2>/dev/null || git checkout -b "$NEXT"
echo "pushed $CUR (msg: '$MSG'); branched $NEXT"
SH
chmod +x ~/.git-next-phase
git config --global alias.next-phase '!~/.git-next-phase'
