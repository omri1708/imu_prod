cat > ~/.git-next-phase <<'SH'
#!/bin/sh
set -e
# ענף נוכחי חייב להיות phase_<n>
CUR=$(git rev-parse --abbrev-ref HEAD); N=${CUR#phase_}; PREV=phase_$((N-1)); NEXT=phase_$((N+1)); \
git fetch -q origin; \
if git show-ref --verify --quiet refs/remotes/origin/$PREV; then BASE=origin/$PREV; \
elif git show-ref --verify --quiet refs/heads/$PREV; then BASE=$PREV; \
else echo "✗ previous branch $PREV not found (push/create it)"; exit 1; fi; \
git add -A && git reset --soft "$BASE" && git add -A && git commit -m "$CUR" && \
git push --force-with-lease -u origin "$CUR" && git switch -c "$NEXT"


SH
chmod +x ~/.git-next-phase
git config --global alias.next-phase '!~/.git-next-phase'
