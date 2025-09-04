#!/bin/bash
set -e

# מוציא את שם הענף הנוכחי
CUR=$(git branch --show-current)

# שולף את המספר מהשם (מניח שהפורמט הוא phase_XX)
NUM=${CUR#phase_}

# אם לא הצליח לזהות מספר
if ! [[ "$NUM" =~ ^[0-9]+$ ]]; then
  echo "Error: Current branch name is not in the format phase_<number>"
  exit 1
fi

NEXT=$((NUM+1))
NEXT_BRANCH="phase_$NEXT"

# קומיט + פוש לענף הנוכחי
git add -A
git commit -m "checkpoint" || true  # לא יקרוס אם אין מה לקומט
git push -u origin "$CUR"

# יצירת הענף החדש
git checkout -b "$NEXT_BRANCH"
echo "✅ moved from $CUR to $NEXT_BRANCH"
