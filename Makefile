SHELL := /bin/bash
.ONESHELL:

.PHONY: next_phase
next_phase:
	# מריץ את כל הבלוק הזה באותו shell, עוצר על כל שגיאה
	set -euo pipefail

	# שם הענף הנוכחי חייב להיות phase_<n>
	CUR="$$(git rev-parse --abbrev-ref HEAD)"
	case "$$CUR" in phase_[0-9]*) ;; *) echo "be on phase_<n> branch (got: $$CUR)"; exit 1;; esac
	N="$${CUR#phase_}"
	PREV="phase_$$(($$N-1))"
	NEXT="phase_$$(($$N+1))"

	# קובע בסיס: origin/PREV אם קיים, אחרת PREV מקומי
	git fetch -q origin || true
	if git show-ref --verify --quiet "refs/remotes/origin/$$PREV"; then
		BASE="origin/$$PREV"
	elif git show-ref --verify --quiet "refs/heads/$$PREV"; then
		BASE="$$PREV"
	else
		echo "✗ previous branch $$PREV not found (push/create it)"
		exit 1
	fi

	# סקווש: כל הדלתא מאז $$BASE + מה שב־working tree → קומיט אחד בשם $$CUR
	git add -A
	git reset --soft "$$BASE"
	git add -A
	# סטייג' סופי לפני קומיט (גם אם משהו התפספס)
	git add -A
	git commit -m "$$CUR" 2>/dev/null || git commit --allow-empty -m "$$CUR"

	# פוש (SSH origin), שומר מפני דריסה לא רצויה
	git push --force-with-lease -u origin "$$CUR"

	# פתיחת/מעבר לענף הבא (אם קיים - רק switch)
	git switch -c "$$NEXT" 2>/dev/null || git switch "$$NEXT"

	echo "✅ pushed $$CUR; opened $$NEXT"
