SHELL := /bin/bash
.ONESHELL:

.PHONY: next_phase_trace
next_phase_trace:
	set -euo pipefail

	# חייבים להיות על ענף phase_<n>
	CUR="$$(git rev-parse --abbrev-ref HEAD)"
	case "$$CUR" in phase_[0-9]*) ;; *) echo "be on phase_<n> branch (got: $$CUR)"; exit 1;; esac
	N="$${CUR#phase_}"
	PREV="phase_$$(($$N-1))"
	NEXT="phase_$$(($$N+1))"

	# origin חייב להיות SSH
	URL="$$(git remote get-url origin 2>/dev/null || true)"; \
	[ -n "$$URL" ] || { echo "missing remote 'origin'"; exit 1; }
	case "$$URL" in git@*|ssh://*) ;; *) echo "origin must be SSH (got: $$URL)"; exit 1;; esac

	# קבע BASE: origin/PREV אם קיים, אחרת PREV לוקלי
	git fetch -q origin || true
	if git show-ref --verify --quiet "refs/remotes/origin/$$PREV"; then BASE="origin/$$PREV"; \
	elif git show-ref --verify --quiet "refs/heads/$$PREV"; then BASE="$$PREV"; \
	else echo "✗ previous branch $$PREV not found (push/create it)"; exit 1; fi

	# אסוף קבצים ששונו מאז BASE (כולל untracked, rename, delete)
	CHG_FILE="$$(mktemp)"
	git diff --name-status --find-renames "$$BASE"..HEAD > "$$CHG_FILE"
	git ls-files --others --exclude-standard | sed 's/^/A\t/' >> "$$CHG_FILE"

	# החזר HEAD ל-BASE אבל השאר את השינויים על הדיסק (soft)
	git reset --soft "$$BASE"

	# עבור כל קובץ ששונה: בנה TRACE והכן קומיט נפרד עם הודעת ה-trace
	while IFS=$'\t' read -r ST P1 P2; do
		[ -z "$$ST" ] && continue
		# אפס סטייג'ינג בכל איטרציה
		git reset -q

		case "$$ST" in
			M|A) git add -A -- "$$P1" ;;
			D)   git rm -f -- "$$P1" || true ;;
			R*)  git rm -f -- "$$P1" || true; git add -A -- "$$P2" ;;
			*)   git add -A -- "$$P1" ;;
		esac

		# בנה TRACE: כל phases שנגעו בקובץ בעבר (לפי הודעות commit) בסדר כרונולוגי, בלי כפילויות
		TRACE="$$( { git log --reverse --pretty=%s -- "$$P1" 2>/dev/null; [ -n "$$P2" ] && git log --reverse --pretty=%s -- "$$P2" 2>/dev/null; } \
			| grep -Eo 'phase_[0-9]+' \
			| awk '!seen[$$0]++' \
			| paste -sd '-->' - )"
		if [ -z "$$TRACE" ]; then TRACE="$$CUR"; else TRACE="$$TRACE-->$$CUR"; fi

		git commit -m "$$TRACE"
	done < "$$CHG_FILE"
	rm -f "$$CHG_FILE"

	# דחיפה בטוחה ופתיחת הענף הבא
	git push --force-with-lease -u origin "$$CUR"
	git switch -c "$$NEXT" 2>/dev/null || git switch "$$NEXT"
	echo "✅ per-file trace commits pushed on $$CUR; opened $$NEXT"

.PHONY: test-auto
test-auto:
	python -m engine.testgen.runtime_cases
	python -m engine.testgen.kpi_cases
	python -m tools.test_orchestrator

.PHONY: phase_audit
phase_audit:
	@/bin/bash scripts/phase_audit.sh

