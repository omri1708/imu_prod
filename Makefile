SHELL := /bin/sh

.PHONY: next_phase
next_phase:
	@set -e; \
	CUR="$$(git rev-parse --abbrev-ref HEAD)"; \
	case "$$CUR" in phase_[0-9]*) ;; *) echo "be on phase_<n> branch (got: $$CUR)"; exit 1;; esac; \
	N="$${CUR#phase_}"; PREV="phase_$$(($$N-1))"; NEXT="phase_$$(($$N+1))"; \
	# ודא רימוֹט SSH
	URL="$$(git remote get-url origin 2>/dev/null || true)"; \
	[ -n "$$URL" ] || { echo "missing remote 'origin'"; exit 1; }; \
	case "$$URL" in git@*|ssh://*) ;; *) echo "origin must be SSH (got: $$URL)"; exit 1;; esac; \
	# קבע בסיס: origin/$$PREV אם קיים, אחרת $$PREV לוקלי
	git fetch -q origin; \
	if git show-ref --verify --quiet "refs/remotes/origin/$$PREV"; then BASE="origin/$$PREV"; \
	elif git show-ref --verify --quiet "refs/heads/$$PREV"; then BASE="$$PREV"; \
	else echo "✗ previous branch $$PREV not found (push/create it)"; exit 1; fi; \
	# סקווש: כל הדלתא מאז $$BASE + מה שב־working tree → קומיט אחד בשם $$CUR
	git add -A; \
	git reset --soft "$$BASE"; \
	git add -A; \
	if git diff --quiet --cached && git diff --quiet; then \
	  git commit --allow-empty -m "$$CUR"; \
	else \
	  git commit -m "$$CUR"; \
	fi; \
	# פוש לענף הנוכחי (SSH), עם הגנה
	git push --force-with-lease -u origin "$$CUR"; \
	# פתח/עבור לענף הבא
	git switch -c "$$NEXT" 2>/dev/null || git switch "$$NEXT"; \
	echo "✅ pushed $$CUR; opened $$NEXT"
