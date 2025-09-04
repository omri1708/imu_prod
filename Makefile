SHELL := /bin/bash

.PHONY: next_phase
next_phase:
	@set -euo pipefail; \
	CUR="$$(git rev-parse --abbrev-ref HEAD)"; \
	case "$$CUR" in phase_[0-9]*) ;; *) echo "be on phase_<n> branch (got: $$CUR)"; exit 1;; esac; \
	N="$${CUR#phase_}"; \
	PREV="phase_$$(($$N-1))"; \
	NEXT="phase_$$(($$N+1))"; \
	# echo "DBG CUR=$$CUR PREV=$$PREV NEXT=$$NEXT" >&2; \
	git fetch -q origin; \
	if git show-ref --verify --quiet "refs/remotes/origin/$$PREV"; then BASE="origin/$$PREV"; \
	elif git show-ref --verify --quiet "refs/heads/$$PREV"; then BASE="$$PREV"; \
	else echo "✗ previous branch $$PREV not found (push/create it)"; exit 1; fi; \
	git add -A; \
	git reset --soft "$$BASE"; \
	git add -A; \
	# הודעת קומיט = שם השלב (למשל phase_91)
	if git diff --quiet --cached && git diff --quiet; then \
	  git commit --allow-empty -m "$$CUR"; \
	else \
	  git commit -m "$$CUR"; \
	fi; \
	# פוש ב-SSH (מניח origin=git@github.com:...)
	URL="$$(git remote get-url origin)"; case "$$URL" in git@*|ssh://*) ;; *) echo "origin must be SSH (got: $$URL)"; exit 1;; esac; \
	git push --force-with-lease -u origin "$$CUR"; \
	# יצירת/מעבר לענף הבא
	git switch -c "$$NEXT" 2>/dev/null || git switch "$$NEXT"; \
	echo "✅ pushed $$CUR; opened $$NEXT"
