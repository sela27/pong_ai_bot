# CLAUDE.md

## Verification

- The pre-commit hook, `.githooks/pre-commit`, runs `ruff format --check`, `ruff check`, `mypy` and the fast tests through `uv run`. Enable it with `git config core.hooksPath .githooks`, and run it by hand with `sh .githooks/pre-commit`.
- Fast tests: `uv run pytest`. They are the default, and the hook runs them.
- Slow tests: `uv run pytest -m slow`. The hook never runs them.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `sela27/pong_ai_bot`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
