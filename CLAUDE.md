# CLAUDE.md

## Verification

- The pre-commit hook, `.githooks/pre-commit`, runs `ruff format --check`, `ruff check`, `mypy` and the fast tests through `uv run`. Enable it with `git config core.hooksPath .githooks`, and run it by hand with `sh .githooks/pre-commit`.
- Fast tests: `uv run pytest`. They are the default, and the hook runs them.
- Slow tests: `uv run pytest -m slow`. The hook never runs them.

## Working rules

- Build test-first with the `tdd` skill. [#11](https://github.com/sela27/pong_ai_bot/issues/11) lists what is exempt.
- Every commit goes through the pre-commit hook. When it fails, fix the cause; `--no-verify` is never an option ([#11](https://github.com/sela27/pong_ai_bot/issues/11)).
- One branch per piece of work (a ticket, a bug fix), merged into `main` locally, without a PR ([#11](https://github.com/sela27/pong_ai_bot/issues/11)).
- A change that makes `README.md` or `CLAUDE.md` wrong updates them in the same branch. Both link to the issue that made a decision instead of copying it ([#16](https://github.com/sela27/pong_ai_bot/issues/16)).

## The window

- An agent can't see the window, and it isn't built test-first. After any change to the window, ask the developer to go through [`docs/checklists/window.md`](docs/checklists/window.md), and don't call the work done before they have ([#16](https://github.com/sela27/pong_ai_bot/issues/16)).

## Working with the developer

Set in [#16](https://github.com/sela27/pong_ai_bot/issues/16).

- Answer in Hebrew, in an RTL HTML card through `mcp__visualize__show_widget` when that tool is available: `dir="rtl"` on the card, English terms and code wrapped LTR (`<span dir="ltr">`). Every substantive answer goes in the card; text outside it stays one short line.
- When grilling, ask one question per message, with your recommendation.
- Implement only on an explicit instruction from the developer.
- Split tickets very small, without overdoing it, so progress is easy to follow.
- Challenge the developer's assumptions: this is their first ML project.
- The pipeline skills, `/wayfinder` (or `/grill-with-docs`) → `/to-spec` → `/to-tickets` → `/implement` (or `/implement-spec`), start only when the developer types them. When one is the next step, name it for them to type.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `sela27/pong_ai_bot`, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
