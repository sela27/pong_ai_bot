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

## Runs

Set in [#16](https://github.com/sela27/pong_ai_bot/issues/16). A real Run, of the Position learner or the Agent, trains for hours on the one graphics card, and its result decides what comes next; a test environment's Run takes seconds.

- Runs live outside the repo, in `~/pong_ai_bot_runs/` or the directory `PONG_AI_BOT_RUNS_DIR` names. You only ever read there, and never delete anything there.
- On your own: everything in [`docs/checklists/before-a-run.md`](docs/checklists/before-a-run.md); tiny Runs of the test environments (`pong train --player test-env-1` to `test-env-4`), with `PONG_AI_BOT_RUNS_DIR` set to a temp directory; and reading a Run's TensorBoard files and `evaluations.jsonl`, to diagnose it line by line against [`docs/checklists/healthy-run.md`](docs/checklists/healthy-run.md).
- Starting, resuming or stopping a real Run waits for the developer's explicit "yes". Then hand them the exact command to run in their own terminal: a process you start can die with the conversation, hours into the Run. Their terminal is Windows PowerShell 5.1, which passes `~` to `uv` unexpanded, so write a Run's path as `$HOME\pong_ai_bot_runs\...`.
- The Confirmation set is played once, at the end of its Run; you never evaluate on it by hand ([#14](https://github.com/sela27/pong_ai_bot/issues/14)).
- After a failed Run, propose a diagnosis and a step up the ladder ([#14](https://github.com/sela27/pong_ai_bot/issues/14)); the developer decides.
- After a finished Run, once the developer says it's over: read its directory, then, on a branch, add its row to the finished Runs in `healthy-run.md` and correct the graph lines by what it showed.

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
