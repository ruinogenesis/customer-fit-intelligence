# Contributing

## One-time setup

- Python 3.12: `python3 -m venv .venv`, `source .venv/bin/activate`, then `python -m pip install -e "fitcore[dev]" -e "api[dev]"`
- Git hooks: `brew install pre-commit`, then `pre-commit install`
- Emulators: Java 21 and the Firebase CLI. The functions environment is created by hand:
  `python3.12 -m venv functions/venv` and `functions/venv/bin/python -m pip install -r functions/requirements.txt`

## Workflow

- One planner task = one branch = one pull request. Branch from the latest `main` as `tNNN-short-name`.
- Never commit straight to `main`.
- Commit small and often. Messages are short (about 50 characters) and imperative: "Add…", "Fix…".
- The PR title becomes the commit on `main` (squash merge), so make it clear.
- Merge only when green: `gh pr checks --watch && gh pr merge --squash --delete-branch`

## Code style

- `ruff format` decides the formatting. `ruff check` runs the defaults plus `I`, `B` and `UP`.
- Every function in `fitcore/src` and `api/src` has type hints, checked by mypy.
- Files end with a newline and have no trailing spaces. The pre-commit hooks enforce all of this on every commit.

## Tests

- Every change in behaviour comes with a test. Write the test first where practical.
- See a new test fail once before trusting it.
- Run every test with `pytest fitcore api`.
- Tests never touch real projects. Firestore tests use the emulator, and are skipped without it:
  `firebase emulators:exec --only firestore --project demo-cfi "pytest fitcore api"`

## Data and security

- Never commit secrets: no `.env` files, no service-account keys.
- Money is whole satang. Timestamps carry a timezone and are stored in UTC.
- Firestore rules stay locked until per-role rules exist (T304).
- Deploys go through CI only, never from a laptop.

## Decisions

- Open questions and decisions live in the planner (D1–D8). Record the answer there once decided.
