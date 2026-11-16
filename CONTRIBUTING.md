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
- Run every test with `pytest fitcore api`, and the dashboard's with `node --test dashboard/tests/*.test.js`.
- Tests never touch real projects. Firestore and Auth tests use the emulators, and are skipped without them:
  `firebase emulators:exec --only firestore,auth --project demo-cfi "pytest fitcore api"`

## Data and security

- Never commit secrets: no `.env` files, no service-account keys.
- Money is whole satang. Timestamps carry a timezone and are stored in UTC.
- Every Firestore document has a schema in `fitcore.documents`, and the API checks each one before writing it. After changing a schema, run `python -m fitcore.documents > fitcore/DOCUMENTS.md`.
- No app writes to Firestore, and staff read only what `firestore.rules` lets their role read (T304). `READERS` in `api/tests/test_firestore_rules.py` is that table; a new collection stays locked until its task adds a row.
- Deploys go through CI only, never from a laptop.

## Decisions

- Open questions and decisions live in the planner (D1–D8). Record the answer there once decided.
