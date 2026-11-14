# api

Backend service that ingests events and serves scores. Deployed to Cloud Run.

One Flask + flask-smorest app under `/api/v1`, in `src/cfi_api`. It needs
fitcore, which is not on PyPI, so install both in one command from the repo
root: `python -m pip install -e "fitcore[dev]" -e "api[dev]"`.

Settings come from `CFI_*` environment variables (the prefix is dropped):

- `CFI_MODEL_PATH` and `CFI_MODEL_VERSION`: the model.json to score with
  and its version; set both or neither. Without a model, `/events` answers 503.
- `CFI_STORE`: `memory` (the default; lost when the server stops) or `firestore`.
- `CFI_FIRESTORE_PROJECT`: the Firestore project, such as `demo-cfi` for the emulator.

Run it locally with the reference model, from the repo root:

    CFI_MODEL_PATH=notebooks/results/t112_model.json CFI_MODEL_VERSION=t112-sim \
      flask --app cfi_api run --port 8000

- Health: http://127.0.0.1:8000/api/v1/health
- OpenAPI docs: http://127.0.0.1:8000/api/v1/docs (JSON at `/api/v1/openapi.json`)
- Events: `POST /api/v1/events`, one event per call (fitcore.schemas, v1).
  It answers 201 with the re-scored order's scores, or 200 with the first
  result if that `event_id` was already received.
- Tests: `pytest api`

Every error is `{"code": ..., "message": ...}`: `code` is the HTTP status
name in snake case, such as `not_found`. Validation errors (400, 422) add
`errors`; an event that does not fit what is stored, such as a delivery
for an order never placed, is a 409 `conflict`.
