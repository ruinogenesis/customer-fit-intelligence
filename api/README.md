# api

Backend service that ingests events and serves scores. Deployed to Cloud Run.

One Flask + flask-smorest app under `/api/v1`, in `src/cfi_api`. Install it
into the repo's `.venv` with `python -m pip install -e "api[dev]"`.

- Run it locally: `flask --app cfi_api run --port 8000`
- Health: http://127.0.0.1:8000/api/v1/health
- OpenAPI docs: http://127.0.0.1:8000/api/v1/docs (JSON at `/api/v1/openapi.json`)
- Tests: `pytest api`

Every error is `{"code": ..., "message": ...}`: `code` is the HTTP status
name in snake case, such as `not_found`. Validation errors add `errors`.
