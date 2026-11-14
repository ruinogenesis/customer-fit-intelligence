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
  It answers 201 with the re-scored order's scores (the five latents,
  `churn_risk` and the love-matrix `quadrant`), or 200 with the first
  result if that `event_id` was already received.
- What-if: `POST /api/v1/score` takes the same event and answers (200) just
  as `/events` would right now, but saves nothing.
- Fake orders: `python -m cfi_api.fake_orders` sends simulated customers'
  orders (fitcore.sim_events) to a running API, to watch the matrix page
  move (dashboard/README.md).
- Tests: `pytest api`

Every error is `{"code": ..., "message": ...}`: `code` is the HTTP status
name in snake case, such as `not_found`. Validation errors (400, 422) add
`errors`; an event that does not fit what is stored, such as a delivery
for an order never placed, is a 409 `conflict`.

## Deploy to Cloud Run (dev)

The API runs on Cloud Run as the service `cfi-api` in `cfi-dev-th26`, region
`asia-southeast3`, next to Firestore. The image comes from the `Dockerfile` at
the repo root: `.gcloudignore` (what is uploaded) and `.dockerignore` (what
the image may copy) let in only fitcore, the API and the model,
`notebooks/results/t112_model.json`. The service runs as the service account
`cfi-api`, which may only read and write Firestore. It is private: every call
needs a Google identity token.

Until T208 deploys from CI, deploy by hand from the repo root:

    gcloud run deploy cfi-api --source . --project cfi-dev-th26 \
      --region asia-southeast3 \
      --service-account cfi-api@cfi-dev-th26.iam.gserviceaccount.com \
      --no-allow-unauthenticated --min-instances 0 --max-instances 2 \
      --cpu 1 --memory 512Mi --set-env-vars CFI_STORE=firestore

Call it through a local proxy that adds your identity token: start the
proxy in one terminal, then call it from another.

    gcloud run services proxy cfi-api --project cfi-dev-th26 \
      --region asia-southeast3 --port 8081
    curl http://127.0.0.1:8081/api/v1/health
    python -m cfi_api.fake_orders --url http://127.0.0.1:8081

A full fake-order run is about 3,100 Firestore writes, of the 20,000 free
per day.

One-time setup of a project (dev: T204):

1. Link a billing account; this moves the Firebase project to Blaze.
2. Enable the APIs: `run.googleapis.com`, `artifactregistry.googleapis.com`
   and `cloudbuild.googleapis.com`.
3. Create the service account `cfi-api` and grant it `roles/datastore.user`.
4. Grant `roles/run.builder` to the Compute Engine default service account,
   which Cloud Build uses for source deploys.
