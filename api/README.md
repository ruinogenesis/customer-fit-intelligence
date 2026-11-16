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
- `CFI_WRITE_KEY`: if set, every request but a read must carry it in the
  header `X-CFI-Key` (T205, until T303); leave it unset locally.

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
`asia-southeast3`, next to Firestore. Firebase Hosting sends every `/api/**`
request to it (`firebase.json`, T205), so it answers on the Hosting domain
too: https://cfi-dev-th26.web.app/api/v1/health. The image comes from the
`Dockerfile` at the repo root, and `.dockerignore` lets in only fitcore, the
API and the model, `notebooks/results/t112_model.json`. The service runs as
the service account `cfi-api`, which may only read and write Firestore and
read the secret `cfi-write-key`.

The service is public, because Hosting can only reach a service that takes
calls without Google credentials. Until T303 checks staff roles, every
request but a read needs the write key in the header `X-CFI-Key`; the
service gets it from Secret Manager as `CFI_WRITE_KEY`. Reads, `/health` and
the docs included, are open.

Every merge to `main` deploys dev (T208). Once every check has passed, the
`deploy` job in `.github/workflows/ci.yml` builds the image, pushes it to the
Artifact Registry repo `cfi` tagged with the commit, deploys it to Cloud Run
with every setting of the service, deploys the page to Firebase Hosting, and
checks that dev serves that commit. To deploy again without a new commit,
re-run the job on GitHub; nobody deploys from a laptop (CONTRIBUTING.md).

The job signs in with Workload Identity Federation, so no key is stored
anywhere: GitHub gives each run a short-lived token naming the repo and the
branch, and Google trades it for the service account `cfi-deployer`. Only
runs on `main` of this repo, matched by its numeric ID, can do that; pull
requests cannot. `cfi-deployer` may deploy Cloud Run revisions that run as
`cfi-api` and deploy Hosting, but not change who may call the service, and
may push only to the repo `cfi`, which keeps the 5 newest images
(`api/artifact-cleanup.json`).

Send fake orders to it with the key from Secret Manager:

    export CFI_WRITE_KEY=$(gcloud secrets versions access latest \
      --secret cfi-write-key --project cfi-dev-th26)
    python -m cfi_api.fake_orders --url https://cfi-dev-th26.web.app

A full fake-order run is about 3,100 Firestore writes, of the 20,000 free
per day.

One-time setup of a project (dev: T204, T205 and T208):

1. Link a billing account; this moves the Firebase project to Blaze.
2. Enable the APIs: `run.googleapis.com`, `artifactregistry.googleapis.com`,
   `iam.googleapis.com`, `iamcredentials.googleapis.com`,
   `sts.googleapis.com` and `secretmanager.googleapis.com`.
3. Create the service account `cfi-api` and grant it `roles/datastore.user`.
4. Create the secret `cfi-write-key` holding a random key, and grant
   `cfi-api` `roles/secretmanager.secretAccessor` on it.
5. Create the Docker repo `cfi` in the region, with the cleanup policies in
   `api/artifact-cleanup.json`.
6. Create the service account `cfi-deployer`. Grant it
   `roles/run.developer`, `roles/firebasehosting.admin`,
   `roles/serviceusage.apiKeysViewer` and
   `roles/serviceusage.serviceUsageConsumer` on the project,
   `roles/iam.serviceAccountUser` on `cfi-api`, and
   `roles/artifactregistry.writer` on the repo `cfi`.
7. Create the workload identity pool `github` and in it the OIDC provider
   `cfi-repo` for `https://token.actions.githubusercontent.com`, letting in
   only this repo's ID on `refs/heads/main`. Grant the pool's principals for
   this repo `roles/iam.workloadIdentityUser` on `cfi-deployer`.
8. After the first deploy creates the service, make it public once:
   `gcloud run services update cfi-api --no-invoker-iam-check`.
