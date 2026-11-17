# api

Backend service that ingests events and serves scores. Deployed to Cloud Run.

One Flask + flask-smorest app under `/api/v1`, in `src/cfi_api`. It needs
fitcore, which is not on PyPI, so install both in one command from the repo
root: `python -m pip install -e "fitcore[dev]" -e "api[dev]"`.

Settings come from `CFI_*` environment variables (the prefix is dropped):

- `CFI_MODEL_PATH` and `CFI_MODEL_VERSION`: the model.json to score with
  and its version; set both or neither. Without a model, `/events` answers 503.
- `CFI_STORE`: `memory` (the default; lost when the server stops) or `firestore`.
- `CFI_PROJECT`: the Google Cloud project, for Firestore and sign-in
  tokens, such as `demo-cfi` for the emulators; unset, the environment's.
- `CFI_AUTH`: `firebase` checks who may call what (below); unset, as
  locally, nothing is checked.
- `CFI_WRITE_KEY`: the integration key that opens `POST /events` in the
  header `X-CFI-Key`; it needs `CFI_AUTH`.
- `CFI_TOPIC`: the Pub/Sub topic that takes every saved document on to
  BigQuery (below); it needs `CFI_PROJECT`. Unset, nothing is exported.

Run it locally with the reference model, from the repo root:

    CFI_MODEL_PATH=notebooks/results/t112_model.json CFI_MODEL_VERSION=t112-sim \
      flask --app cfi_api run --port 8000

- Health: http://127.0.0.1:8000/api/v1/health
- OpenAPI docs: http://127.0.0.1:8000/api/v1/docs (JSON at `/api/v1/openapi.json`)
- Events: `POST /api/v1/events`, one event per call (fitcore.schemas, v1).
  It answers 201 with the re-scored order's scores (the five latents,
  `churn_risk` and the love-matrix `quadrant`), or 200 with the first
  result if that `event_id` was already received. Each event is read,
  scored and saved in one Firestore transaction (T302), so events for one
  customer at the same moment are applied one after the other. If too many
  arrive at once it answers 503 with `Retry-After`: send it again then.
- What-if: `POST /api/v1/score` takes the same event and answers (200) just
  as `/events` would right now, but saves nothing.
- Reads for the dashboard (T311): `GET /api/v1/customers` (riskiest first;
  filter with `quadrant` and `min_churn_risk`), `/customers/{id}`,
  `/customers/{id}/scores` and `/customers/{id}/orders` (newest first), and
  `/matrix` (customers per quadrant). Lists come a page at a time: `limit`
  (default 50, at most 200), then the answer's `next_cursor` as `cursor`
  for the next page. An unknown query parameter is a 422.
- Fake orders: `python -m cfi_api.fake_orders` sends simulated customers'
  orders (fitcore.sim_events) to a running API, to watch the matrix page
  move (dashboard/README.md).
- Tests: `pytest api`

What the API keeps in Firestore, document by document, is in
`fitcore/DOCUMENTS.md` (T301). Before it writes a document, it checks it
against its schema in `fitcore.documents`; one that does not fit fails the
request with a 500, and nothing of that event is saved.

Every error is `{"code": ..., "message": ...}`: `code` is the HTTP status
name in snake case, such as `not_found`. Validation errors (400, 422) add
`errors`; an event that does not fit what is stored, such as a delivery
for an order never placed, is a 409 `conflict`.

## Export to BigQuery (T306)

Every document `POST /events` saves also goes to BigQuery, as a row of
`raw.changes`: its collection, path, event, the time the API saved it, and
the document itself as JSON. `bigquery/raw.sql` makes the table, and a view
per collection holding the latest version of each document: `raw.events`,
`raw.customers`, `raw.orders` and `raw.scores`.

The rows travel as Pub/Sub messages, one per document, to the topic
`raw-changes`; its BigQuery subscription `raw-changes-bigquery` writes them
to the table. Each event's messages are first saved in an `outbox` document
in the event's own transaction (`cfi_api.export`), so a message exists
exactly when its document does. After the commit the API publishes them and
deletes the entry. If Pub/Sub fails, the event is still answered and the
entry stays; a later `POST /events` sends entries over a minute old, ten at
a time. So a row can arrive twice, and the views keep one per
`(event_id, path)`. The messages and the topic stay in `asia-southeast3`.

## Who may call what (T303)

With `CFI_AUTH=firebase`, as on Cloud Run, the API checks each request
against the table `ACCESS` in `src/cfi_api/access.py`:

| Request | Who may |
| --- | --- |
| `GET /health`, `/docs`, `/openapi.json` | anyone |
| `GET /customers`, `/customers/{id}` and its `/scores` and `/orders`, `/matrix` | every staff role |
| `POST /score` | analysts and admins |
| `POST /events` | admins, and the ordering app with the integration key |
| anything the table does not list | admins |

Staff send a Firebase ID token as `Authorization: Bearer <token>`. Its
`role` claim ranks viewer < analyst < admin, and a role may do what the
roles below it may. The ordering app sends the key in `X-CFI-Key` instead.
No valid token (or key, where one is enough) gets 401 `unauthorized`; a
role that ranks too low gets 403 `forbidden`. Only the token is checked, so
a role taken away works until that token expires, within an hour.

Give roles with `python -m cfi_api.roles`. It runs as you, with
Application Default Credentials: once, `gcloud auth application-default
login`, then `gcloud auth application-default set-quota-project
cfi-dev-th26`.

    python -m cfi_api.roles --project cfi-dev-th26 grant ann@example.com analyst --create
    python -m cfi_api.roles --project cfi-dev-th26 revoke ann@example.com
    python -m cfi_api.roles --project cfi-dev-th26 list

`grant` sets the claim and writes `users/{uid}` (`fitcore/DOCUMENTS.md`);
with `--create` it first makes the account, with a password you type.
`revoke` takes both away and signs the account out everywhere. A new role
reaches a person's token at their next sign-in, or within an hour.

## Deploy to Cloud Run (dev)

The API runs on Cloud Run as the service `cfi-api` in `cfi-dev-th26`, region
`asia-southeast3`, next to Firestore. Firebase Hosting sends every `/api/**`
request to it (`firebase.json`, T205), so it answers on the Hosting domain
too: https://cfi-dev-th26.web.app/api/v1/health. The image comes from the
`Dockerfile` at the repo root, and `.dockerignore` lets in only fitcore, the
API and the model, `notebooks/results/t112_model.json`. The service runs as
the service account `cfi-api`, which may only read and write Firestore,
read the secret `cfi-write-key` and publish to the topic `raw-changes`.

The service is public, because Hosting can only reach a service that takes
calls without Google credentials. So the API checks its callers itself
(above): the deploy sets `CFI_AUTH=firebase`, and the service gets the
integration key from Secret Manager as `CFI_WRITE_KEY`.

Every merge to `main` deploys dev (T208). Once every check has passed, the
`deploy` job in `.github/workflows/ci.yml` builds the image, pushes it to the
Artifact Registry repo `cfi` tagged with the commit, deploys it to Cloud Run
with every setting of the service, deploys the page to Firebase Hosting and
`firestore.rules` to Firestore (T304), and checks that dev serves that
commit, refuses `POST /score` without a token and refuses a Firestore read
without one. To deploy again without a new commit, re-run the job on GitHub;
nobody deploys from a laptop (CONTRIBUTING.md).

The job signs in with Workload Identity Federation, so no key is stored
anywhere: GitHub gives each run a short-lived token naming the repo and the
branch, and Google trades it for the service account `cfi-deployer`. Only
runs on `main` of this repo, matched by its numeric ID, can do that; pull
requests cannot. `cfi-deployer` may deploy Cloud Run revisions that run as
`cfi-api`, Hosting and the Firestore rules, but not change who may call the
service, and
may push only to the repo `cfi`, which keeps the 5 newest images
(`api/artifact-cleanup.json`).

Send fake orders to it with the key from Secret Manager:

    export CFI_WRITE_KEY=$(gcloud secrets versions access latest \
      --secret cfi-write-key --project cfi-dev-th26)
    python -m cfi_api.fake_orders --url https://cfi-dev-th26.web.app

A full fake-order run is about 4,000 Firestore writes and 800 deletes, of
the 20,000 of each free per day.

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
   `roles/firebaserules.admin` (T304), `roles/datastore.indexAdmin` (T311),
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
9. In the Firebase console, start Authentication and turn on the
   Email/Password provider. In its settings, untick "Enable create
   (sign-up)": accounts come only from `python -m cfi_api.roles` (T303).
10. Export to BigQuery (T306): enable `pubsub.googleapis.com` and
    `bigquery.googleapis.com`, and run `bigquery/raw.sql` (the command is at
    its top). Create the topic `raw-changes` with
    `--message-storage-policy-allowed-regions=asia-southeast3`, and grant
    `cfi-api` `roles/pubsub.publisher` on it. Grant the Pub/Sub service agent
    `service-PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com`
    `roles/bigquery.dataEditor` on the table `raw.changes`. Then create the
    subscription `raw-changes-bigquery` on the topic with
    `--bigquery-table=PROJECT:raw.changes --use-table-schema --write-metadata`.
11. Labels (T307): run `bigquery/labels.sql` the same way, after `raw.sql`
    (bigquery/README.md).
