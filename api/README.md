# api

Backend service that ingests events and serves scores. Deployed to Cloud Run.

One Flask + flask-smorest app under `/api/v1`, in `src/cfi_api`. It needs
fitcore, which is not on PyPI, so install both in one command from the repo
root: `python -m pip install -e "fitcore[dev]" -e "api[dev]"`.

Settings come from `CFI_*` environment variables (the prefix is dropped):

- `CFI_MODEL_PATH` and `CFI_MODEL_VERSION`: the model.json to score with
  and its version; set both or neither. Without a model, `/events` answers 503.
- `CFI_REGISTRY`: `gcs` scores with the model registry's live version
  instead (T310, below); not with `CFI_MODEL_PATH`.
- `CFI_STORE`: `memory` (the default; lost when the server stops) or `firestore`.
- `CFI_PROJECT`: the Google Cloud project, for Firestore and sign-in
  tokens, such as `demo-cfi` for the emulators; unset, the environment's.
- `CFI_AUTH`: `firebase` checks who may call what (below); unset, as
  locally, nothing is checked.
- `CFI_WRITE_KEY`: the integration key that opens `POST /events` in the
  header `X-CFI-Key`; it needs `CFI_AUTH`.
- `CFI_TOPIC`: the Pub/Sub topic that takes every saved document on to
  BigQuery (below); it needs `CFI_PROJECT`. Unset, nothing is exported.
- `CFI_HISTORY`: `bigquery` reads BigQuery in `CFI_PROJECT`: the daily
  snapshots for movement on `/matrix`, and raw for `/drivers`. Unset, both
  answer 503.

Run it locally with the reference model, from the repo root:

    CFI_MODEL_PATH=notebooks/results/t112_model.json CFI_MODEL_VERSION=t112-sim \
      flask --app cfi_api run --port 8000

- Health: http://127.0.0.1:8000/api/v1/health
- OpenAPI docs: http://127.0.0.1:8000/api/v1/docs (JSON at `/api/v1/openapi.json`)
- Events: `POST /api/v1/events`, one event per call (fitcore.schemas, v1).
  It answers 201 with the re-scored order's scores (the five latents,
  `churn_risk` and the love-matrix `quadrant`) and `drivers`, the inputs
  that moved its fit most (T316), or 200 with the first result if that
  `event_id` was already received. Each scoring's history entry also keeps
  every input's share of every score (`contributions`, from
  `fitcore.model.attributions`). Each event is read,
  scored and saved in one Firestore transaction (T302), so events for one
  customer at the same moment are applied one after the other. If too many
  arrive at once it answers 503 with `Retry-After`: send it again then.
  Firestore keeps each `events/{event_id}` for 30 days after the API
  received it (T305): `expires_at` is the TTL field set in
  `firestore.indexes.json`, which CI deploys, and Firestore deletes the
  document usually within a day after that. A repeat is recognised only
  while it is there; sent again later, an event counts as new (an
  `order_placed` is still refused). BigQuery keeps every event.
- What-if: `POST /api/v1/score` takes the same event and answers (200) just
  as `/events` would right now, but saves nothing.
- Reads for the dashboard (T311): `GET /api/v1/customers` (riskiest first;
  filter with `quadrant` and `min_churn_risk`), `/customers/{id}`,
  `/customers/{id}/scores` and `/customers/{id}/orders` (newest first), and
  `/matrix` (customers per quadrant). Lists come a page at a time: `limit`
  (default 50, at most 200), then the answer's `next_cursor` as `cursor`
  for the next page. An unknown query parameter is a 422.
  Score history is indexed on `at` alone (T317): `firestore.indexes.json`
  leaves every other field of a score entry out, as nothing queries it in
  Firestore; BigQuery's `raw.scores` answers questions across scorings.
- Overview (T402): `GET /api/v1/overview` is now: customers per quadrant,
  the scored customers' average fit and churn risk, and how many are at or
  above `at_risk` (default 0.1). Each number is a Firestore aggregation,
  billed by index entries read (a read per 1,000), so the dashboard asks
  every 30 seconds. `GET /api/v1/overview/trend` gives the same for each
  Bangkok day, `days` long (default 28, at most 90) and ending on `to`
  (default yesterday), from `features.daily` (needs `CFI_HISTORY`).
- Movement (T313): `GET /api/v1/matrix?from=2026-10-01&to=2026-11-17` adds
  `movement`: the customers scored by the end of `to`, by their quadrant at
  the end of `from` (or `unscored`), then at the end of `to`. Days are
  Bangkok days, read from BigQuery's `features.daily`, so `to` must be
  before today; a day with no snapshot yet (yesterday before the 02:00 run)
  is a 422.
- Drivers (T315): `GET /api/v1/drivers?by=courier` (or `by=restaurant`)
  averages each input's contribution to fit (T316's split) over each
  courier's orders placed from `from` to `to`, Bangkok days (by default the
  last 28, today included), each order by its latest scoring with a split.
  `excess` compares each group with all those orders, and `drivers` names
  the inputs worse there than overall, worst first. Groups with fewer than
  `min_orders` (5) orders are left out; the worst `limit` (50) come first.
- Models (T310): `GET /api/v1/models` lists every version in the registry
  with its status (candidate, live or retired) and metrics; `POST
  /api/v1/models/{version}/promote` (admins) makes one live, after checking
  that its file loads, and retires the one before. Promoting an older version
  rolls back. `GET /api/v1/models/{version}/card` (every staff role) gives
  its `model_card.md` as markdown (T407). Each event reads the live version inside its own transaction,
  so the next event on every instance uses it, with no redeploy. Admins add
  versions from the command line, which runs as they do, with Application
  Default Credentials, as `cfi_api.roles` does (below):

      python -m cfi_api.registry --project cfi-dev-th26 register t112-sim \
          notebooks/results/t112_model.json \
          --card notebooks/results/t112_model_card.md --bucket cfi-dev-th26-models \
          --trained 2026-11-14 --metric churn_auc=0.789 --metric reorder_auc=0.711
      python -m cfi_api.registry --project cfi-dev-th26 promote t112-sim
      python -m cfi_api.registry --project cfi-dev-th26 list

  The files go to `gs://BUCKET/models/VERSION/` (`model.json` and
  `model_card.md`), and `models/VERSION` in Firestore keeps the model.json's
  SHA-256, which every instance checks before it uses the file.
- Rules (T406): `GET /api/v1/rules` (every staff role) lists the
  automation rules by name, each with its kind (service or marketing),
  status (off, dry_run, test or live) and holdout share; the dashboard's
  Automations screen shows them. Apps cannot read `rules/` in Firestore.
  Until the rules engine (T501) nothing changes a rule: admins add the
  design doc's six starter rules, each off, from the command line, which
  runs as `cfi_api.roles` does. Seeding again adds only the ones missing
  and never changes one that is there:

      python -m cfi_api.rules --project cfi-dev-th26 seed
      python -m cfi_api.rules --project cfi-dev-th26 list
      python -m cfi_api.rules --project cfi-dev-th26 set fee-sensitivity dry_run
      python -m cfi_api.rules --project cfi-dev-th26 actions --limit 10
      python -m cfi_api.rules --project cfi-dev-th26 guardrails
      python -m cfi_api.rules --project cfi-dev-th26 pause
      python -m cfi_api.rules --project cfi-dev-th26 resume

- Rules engine (T501): every saved document goes to the topic `raw-changes`
  with its collection as the attribute `collection` (T306). The push
  subscription `raw-changes-rules` takes the score entries and sends each to
  the rules service, `cfi-rules`: the same image, running
  `cfi_api.rules_service`, private, so only the subscription's account
  `cfi-rules-push` may call it. Each rule that is not off and has its
  logic in `cfi_api.engine` is asked whether it fires:

  - Fee sensitivity: an order in the Product quadrant whose fees take at
    least 0.05 off fit (`min_fee_drag`): a fee-free delivery, 40 baht.
  - Brand → Lovemark (T506): the customer's latest 3 orders (`orders`), as
    placed, all in the Brand quadrant: a personal touch in the app. It is
    about the streak's first order, so it decides once per streak.
  - Fad protection (T506): the customer's newest order drops into the Fad
    quadrant from one with high Respect (Lovemark or Brand): reliable
    restaurants, in the app, with no discount.

  A rule that fires is logged to `actions/{rule}--{order}` with the customer's
  arm: a stable hash of rule and customer puts the rule's `holdout_share` in
  the holdout (T503). A rule decides at most once per order, so a retry or a
  later scoring of the same order logs nothing more. A condition overrides the
  logic's numbers (`{"min_fee_drag": 0.02}`), and an action map what a
  decision is worth (`{"value": 2500}`, in satang; Fee sensitivity's voucher
  is 40 baht). Until the channels (T504), every decision is a dry run,
  whatever the rule's status: nothing is sent. Pub/Sub retries a push the
  service answers with 500, for up to 7 days.
- Guardrails (T502): before a decision is logged it passes, in order, the
  global kill switch (`pause`, `resume`), the rule's own (status `off`),
  marketing consent (T610), the frequency cap (by default 2 marketing
  decisions per customer in 7 days, both arms counted; service rules are
  exempt) and the daily budget (by default 5,000 baht a Bangkok day: only the
  treatment arm spends, counted in `budget/{day}` in the decision's own
  transaction, and once a decision would pass it neither arm decides that
  day). One that fails logs a "held back" line and nothing else. A marketing
  decision made in quiet hours (by default 21:00 to 08:00 Bangkok time) gets
  `send_at` at their end; service decisions go at once. The settings live
  in `settings/automations`; `guardrails --cap N --cap-days D --quiet
  HH:MM-HH:MM --budget BAHT` changes them, and with no settings saved the
  defaults hold. The cap reads a customer's actions by time, through the
  composite index on `actions` (customer_id, decided_at) that CI deploys.
- Daily sweep (T505): some rules fire on what has not happened, which no
  scoring brings. Each day at 10:00 Bangkok time, the Cloud Scheduler job
  `cfi-sweep` calls `POST /sweep` on `cfi-rules` as the service account
  `cfi-sweep`; each rule that sweeps finds its customers, and each decision
  passes the same guardrails. So far Win-back: a customer with no order for
  14 days whose latest order's churn risk is at least 10%
  (`{"quiet_days": 21, "min_churn": 0.2}` overrides either) gets a 50-baht
  voucher, riskiest first, so the daily budget goes to them first. The
  decision is about their latest order (`actions/win-back--{order}`), so it
  is logged once per quiet spell however many sweeps see them, and again
  only after a new order and another 14 quiet days. The sweep answers 200
  with `{"decided": n}`, or 500, and Scheduler tries again: sweeping again
  logs nothing twice. To sweep now: `gcloud scheduler jobs run cfi-sweep
  --location asia-southeast3 --project cfi-dev-th26`.
- Fake orders: `python -m cfi_api.fake_orders` sends simulated customers'
  orders (fitcore.sim_events) to a running API, to watch the matrix page
  move (dashboard/README.md). With `--plant`, most late and cold orders go to
  a few restaurants and most rude ones to a few couriers, for `/drivers` to
  find. With `--consent`, every customer answers the marketing consent
  question at sign-up (60% say yes) and some withdraw it later (T610).
- Marketing consent (T610): a `consent_changed` event (`marketing` true or
  false; `source` `sign_up`, `settings` or `support`) sets the customer's
  `consent` (`customers/{id}`, `GET /customers/{id}`). The latest answer
  wins, so an older one arriving late changes nothing. A customer never
  asked has `null`, which counts as no consent. The rules engine logs no
  decision of a marketing rule for a customer without consent; service rules
  need none.
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
| `GET /customers`, `/customers/{id}` and its `/scores` and `/orders`, `/matrix`, `/overview` and `/overview/trend`, `/drivers`, `/models` and `/models/{version}/card` | every staff role |
| `POST /score` | analysts and admins |
| `POST /events` | admins, and the ordering app with the integration key |
| `POST /models/{version}/promote` | admins |
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
`Dockerfile` at the repo root, and `.dockerignore` lets in only fitcore and
the API: the model comes from the registry (`CFI_REGISTRY=gcs`, T310). The
service runs as the service account `cfi-api`, which may only read and write
Firestore, read the secret `cfi-write-key`, publish to the topic
`raw-changes`, read BigQuery's `features` and `raw`, and read the models
bucket.

The service is public, because Hosting can only reach a service that takes
calls without Google credentials. So the API checks its callers itself
(above): the deploy sets `CFI_AUTH=firebase`, and the service gets the
integration key from Secret Manager as `CFI_WRITE_KEY`.

Every merge to `main` deploys dev (T208). Once every check has passed, the
`deploy` job in `.github/workflows/ci.yml` builds the image, pushes it to the
Artifact Registry repo `cfi` tagged with the commit, deploys it to Cloud Run
with every setting of the service, deploys the page to Firebase Hosting and
`firestore.rules` to Firestore (T304), deploys the same image as the rules
service `cfi-rules` (T501), and checks that dev serves that commit, refuses
`POST /score` without a token, refuses a Firestore read without one, and
that `cfi-rules` runs the commit and refuses a call without the push
account's token. To deploy again without a new commit, re-run the job on GitHub;
nobody deploys from a laptop (CONTRIBUTING.md).

The job signs in with Workload Identity Federation, so no key is stored
anywhere: GitHub gives each run a short-lived token naming the repo and the
branch, and Google trades it for the service account `cfi-deployer`. Only
runs on `main` of this repo, matched by its numeric ID, can do that; pull
requests cannot. `cfi-deployer` may deploy Cloud Run revisions that run as
`cfi-api` or `cfi-rules`, Hosting and the Firestore rules, but not change who may call the
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
12. Daily snapshots (T308): run `bigquery/features.sql`, then schedule
    `bigquery/features_daily.sql` as the service account `cfi-features`
    (bigquery/README.md has the steps).
13. Movement (T313): grant `cfi-api` `roles/bigquery.jobUser` on the
    project and `roles/bigquery.dataViewer` on the dataset `features`
    (`GRANT ... ON SCHEMA features`).
14. Drivers (T315): grant `cfi-api` `roles/bigquery.dataViewer` on the
    dataset `raw` too (`GRANT ... ON SCHEMA raw`). It already reads the same
    documents in Firestore.
15. Model registry (T310), before the first deploy with `CFI_REGISTRY=gcs`:
    create the bucket `PROJECT-models` in `asia-southeast3` with uniform
    access and public access prevention, grant `cfi-api`
    `roles/storage.objectViewer` on it, then register and promote the
    first version (above). Until a version is live, `/events` answers 503.
16. Automation rules (T406): seed the starter rules (above).
17. Rules engine (T501). Before the first deploy of `cfi-rules`: create the
    service account `cfi-rules`, grant it `roles/datastore.user`, and let
    `cfi-deployer` act as it (`roles/iam.serviceAccountUser` on it). After
    CI has deployed it: create the service account `cfi-rules-push`, grant
    it `roles/run.invoker` on `cfi-rules`, then create the push
    subscription on `raw-changes`, filtered to score entries:

        gcloud pubsub subscriptions create raw-changes-rules \
            --topic raw-changes --project cfi-dev-th26 \
            --push-endpoint "$(gcloud run services describe cfi-rules \
                --project cfi-dev-th26 --region asia-southeast3 \
                --format 'value(status.url)')/push" \
            --push-auth-service-account \
                cfi-rules-push@cfi-dev-th26.iam.gserviceaccount.com \
            --message-filter 'attributes.collection = "scores"' \
            --ack-deadline 60 --min-retry-delay 10s --max-retry-delay 600s
18. Daily sweep (T505), after CI has deployed `cfi-rules` with `/sweep`:
    enable the Cloud Scheduler API, create the service account `cfi-sweep`,
    grant it `roles/run.invoker` on `cfi-rules`, then create the job. It
    signs in to the service's own URL, and tries a failed sweep 3 more times:

        URL=$(gcloud run services describe cfi-rules --project cfi-dev-th26 \
            --region asia-southeast3 --format 'value(status.url)')
        gcloud scheduler jobs create http cfi-sweep \
            --location asia-southeast3 --project cfi-dev-th26 \
            --schedule "0 10 * * *" --time-zone Asia/Bangkok \
            --uri "$URL/sweep" --http-method POST \
            --oidc-service-account-email \
                cfi-sweep@cfi-dev-th26.iam.gserviceaccount.com \
            --oidc-token-audience "$URL" \
            --attempt-deadline 5m --max-retry-attempts 3 --min-backoff 5m
