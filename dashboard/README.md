# dashboard

The ops dashboard for the business team: love matrix, customer profiles,
drivers, automations and models. React + TypeScript, built by Vite, served
by Firebase Hosting at https://cfi-dev-th26.web.app (CI deploys it on every
merge).

## The shell (T401)

Staff sign in with Firebase Auth (email and password; there is no sign-up).
The app reads the `role` claim on the person's ID token (T303: viewer,
analyst or admin, granted with `python -m cfi_api.roles`) and shows every
screen of the design doc. Each screen says whether this role may take its
main action; an account with no role sees "No dashboard access". The
screens fill in with T402 to T407.

Screens call the API as the person signed in (`src/api.ts`): same origin,
`/api/v1/...`, with their ID token as `Authorization: Bearer`.

## Overview (T402)

Today's numbers from `GET /overview`: customers, average fit, average churn
risk, how many are at risk (churn 10% or more), and customers per quadrant.
They refresh every 30 seconds while the tab is visible, and as soon as it
shows again; a failed refresh keeps the last numbers and says so. Below
them, the last 28 days through yesterday from `GET /overview/trend` (read
once: a day's snapshot is written at 02:00 Bangkok), as two line charts
with a crosshair on hover and arrow keys, and a table of the numbers; then
the 10 riskiest customers from `GET /customers`.

## Models (T407)

Every version in the registry, newest first, with its status, metrics and
dates (Bangkok time), and its model card rendered from markdown (DOMPurify
removes anything unsafe). Admins promote a candidate, or roll back by
promoting a retired version, after a confirm step that says which version
goes live and which is retired. The screen loads as its own chunk, so the
markdown libraries stay out of the first page load.

The Firebase config, Web API key included, is never in the repo or the
build: on Hosting the app fetches the project's own from
`/__/firebase/init.json`, which needs a registered web app (one-time setup
below). `npm run dev` uses the emulators' demo project instead.

## Work on it

Node 24, from `dashboard/`:

    npm ci            # the packages exactly as package-lock.json pins them
    npm test          # Vitest (tests/*.test.ts[x])
    npm run lint      # oxlint
    npm run build     # type-check (tsc -b), then build dist/

Two ways to run it locally:

- `npm run dev:cloud`: the dev project, as on Hosting. The dev server
  proxies `/api` and `/__/firebase/init.json` to https://cfi-dev-th26.web.app,
  so you sign in with a real dev staff account and see dev data; no key is
  copied anywhere. What you do there (a promote, say) happens on dev.
- `npm run dev`: the emulators, as below. `/api` goes to an API you run
  locally on port 8000.

With the emulators:

1. From the repo root: `firebase emulators:start --only auth,firestore --project demo-cfi`
2. Make a staff account in the emulators (the role is a claim, plus
   `users/{uid}` in Firestore), from the repo root:

       FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099 FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 \
         python -m cfi_api.roles --project demo-cfi grant you@example.com admin --create

3. For screens that call the API, run it against the emulators, from the
   repo root (its registry starts empty):

       FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099 FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 \
         CFI_AUTH=firebase CFI_PROJECT=demo-cfi CFI_STORE=firestore CFI_REGISTRY=memory \
         flask --app cfi_api run --port 8000

4. `npm run dev` here, then open http://localhost:5173.

## One-time setup per project

Register the dashboard as a web app, so Hosting can serve its config:

    firebase apps:create WEB "CFI dashboard" --project cfi-dev-th26

## The live customer feed (T403)

The love matrix watches every customer through a Firestore listener
(`src/feed.ts`, `src/firestoreFeed.ts`), as the signed-in staff member:
firestore.rules lets every staff role read customers (T304). Firestore's SDK
loads only when the matrix opens. It costs one read per customer when the
matrix opens, then one per customer that changes. `npm run dev` points it at
the Firestore emulator (port 8080), so run that with `--only auth,firestore`
as above.
