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

The Firebase config, Web API key included, is never in the repo or the
build: on Hosting the app fetches the project's own from
`/__/firebase/init.json`, which needs a registered web app (one-time setup
below). `npm run dev` uses the emulators' demo project instead.

## Work on it

Node 24, from `dashboard/`:

    npm ci            # the packages exactly as package-lock.json pins them
    npm test          # Vitest (tests/*.test.ts[x]) and the legacy matrix tests
    npm run lint      # oxlint
    npm run build     # type-check (tsc -b), then build dist/

To click through it locally against the Auth emulator:

1. From the repo root: `firebase emulators:start --only auth,firestore --project demo-cfi`
2. Make a staff account in the emulators (the role is a claim, plus
   `users/{uid}` in Firestore), from the repo root:

       FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099 FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 \
         python -m cfi_api.roles --project demo-cfi grant you@example.com admin --create

3. `npm run dev` here, then open http://localhost:5173.

## One-time setup per project

Register the dashboard as a web app, so Hosting can serve its config:

    firebase apps:create WEB "CFI dashboard" --project cfi-dev-th26

## legacy/ (T206)

The first, no-build love matrix: live dots from the Firestore emulator. It
is no longer served; T403 brings the matrix into the app, starting from
`legacy/matrix.js` and its tests (`tests/matrix.test.js`, Node's own runner).
