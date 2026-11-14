# dashboard

Here will be love matrix, customer profiles, drivers, automations and models. This is the ops dashboard for the business team.

## Love matrix (T206, bare)

`public/` holds a plain page (no build step) that plots every customer as a
dot, Respect across and Love up, coloured by quadrant, live from Firestore.
For now it reads the local emulator only, as a stand-in "viewer"; staff
sign-in comes with the React dashboard (T401).

Watch it move, from the repo root:

1. `firebase emulators:start --only firestore,hosting --project demo-cfi`
2. In a second terminal, the API writing to the emulator:

       FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 CFI_STORE=firestore \
         CFI_FIRESTORE_PROJECT=demo-cfi \
         CFI_MODEL_PATH=notebooks/results/t112_model.json CFI_MODEL_VERSION=t112-sim \
         flask --app cfi_api run --port 8000

3. Open http://127.0.0.1:5002 and post events to `/api/v1/events`.

Tests: `node --test dashboard/tests/*.test.js` (Node 20 or newer, no packages; CI uses 22).
