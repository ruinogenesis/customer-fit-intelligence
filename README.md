# Customer Fit Intelligence

**Live dashboard: https://cfi-dev-th26.web.app** (sign-in only; all data is simulated)

The product measures how the services fit each customer.
It points out why each customer's fit changes.
Automated actions, tested against a holdout before they go live, are planned but not built yet.

## Live dashboard

- Sign-in only. An admin creates each account, and there is no sign-up.
  To ask for a read-only viewer account, open an issue on this repository.
- Every customer there is simulated. No real customer data is stored.
- The API's health check is open to all: https://cfi-dev-th26.web.app/api/v1/health

## Layout

- `api/`: backend service that ingests events
- `app/`: the product for customer which produces every data the model needs
- `dashboard/`: dashboard for the business team
- `fitcore/`: shared python package
- `functions/`: cloud functions
- `notebooks/`: colab research
- `paper/`: the paper, its figures and where each number comes from

## Local development

Start the Firestore, Auth, Pub/Sub, Functions and Hosting emulators:

    firebase emulators:start --project demo-cfi

Emulator UI: http://127.0.0.1:4000

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow and conventions.
