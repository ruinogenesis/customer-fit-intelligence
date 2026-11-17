# Customer Fit Intelligence

The product measures how the services fit each customer.
It points out why each customer's fit changes.
It does all the act automatically with measured actions.

## Layout

- `api/`: backend service that ingests events
- `app/`: the product for customer which produces every data the model needs
- `dashboard/`: dashboard for the business team
- `fitcore/`: shared python package
- `functions/`: cloud functions
- `notebooks/`: colab research

## Local development

Start the Firestore, Auth, Pub/Sub, Functions and Hosting emulators:

    firebase emulators:start --project demo-cfi

Emulator UI: http://127.0.0.1:4000

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow and conventions.
