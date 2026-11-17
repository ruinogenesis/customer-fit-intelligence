-- T306: BigQuery's side of the export (cfi_api.export). Run it once per
-- project, and again after changing a view; it keeps every row already in
-- raw.changes. A change to the table itself needs its own ALTER TABLE.
--
--   bq query --project_id=cfi-dev-th26 --location=asia-southeast3 \
--     --use_legacy_sql=false < bigquery/raw.sql

CREATE SCHEMA IF NOT EXISTS raw
OPTIONS (
  location = 'asia-southeast3',
  description = 'Every document the API saves, as Pub/Sub delivers it (T306)'
);

-- One row per document an event saved, written by the BigQuery subscription
-- raw-changes-bigquery. The first five columns are the message
-- (cfi_api.export.message); the last four are Pub/Sub's. A row can arrive
-- twice: the views keep one per (event_id, path).
CREATE TABLE IF NOT EXISTS raw.changes (
  collection STRING NOT NULL OPTIONS (description = 'events, customers, orders or scores'),
  path STRING NOT NULL OPTIONS (description = 'The Firestore document path'),
  event_id STRING NOT NULL OPTIONS (description = 'The event that saved the document'),
  written_at TIMESTAMP NOT NULL OPTIONS (description = 'When the API saved it'),
  document JSON NOT NULL OPTIONS (description = 'The document as saved; timestamps as ISO 8601 text in UTC'),
  subscription_name STRING OPTIONS (description = 'Pub/Sub: the subscription that wrote the row'),
  message_id STRING OPTIONS (description = 'Pub/Sub: the message id'),
  publish_time TIMESTAMP OPTIONS (description = 'Pub/Sub: when the API published it'),
  attributes STRING OPTIONS (description = 'Pub/Sub: the message attributes, as JSON')
)
PARTITION BY DATE(written_at)
CLUSTER BY collection, path
OPTIONS (description = 'Every version of every document the API saved (T306)');

-- Each collection as it is now: the latest version of each document. Of two
-- events that changed one document, the one that committed later has the
-- later written_at (cfi_api.firestore_store).

CREATE OR REPLACE VIEW raw.events AS
SELECT path, event_id, written_at, document
FROM raw.changes
WHERE collection = 'events'
QUALIFY ROW_NUMBER() OVER (PARTITION BY path ORDER BY written_at DESC) = 1;

CREATE OR REPLACE VIEW raw.customers AS
SELECT path, event_id, written_at, document
FROM raw.changes
WHERE collection = 'customers'
QUALIFY ROW_NUMBER() OVER (PARTITION BY path ORDER BY written_at DESC) = 1;

CREATE OR REPLACE VIEW raw.orders AS
SELECT path, event_id, written_at, document
FROM raw.changes
WHERE collection = 'orders'
QUALIFY ROW_NUMBER() OVER (PARTITION BY path ORDER BY written_at DESC) = 1;

CREATE OR REPLACE VIEW raw.scores AS
SELECT path, event_id, written_at, document
FROM raw.changes
WHERE collection = 'scores'
QUALIFY ROW_NUMBER() OVER (PARTITION BY path ORDER BY written_at DESC) = 1;
