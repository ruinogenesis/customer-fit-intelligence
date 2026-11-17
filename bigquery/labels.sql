-- T307: the outcome labels training needs, read from raw (bigquery/raw.sql).
-- Run it after raw.sql, once per project and again after a change:
--
--   bq query --project_id=cfi-dev-th26 --location=asia-southeast3 \
--     --use_legacy_sql=false < bigquery/labels.sql
--
-- api/tests/test_labels.py runs this file on DuckDB (through sqlglot) over
-- the golden log, so keep to SQL that sqlglot translates.

CREATE SCHEMA IF NOT EXISTS labels
OPTIONS (
  location = 'asia-southeast3',
  description = 'Outcome labels for training, read from raw (T307)'
);

-- One row per order placed by horizon, with its labels as known at horizon,
-- as fitcore.sim_orders defines them:
--   reorder_14d: the customer ordered again within 14 days of this order.
--   churn_30d: 30 days passed with no new order.
-- A label is NULL while its window is still open at horizon. Real data uses
-- the view below (horizon = now); simulated history passes the end of its
-- simulated period, because its customers stop when the simulation does.
CREATE OR REPLACE TABLE FUNCTION labels.order_labels(horizon TIMESTAMP) AS
SELECT
  order_id,
  customer_id,
  placed_at,
  next_placed_at,
  CASE
    WHEN next_placed_at <= TIMESTAMP_ADD(placed_at, INTERVAL 14 DAY) THEN TRUE
    WHEN next_placed_at IS NOT NULL THEN FALSE
    WHEN TIMESTAMP_ADD(placed_at, INTERVAL 14 DAY) <= horizon THEN FALSE
  END AS reorder_14d,
  CASE
    WHEN next_placed_at <= TIMESTAMP_ADD(placed_at, INTERVAL 30 DAY) THEN FALSE
    WHEN next_placed_at IS NOT NULL THEN TRUE
    WHEN TIMESTAMP_ADD(placed_at, INTERVAL 30 DAY) <= horizon THEN TRUE
  END AS churn_30d
FROM (
  -- Each order with the customer's next one placed by horizon, if any.
  SELECT
    *,
    LEAD(placed_at) OVER (
      PARTITION BY customer_id ORDER BY placed_at, order_id
    ) AS next_placed_at
  FROM (
    SELECT
      JSON_VALUE(document, '$.order_id') AS order_id,
      JSON_VALUE(document, '$.customer_id') AS customer_id,
      TIMESTAMP(JSON_VALUE(document, '$.placed_at')) AS placed_at
    FROM raw.orders
  )
  WHERE placed_at <= horizon
);

-- Every order's labels as known now.
CREATE OR REPLACE VIEW labels.orders AS
SELECT * FROM labels.order_labels(CURRENT_TIMESTAMP());
