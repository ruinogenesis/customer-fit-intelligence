-- T308: every customer's state on every day, read from raw (bigquery/raw.sql).
-- Run it after raw.sql, once per project and again after a change; the
-- table keeps its rows:
--
--   bq query --project_id=cfi-dev-th26 --location=asia-southeast3 \
--     --use_legacy_sql=false < bigquery/features.sql
--
-- bigquery/features_daily.sql, scheduled, adds each day's rows.
-- api/tests/test_features.py runs this file on DuckDB (through sqlglot) over
-- the golden log, so keep to SQL that sqlglot translates.

CREATE SCHEMA IF NOT EXISTS features
OPTIONS (
  location = 'asia-southeast3',
  description = 'Daily snapshots of every customer, read from raw (T308)'
);

-- One row per customer per day, from the day of their first scoring: their
-- latest scoring by the end of that day, and the gaps behind it. Days are
-- Bangkok calendar days, and a scoring belongs to the day its event happened
-- (its at), so simulated history keeps its own dates.
CREATE TABLE IF NOT EXISTS features.daily (
  day DATE NOT NULL OPTIONS (description = 'A Bangkok calendar day'),
  customer_id STRING NOT NULL OPTIONS (description = 'The customer'),
  scored_at TIMESTAMP NOT NULL OPTIONS (description = 'When the event behind the latest scoring by the end of day happened'),
  event_id STRING NOT NULL OPTIONS (description = 'That event'),
  order_id STRING NOT NULL OPTIONS (description = 'The order it scored'),
  respect FLOAT64 NOT NULL,
  love FLOAT64 NOT NULL,
  convenience FLOAT64 NOT NULL,
  trust FLOAT64 NOT NULL,
  fit FLOAT64 NOT NULL,
  churn_risk FLOAT64 NOT NULL,
  quadrant STRING NOT NULL,
  model_version STRING NOT NULL OPTIONS (description = 'The model that scored it'),
  x1 FLOAT64 OPTIONS (description = 'Gaps of that order after that event, x1 to x8; null: not measured yet'),
  x2 FLOAT64,
  x3 FLOAT64,
  x4 FLOAT64,
  x5 FLOAT64,
  x6 FLOAT64,
  x7 FLOAT64,
  x8 FLOAT64
)
PARTITION BY day
CLUSTER BY customer_id
OPTIONS (description = 'Every customer on every day since their first scoring (T308)');

-- The rows features.daily holds for each day from first_day to last_day.
-- A scoring is a customer's row from the day it happened until the day
-- before their next scoring's day; of several on one day, the last counts.
CREATE OR REPLACE TABLE FUNCTION features.snapshots(first_day DATE, last_day DATE) AS
SELECT
  day,
  customer_id,
  scored_at,
  event_id,
  order_id,
  respect,
  love,
  convenience,
  trust,
  fit,
  churn_risk,
  quadrant,
  model_version,
  x1,
  x2,
  x3,
  x4,
  x5,
  x6,
  x7,
  x8
FROM (
  SELECT
    *,
    LEAD(scored_at) OVER (
      PARTITION BY customer_id ORDER BY scored_at, event_id
    ) AS next_scored_at
  FROM (
    -- Each scoring, with the gaps its order had after the same event.
    SELECT
      SPLIT(scores.path, '/')[OFFSET(1)] AS customer_id,
      TIMESTAMP(JSON_VALUE(scores.document, '$.at')) AS scored_at,
      scores.event_id,
      JSON_VALUE(scores.document, '$.order_id') AS order_id,
      CAST(JSON_VALUE(scores.document, '$.scores.respect') AS FLOAT64) AS respect,
      CAST(JSON_VALUE(scores.document, '$.scores.love') AS FLOAT64) AS love,
      CAST(JSON_VALUE(scores.document, '$.scores.convenience') AS FLOAT64) AS convenience,
      CAST(JSON_VALUE(scores.document, '$.scores.trust') AS FLOAT64) AS trust,
      CAST(JSON_VALUE(scores.document, '$.scores.fit') AS FLOAT64) AS fit,
      CAST(JSON_VALUE(scores.document, '$.churn_risk') AS FLOAT64) AS churn_risk,
      JSON_VALUE(scores.document, '$.quadrant') AS quadrant,
      JSON_VALUE(scores.document, '$.model_version') AS model_version,
      CAST(JSON_VALUE(orders.document, '$.gaps.x1') AS FLOAT64) AS x1,
      CAST(JSON_VALUE(orders.document, '$.gaps.x2') AS FLOAT64) AS x2,
      CAST(JSON_VALUE(orders.document, '$.gaps.x3') AS FLOAT64) AS x3,
      CAST(JSON_VALUE(orders.document, '$.gaps.x4') AS FLOAT64) AS x4,
      CAST(JSON_VALUE(orders.document, '$.gaps.x5') AS FLOAT64) AS x5,
      CAST(JSON_VALUE(orders.document, '$.gaps.x6') AS FLOAT64) AS x6,
      CAST(JSON_VALUE(orders.document, '$.gaps.x7') AS FLOAT64) AS x7,
      CAST(JSON_VALUE(orders.document, '$.gaps.x8') AS FLOAT64) AS x8
    FROM raw.scores AS scores
    LEFT JOIN (
      -- The order document each event saved: one per event, however often
      -- Pub/Sub delivered it.
      SELECT event_id, ANY_VALUE(document) AS document
      FROM raw.changes
      WHERE collection = 'orders'
      GROUP BY event_id
    ) AS orders
      ON orders.event_id = scores.event_id
  )
)
CROSS JOIN UNNEST(
  GENERATE_DATE_ARRAY(
    GREATEST(first_day, DATE(scored_at, 'Asia/Bangkok')),
    LEAST(
      last_day,
      IFNULL(DATE_SUB(DATE(next_scored_at, 'Asia/Bangkok'), INTERVAL 1 DAY), last_day)
    )
  )
) AS day;
