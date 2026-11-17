# BigQuery

History and training data, in `asia-southeast3`. Each file makes one dataset. Run it with
`bq query` (the command is at its top), once per project and again after a change:

| File | Dataset | What it holds |
| --- | --- | --- |
| `raw.sql` | `raw` | `raw.changes`: every document the API saves, as Pub/Sub delivers it (T306); a view per collection with each document's latest version |
| `labels.sql` | `labels` | `labels.order_labels(horizon)`: `reorder_14d` and `churn_30d` for every order, as known at `horizon`; the view `labels.orders` uses now (T307) |
| `features.sql` | `features` | `features.daily`: every customer on every Bangkok day since their first scoring, with their latest scores and the gaps behind them; `features.snapshots(first_day, last_day)` gives those rows for any days (T308). The API reads `features.daily` for movement on `/matrix` (T313) |

Run `raw.sql` first: the others read its views.

For simulated history, call `labels.order_labels` with the end of the simulated period: its
customers stop ordering when the simulation stops, which the view would read as churn.

## The daily snapshot (T308)

`features_daily.sql` is a scheduled query: every day at 02:00 in Bangkok it replaces
yesterday's rows of `features.daily`. A day's row holds the latest scoring whose event
happened by that day's midnight, so simulated history keeps its own dates. An event that
arrives after the run doesn't change days already written; write them again with a backfill.

One-time setup, after `features.sql`:

1. Enable `bigquerydatatransfer.googleapis.com`.
2. Create the service account `cfi-features`. Grant it `roles/bigquery.jobUser` on the
   project, and read `raw` and write `features` (`GRANT ... ON SCHEMA`).
3. Schedule `features_daily.sql` as `cfi-features`:

       bq mk --transfer_config --project_id=PROJECT --location=asia-southeast3 \
         --data_source=scheduled_query --display_name="features.daily (T308)" \
         --schedule="every day 19:00" \
         --service_account_name=cfi-features@PROJECT.iam.gserviceaccount.com \
         --params="$(python3 -c 'import json; print(json.dumps({"query": open("bigquery/features_daily.sql").read()}))')"

   The schedule is in UTC: 19:00 is 02:00 in Bangkok.
4. Backfill the days before the first run, in one query:

       DELETE FROM features.daily WHERE day <= LAST;
       INSERT INTO features.daily SELECT * FROM features.snapshots(FIRST, LAST);

## Tests

BigQuery has no emulator, so `api/tests/bigquery_on_duckdb.py` runs these files on DuckDB,
translated by sqlglot, over the rows the golden log (T309) sends to `raw.changes`:
`test_labels.py` and `test_features.py` compare the results with the same rows read in
Python. Keep to SQL that sqlglot translates: a statement it can't parse fails those tests.
The scheduled script `features_daily.sql` is not run by the tests; it only calls
`features.snapshots` for one day.
