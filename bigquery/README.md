# BigQuery

History and training data, in `asia-southeast3`. Each file makes one dataset. Run it with
`bq query` (the command is at its top), once per project and again after a change:

| File | Dataset | What it holds |
| --- | --- | --- |
| `raw.sql` | `raw` | `raw.changes`: every document the API saves, as Pub/Sub delivers it (T306); a view per collection with each document's latest version |
| `labels.sql` | `labels` | `labels.order_labels(horizon)`: `reorder_14d` and `churn_30d` for every order, as known at `horizon`; the view `labels.orders` uses now (T307) |

Run `raw.sql` first: `labels.sql` reads its views.

For simulated history, call `labels.order_labels` with the end of the simulated period: its
customers stop ordering when the simulation stops, which the view would read as churn.

BigQuery has no emulator, so `api/tests/test_labels.py` runs these files on DuckDB, translated
by sqlglot, over the rows the golden log (T309) sends to `raw.changes`. Keep to SQL that sqlglot
translates: a statement it can't parse fails that test.
