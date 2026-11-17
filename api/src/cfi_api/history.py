"""What the API reads from BigQuery: history too long to read from Firestore
on every request (T313).

T308's scheduled query writes features.daily: every customer's latest
scoring at the end of each Bangkok day, up to yesterday. GET /matrix reads
two of its days to show how customers moved between quadrants: two day
partitions, a few MB at 10,000 customers, where Firestore would read every
score entry since the first day.

The SQL lives here, so it ships with the API it answers. Its tests run it
on the DuckDB stand-in for BigQuery (api/tests/test_movement.py).
"""

from datetime import date, datetime
from typing import Any, NamedTuple, Protocol
from zoneinfo import ZoneInfo

# Where the datasets live, so where BigQuery runs the queries.
LOCATION = "asia-southeast3"

# features.daily's days are Bangkok calendar days (T308).
BANGKOK = ZoneInfo("Asia/Bangkok")

# Each customer scored by @last_day, counted by their quadrant on @first_day
# (NULL: not scored yet) and on @last_day. A customer stays in features.daily
# every day after their first scoring, so no one scored on @first_day is
# missing on @last_day.
MOVEMENT = """
SELECT earlier.quadrant AS was, later.quadrant AS now, COUNT(*) AS customers
FROM (
  SELECT customer_id, quadrant FROM features.daily WHERE day = @last_day
) AS later
LEFT JOIN (
  SELECT customer_id, quadrant FROM features.daily WHERE day = @first_day
) AS earlier
  ON earlier.customer_id = later.customer_id
GROUP BY was, now
"""


def today() -> date:
    """Today in Bangkok: features.daily has every day before it, once its
    02:00 run has written yesterday."""
    return datetime.now(BANGKOK).date()


class Move(NamedTuple):
    """Customers who were in one quadrant on the first day and in another,
    or the same, on the last; was is None for those not scored yet."""

    was: str | None
    now: str
    customers: int


class History(Protocol):
    """Reads from BigQuery's history."""

    def movement(self, first: date, last: date) -> list[Move]:
        """Every customer in features.daily on last, by quadrant on both days.

        No moves: features.daily has no customers on last.
        """
        ...


class BigQueryHistory:
    """A History through BigQuery's client (google.cloud.bigquery.Client), or
    anything that answers query() as it does."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def movement(self, first: date, last: date) -> list[Move]:
        rows = self._query(MOVEMENT, first_day=first, last_day=last)
        return [Move(row["was"], row["now"], int(row["customers"])) for row in rows]

    def _query(self, sql: str, **days: date) -> list[Any]:
        # Imported only when used: it is slow to import.
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(name, "DATE", day)
                for name, day in days.items()
            ]
        )
        job = self._client.query(sql, job_config=config, location=LOCATION)
        return list(job.result())
