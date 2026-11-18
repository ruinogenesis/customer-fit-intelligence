"""What the API reads from BigQuery: history too long to read from Firestore
on every request (T313, T315, T402).

T308's scheduled query writes features.daily: every customer's latest
scoring at the end of each Bangkok day, up to yesterday. GET /matrix reads
two of its days to show how customers moved between quadrants: two day
partitions, a few MB at 10,000 customers, where Firestore would read every
score entry since the first day.

GET /overview/trend gives each day's averages and quadrant counts over a
span of features.daily (T402): one row per day, whatever the customers.

GET /drivers averages each restaurant's or courier's orders' contributions
to fit (T316's splits) from raw.scores and raw.orders, which are live.

The SQL lives here, so it ships with the API it answers. Its tests run it
on the DuckDB stand-in for BigQuery (api/tests/test_movement.py).
"""

from datetime import date, datetime
from typing import Any, NamedTuple, Protocol
from zoneinfo import ZoneInfo

from fitcore.features import INPUTS
from fitcore.model import QUADRANTS

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


# Each Bangkok day from @first_day to @last_day that features.daily has: how
# many customers it holds, their average fit and churn risk, and how many
# are in each quadrant (QUADRANTS). A day before anyone's first scoring has
# no rows, so no answer.
TREND = (
    """
SELECT
  day,
  COUNT(*) AS customers,
  AVG(fit) AS fit,
  AVG(churn_risk) AS churn_risk,
"""
    + ",\n".join(f"  COUNTIF(quadrant = '{q}') AS {q}" for q in QUADRANTS)
    + """
FROM features.daily
WHERE day BETWEEN @first_day AND @last_day
GROUP BY day
ORDER BY day
"""
)


# Each restaurant (@by = 'restaurant') or courier (@by = 'courier') with
# orders placed from @first_day to @last_day (Bangkok days): how many, their
# average fit, and each input's average contribution to it. An order counts
# once, by its latest scoring that carries a split (T316); scorings from
# before T316 carry none, and an order not delivered yet has no courier.
DRIVERS = """
SELECT
  served.group_id,
  COUNT(*) AS orders,
  AVG(scored.fit) AS fit,
  AVG(scored.x1) AS x1,
  AVG(scored.x2) AS x2,
  AVG(scored.x3) AS x3,
  AVG(scored.x4) AS x4,
  AVG(scored.x5) AS x5,
  AVG(scored.x6) AS x6,
  AVG(scored.x7) AS x7,
  AVG(scored.x8) AS x8
FROM (
  SELECT
    order_id,
    fit,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8
  FROM (
    -- Every scoring with a split, newest first within its order.
    SELECT
      JSON_VALUE(document, '$.order_id') AS order_id,
      CAST(JSON_VALUE(document, '$.scores.fit') AS FLOAT64) AS fit,
      CAST(JSON_VALUE(document, '$.contributions.fit.x1') AS FLOAT64) AS x1,
      CAST(JSON_VALUE(document, '$.contributions.fit.x2') AS FLOAT64) AS x2,
      CAST(JSON_VALUE(document, '$.contributions.fit.x3') AS FLOAT64) AS x3,
      CAST(JSON_VALUE(document, '$.contributions.fit.x4') AS FLOAT64) AS x4,
      CAST(JSON_VALUE(document, '$.contributions.fit.x5') AS FLOAT64) AS x5,
      CAST(JSON_VALUE(document, '$.contributions.fit.x6') AS FLOAT64) AS x6,
      CAST(JSON_VALUE(document, '$.contributions.fit.x7') AS FLOAT64) AS x7,
      CAST(JSON_VALUE(document, '$.contributions.fit.x8') AS FLOAT64) AS x8,
      ROW_NUMBER() OVER (
        PARTITION BY JSON_VALUE(document, '$.order_id')
        ORDER BY TIMESTAMP(JSON_VALUE(document, '$.at')) DESC, event_id DESC
      ) AS newest
    FROM raw.scores
    WHERE JSON_QUERY(document, '$.contributions.fit') IS NOT NULL
  )
  WHERE newest = 1
) AS scored
JOIN (
  -- Each order placed in the window, by its restaurant or its courier.
  SELECT
    JSON_VALUE(document, '$.order_id') AS order_id,
    IF(
      @by = 'restaurant',
      JSON_VALUE(document, '$.restaurant_id'),
      JSON_VALUE(document, '$.courier_id')
    ) AS group_id
  FROM raw.orders
  WHERE DATE(TIMESTAMP(JSON_VALUE(document, '$.placed_at')), 'Asia/Bangkok')
    BETWEEN @first_day AND @last_day
) AS served
  ON served.order_id = scored.order_id
WHERE served.group_id IS NOT NULL
GROUP BY served.group_id
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


class Group(NamedTuple):
    """One restaurant's or courier's orders: how many, their average fit, and
    each input's average contribution to it (INPUTS order)."""

    id: str
    orders: int
    fit: float
    drag: tuple[float, ...]


class Day(NamedTuple):
    """One day of features.daily, summed up."""

    day: date
    customers: int
    fit: float
    churn_risk: float
    quadrants: dict[str, int]


class History(Protocol):
    """Reads from BigQuery's history."""

    def movement(self, first: date, last: date) -> list[Move]:
        """Every customer in features.daily on last, by quadrant on both days.

        No moves: features.daily has no customers on last.
        """
        ...

    def drivers(self, by: str, first: date, last: date) -> list[Group]:
        """Every restaurant (by "restaurant") or courier (by "courier") with
        orders placed from first to last, Bangkok days, that have a split."""
        ...

    def trend(self, first: date, last: date) -> list[Day]:
        """Each day from first to last that features.daily has, in order."""
        ...


class BigQueryHistory:
    """A History through BigQuery's client (google.cloud.bigquery.Client), or
    anything that answers query() as it does."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def movement(self, first: date, last: date) -> list[Move]:
        rows = self._query(MOVEMENT, first_day=first, last_day=last)
        return [Move(row["was"], row["now"], int(row["customers"])) for row in rows]

    def drivers(self, by: str, first: date, last: date) -> list[Group]:
        if by not in ("restaurant", "courier"):
            raise ValueError(f"by must be restaurant or courier, got {by!r}")
        rows = self._query(DRIVERS, by=by, first_day=first, last_day=last)
        return [
            Group(
                row["group_id"],
                int(row["orders"]),
                float(row["fit"]),
                tuple(float(row[key]) for key in INPUTS),
            )
            for row in rows
        ]

    def trend(self, first: date, last: date) -> list[Day]:
        rows = self._query(TREND, first_day=first, last_day=last)
        return [
            Day(
                row["day"],
                int(row["customers"]),
                float(row["fit"]),
                float(row["churn_risk"]),
                {quadrant: int(row[quadrant]) for quadrant in QUADRANTS},
            )
            for row in rows
        ]

    def _query(self, sql: str, **params: date | str) -> list[Any]:
        # Imported only when used: it is slow to import.
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    name, "STRING" if isinstance(value, str) else "DATE", value
                )
                for name, value in params.items()
            ]
        )
        job = self._client.query(sql, job_config=config, location=LOCATION)
        return list(job.result())
