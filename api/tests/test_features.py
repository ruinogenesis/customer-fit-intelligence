"""T308: features.snapshots gives each customer's latest scoring on every day.

These tests run raw.sql and features.sql on the DuckDB stand-in for
BigQuery (bigquery_on_duckdb), over the rows the golden log (T309) sends to
raw.changes, and compare with the same rows read here in Python: per
customer and Bangkok day, the latest scoring whose event happened by
midnight, and the gaps its order had after that event.
"""

import json
import uuid
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest
import sqlglot
from bigquery_on_duckdb import ROOT, exported, stand_in
from fitcore.golden import read_events
from fitcore.model import LATENTS
from sqlglot import exp

BANGKOK = ZoneInfo("Asia/Bangkok")
EVENTS = ROOT / "notebooks/results/t309_events.jsonl"
# The golden log's Bangkok days: its 60 simulated days start 2026-01-01 UTC.
FIRST, LAST = date(2026, 1, 1), date(2026, 3, 2)
GAPS = tuple(f"x{i}" for i in range(1, 9))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

Row = tuple[Any, ...]
Snapshot = dict[tuple[date, str], Row]


def bigquery(rows: list[dict[str, Any]]) -> duckdb.DuckDBPyConnection:
    """The stand-in with raw.changes holding rows, and features.sql's views."""
    return stand_in(rows, "features.sql")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    """raw.changes after the golden log: one row per document the API saved."""
    return exported(read_events(EVENTS))


@pytest.fixture(scope="module")
def db(rows: list[dict[str, Any]]) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = bigquery(rows)
    yield connection
    connection.close()


def snapshots(db: duckdb.DuckDBPyConnection, first: date, last: date) -> Snapshot:
    """features.snapshots(first, last), by (day, customer); scored_at in µs."""
    found = db.execute(
        "SELECT day, customer_id, epoch_us(scored_at), * EXCLUDE "
        "(day, customer_id, scored_at) FROM features.snapshots("
        "CAST(? AS DATE), CAST(? AS DATE))",
        [first.isoformat(), last.isoformat()],
    ).fetchall()
    snapshot = {(row[0], row[1]): tuple(row) for row in found}
    assert len(snapshot) == len(found)  # one row per customer per day
    return snapshot


def _microseconds(at: str) -> int:
    return (datetime.fromisoformat(at) - EPOCH) // timedelta(microseconds=1)


def expected(rows: list[dict[str, Any]], first: date, last: date) -> Snapshot:
    """What features.snapshots should give, read from rows in Python."""
    scorings: dict[str, dict[str, Any]] = {}  # by path: a row can come twice
    gaps: dict[str, dict[str, float]] = {}  # each event's order, after it
    for row in rows:
        document = json.loads(row["document"])
        if row["collection"] == "scores":
            scorings[row["path"]] = document | {"customer": row["path"].split("/")[1]}
        elif row["collection"] == "orders":
            gaps[row["event_id"]] = document["gaps"]
    history = defaultdict(list)
    for scoring in scorings.values():
        history[scoring["customer"]].append(scoring)
    want: Snapshot = {}
    day = first
    while day <= last:
        midnight = datetime.combine(day + timedelta(days=1), time(), BANGKOK)
        for customer, entries in history.items():
            done = [e for e in entries if datetime.fromisoformat(e["at"]) < midnight]
            if not done:
                continue
            latest = max(
                done, key=lambda e: (datetime.fromisoformat(e["at"]), e["event_id"])
            )
            measured = gaps.get(latest["event_id"], {})
            want[day, customer] = (
                day,
                customer,
                _microseconds(latest["at"]),
                latest["event_id"],
                latest["order_id"],
                *(latest["scores"][name] for name in LATENTS),
                latest["churn_risk"],
                latest["quadrant"],
                latest["model_version"],
                *(measured.get(name) for name in GAPS),
            )
        day += timedelta(days=1)
    return want


def test_every_day_holds_each_customers_latest_scoring(
    rows: list[dict[str, Any]], db: duckdb.DuckDBPyConnection
) -> None:
    got = snapshots(db, FIRST, LAST)
    assert got == expected(rows, FIRST, LAST)
    # 25 customers, each from the day of their first scoring to the last day.
    assert len(got) == 1275
    assert len({customer for _, customer in got}) == 25


def test_one_day_is_the_same_slice_of_a_longer_run(
    db: duckdb.DuckDBPyConnection,
) -> None:
    # What the scheduled query writes for one day, and what a backfill of
    # the whole period writes for that day, are the same rows.
    everything = snapshots(db, FIRST, LAST)
    for day in (FIRST, date(2026, 1, 20), date(2026, 2, 14), LAST):
        one = snapshots(db, day, day)
        assert one == {key: row for key, row in everything.items() if key[0] == day}


def test_a_row_written_twice_changes_nothing(
    rows: list[dict[str, Any]], db: duckdb.DuckDBPyConnection
) -> None:
    twice = bigquery(rows + rows)
    assert snapshots(twice, FIRST, LAST) == snapshots(db, FIRST, LAST)


def test_the_rows_fit_features_daily(db: duckdb.DuckDBPyConnection) -> None:
    table = next(
        statement
        for statement in sqlglot.parse(
            (ROOT / "bigquery/features.sql").read_text(), read="bigquery"
        )
        if statement.args.get("kind") == "TABLE"
    )
    columns = list(table.this.expressions)
    run = "features.snapshots(DATE '2026-01-01', DATE '2026-03-02')"
    found = db.execute(f"SELECT * FROM {run} LIMIT 0")
    names = [column[0] for column in found.description]
    assert [column.name for column in columns] == names  # INSERT ... SELECT *
    for column in columns:
        if any(
            isinstance(constraint.kind, exp.NotNullColumnConstraint)
            for constraint in column.constraints
        ):
            nulls = db.execute(
                f"SELECT count(*) FROM {run} WHERE {column.name} IS NULL"
            ).fetchone()
            assert nulls == (0,), column.name


# --- Days end at midnight in Bangkok ---


def _placed(customer: str, at: datetime) -> dict[str, Any]:
    """An order placed at `at`: scored at once, with only x3 (fees) measured."""
    return {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"t308/{customer}")),
        "type": "order_placed",
        "customer_id": customer,
        "occurred_at": at.isoformat(),
        "schema_version": 1,
        "payload": {
            "order_id": f"{customer}-001",
            "restaurant_id": "rest_1",
            "basket_value": 35_000,
            "fees": 4_000,
            "promised_eta": (at + timedelta(minutes=30)).isoformat(),
            "promo_used": False,
        },
    }


@pytest.fixture(scope="module")
def midnight() -> duckdb.DuckDBPyConnection:
    """Two orders 30 minutes either side of midnight in Bangkok, 1 March."""
    events = [
        _placed("evening", datetime(2026, 3, 1, 16, 30, tzinfo=UTC)),  # 23:30 there
        _placed("night", datetime(2026, 3, 1, 17, 30, tzinfo=UTC)),  # 00:30 there
    ]
    return bigquery(exported(events))


def test_a_scoring_belongs_to_its_bangkok_day(
    midnight: duckdb.DuckDBPyConnection,
) -> None:
    got = snapshots(midnight, date(2026, 3, 1), date(2026, 3, 2))
    assert sorted(got) == [
        (date(2026, 3, 1), "evening"),
        (date(2026, 3, 2), "evening"),
        (date(2026, 3, 2), "night"),
    ]


def test_a_gap_not_measured_yet_is_null(midnight: duckdb.DuckDBPyConnection) -> None:
    for row in snapshots(midnight, date(2026, 3, 2), date(2026, 3, 2)).values():
        gaps = dict(zip(GAPS, row[-len(GAPS) :], strict=True))
        assert gaps["x3"] is not None  # the fees, known when it was placed
        assert gaps["x1"] is None  # speed, known only once delivered
