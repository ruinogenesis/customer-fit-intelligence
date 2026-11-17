"""T313: GET /matrix?from=&to= shows how customers moved between quadrants.

The movement comes from features.daily (T308) through cfi_api.history. Its
SQL runs here on the DuckDB stand-in for BigQuery (bigquery_on_duckdb):
DuckDBClient takes the place of BigQuery's client, so BigQueryHistory runs
as it does on Cloud Run, query parameters included. features.daily is
filled from features.snapshots over the golden log (T309), and each answer
is compared with the same rows read in Python (test_features.expected).
"""

import os
from collections import Counter
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any

import duckdb
import pytest
import sqlglot
from bigquery_on_duckdb import MODEL, exported, stand_in
from fitcore.golden import read_events
from fitcore.model import QUADRANTS
from flask import Flask
from flask.testing import FlaskClient
from test_features import BANGKOK, EVENTS, FIRST, LAST, expected

from cfi_api import create_app
from cfi_api.history import LOCATION, BigQueryHistory, Move

API = "/api/v1"
ROWS = (*QUADRANTS, "unscored")
QUADRANT = 11  # a snapshot row's quadrant (test_features.expected)


class DuckDBClient:
    """BigQuery's client, as BigQueryHistory uses it, on the DuckDB stand-in."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self.db = db

    def query(self, sql: str, *, job_config: Any, location: str) -> Any:
        assert location == LOCATION
        params = {}
        for param in job_config.query_parameters:
            assert param.type_ == "DATE"
            params[param.name] = param.value
        found = self.db.execute(
            sqlglot.transpile(sql, read="bigquery", write="duckdb")[0], params
        )
        names = [column[0] for column in found.description]
        rows = [dict(zip(names, row, strict=True)) for row in found.fetchall()]

        class Job:
            def result(self) -> list[dict[str, Any]]:
                return rows

        return Job()


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    """raw.changes after the golden log."""
    return exported(read_events(EVENTS))


@pytest.fixture(scope="module")
def db(rows: list[dict[str, Any]]) -> Iterator[duckdb.DuckDBPyConnection]:
    """The stand-in, with features.daily written for every day of the log."""
    connection = stand_in(rows, "features.sql")
    connection.execute(
        "CREATE TABLE features.daily AS SELECT * FROM features.snapshots(?, ?)",
        [FIRST, LAST],
    )
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def snapshot(rows: list[dict[str, Any]]) -> dict[tuple[date, str], tuple[Any, ...]]:
    """Every customer's row on every day, read in Python."""
    return expected(rows, FIRST, LAST)


def want(
    snapshot: dict[tuple[date, str], tuple[Any, ...]], first: date, last: date
) -> dict[str, dict[str, int]]:
    """The movement grid from first to last, read from the Python snapshot."""
    grid = {row: dict.fromkeys(QUADRANTS, 0) for row in ROWS}
    for (day, customer), row in snapshot.items():
        if day == last:
            before = snapshot.get((first, customer))
            was = "unscored" if before is None else before[QUADRANT]
            grid[was][row[QUADRANT]] += 1
    return grid


def _app(db: duckdb.DuckDBPyConnection | None) -> Flask:
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        app = create_app({"MODEL_PATH": str(MODEL), "MODEL_VERSION": "t112-sim"})
    if db is not None:
        app.extensions["cfi_history"] = BigQueryHistory(DuckDBClient(db))
    return app


@pytest.fixture(scope="module")
def client(db: duckdb.DuckDBPyConnection) -> FlaskClient:
    """An app whose history is the stand-in."""
    return _app(db).test_client()


def movement(client: FlaskClient, first: date, last: date) -> Any:
    answer = client.get(f"{API}/matrix", query_string={"from": first, "to": last})
    assert answer.status_code == 200, answer.get_json()
    return answer.get_json()["movement"]


# --- The SQL ---

SPANS = [
    (date(2026, 1, 1), date(2026, 1, 2)),
    (date(2026, 1, 10), date(2026, 2, 10)),
    (date(2026, 2, 1), LAST),
    (FIRST, LAST),
]


@pytest.mark.parametrize("first, last", SPANS, ids=str)
def test_history_counts_each_customer_once_by_both_quadrants(
    db: duckdb.DuckDBPyConnection,
    snapshot: dict[tuple[date, str], tuple[Any, ...]],
    first: date,
    last: date,
) -> None:
    moves = BigQueryHistory(DuckDBClient(db)).movement(first, last)
    got = {row: dict.fromkeys(QUADRANTS, 0) for row in ROWS}
    for move in moves:
        assert isinstance(move, Move) and move.customers > 0
        got[move.was or "unscored"][move.now] += move.customers
    assert got == want(snapshot, first, last)


def test_the_golden_log_has_movement_to_check(
    snapshot: dict[tuple[date, str], tuple[Any, ...]],
) -> None:
    # Customers who changed quadrant and customers new in the span: without
    # both, the tests above would pass on a query that ignored either.
    grid = want(snapshot, date(2026, 1, 10), date(2026, 2, 10))
    moved = sum(n for was, row in grid.items() for now, n in row.items() if was != now)
    new = sum(grid["unscored"].values())
    assert moved - new > 0 and new > 0


# --- GET /matrix?from=&to= ---


def test_the_matrix_gives_the_movement_between_two_days(
    client: FlaskClient, snapshot: dict[tuple[date, str], tuple[Any, ...]]
) -> None:
    first, last = date(2026, 1, 10), date(2026, 2, 10)
    got = movement(client, first, last)
    assert got["from"] == "2026-01-10" and got["to"] == "2026-02-10"
    assert got["moves"] == want(snapshot, first, last)
    counted = sum(n for row in got["moves"].values() for n in row.values())
    assert got["customers"] == counted == Counter(day for day, _ in snapshot)[last]


def test_before_the_first_scoring_everyone_is_unscored(
    client: FlaskClient, snapshot: dict[tuple[date, str], tuple[Any, ...]]
) -> None:
    got = movement(client, date(2025, 12, 1), LAST)
    assert all(n == 0 for was in QUADRANTS for n in got["moves"][was].values())
    assert sum(got["moves"]["unscored"].values()) == got["customers"] > 0


def test_without_dates_the_matrix_is_as_before(client: FlaskClient) -> None:
    answer = client.get(f"{API}/matrix")
    assert answer.status_code == 200
    assert set(answer.get_json()) == {"quadrants", "unscored", "customers"}


def _today() -> date:
    return datetime.now(BANGKOK).date()


BAD_SPANS = {
    "from only": {"from": "2026-01-10"},
    "to only": {"to": "2026-02-10"},
    "same day": {"from": "2026-02-10", "to": "2026-02-10"},
    "backwards": {"from": "2026-02-10", "to": "2026-01-10"},
    "not a date": {"from": "2026-01-10", "to": "next-week"},
    "today": {"from": "2026-01-10", "to": str(_today())},
    "future": {"from": "2026-01-10", "to": str(_today() + timedelta(days=30))},
}


@pytest.mark.parametrize("query", BAD_SPANS.values(), ids=BAD_SPANS.keys())
def test_a_bad_span_is_422(client: FlaskClient, query: dict[str, str]) -> None:
    answer = client.get(f"{API}/matrix", query_string=query)
    assert answer.status_code == 422, answer.get_json()
    assert answer.get_json()["code"] == "unprocessable_entity"


def test_a_day_without_a_snapshot_is_422(client: FlaskClient) -> None:
    # features.daily ends at LAST here, as it ends yesterday on BigQuery
    # until 02:00 Bangkok writes it.
    later = LAST + timedelta(days=1)
    answer = client.get(f"{API}/matrix", query_string={"from": FIRST, "to": later})
    assert answer.status_code == 422
    assert str(later) in answer.get_json()["message"]


def test_without_bigquery_movement_is_503() -> None:
    client = _app(None).test_client()
    answer = client.get(f"{API}/matrix", query_string={"from": FIRST, "to": LAST})
    assert answer.status_code == 503
    assert client.get(f"{API}/matrix").status_code == 200


def test_an_unknown_history_is_refused() -> None:
    with pytest.raises(ValueError, match="HISTORY"):
        create_app({"HISTORY": "spreadsheet"})


def test_the_dates_are_documented(client: FlaskClient) -> None:
    get = client.get(f"{API}/openapi.json").get_json()["paths"][f"{API}/matrix"]["get"]
    query = {param["name"] for param in get["parameters"] if param["in"] == "query"}
    assert query == {"from", "to"}
    assert {"200", "401", "403", "422", "503"} <= set(get["responses"])
