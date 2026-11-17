"""T307: bigquery/labels.sql gives the simulator's own labels on simulated history.

BigQuery has no emulator, so these tests run the committed SQL on DuckDB,
translated by sqlglot: raw.sql's views, then labels.sql, over the rows the
golden log (T309) sends to raw.changes through the API. They check the
SQL's logic on every merge; the T307 dev check ran the same files on
BigQuery itself.
"""

import json
import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
import sqlglot
from fitcore.golden import read_events
from fitcore.sim_orders import START, simulate_orders, simulate_profiles

from cfi_api import create_app
from cfi_api.export import MemoryPublisher
from cfi_api.store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "notebooks/results/t309_events.jsonl"
MODEL = ROOT / "notebooks/results/t112_model.json"
# The golden log: 25 simulated customers' first 60 days, seed 309.
CUSTOMERS, SEED, DAYS = 25, 309, 60.0
COLUMNS = ("collection", "path", "event_id", "written_at", "document")

Labels = dict[str, tuple[bool | None, bool | None]]


def duckdb_sql(path: Path) -> list[str]:
    """The views and table functions in a BigQuery SQL file, as DuckDB SQL.

    A table function becomes a DuckDB table macro. Datasets and tables are
    left out: the caller makes them as DuckDB schemas and tables.
    """
    statements = []
    for statement in sqlglot.parse(path.read_text(), read="bigquery"):
        kind = statement.args.get("kind")
        if kind == "VIEW":
            statements.append(statement.sql(dialect="duckdb"))
        elif kind == "FUNCTION":
            function = statement.this
            name = function.this.sql(dialect="duckdb")
            params = ", ".join(param.name for param in function.expressions)
            body = statement.expression.sql(dialect="duckdb")
            statements.append(f"CREATE MACRO {name}({params}) AS TABLE {body}")
    return statements


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    """raw.changes after the golden log: one row per document the API saved."""
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        app = create_app({"MODEL_PATH": str(MODEL), "MODEL_VERSION": "t112-sim"})
    app.extensions["cfi_store"] = MemoryStore(outbox=True)
    publisher = app.extensions["cfi_publisher"] = MemoryPublisher()
    client = app.test_client()
    for event in read_events(EVENTS):
        assert client.post("/api/v1/events", json=event).status_code == 201
    return [json.loads(message) for message in publisher.sent]


def bigquery(rows: list[dict[str, str]]) -> duckdb.DuckDBPyConnection:
    """A DuckDB stand-in for BigQuery: raw.changes holding rows, and the views
    and table functions of raw.sql and labels.sql."""
    db = duckdb.connect()
    db.execute("CREATE SCHEMA raw; CREATE SCHEMA labels")
    db.execute(
        "CREATE TABLE raw.changes (collection VARCHAR, path VARCHAR, "
        "event_id VARCHAR, written_at TIMESTAMPTZ, document JSON)"
    )
    # One column per parameter: far faster than a row at a time.
    db.execute(
        "INSERT INTO raw.changes SELECT "
        + ", ".join(f"unnest(${i + 1})" for i in range(len(COLUMNS))),
        [[row[column] for row in rows] for column in COLUMNS],
    )
    for statement in duckdb_sql(ROOT / "bigquery/raw.sql"):
        db.execute(statement)
    for statement in duckdb_sql(ROOT / "bigquery/labels.sql"):
        db.execute(statement)
    return db


@pytest.fixture(scope="module")
def db(rows: list[dict[str, str]]) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = bigquery(rows)
    yield connection
    connection.close()


def order_labels(db: duckdb.DuckDBPyConnection, horizon: datetime) -> Labels:
    """labels.order_labels(horizon), by order."""
    found = db.execute(
        "SELECT order_id, reorder_14d, churn_30d "
        "FROM labels.order_labels(CAST(? AS TIMESTAMPTZ))",
        [horizon.isoformat()],
    ).fetchall()
    return {order: (reorder, churn) for order, reorder, churn in found}


def simulated(days: float) -> Labels:
    """The simulator's own labels for the golden log's customers over days."""
    profiles = simulate_profiles(CUSTOMERS, SEED)
    position = {profile.customer_id: i for i, profile in enumerate(profiles)}
    orders = simulate_orders(profiles, SEED, start=START, days=days)
    return {
        f"sim{SEED}-{position[customer]:06d}-{number:03d}": (reorder, churn)
        for customer, number, reorder, churn in zip(
            orders.customer_ids,
            orders.order_numbers,
            orders.reorder_14d,
            orders.churn_30d,
            strict=True,
        )
    }


def test_labels_match_the_simulator_at_the_end_of_its_period(
    db: duckdb.DuckDBPyConnection,
) -> None:
    got = order_labels(db, START + timedelta(days=DAYS))
    assert got == simulated(DAYS)
    reorders = [reorder for reorder, _ in got.values()]
    churns = [churn for _, churn in got.values()]
    # Every kind of label occurs, so the comparison is not a trivial one.
    assert {True, False, None} <= set(reorders)
    assert {True, False, None} <= set(churns)


@pytest.mark.parametrize("days", [10.0, 30.0])
def test_earlier_labels_match_a_shorter_simulation(
    db: duckdb.DuckDBPyConnection, days: float
) -> None:
    # A shorter simulation is the first part of a longer one, so the labels
    # as known at its end are what the SQL knows at that time.
    assert order_labels(db, START + timedelta(days=days)) == simulated(days)


def test_a_row_written_twice_changes_no_label(
    rows: list[dict[str, str]], db: duckdb.DuckDBPyConnection
) -> None:
    twice = bigquery(rows + rows)
    end = START + timedelta(days=DAYS)
    assert order_labels(twice, end) == order_labels(db, end)


def test_the_view_knows_every_label_of_old_history(
    db: duckdb.DuckDBPyConnection,
) -> None:
    orders, reorders, churns = db.execute(
        "SELECT count(*), count(reorder_14d), count(churn_30d) FROM labels.orders"
    ).fetchone() or (0, 0, 0)
    assert orders == len(simulated(DAYS))
    assert reorders == churns == orders


def _order_row(order_id: str, placed_at: datetime) -> dict[str, str]:
    """A raw.changes row for an order document holding only what labels read."""
    document = {
        "order_id": order_id,
        "customer_id": "cust_1",
        "placed_at": placed_at.isoformat(),
    }
    return {
        "collection": "orders",
        "path": f"orders/{order_id}",
        "event_id": f"ev_{order_id}",
        "written_at": placed_at.isoformat(),
        "document": json.dumps(document),
    }


def test_orders_go_by_when_they_were_placed_not_by_their_ids() -> None:
    # The simulator numbers orders in time order; real order ids need not be.
    first, second = START, START + timedelta(days=20)
    db = bigquery([_order_row("ord_z", first), _order_row("ord_a", second)])
    assert order_labels(db, START + timedelta(days=DAYS)) == {
        "ord_z": (False, False),  # next order 20 days later
        "ord_a": (False, True),  # none in the 30 days after
    }
