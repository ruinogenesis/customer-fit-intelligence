"""T307: bigquery/labels.sql gives the simulator's own labels on simulated history.

These tests run raw.sql and labels.sql on the DuckDB stand-in for BigQuery
(bigquery_on_duckdb), over the rows the golden log (T309) sends to
raw.changes. The T307 dev check ran the same files on BigQuery itself.
"""

import json
from collections.abc import Iterator
from datetime import datetime, timedelta

import duckdb
import pytest
from bigquery_on_duckdb import ROOT, exported, stand_in
from fitcore.golden import read_events
from fitcore.sim_orders import START, simulate_orders, simulate_profiles

EVENTS = ROOT / "notebooks/results/t309_events.jsonl"
# The golden log: 25 simulated customers' first 60 days, seed 309.
CUSTOMERS, SEED, DAYS = 25, 309, 60.0

Labels = dict[str, tuple[bool | None, bool | None]]


def bigquery(rows: list[dict[str, str]]) -> duckdb.DuckDBPyConnection:
    """The stand-in with raw.changes holding rows, and labels.sql's views."""
    return stand_in(rows, "labels.sql")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    """raw.changes after the golden log: one row per document the API saved."""
    return exported(read_events(EVENTS))


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
