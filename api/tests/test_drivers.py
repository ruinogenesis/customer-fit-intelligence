"""T315: GET /drivers ranks the inputs pulling fit down at each restaurant or
courier, from BigQuery.

The drivers SQL (cfi_api.history.DRIVERS) runs on the DuckDB stand-in over
a planted simulated run (fitcore.sim_events, plant=True): 100 customers, 90
days. Its answer is compared with the same rows read in Python (expected),
and the endpoint must find what was planted: the slow restaurants behind
most late orders (x1), the one behind most cold food (x6), and the rude
couriers (x5), whose lateness alone would look like everyone else's.
"""

import json
import os
from collections import defaultdict
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pytest
from bigquery_on_duckdb import MODEL, DuckDBClient, exported, stand_in
from fitcore.features import INPUTS
from fitcore.sim_events import PLANTED, simulate_events
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.history import BigQueryHistory, Group

API = "/api/v1/drivers"
BANGKOK = ZoneInfo("Asia/Bangkok")
RUN = (100, 315, 90.0)  # customers, seed, days
FIRST, LAST = date(2026, 1, 1), date(2026, 4, 1)
FIELD = {"restaurant": "restaurant_id", "courier": "courier_id"}
# What each planted restaurant or courier should be found for.
WANT = {
    name: {"late": "x1", "cold_food": "x6", "rude_courier": "x5"}[incident]
    for incident, (_, names) in PLANTED.items()
    for name in names
}


def expected(
    rows: list[dict[str, Any]], by: str, first: date, last: date
) -> dict[str, tuple[int, float, tuple[float, ...]]]:
    """Each group's orders, average fit and average drag, read in Python:
    every order placed from first to last, by its latest scoring with a split."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:  # raw.changes: each document's versions, oldest first
        latest[row["path"]] = json.loads(row["document"])
    newest: dict[str, dict[str, Any]] = {}
    for path, doc in latest.items():
        if "/scores/" in path and "contributions" in doc:
            key = (datetime.fromisoformat(doc["at"]), doc["event_id"])
            old = newest.get(doc["order_id"])
            if old is None or key > (
                datetime.fromisoformat(old["at"]),
                old["event_id"],
            ):
                newest[doc["order_id"]] = doc
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, doc in latest.items():
        if path.startswith("orders/") and doc["order_id"] in newest:
            placed = datetime.fromisoformat(doc["placed_at"]).astimezone(BANGKOK).date()
            group = doc.get(FIELD[by])
            if group is not None and first <= placed <= last:
                groups[group].append(newest[doc["order_id"]])
    return {
        group: (
            len(docs),
            sum(doc["scores"]["fit"] for doc in docs) / len(docs),
            tuple(
                sum(doc["contributions"]["fit"][key] for doc in docs) / len(docs)
                for key in INPUTS
            ),
        )
        for group, docs in groups.items()
    }


def alike(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> bool:
    """Two answers' groups alike: ids, order counts and drivers exactly, the
    numbers within 1e-12. DuckDB, like BigQuery, adds up on several threads,
    so two runs of one query can differ in the last digit."""

    def exact(groups: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return [(g["id"], g["orders"], g["drivers"]) for g in groups]

    return exact(got) == exact(want) and all(
        abs(g["fit"] - w["fit"]) < 1e-12
        and all(
            abs(g[part][key] - w[part][key]) < 1e-12
            for part in ("drag", "excess")
            for key in INPUTS
        )
        for g, w in zip(got, want, strict=True)
    )


def close(got: dict[str, Group], want: dict[str, Any]) -> bool:
    return set(got) == set(want) and all(
        got[name].orders == want[name][0]
        and abs(got[name].fit - want[name][1]) < 1e-12
        and all(
            abs(a - b) < 1e-12
            for a, b in zip(got[name].drag, want[name][2], strict=True)
        )
        for name in want
    )


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    """raw.changes after the planted run."""
    events = simulate_events(*RUN, plant=True)
    return exported(json.loads(json.dumps(events, default=str)))


@pytest.fixture(scope="module")
def db(rows: list[dict[str, Any]]) -> Iterator[duckdb.DuckDBPyConnection]:
    connection = stand_in(rows)
    yield connection
    connection.close()


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
    return _app(db).test_client()


def drivers(client: FlaskClient, **query: Any) -> Any:
    answer = client.get(API, query_string={"from": FIRST, "to": LAST, **query})
    assert answer.status_code == 200, answer.get_json()
    return answer.get_json()


# --- The SQL ---


SPANS = [(FIRST, LAST), (date(2026, 2, 1), date(2026, 2, 28)), (LAST, LAST)]


@pytest.mark.parametrize("by", ["restaurant", "courier"])
@pytest.mark.parametrize("first, last", SPANS, ids=str)
def test_history_averages_each_groups_orders_as_python_does(
    db: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
    by: str,
    first: date,
    last: date,
) -> None:
    client = DuckDBClient(db)
    got = {
        group.id: group for group in BigQueryHistory(client).drivers(by, first, last)
    }
    assert client.types == ["STRING", "DATE", "DATE"]
    assert close(got, expected(rows, by, first, last))


def test_an_order_counts_by_its_latest_scoring_with_a_split(
    rows: list[dict[str, Any]],
) -> None:
    # Scorings from before T316 carry no split. Take it off every scoring of
    # one customer and off the newest scoring of another customer's orders:
    # the first one's orders drop out, the second one's count by the
    # scoring before.
    first_customer, second = "sim315-000000", "sim315-000001"
    newest: dict[str, str] = {}
    for row in rows:
        doc = json.loads(row["document"])
        if row["collection"] == "scores" and doc["order_id"].startswith(second):
            newest[doc["order_id"]] = row["path"]  # rows come in time order
    changed = []
    for row in rows:
        doc = json.loads(row["document"])
        if row["collection"] == "scores" and (
            row["path"].startswith(f"customers/{first_customer}/")
            or row["path"] in newest.values()
        ):
            doc.pop("contributions")
            row = {**row, "document": json.dumps(doc)}
        changed.append(row)
    with stand_in(changed) as connection:
        found = BigQueryHistory(DuckDBClient(connection)).drivers(
            "courier", FIRST, LAST
        )
    want = expected(changed, "courier", FIRST, LAST)
    assert close({group.id: group for group in found}, want)
    assert sum(n for n, _, _ in want.values()) < sum(
        n for n, _, _ in expected(rows, "courier", FIRST, LAST).values()
    )


# --- GET /drivers ---


@pytest.mark.parametrize("by", ["restaurant", "courier"])
def test_each_group_is_compared_with_all_orders(
    client: FlaskClient, rows: list[dict[str, Any]], by: str
) -> None:
    got = drivers(client, by=by, min_orders=1, limit=200)
    want = expected(rows, by, FIRST, LAST)
    orders = sum(n for n, _, _ in want.values())
    assert (got["by"], got["from"], got["to"]) == (by, str(FIRST), str(LAST))
    assert got["overall"]["orders"] == orders
    overall = got["overall"]["drag"]
    for i, key in enumerate(INPUTS):
        mean = sum(n * drag[i] for n, _, drag in want.values()) / orders
        assert overall[key] == pytest.approx(mean, abs=1e-12)
    assert len(got["groups"]) == len(want)
    for group in got["groups"]:
        assert group["orders"] == want[group["id"]][0]
        for key in INPUTS:
            excess = group["drag"][key] - overall[key]
            assert group["excess"][key] == pytest.approx(excess, abs=1e-12)
        worse = sorted(
            (k for k in INPUTS if group["excess"][k] < 0), key=group["excess"].get
        )
        assert group["drivers"] == worse[:3]
    worst = [min(group["excess"].values()) for group in got["groups"]]
    assert worst == sorted(worst)


@pytest.mark.parametrize("by, top", [("restaurant", 3), ("courier", 6)])
def test_it_finds_what_was_planted(client: FlaskClient, by: str, top: int) -> None:
    groups = drivers(client, by=by)["groups"]
    planted = [name for name in WANT if name.startswith(by[:4])]
    ranked = [group["id"] for group in groups]
    for name in planted:
        group = groups[ranked.index(name)]
        assert group["drivers"][0] == WANT[name], (name, group["excess"])
        assert ranked.index(name) < top, (name, ranked[:top])


def test_small_groups_and_the_limit(client: FlaskClient) -> None:
    every = drivers(client, by="courier", min_orders=1, limit=200)["groups"]
    some = drivers(client, by="courier", min_orders=25, limit=200)["groups"]
    assert alike(some, [group for group in every if group["orders"] >= 25])
    assert 0 < len(some) < len(every)
    first_three = drivers(client, by="courier", min_orders=1, limit=3)["groups"]
    assert alike(first_three, every[:3])
    default = drivers(client, by="courier")
    assert default["min_orders"] == 5
    assert alike(default["groups"], [g for g in every if g["orders"] >= 5][:50])


def test_by_default_the_last_28_days(client: FlaskClient) -> None:
    today = datetime.now(BANGKOK).date()
    got = client.get(API, query_string={"by": "courier"}).get_json()
    assert (got["from"], got["to"]) == (str(today - timedelta(days=27)), str(today))
    # The simulated run ended in April: nothing in the last 28 days.
    assert got["overall"] == {
        "orders": 0,
        "fit": None,
        "drag": dict.fromkeys(INPUTS, 0.0),
    }
    assert got["groups"] == []
    only_to = client.get(API, query_string={"by": "courier", "to": LAST}).get_json()
    assert only_to["from"] == str(LAST - timedelta(days=27))


def _today() -> date:
    return datetime.now(BANGKOK).date()


BAD = {
    "no by": {"from": "2026-01-01"},
    "unknown by": {"by": "zone"},
    "backwards": {"by": "courier", "from": "2026-02-01", "to": "2026-01-01"},
    "tomorrow": {"by": "courier", "to": str(_today() + timedelta(days=1))},
    "no orders": {"by": "courier", "min_orders": 0},
    "too many": {"by": "courier", "limit": 201},
    "unknown": {"by": "courier", "zone": "north"},
}


@pytest.mark.parametrize("query", BAD.values(), ids=BAD.keys())
def test_bad_arguments_are_422(client: FlaskClient, query: dict[str, Any]) -> None:
    answer = client.get(API, query_string=query)
    assert answer.status_code == 422, answer.get_json()


def test_without_bigquery_drivers_is_503() -> None:
    answer = _app(None).test_client().get(API, query_string={"by": "courier"})
    assert answer.status_code == 503


def test_history_refuses_an_unknown_grouping(db: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError, match="restaurant or courier"):
        BigQueryHistory(DuckDBClient(db)).drivers("zone", FIRST, LAST)


def test_drivers_is_documented(client: FlaskClient) -> None:
    get = client.get("/api/v1/openapi.json").get_json()["paths"][API]["get"]
    query = {param["name"] for param in get["parameters"] if param["in"] == "query"}
    assert query == {"by", "from", "to", "min_orders", "limit"}
    assert {"200", "401", "403", "422", "503"} <= set(get["responses"])
