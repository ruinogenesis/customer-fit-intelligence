"""T311: GET /customers, one customer's state, scores and orders, and /matrix.

Each test runs on the memory store and, under the Firestore emulator, on
FirestoreStore too, after the same simulated events. The emulator also holds
other tests' customers, so checks there hold for whatever is listed: the
order, the filters, the pages. The parity tests compare Firestore's answers
with the memory store's for this file's own customers.
"""

import json
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fitcore.model import QUADRANTS
from fitcore.sim_events import simulate_events
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.customers import encode_cursor
from cfi_api.firestore_store import COMPOSITE_INDEXES

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "notebooks/results/t112_model.json"
API = "/api/v1"
T0 = datetime(2026, 3, 1, 12, tzinfo=UTC)


def _event(kind: str, customer: str, minutes: int, payload: dict[str, Any]) -> Any:
    # The same id on every run: the emulator keeps what an earlier run sent.
    event_id = uuid.uuid5(uuid.NAMESPACE_URL, f"t311/{customer}/{kind}/{minutes}")
    return {
        "event_id": str(event_id),
        "type": kind,
        "customer_id": customer,
        "occurred_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


def _order(customer: str, late: int, courier: int, food: int) -> list[Any]:
    """One order delivered late minutes after its promise, then rated."""
    order_id = f"{customer}-001"
    promised = (T0 + timedelta(minutes=30)).isoformat()
    placed = {
        "order_id": order_id,
        "restaurant_id": "rest_1",
        "basket_value": 35_000,
        "fees": 4_000,
        "promised_eta": promised,
        "promo_used": False,
    }
    delivered = {"order_id": order_id, "courier_id": "courier_1"}
    rated = {"order_id": order_id, "food": food, "courier": courier}
    return [
        _event("order_placed", customer, 0, placed),
        _event("order_delivered", customer, 30 + late, delivered),
        _event("rating_submitted", customer, 45 + late, rated),
    ]


# Six simulated customers' first 30 days (47 events, all lovemarks), and
# customers made for the other quadrants: two alike, to check the id breaks
# a tie, and one who only answered a survey, so is not scored yet.
EVENTS = [
    *simulate_events(6, seed=3111, days=30.0),
    *_order("t311-product-a", late=60, courier=1, food=1),
    *_order("t311-product-b", late=60, courier=1, food=1),
    *_order("t311-fad", late=0, courier=1, food=5),
    *_order("t311-brand", late=-20, courier=5, food=3),
    _event("survey_response", "t311-unscored", 0, {"item_id": "love", "value": 4}),
]
OURS = sorted({event["customer_id"] for event in EVENTS})

needs_emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


def _app_with_events(store: str) -> FlaskClient:
    config: dict[str, Any] = {"MODEL_PATH": str(MODEL), "MODEL_VERSION": "t112-sim"}
    if store == "firestore":
        config |= {"STORE": "firestore", "PROJECT": "demo-cfi"}
    with pytest.MonkeyPatch.context() as env:
        for name in [name for name in os.environ if name.startswith("CFI_")]:
            env.delenv(name)
        client = create_app(config).test_client()
    for event in EVENTS:
        # 200 too: the emulator keeps the events of an earlier run.
        assert client.post(f"{API}/events", json=event).status_code in (200, 201)
    return client


@pytest.fixture(
    scope="module", params=["memory", pytest.param("firestore", marks=needs_emulator)]
)
def client(request: pytest.FixtureRequest) -> FlaskClient:
    """An app on each store, after this file's events."""
    return _app_with_events(request.param)


@pytest.fixture(scope="module")
def reference() -> FlaskClient:
    """The memory store after the same events: what Firestore must answer."""
    return _app_with_events("memory")


def pages(
    client: FlaskClient, path: str, name: str, limit: int, query: dict[str, Any]
) -> Iterator[list[Any]]:
    """Every page of a list, following next_cursor until it is null."""
    cursor = None
    while True:
        page = {"limit": limit} | ({} if cursor is None else {"cursor": cursor})
        answer = client.get(path, query_string=query | page)
        assert answer.status_code == 200, answer.get_json()
        body = answer.get_json()
        assert len(body[name]) <= limit
        yield body[name]
        cursor = body["next_cursor"]
        if cursor is None:
            return


def everything(
    client: FlaskClient,
    path: str,
    name: str,
    limit: int = 4,
    query: dict[str, Any] | None = None,
) -> list[Any]:
    """Every item of a list, read a page of limit at a time."""
    return [
        item for page in pages(client, path, name, limit, query or {}) for item in page
    ]


def customer_rank(customer: dict[str, Any]) -> tuple[bool, float, str]:
    risk = customer["churn_risk"]
    return risk is not None, risk or 0.0, customer["customer_id"]


def time_rank(item: dict[str, Any], when: str, item_id: str) -> tuple[str, str]:
    # ISO 8601 in UTC sorts as text in time order.
    return item[when], item[item_id]


# --- /customers ---


def test_customers_come_riskiest_first_a_page_at_a_time(client: FlaskClient) -> None:
    listed = everything(client, f"{API}/customers", "customers")
    ids = [customer["customer_id"] for customer in listed]
    assert len(ids) == len(set(ids))  # no page repeats one
    assert set(OURS) <= set(ids)  # and none is skipped
    ranks = [customer_rank(customer) for customer in listed]
    assert ranks == sorted(ranks, reverse=True)


@pytest.mark.parametrize("quadrant", [None, *QUADRANTS])
@pytest.mark.parametrize("min_churn_risk", [None, 0.02, 0.1])
def test_filters_keep_exactly_the_customers_that_match(
    client: FlaskClient, quadrant: str | None, min_churn_risk: float | None
) -> None:
    filters = {"quadrant": quadrant, "min_churn_risk": min_churn_risk}
    query = {name: value for name, value in filters.items() if value is not None}
    listed = everything(client, f"{API}/customers", "customers", query=query)
    every = everything(client, f"{API}/customers", "customers", limit=200)

    def matches(customer: dict[str, Any]) -> bool:
        risk = customer["churn_risk"]
        return (quadrant is None or customer["quadrant"] == quadrant) and (
            min_churn_risk is None or (risk is not None and risk >= min_churn_risk)
        )

    assert listed == [customer for customer in every if matches(customer)]


def test_a_listed_customer_leaves_out_the_expectations(client: FlaskClient) -> None:
    listed = everything(client, f"{API}/customers", "customers", limit=200)
    ours = {c["customer_id"]: c for c in listed if c["customer_id"] in OURS}
    for customer_id, summary in ours.items():
        full = client.get(f"{API}/customers/{customer_id}").get_json()
        assert full.pop("expectations")
        assert full == summary


# --- One customer ---


def test_a_customers_scores_come_newest_first(client: FlaskClient) -> None:
    for customer_id in OURS:
        path = f"{API}/customers/{customer_id}/scores"
        scores = everything(client, path, "scores", limit=3)
        ranks = [time_rank(score, "at", "event_id") for score in scores]
        assert ranks == sorted(ranks, reverse=True)
        assert len({score["event_id"] for score in scores}) == len(scores)
        latest = client.get(f"{API}/customers/{customer_id}").get_json()
        if scores:
            assert scores[0]["churn_risk"] == latest["churn_risk"]


def test_a_customers_orders_come_newest_first(client: FlaskClient) -> None:
    for customer_id in OURS:
        path = f"{API}/customers/{customer_id}/orders"
        orders = everything(client, path, "orders", limit=2)
        placed = sorted(
            event["payload"]["order_id"]
            for event in EVENTS
            if event["customer_id"] == customer_id and event["type"] == "order_placed"
        )
        assert sorted(order["order_id"] for order in orders) == placed
        ranks = [time_rank(order, "placed_at", "order_id") for order in orders]
        assert ranks == sorted(ranks, reverse=True)


@pytest.mark.parametrize("tail", ["", "/scores", "/orders"])
@pytest.mark.parametrize("customer_id", ["nobody", "__name__", ".."])
def test_an_unknown_customer_is_404(
    client: FlaskClient, tail: str, customer_id: str
) -> None:
    answer = client.get(f"{API}/customers/{customer_id}{tail}")
    assert answer.status_code == 404
    assert answer.get_json()["code"] == "not_found"


# --- /matrix ---


def test_the_matrix_counts_the_listed_customers(client: FlaskClient) -> None:
    matrix = client.get(f"{API}/matrix").get_json()
    listed = everything(client, f"{API}/customers", "customers", limit=200)
    for quadrant in QUADRANTS:
        here = [c for c in listed if c["quadrant"] == quadrant]
        assert matrix["quadrants"][quadrant] == len(here)
    assert matrix["unscored"] == sum(c["quadrant"] is None for c in listed)
    assert matrix["customers"] == len(listed)


def test_the_matrix_counts_this_files_customers_exactly(reference: FlaskClient) -> None:
    matrix = reference.get(f"{API}/matrix").get_json()
    quadrants = [
        reference.get(f"{API}/customers/{customer_id}").get_json()["quadrant"]
        for customer_id in OURS
    ]
    assert matrix["customers"] == len(OURS)
    assert sum(matrix["quadrants"].values()) + matrix["unscored"] == len(OURS)
    for quadrant in QUADRANTS:
        assert matrix["quadrants"][quadrant] == quadrants.count(quadrant) > 0
    assert matrix["unscored"] == quadrants.count(None) == 1


# --- Bad requests ---


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=201",
        "quadrant=happy",
        "min_churn_risk=2",
        "cursor=not-a-cursor",
        f"cursor={encode_cursor(('2026-01-01T00:00:00+00:00', 'x'))}",  # a time key
        "colour=red",
    ],
)
def test_a_bad_query_is_422(reference: FlaskClient, query: str) -> None:
    answer = reference.get(f"{API}/customers?{query}")
    assert answer.status_code == 422
    assert answer.get_json()["code"] == "unprocessable_entity"


def test_a_customer_cursor_is_refused_by_a_time_list(reference: FlaskClient) -> None:
    cursor = encode_cursor((0.5, OURS[0]))
    answer = reference.get(f"{API}/customers/{OURS[0]}/scores?cursor={cursor}")
    assert answer.status_code == 422


# --- Firestore against the memory store ---


@needs_emulator
def test_firestore_answers_as_the_memory_store_does(reference: FlaskClient) -> None:
    firestore = _app_with_events("firestore")
    for customer_id in OURS:
        for tail, name in (("", None), ("/scores", "scores"), ("/orders", "orders")):
            path = f"{API}/customers/{customer_id}{tail}"
            if name is None:
                got, want = firestore.get(path), reference.get(path)
                assert got.get_json() == want.get_json()
            else:
                assert everything(firestore, path, name) == everything(
                    reference, path, name
                )
    listed = everything(firestore, f"{API}/customers", "customers")
    ours = [c["customer_id"] for c in listed if c["customer_id"] in OURS]
    want = everything(reference, f"{API}/customers", "customers")
    assert ours == [c["customer_id"] for c in want]


# --- Documentation and indexes ---


def test_the_reads_are_documented_with_their_sign_in(reference: FlaskClient) -> None:
    spec = reference.get(f"{API}/openapi.json").get_json()
    for path in (
        "/customers",
        "/customers/{customer_id}",
        "/customers/{customer_id}/scores",
        "/customers/{customer_id}/orders",
        "/matrix",
    ):
        get = spec["paths"][f"{API}{path}"]["get"]
        assert get["security"] == [{"firebase": []}]
        assert {"200", "401", "403"} <= set(get["responses"])


def test_firestore_indexes_json_has_the_composite_indexes_the_reads_need() -> None:
    listed = json.loads((ROOT / "firestore.indexes.json").read_text())["indexes"]
    found = {
        (
            index["collectionGroup"],
            tuple((field["fieldPath"], field["order"]) for field in index["fields"]),
        )
        for index in listed
        if index["queryScope"] == "COLLECTION"
    }
    assert found == set(COMPOSITE_INDEXES)


def test_a_tie_in_churn_risk_goes_by_customer_id(reference: FlaskClient) -> None:
    listed = everything(reference, f"{API}/customers", "customers", limit=1)
    twins = [c for c in listed if c["customer_id"].startswith("t311-product-")]
    assert twins[0]["churn_risk"] == twins[1]["churn_risk"]
    assert [c["customer_id"] for c in twins] == ["t311-product-b", "t311-product-a"]
