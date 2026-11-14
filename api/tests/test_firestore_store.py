"""T202: the Firestore store.

The document conversions run everywhere. The tests marked needs_emulator
run only against the Firestore emulator (CI's emulator job, or locally with
firebase emulators:exec); elsewhere they are skipped.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fitcore.expectations import Baseline, update
from fitcore.scoring import Customer, Order, Scores
from flask.testing import FlaskClient
from google.cloud import firestore

from cfi_api import create_app
from cfi_api.firestore_store import (
    customer_document,
    customer_from_document,
    history_document,
    order_document,
    order_from_document,
)
from cfi_api.store import HistoryEntry

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SCORES = Scores(0.9, 0.8, 0.7, 0.6, 0.5, churn_risk=0.04)

needs_emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


# --- Documents ---


def test_a_customer_round_trips_through_its_document() -> None:
    new = Customer.new("cust_1")
    baselines = dict(new.baselines)
    baselines["x5"] = update(Baseline(), 4.0, T0)
    customer = Customer("cust_1", baselines, "ord_1", T0, SCORES)
    document = customer_document(customer, "t112-sim")
    assert document["scores"] == {
        "respect": 0.9,
        "love": 0.8,
        "convenience": 0.7,
        "trust": 0.6,
        "fit": 0.5,
    }
    assert (document["churn_risk"], document["model_version"]) == (0.04, "t112-sim")
    assert document["expectations"]["x5"] == {
        "mean": 4.0,
        "weight": 1.0,
        "last_seen": T0,
    }
    assert customer_from_document(document) == customer
    assert customer_from_document(customer_document(new, "v")) == new


def test_an_order_round_trips_through_its_document() -> None:
    order = Order("ord_1", "cust_1", T0, T0 + timedelta(minutes=30), {"x1": -5.0})
    assert order_from_document(order_document(order)) == order


def test_a_history_entry_keeps_scores_and_churn_risk_apart() -> None:
    entry = HistoryEntry("ev_1", "ord_1", T0, SCORES, "t112-sim")
    document = history_document(entry)
    assert document["churn_risk"] == 0.04
    assert set(document["scores"]) == {"respect", "love", "convenience", "trust", "fit"}
    assert (document["order_id"], document["at"]) == ("ord_1", T0)


# --- Against the emulator ---


def _event(
    kind: str, payload: dict[str, Any], customer: str, minutes: int
) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": kind,
        "customer_id": customer,
        "occurred_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


@pytest.fixture
def firestore_client(reference_model: str) -> FlaskClient:
    app = create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "STORE": "firestore",
            "FIRESTORE_PROJECT": "demo-cfi",
        }
    )
    return app.test_client()


@needs_emulator
def test_events_are_stored_and_scored_in_firestore(
    firestore_client: FlaskClient,
) -> None:
    tag = uuid.uuid4().hex[:8]
    customer, order = f"cust_{tag}", f"ord_{tag}"
    placed = _event(
        "order_placed",
        {
            "order_id": order,
            "restaurant_id": "rest_1",
            "basket_value": 50_000,
            "fees": 6_000,
            "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
            "promo_used": False,
        },
        customer,
        0,
    )
    delivered = _event(
        "order_delivered", {"order_id": order, "courier_id": "c1"}, customer, 45
    )
    first = firestore_client.post("/api/v1/events", json=placed)
    late = firestore_client.post("/api/v1/events", json=delivered)
    again = firestore_client.post("/api/v1/events", json=delivered)
    assert (first.status_code, late.status_code, again.status_code) == (201, 201, 200)
    assert again.get_json() == {**late.get_json(), "duplicate": True}
    scores = late.get_json()["scores"]
    assert scores["fit"] < first.get_json()["scores"]["fit"]

    db = firestore.Client(project="demo-cfi")
    stored = db.document(f"customers/{customer}").get().to_dict()
    assert stored is not None
    assert stored["churn_risk"] == scores["churn_risk"]
    assert stored["scores"]["fit"] == scores["fit"]
    assert stored["model_version"] == "t112-sim"
    gaps = db.document(f"orders/{order}").get().to_dict()["gaps"]
    assert set(gaps) == {"x1", "x3"}
    history = db.collection(f"customers/{customer}/scores").get()
    assert {doc.id for doc in history} == {placed["event_id"], delivered["event_id"]}


@needs_emulator
def test_a_conflict_writes_nothing_to_firestore(firestore_client: FlaskClient) -> None:
    tag = uuid.uuid4().hex[:8]
    event = _event(
        "order_delivered",
        {"order_id": f"ord_{tag}", "courier_id": "c1"},
        f"cust_{tag}",
        5,
    )
    response = firestore_client.post("/api/v1/events", json=event)
    assert response.status_code == 409
    db = firestore.Client(project="demo-cfi")
    assert not db.document(f"events/{event['event_id']}").get().exists
    assert not db.document(f"customers/cust_{tag}").get().exists


@needs_emulator
def test_a_what_if_saves_nothing_in_firestore(firestore_client: FlaskClient) -> None:
    tag = uuid.uuid4().hex[:8]
    customer, order = f"cust_{tag}", f"ord_{tag}"
    placed = _event(
        "order_placed",
        {
            "order_id": order,
            "restaurant_id": "rest_1",
            "basket_value": 50_000,
            "fees": 6_000,
            "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
            "promo_used": False,
        },
        customer,
        0,
    )
    late = _event(
        "order_delivered", {"order_id": order, "courier_id": "c1"}, customer, 45
    )
    now = firestore_client.post("/api/v1/events", json=placed).get_json()["scores"]
    what_if = firestore_client.post("/api/v1/score", json=late)
    assert what_if.status_code == 200
    assert what_if.get_json()["scores"]["fit"] < now["fit"]

    db = firestore.Client(project="demo-cfi")
    assert not db.document(f"events/{late['event_id']}").get().exists
    assert set(db.document(f"orders/{order}").get().to_dict()["gaps"]) == {"x3"}
    stored = db.document(f"customers/{customer}").get().to_dict()
    assert stored["scores"]["fit"] == now["fit"]
    assert len(db.collection(f"customers/{customer}/scores").get()) == 1
