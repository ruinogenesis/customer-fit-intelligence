"""T202: the Firestore store; T301: the documents every event writes.

writes() and the document checks run everywhere. The tests marked
needs_emulator run only against the Firestore emulator (CI's emulator job,
or locally with firebase emulators:exec); elsewhere they are skipped.

There is no emulator test of many events for one customer at once (T302):
the emulator deadlocks on concurrent transactions over the same documents
and breaks the deadlock only after about 30 s (firebase-tools issue #2452),
which production Firestore does not. test_one_at_a_time.py tests that logic,
and the T302 check on dev measures it against real Firestore.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fitcore.documents import DocumentError
from fitcore.schemas import load_event
from fitcore.scoring import Applied, Customer, Scores
from flask.testing import FlaskClient
from google.cloud import firestore

from cfi_api import create_app
from cfi_api.store import (
    EventRecord,
    HistoryEntry,
    MemoryStore,
    history_document,
    writes,
)

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SCORES = Scores(0.9, 0.8, 0.7, 0.6, 0.5, churn_risk=0.04, quadrant="lovemark")

needs_emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


# --- Documents ---


def test_a_history_entry_keeps_scores_churn_risk_and_quadrant_apart() -> None:
    entry = HistoryEntry("ev_1", "ord_1", T0, SCORES, "t112-sim")
    document = history_document(entry)
    assert (document["churn_risk"], document["quadrant"]) == (0.04, "lovemark")
    assert set(document["scores"]) == {"respect", "love", "convenience", "trust", "fit"}
    assert (document["order_id"], document["at"]) == ("ord_1", T0)


def _survey() -> EventRecord:
    event_id = str(uuid.uuid4())
    event = load_event(
        {
            "event_id": event_id,
            "type": "survey_response",
            "customer_id": "cust_1",
            "occurred_at": T0.isoformat(),
            "schema_version": 1,
            "payload": {"item_id": "trust", "value": 6},
        }
    )
    result = {
        "event_id": event_id,
        "customer_id": "cust_1",
        "duplicate": False,
        "order_id": None,
        "scores": None,
        "model_version": "t112-sim",
    }
    return EventRecord({**event, "event_id": event_id}, result)


def test_an_event_writes_its_event_and_customer_documents() -> None:
    record = _survey()
    applied = Applied(Customer.new("cust_1"), order=None, scored=None)
    paths = [path for path, _ in writes(record, applied, "t112-sim")]
    event_id = record.event["event_id"]
    assert paths == [f"events/{event_id}", "customers/cust_1"]


def test_a_document_of_the_wrong_shape_is_never_saved() -> None:
    record = _survey()
    applied = Applied(Customer.new("cust_1"), order=None, scored=None)
    store = MemoryStore()
    with pytest.raises(DocumentError, match="customers.*model_version"):
        store.save(record, applied, "")  # a customer document needs a model
    assert (store.events, store.customers) == ({}, {})


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
            "PROJECT": "demo-cfi",
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
    assert stored["quadrant"] == scores["quadrant"]
    stored_order = db.document(f"orders/{order}").get().to_dict()
    assert set(stored_order["gaps"]) == {"x1", "x3"}
    assert (stored_order["restaurant_id"], stored_order["courier_id"]) == (
        "rest_1",
        "c1",
    )
    assert stored_order["delivered_at"] == T0 + timedelta(minutes=45)
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
