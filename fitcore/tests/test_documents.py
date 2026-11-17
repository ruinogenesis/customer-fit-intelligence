"""T301: the Firestore documents, their checks, and their reference."""

import copy
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fitcore.documents import (
    COLLECTIONS,
    DocumentError,
    check,
    customer_document,
    customer_from_document,
    order_document,
    order_from_document,
    reference,
)
from fitcore.expectations import Baseline, update
from fitcore.model import ModelParams
from fitcore.schemas import load_event
from fitcore.scoring import (
    Customer,
    Issue,
    Order,
    Scores,
    apply_event,
    order_id_for,
)
from fitcore.sim_events import simulate_events

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SCORES = Scores(0.9, 0.8, 0.7, 0.6, 0.5, churn_risk=0.04, quadrant="lovemark")
LATENTS = {"respect": 0.9, "love": 0.8, "convenience": 0.7, "trust": 0.6, "fit": 0.5}
EVENT_ID = "6f1c2c8e-8a52-4a8e-9f0e-6b1d5b0c4a11"

ORDER = Order(
    "ord_1",
    "cust_1",
    T0,
    T0 + timedelta(minutes=30),
    {"x1": -5.0, "x3": 0.02},
    restaurant_id="rest_1",
    basket_value=50_000,
    fees=6_000,
    promo_used=False,
    courier_id="cour_1",
    delivered_at=T0 + timedelta(minutes=35),
    issues=(Issue("cold", T0 + timedelta(minutes=40)),),
    ratings={"food": 2, "courier": 5},
)


def _customer() -> Customer:
    baselines = dict(Customer.new("cust_1").baselines)
    baselines["x5"] = update(Baseline(), 4.0, T0)
    return Customer("cust_1", baselines, "ord_1", T0, SCORES)


def _examples() -> dict[str, dict[str, Any]]:
    """One valid document per collection."""
    event = load_event(
        {
            "event_id": EVENT_ID,
            "type": "order_issue",
            "customer_id": "cust_1",
            "occurred_at": T0.isoformat(),
            "schema_version": 1,
            "payload": {"order_id": "ord_1", "issue_type": "cold"},
        }
    )
    return {
        "events": {
            "event": {**event, "event_id": EVENT_ID},
            "result": {
                "event_id": EVENT_ID,
                "customer_id": "cust_1",
                "duplicate": False,
                "order_id": None,
                "scores": None,
                "model_version": "t112-sim",
            },
        },
        "customers": customer_document(_customer(), "t112-sim"),
        "scores": {
            "event_id": EVENT_ID,
            "order_id": "ord_1",
            "at": T0,
            "scores": LATENTS,
            "churn_risk": 0.04,
            "quadrant": "lovemark",
            "model_version": "t112-sim",
        },
        "orders": order_document(ORDER),
        "rules": {
            "rule_id": "service_recovery",
            "name": "Service recovery",
            "kind": "service",
            "status": "dry_run",
            "holdout_share": 0.5,
            "condition": {},
            "action": {},
            "caps": {},
            "updated_at": T0,
        },
        "actions": {
            "action_id": "act_1",
            "rule_id": "service_recovery",
            "customer_id": "cust_1",
            "arm": "holdout",
            "channel": "push",
            "dry_run": True,
            "decided_at": T0,
            "outcome": None,
        },
        "models": {
            "version": "t112-sim",
            "status": "live",
            "path": "gs://cfi-dev-th26-models/models/t112-sim/model.json",
            "metrics": {"churn_auc": 0.707},
            "created_at": T0,
            "promoted_at": T0,
        },
        "users": {
            "uid": "u_123",
            "email": "analyst@example.com",
            "role": "analyst",
            "updated_at": T0,
        },
        "outbox": {
            "event_id": EVENT_ID,
            "messages": ['{"collection": "events"}'],
            "created_at": T0,
        },
    }


# --- Every collection ---


def test_there_are_nine_collections_with_an_example_each() -> None:
    assert (
        set(COLLECTIONS)
        == set(_examples())
        == {
            "events",
            "customers",
            "scores",
            "orders",
            "rules",
            "actions",
            "models",
            "users",
            "outbox",
        }
    )


@pytest.mark.parametrize("collection", list(COLLECTIONS))
def test_a_valid_document_passes(collection: str) -> None:
    check(collection, _examples()[collection])


@pytest.mark.parametrize("collection", list(COLLECTIONS))
def test_an_unknown_field_is_refused(collection: str) -> None:
    document = {**_examples()[collection], "surprise": 1}
    with pytest.raises(DocumentError, match="surprise"):
        check(collection, document)


@pytest.mark.parametrize(
    ("collection", "field"),
    [(name, field) for name, c in COLLECTIONS.items() for field in c.schema().fields],
)
def test_every_field_is_required(collection: str, field: str) -> None:
    document = copy.deepcopy(_examples()[collection])
    del document[field]
    with pytest.raises(DocumentError, match=field):
        check(collection, document)


def test_every_field_is_described() -> None:
    for collection in COLLECTIONS.values():
        for name, field in collection.schema().fields.items():
            assert field.metadata.get("description"), f"{collection.path}: {name}"


# --- Rules the schemas enforce ---


def _refused(collection: str, change: dict[str, Any], match: str) -> None:
    document = {**_examples()[collection], **change}
    with pytest.raises(DocumentError, match=match):
        check(collection, document)


@pytest.mark.parametrize(
    ("collection", "change", "match"),
    [
        ("customers", {"customer_id": "a/b"}, "customer_id"),
        ("customers", {"latest_order_id": None}, "go together"),
        ("customers", {"quadrant": None}, "go together"),
        ("customers", {"churn_risk": 1.5}, "churn_risk"),
        ("customers", {"latest_placed_at": datetime(2026, 11, 14)}, "aware"),
        ("scores", {"quadrant": "cult"}, "quadrant"),
        ("orders", {"fees": -1}, "fees"),
        ("orders", {"fees": 1.5}, "fees"),
        ("orders", {"promo_used": 1}, "promo_used"),
        ("orders", {"courier_id": None}, "go together"),
        ("orders", {"gaps": {"x9": 1.0}}, "gaps"),
        ("orders", {"ratings": {"food": 6}}, "ratings"),
        ("orders", {"ratings": {"taste": 3}}, "ratings"),
        ("orders", {"issues": [{"issue_type": "rude", "at": T0}]}, "issues"),
        ("rules", {"status": "paused"}, "status"),
        ("rules", {"holdout_share": 10}, "holdout_share"),
        ("actions", {"arm": "control"}, "arm"),
        ("models", {"path": "/tmp/model.json"}, "path"),
        ("users", {"role": "owner"}, "role"),
        ("users", {"email": "not an email"}, "email"),
        ("events", {"result": {}}, "result"),
        ("outbox", {"messages": []}, "messages"),
        ("outbox", {"messages": [""]}, "messages"),
    ],
)
def test_a_document_that_breaks_a_rule_is_refused(
    collection: str, change: dict[str, Any], match: str
) -> None:
    _refused(collection, change, match)


def test_a_customer_needs_an_expectation_for_every_input() -> None:
    document = customer_document(_customer(), "t112-sim")
    del document["expectations"]["x8"]
    with pytest.raises(DocumentError, match="x2, x3, x4, x5, x6, x7, x8"):
        check("customers", document)


def test_an_events_document_checks_the_event_itself() -> None:
    document = _examples()["events"]
    document["event"]["payload"]["issue_type"] = "rude"
    with pytest.raises(DocumentError, match="issue_type"):
        check("events", document)


# --- Customers and orders, to documents and back ---


def test_a_customer_round_trips_through_its_document() -> None:
    customer, new = _customer(), Customer.new("cust_1")
    document = customer_document(customer, "t112-sim")
    assert document["scores"] == LATENTS
    assert (document["churn_risk"], document["quadrant"]) == (0.04, "lovemark")
    assert document["expectations"]["x5"] == {
        "mean": 4.0,
        "weight": 1.0,
        "last_seen": T0,
    }
    assert customer_from_document(document) == customer
    assert customer_from_document(customer_document(new, "v")) == new
    check("customers", customer_document(new, "v"))


def test_an_order_round_trips_through_its_document() -> None:
    document = order_document(ORDER)
    assert document["issues"] == [{"issue_type": "cold", "at": ORDER.issues[0].at}]
    assert order_from_document(document) == ORDER


def test_every_document_a_simulated_replay_makes_passes() -> None:
    params = ModelParams.from_json(
        (ROOT / "notebooks/results/t112_model.json").read_bytes()
    )
    customers: dict[str, Customer] = {}
    orders: dict[str, Order] = {}
    for raw in simulate_events(20, 5, 60.0):
        event = load_event(raw)
        cid = event["customer_id"]
        customer = customers.get(cid) or Customer.new(cid)
        order_id = order_id_for(event, customer)
        applied = apply_event(params, event, customer, orders.get(order_id or ""))
        customers[cid] = applied.customer
        if applied.order is not None:
            orders[applied.order.order_id] = applied.order
    for customer in customers.values():
        check("customers", customer_document(customer, "t112-sim"))
    for order in orders.values():
        check("orders", order_document(order))
    assert len(orders) > 100
    assert any(order.issues for order in orders.values())
    assert any(order.ratings for order in orders.values())


# --- The reference ---


def test_the_reference_is_up_to_date() -> None:
    """If this fails: python -m fitcore.documents > fitcore/DOCUMENTS.md"""
    assert (ROOT / "fitcore/DOCUMENTS.md").read_text() == reference()


def test_the_reference_has_every_collection_and_field() -> None:
    text = reference()
    for collection in COLLECTIONS.values():
        assert f"## `{collection.path}`" in text
        for name in collection.schema().fields:
            assert f"| `{name}` |" in text


def test_an_event_id_is_kept_as_text() -> None:
    document = _examples()["scores"]
    document["event_id"] = str(uuid.uuid4())
    check("scores", document)
