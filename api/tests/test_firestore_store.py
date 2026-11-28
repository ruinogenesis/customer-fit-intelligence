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
from fitcore.features import INPUTS
from fitcore.model import SCORES as SCORE_NAMES
from fitcore.schemas import load_event
from fitcore.scoring import Applied, Customer, Explanation, Scores
from flask.testing import FlaskClient
from google.cloud import firestore

from cfi_api import create_app
from cfi_api.store import (
    EVENT_RETENTION,
    EXPIRES_AT,
    EventRecord,
    HistoryEntry,
    MemoryStore,
    history_document,
    writes,
)

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SCORES = Scores(0.9, 0.8, 0.7, 0.6, 0.5, churn_risk=0.04, quadrant="lovemark")
SHARES = {name: {**dict.fromkeys(INPUTS, 0.0), "x1": -0.1} for name in SCORE_NAMES}
EXPLAINED = Explanation(SHARES, ("x1",))

needs_emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


# --- Documents ---


def test_a_history_entry_keeps_scores_churn_risk_and_quadrant_apart() -> None:
    entry = HistoryEntry("ev_1", "ord_1", T0, SCORES, EXPLAINED, "t112-sim")
    document = history_document(entry)
    assert (document["churn_risk"], document["quadrant"]) == (0.04, "lovemark")
    assert set(document["scores"]) == {"respect", "love", "convenience", "trust", "fit"}
    assert (document["order_id"], document["at"]) == ("ord_1", T0)
    assert (document["contributions"], document["drivers"]) == (SHARES, ["x1"])


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
        "drivers": None,
        "model_version": "t112-sim",
    }
    return EventRecord({**event, "event_id": event_id}, result)


def test_an_event_writes_its_event_and_customer_documents() -> None:
    record = _survey()
    applied = Applied(Customer.new("cust_1"), order=None, scored=None)
    paths = [path for path, _ in writes(record, applied, "t112-sim", T0)]
    event_id = record.event["event_id"]
    assert paths == [f"events/{event_id}", "customers/cust_1"]


def test_a_document_of_the_wrong_shape_is_never_saved() -> None:
    record = _survey()
    applied = Applied(Customer.new("cust_1"), order=None, scored=None)
    store = MemoryStore()
    with pytest.raises(DocumentError, match="customers.*model_version"):
        store.save(record, applied, "", T0)  # a customer document needs a model
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
    entry = db.document(f"customers/{customer}/scores/{delivered['event_id']}").get()
    stored_entry = entry.to_dict()
    assert stored_entry is not None
    assert stored_entry["drivers"] == late.get_json()["drivers"] == ["x1"]
    assert set(stored_entry["contributions"]) == set(SCORE_NAMES)


@needs_emulator
def test_firestore_keeps_an_events_expiry_as_a_timestamp(
    firestore_client: FlaskClient,
) -> None:
    # TTL deletes only by a timestamp; a string there would never expire.
    tag = uuid.uuid4().hex[:8]
    event = _event(
        "order_placed",
        {
            "order_id": f"ord_{tag}",
            "restaurant_id": "rest_1",
            "basket_value": 50_000,
            "fees": 6_000,
            "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
            "promo_used": False,
        },
        f"cust_{tag}",
        0,
    )
    before = datetime.now(UTC)
    assert firestore_client.post("/api/v1/events", json=event).status_code == 201
    after = datetime.now(UTC)
    db = firestore.Client(project="demo-cfi")
    stored = db.document(f"events/{event['event_id']}").get().to_dict()
    assert stored is not None
    expires = stored[EXPIRES_AT]
    assert isinstance(expires, datetime)
    # Firestore keeps microseconds, so the stored time is within the window.
    assert before + EVENT_RETENTION <= expires <= after + EVENT_RETENTION


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


# --- The model registry in Firestore (T310) ---


@needs_emulator
def test_the_registry_promotes_and_rolls_back_in_firestore() -> None:
    from cfi_api.firestore_store import FirestoreStore
    from cfi_api.model_registry import ModelEntry, UnknownVersion

    store = FirestoreStore(firestore.Client(project="demo-cfi"))
    tag = uuid.uuid4().hex[:8]
    first, second = f"a-{tag}", f"b-{tag}"

    def entry(version: str, hours: int) -> ModelEntry:
        return ModelEntry(
            version=version,
            status="candidate",
            path=f"gs://cfi-test-models/models/{version}/model.json",
            sha256="0" * 64,
            metrics={"churn_auc": 0.7},
            created_at=T0 + timedelta(hours=hours),
            promoted_at=None,
        )

    store.add_model(entry(first, 0))
    store.add_model(entry(second, 1))
    with pytest.raises(ValueError, match="already registered"):
        store.add_model(entry(first, 2))
    assert store.promote_model(first, T0).status == "live"
    found = store.find_live_model()
    assert found is not None and found.version == first
    store.promote_model(second, T0)
    mine = {e.version: e.status for e in store.list_models() if e.version.endswith(tag)}
    assert mine == {first: "retired", second: "live"}
    store.promote_model(first, T0)  # rolling back
    mine = {e.version: e.status for e in store.list_models() if e.version.endswith(tag)}
    assert mine == {first: "live", second: "retired"}
    with pytest.raises(UnknownVersion):
        store.promote_model(f"none-{tag}", T0)


@needs_emulator
def test_an_event_reads_the_live_version_inside_its_transaction(
    reference_model: str,
) -> None:
    from cfi_api.model_registry import Models, register

    app = create_app(
        {"STORE": "firestore", "PROJECT": "demo-cfi", "REGISTRY": "memory"}
    )
    models = app.extensions["cfi_models"]
    assert isinstance(models, Models)
    tag = uuid.uuid4().hex[:8]
    with open(reference_model, "rb") as model:
        register(
            app.extensions["cfi_store"],
            models.files,
            version=f"live-{tag}",
            model_json=model.read(),
            card=b"# card",
            bucket="cfi-test-models",
            trained=T0,
            metrics={},
        )
    client = app.test_client()
    assert client.post(f"/api/v1/models/live-{tag}/promote").status_code == 200
    placed = _event(
        "order_placed",
        {
            "order_id": f"ord_{tag}",
            "restaurant_id": "rest_1",
            "basket_value": 50_000,
            "fees": 6_000,
            "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
            "promo_used": False,
        },
        f"cust_{tag}",
        0,
    )
    answer = client.post("/api/v1/events", json=placed)
    assert answer.status_code == 201
    assert answer.get_json()["model_version"] == f"live-{tag}"


# --- Automation rules in Firestore (T406) ---


@needs_emulator
def test_rules_are_added_once_and_listed_by_name_in_firestore() -> None:
    from cfi_api.firestore_store import FirestoreStore
    from cfi_api.rules import StarterRule, rule_document, seed

    store = FirestoreStore(firestore.Client(project="demo-cfi"))
    tag = uuid.uuid4().hex[:8]
    later = rule_document(StarterRule(f"b-{tag}", f"B rule {tag}", "service"), T0)
    sooner = rule_document(StarterRule(f"a-{tag}", f"a rule {tag}", "marketing"), T0)
    store.add_rule(later)
    store.add_rule(sooner)
    with pytest.raises(ValueError, match="there already"):
        store.add_rule(later)
    with pytest.raises(DocumentError):
        store.add_rule({**later, "rule_id": f"c-{tag}", "status": "on"})
    mine = [rule for rule in store.list_rules() if rule["rule_id"].endswith(tag)]
    assert mine == [sooner, later]  # by name, whatever its case
    seed(store, T0)
    before = store.list_rules()
    assert seed(store, T0) == []
    assert store.list_rules() == before


@needs_emulator
def test_rule_status_and_actions_in_firestore() -> None:
    from cfi_api.engine import Scoring, decide
    from cfi_api.firestore_store import FirestoreStore
    from cfi_api.rules import StarterRule, rule_document
    from cfi_api.store import UnknownRule

    store = FirestoreStore(firestore.Client(project="demo-cfi"))
    tag = uuid.uuid4().hex[:8]
    rule = rule_document(StarterRule(f"r-{tag}", f"Rule {tag}", "marketing"), T0)
    store.add_rule(rule)
    later = T0 + timedelta(hours=1)
    changed = store.set_rule_status(f"r-{tag}", "dry_run", later)
    assert (changed["status"], changed["updated_at"]) == ("dry_run", later)
    found = next(r for r in store.list_rules() if r["rule_id"] == f"r-{tag}")
    assert (found["status"], found["updated_at"]) == ("dry_run", later)
    with pytest.raises(UnknownRule):
        store.set_rule_status(f"none-{tag}", "off", T0)
    with pytest.raises(DocumentError):
        store.set_rule_status(f"r-{tag}", "on", T0)

    newest = datetime.now(UTC) + timedelta(days=3650)
    action = {
        "action_id": f"r-{tag}--ord_1",
        "rule_id": f"r-{tag}",
        "customer_id": f"cust_{tag}",
        "arm": "treatment",
        "channel": "push",
        "dry_run": True,
        "decided_at": newest,
        "cost": 0,
        "send_at": newest,
        "outcome": None,
    }
    assert store.add_action(action)
    assert not store.add_action({**action, "arm": "holdout"})
    assert store.list_actions(1) == [action]  # newest first, and kept as first written

    # The engine on Firestore: a fee-sensitivity decision, once per order.
    if not any(r["rule_id"] == "fee-sensitivity" for r in store.list_rules()):
        from cfi_api.rules import seed

        seed(store, T0)
    store.set_rule_status("fee-sensitivity", "dry_run", T0)
    entry = {
        "order_id": f"ord_{tag}",
        "quadrant": "product",
        "contributions": {"fit": {"x3": -0.2}},
    }
    new = Scoring(f"cust_{tag}", "00000000-0000-4000-8000-000000000001", entry)
    # A marketing rule decides only for a customer with consent (T610).
    assert decide(store, new, T0) == []
    given = {"marketing": True, "at": T0, "source": "sign_up"}
    customer = firestore.Client(project="demo-cfi").document(f"customers/cust_{tag}")
    customer.set({"customer_id": f"cust_{tag}", "consent": given})
    assert len(decide(store, new, T0)) == 1
    assert decide(store, new, T0) == []
    store.set_rule_status("fee-sensitivity", "off", T0)


@needs_emulator
def test_the_guardrails_in_firestore() -> None:
    from cfi_api.engine import Guardrails
    from cfi_api.firestore_store import FirestoreStore
    from cfi_api.store import OverBudget

    store = FirestoreStore(firestore.Client(project="demo-cfi"))
    tag = uuid.uuid4().hex[:8]
    settings = Guardrails(cap_actions=3).document(T0)
    store.set_automation_settings(settings)
    assert store.automation_settings() == settings
    store.set_automation_settings(Guardrails().document(T0))  # back to defaults

    def action(n: int, arm: str, at: datetime) -> dict[str, Any]:
        return {
            "action_id": f"r-{tag}--ord_{n}",
            "rule_id": f"r-{tag}",
            "customer_id": f"cust_{tag}",
            "arm": arm,
            "channel": "voucher",
            "dry_run": True,
            "decided_at": at,
            "cost": 4_000,
            "send_at": at,
            "outcome": None,
        }

    day = "2099-12-31"  # a day no other test spends on
    assert store.add_action(action(1, "treatment", T0), day=day, budget=10_000)
    assert store.add_action(action(2, "holdout", T0), day=day, budget=10_000)
    assert store.add_action(action(3, "treatment", T0), day=day, budget=10_000)
    assert store.spent_on(day) == 8_000
    assert not store.add_action(action(1, "treatment", T0), day=day, budget=10_000)
    with pytest.raises(OverBudget):
        store.add_action(action(4, "holdout", T0), day=day, budget=10_000)
    assert store.spent_on(day) == 8_000
    found = store.customer_actions(f"cust_{tag}", T0)
    assert sorted(a["action_id"] for a in found) == [
        f"r-{tag}--ord_1",
        f"r-{tag}--ord_2",
        f"r-{tag}--ord_3",
    ]
    assert store.customer_actions(f"cust_{tag}", T0 + timedelta(seconds=1)) == []


@needs_emulator
def test_quiet_customers_in_firestore() -> None:
    from cfi_api.firestore_store import FirestoreStore

    client = firestore.Client(project="demo-cfi")
    store = FirestoreStore(client)
    tag = uuid.uuid4().hex[:8]
    long_ago = datetime(1990, 1, 1, tzinfo=UTC)  # before any other test's orders
    placed = {
        "quiet": long_ago,
        "edge": long_ago + timedelta(days=1),
        "recent": long_ago + timedelta(days=2),
        "never": None,
    }
    for name, at in placed.items():
        client.document(f"customers/cust_{tag}_{name}").set(
            {
                "customer_id": f"cust_{tag}_{name}",
                "latest_order_id": None if at is None else f"ord_{tag}_{name}",
                "latest_placed_at": at,
            }
        )
    found = store.quiet_customers(long_ago + timedelta(days=1))
    mine = sorted(c["customer_id"] for c in found if tag in c["customer_id"])
    assert mine == [f"cust_{tag}_edge", f"cust_{tag}_quiet"]


@needs_emulator
def test_a_rule_reads_the_customers_history_in_firestore() -> None:
    from cfi_api.engine import Scoring, decide
    from cfi_api.firestore_store import FirestoreStore
    from cfi_api.rules import seed

    client = firestore.Client(project="demo-cfi")
    store = FirestoreStore(client)
    tag = uuid.uuid4().hex[:8]
    customer = f"cust_{tag}"
    given = {"marketing": True, "at": T0, "source": "sign_up"}
    client.document(f"customers/{customer}").set(
        {"customer_id": customer, "consent": given}
    )
    scorings = []
    for n in range(1, 4):  # three Brand orders in a row (T506)
        order_id, at = f"ord_{tag}_{n}", T0 + timedelta(days=n)
        client.document(f"orders/{order_id}").set(
            {"order_id": order_id, "customer_id": customer, "placed_at": at}
        )
        event_id = str(uuid.uuid4())
        entry = {
            "event_id": event_id,
            "order_id": order_id,
            "at": at,
            "quadrant": "brand",
            "contributions": {"fit": {"x3": 0.0}},
        }
        client.document(f"customers/{customer}/scores/{event_id}").set(entry)
        scorings.append(Scoring(customer, event_id, entry))
    if not any(r["rule_id"] == "brand-to-lovemark" for r in store.list_rules()):
        seed(store, T0)
    store.set_rule_status("brand-to-lovemark", "dry_run", T0)
    try:
        decided = decide(store, scorings[-1], T0 + timedelta(days=3))
    finally:
        store.set_rule_status("brand-to-lovemark", "off", T0)
    assert [a["action_id"] for a in decided] == [f"brand-to-lovemark--ord_{tag}_1"]
