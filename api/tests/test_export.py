"""T306: every document POST /events saves goes on to BigQuery, through Pub/Sub.

Most tests use a MemoryStore with an outbox and a MemoryPublisher. Those
marked needs_emulators run against the Firestore and Pub/Sub emulators (CI's
emulator job, or locally with firebase emulators:exec); elsewhere they are
skipped.
"""

import dataclasses
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fitcore.documents import COLLECTIONS
from flask import Flask
from flask.testing import FlaskClient
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import firestore
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient

from cfi_api import create_app
from cfi_api.export import (
    LEFTOVERS_PER_REQUEST,
    MemoryPublisher,
    OutboxEntry,
    attributes,
    export,
    message,
    outbox_document,
)
from cfi_api.store import MemoryStore

URL = "/api/v1/events"
T0 = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
RAW_SQL = Path(__file__).resolve().parents[2] / "bigquery/raw.sql"

# The collections POST /events writes, each sent on to BigQuery with a view
# in bigquery/raw.sql, and the rest, each with why not. A new collection
# must join one or the other.
EXPORTED = frozenset({"events", "customers", "orders", "scores"})
NOT_EXPORTED = {
    "rules": "admins' settings (T406); the dashboard reads them through the API",
    "actions": "the uplift report (T507) decides how decisions reach BigQuery",
    "settings": "admins' guardrail settings (T502), not customer data",
    "budget": "a running total the guardrails keep (T502); actions hold the costs",
    "models": "nothing writes it yet (T310); its task decides",
    "users": "staff accounts (T303) stay out of analytics",
    "outbox": "the export's own queue",
}

needs_emulators = pytest.mark.skipif(
    not {"FIRESTORE_EMULATOR_HOST", "PUBSUB_EMULATOR_HOST"} <= set(os.environ),
    reason="needs the Firestore and Pub/Sub emulators: run via firebase emulators:exec",
)


def _event(kind: str, payload: dict[str, Any], customer: str) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": kind,
        "customer_id": customer,
        "occurred_at": T0.isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


def placed(order: str = "ord_1", customer: str = "cust_1") -> dict[str, Any]:
    payload = {
        "order_id": order,
        "restaurant_id": "rest_1",
        "basket_value": 50_000,
        "fees": 6_000,
        "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
        "promo_used": False,
    }
    return _event("order_placed", payload, customer)


def survey(customer: str = "cust_1") -> dict[str, Any]:
    return _event("survey_response", {"item_id": "trust", "value": 6}, customer)


def rows(messages: list[str]) -> list[dict[str, Any]]:
    return [json.loads(text) for text in messages]


# --- The messages and raw.changes ---


def _columns() -> list[str]:
    """raw.changes's columns, as bigquery/raw.sql makes them."""
    table = re.search(
        r"CREATE TABLE IF NOT EXISTS raw\.changes \((.*?)\n\)",
        RAW_SQL.read_text(),
        re.DOTALL,
    )
    assert table is not None
    return re.findall(r"^  (\w+) ", table.group(1), re.MULTILINE)


def test_a_message_is_a_row_of_raw_changes() -> None:
    row = json.loads(message("customers/cust_1/scores/ev_1", {"at": T0}, "ev_1", T0))
    metadata = ["subscription_name", "message_id", "publish_time", "attributes"]
    assert sorted(_columns()) == sorted([*row, *metadata])
    assert (row["collection"], row["path"]) == (
        "scores",
        "customers/cust_1/scores/ev_1",
    )


def test_a_message_names_its_collection_as_an_attribute() -> None:
    sent = message("customers/cust_1/scores/ev_1", {"at": T0}, "ev_1", T0)
    assert attributes(sent) == {"collection": "scores"}
    assert attributes(message("orders/ord_1", {}, "ev_1", T0)) == {
        "collection": "orders"
    }


def test_timestamps_go_to_bigquery_in_utc() -> None:
    bangkok = datetime(2026, 11, 17, 19, 0, tzinfo=timezone(timedelta(hours=7)))
    row = json.loads(message("orders/ord_1", {"placed_at": bangkok}, "ev_1", bangkok))
    assert row["written_at"] == "2026-11-17T12:00:00+00:00"
    assert json.loads(row["document"]) == {"placed_at": "2026-11-17T12:00:00+00:00"}


def test_every_collection_is_either_exported_or_says_why_not() -> None:
    assert EXPORTED.isdisjoint(NOT_EXPORTED)
    assert EXPORTED | set(NOT_EXPORTED) == set(COLLECTIONS)


@pytest.mark.parametrize("collection", sorted(EXPORTED))
def test_every_exported_collection_has_a_view(collection: str) -> None:
    view = f"CREATE OR REPLACE VIEW raw.{collection} AS"
    assert view in RAW_SQL.read_text()
    assert f"WHERE collection = '{collection}'" in RAW_SQL.read_text()


# --- POST /events, with a memory store ---


@pytest.fixture
def app(reference_model: str) -> Flask:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    app.extensions["cfi_store"] = MemoryStore(outbox=True)
    app.extensions["cfi_publisher"] = MemoryPublisher()
    return app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def stored(app: Flask) -> MemoryStore:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return store


def published(app: Flask) -> MemoryPublisher:
    publisher = app.extensions["cfi_publisher"]
    assert isinstance(publisher, MemoryPublisher)
    return publisher


def test_an_order_sends_every_document_it_saved(
    app: Flask, client: FlaskClient
) -> None:
    event = placed()
    answer = client.post(URL, json=event).get_json()
    sent = rows(published(app).sent)
    assert sorted(row["path"] for row in sent) == sorted(
        [
            f"events/{event['event_id']}",
            "customers/cust_1",
            "orders/ord_1",
            f"customers/cust_1/scores/{event['event_id']}",
        ]
    )
    assert {row["event_id"] for row in sent} == {event["event_id"]}
    assert len({row["written_at"] for row in sent}) == 1
    customer = next(row for row in sent if row["collection"] == "customers")
    document = json.loads(customer["document"])
    assert document["churn_risk"] == answer["scores"]["churn_risk"]
    assert stored(app).outbox == {}


def test_an_event_without_an_order_sends_its_event_and_customer(
    app: Flask, client: FlaskClient
) -> None:
    client.post(URL, json=survey())
    assert sorted(row["collection"] for row in rows(published(app).sent)) == [
        "customers",
        "events",
    ]


def test_a_duplicate_sends_nothing(app: Flask, client: FlaskClient) -> None:
    event = placed()
    client.post(URL, json=event)
    count = len(published(app).sent)
    assert client.post(URL, json=event).status_code == 200
    assert len(published(app).sent) == count


def test_a_conflict_saves_and_sends_nothing(app: Flask, client: FlaskClient) -> None:
    late = _event(
        "order_delivered", {"order_id": "ord_9", "courier_id": "c1"}, "cust_1"
    )
    assert client.post(URL, json=late).status_code == 409
    assert (published(app).sent, stored(app).outbox) == ([], {})


def test_when_pub_sub_fails_the_event_is_saved_and_waits_in_the_outbox(
    app: Flask, client: FlaskClient, caplog: pytest.LogCaptureFixture
) -> None:
    published(app).failing = True
    event = placed()
    with caplog.at_level(logging.ERROR, logger="cfi_api.export"):
        assert client.post(URL, json=event).status_code == 201
    assert "stays in the outbox" in caplog.text
    assert stored(app).find_event(event["event_id"]) is not None
    assert list(stored(app).outbox) == [event["event_id"]]
    assert published(app).sent == []


def test_a_later_request_sends_leftovers_over_a_minute_old(
    app: Flask, client: FlaskClient
) -> None:
    store, publisher = stored(app), published(app)
    publisher.failing = True
    first = placed()
    client.post(URL, json=first)
    publisher.failing = False

    client.post(URL, json=survey("cust_2"))
    assert list(store.outbox) == [first["event_id"]]  # not a leftover yet

    entry = store.outbox[first["event_id"]]
    store.outbox[entry.event_id] = dataclasses.replace(
        entry, created_at=entry.created_at - timedelta(minutes=2)
    )
    client.post(URL, json=survey("cust_3"))
    assert store.outbox == {}
    assert {row["event_id"] for row in rows(publisher.sent)} >= {first["event_id"]}
    assert publisher.sent.count(entry.messages[0]) == 1


def test_without_a_topic_nothing_is_exported(reference_model: str) -> None:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    assert app.extensions["cfi_publisher"] is None
    app.test_client().post(URL, json=placed())
    assert stored(app).outbox == {}


def test_a_topic_needs_a_project() -> None:
    with pytest.raises(ValueError, match="TOPIC needs PROJECT"):
        create_app({"TOPIC": "raw-changes"})


# --- export() on its own ---


def _entry(minutes_ago: int, now: datetime = T0) -> OutboxEntry:
    created = now - timedelta(minutes=minutes_ago)
    event_id = str(uuid.uuid4())
    return OutboxEntry(
        event_id, (message(f"events/{event_id}", {}, event_id, created),), created
    )


def test_leftovers_go_oldest_first_and_a_few_at_a_time() -> None:
    store, publisher = MemoryStore(outbox=True), MemoryPublisher()
    entries = [_entry(minutes) for minutes in range(2, 2 + LEFTOVERS_PER_REQUEST + 2)]
    store.outbox = {entry.event_id: entry for entry in entries}
    export(store, publisher, None, T0)
    oldest = sorted(entries, key=lambda entry: entry.created_at)
    assert publisher.sent == [e.messages[0] for e in oldest[:LEFTOVERS_PER_REQUEST]]
    assert set(store.outbox) == {e.event_id for e in oldest[LEFTOVERS_PER_REQUEST:]}


class FailsAfter(MemoryPublisher):
    """Takes the first n publishes, then fails every one after."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.calls, self.n = 0, n

    def publish(self, messages: Sequence[str]) -> None:
        self.calls += 1
        self.failing = self.calls > self.n
        super().publish(messages)


def test_export_stops_at_the_first_failure() -> None:
    store, publisher = MemoryStore(outbox=True), FailsAfter(1)
    entries = [_entry(minutes) for minutes in (5, 4, 3)]
    store.outbox = {entry.event_id: entry for entry in entries}
    export(store, publisher, None, T0)
    assert publisher.calls == 2  # one sent, one failed, none tried after
    assert set(store.outbox) == {entries[1].event_id, entries[2].event_id}


def test_an_outbox_document_fits_its_schema() -> None:
    document = outbox_document(_entry(0))
    assert set(document) == {"event_id", "messages", "created_at"}


# --- Against the emulators ---


@pytest.fixture
def topic() -> Iterator[tuple[str, str]]:
    """A new topic in the Pub/Sub emulator with a pull subscription: their names."""
    name = f"raw-changes-{uuid.uuid4().hex[:8]}"
    publisher, subscriber = PublisherClient(), SubscriberClient()
    topic_path = publisher.topic_path("demo-cfi", name)
    subscription = subscriber.subscription_path("demo-cfi", name)
    publisher.create_topic(name=topic_path)
    subscriber.create_subscription(name=subscription, topic=topic_path)
    yield name, subscription
    subscriber.delete_subscription(subscription=subscription)
    publisher.delete_topic(topic=topic_path)


def pulled(subscription: str, count: int) -> list[dict[str, Any]]:
    """At least count messages from subscription, acknowledged, as rows."""
    subscriber = SubscriberClient()
    got: list[dict[str, Any]] = []
    deadline = time.monotonic() + 30
    while len(got) < count and time.monotonic() < deadline:
        try:
            response = subscriber.pull(
                subscription=subscription, max_messages=100, timeout=5
            )
        except DeadlineExceeded:
            continue
        received = response.received_messages
        got += [json.loads(r.message.data) for r in received]
        if received:
            subscriber.acknowledge(
                subscription=subscription, ack_ids=[r.ack_id for r in received]
            )
    return got


@pytest.fixture
def exporting(reference_model: str, topic: tuple[str, str]) -> FlaskClient:
    app = create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "STORE": "firestore",
            "PROJECT": "demo-cfi",
            "TOPIC": topic[0],
        }
    )
    return app.test_client()


@needs_emulators
def test_saved_documents_reach_the_topic(
    exporting: FlaskClient, topic: tuple[str, str]
) -> None:
    tag = uuid.uuid4().hex[:8]
    event = placed(f"ord_{tag}", f"cust_{tag}")
    assert exporting.post(URL, json=event).status_code == 201
    got = [row for row in pulled(topic[1], 4) if row["event_id"] == event["event_id"]]
    assert sorted(row["collection"] for row in got) == [
        "customers",
        "events",
        "orders",
        "scores",
    ]
    db = firestore.Client(project="demo-cfi")
    order = db.document(f"orders/ord_{tag}").get().to_dict()
    sent = next(row for row in got if row["collection"] == "orders")
    assert json.loads(sent["document"])["basket_value"] == order["basket_value"]
    assert not db.document(f"outbox/{event['event_id']}").get().exists


@needs_emulators
def test_each_message_reaches_the_topic_with_its_collection(
    exporting: FlaskClient, topic: tuple[str, str]
) -> None:
    # The rules service's subscription takes score entries by it (T501).
    tag = uuid.uuid4().hex[:8]
    event = placed(f"ord_{tag}", f"cust_{tag}")
    assert exporting.post(URL, json=event).status_code == 201
    subscriber = SubscriberClient()
    seen: dict[str, dict[str, str]] = {}
    deadline = time.monotonic() + 30
    while len(seen) < 4 and time.monotonic() < deadline:
        try:
            response = subscriber.pull(
                subscription=topic[1], max_messages=100, timeout=5
            )
        except DeadlineExceeded:
            continue
        for received in response.received_messages:
            row = json.loads(received.message.data)
            if row["event_id"] == event["event_id"]:
                seen[row["collection"]] = dict(received.message.attributes)
        if response.received_messages:
            subscriber.acknowledge(
                subscription=topic[1],
                ack_ids=[r.ack_id for r in response.received_messages],
            )
    assert seen == {name: {"collection": name} for name in seen}
    assert sorted(seen) == ["customers", "events", "orders", "scores"]


@needs_emulators
def test_a_leftover_in_firestore_is_sent_by_a_later_request(
    exporting: FlaskClient, topic: tuple[str, str]
) -> None:
    db = firestore.Client(project="demo-cfi")
    leftover = _entry(2, now=datetime.now(UTC))
    db.document(f"outbox/{leftover.event_id}").set(outbox_document(leftover))

    tag = uuid.uuid4().hex[:8]
    assert exporting.post(URL, json=survey(f"cust_{tag}")).status_code == 201
    got = pulled(topic[1], 3)
    assert leftover.event_id in {row["event_id"] for row in got}
    assert not db.document(f"outbox/{leftover.event_id}").get().exists
