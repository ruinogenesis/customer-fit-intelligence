"""T302: one event's reads and writes are one unit, so events for one
customer at the same moment never erase each other's changes."""

import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fitcore.scoring import Applied, Customer, Order
from flask import Flask
from google.api_core.exceptions import Aborted

from cfi_api import create_app
from cfi_api.firestore_store import gave_up
from cfi_api.store import Busy, EventRecord, Handled, MemoryStore, Work

T0 = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
ROUND_TRIP = 0.01  # seconds, about one Firestore call from Cloud Run


def _event(kind: str, payload: dict[str, Any], minutes: int) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": kind,
        "customer_id": "cust_1",
        "occurred_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


PLACED = {
    "order_id": "ord_1",
    "restaurant_id": "rest_1",
    "basket_value": 50_000,
    "fees": 6_000,
    "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
    "promo_used": False,
}


class SlowStore(MemoryStore):
    """A MemoryStore whose reads and saves take as long as Firestore's, so
    the gap between reading and writing is as wide as in production."""

    def find_event(self, event_id: str) -> EventRecord | None:
        time.sleep(ROUND_TRIP)
        return super().find_event(event_id)

    def find_customer(self, customer_id: str) -> Customer | None:
        time.sleep(ROUND_TRIP)
        return super().find_customer(customer_id)

    def find_order(self, order_id: str) -> Order | None:
        time.sleep(ROUND_TRIP)
        return super().find_order(order_id)

    def save(self, record: EventRecord, applied: Applied, model_version: str) -> None:
        time.sleep(ROUND_TRIP)
        super().save(record, applied, model_version)


@pytest.fixture
def slow_app(reference_model: str) -> tuple[Flask, SlowStore]:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    store = SlowStore()
    app.extensions["cfi_store"] = store
    return app, store


def at_once(app: Flask, events: list[dict[str, Any]]) -> list[int]:
    """POST every event from its own thread, all released together."""
    start = threading.Barrier(len(events))
    statuses: list[int] = []

    def send(event: dict[str, Any]) -> None:
        client = app.test_client()
        start.wait()
        statuses.append(client.post("/api/v1/events", json=event).status_code)

    threads = [threading.Thread(target=send, args=(e,)) for e in events]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return statuses


def test_events_for_one_customer_at_once_all_count(
    slow_app: tuple[Flask, SlowStore],
) -> None:
    app, store = slow_app
    assert (
        app.test_client()
        .post("/api/v1/events", json=_event("order_placed", PLACED, 0))
        .status_code
        == 201
    )
    ratings = [
        _event("rating_submitted", {"order_id": "ord_1", "courier": 5}, 40)
        for _ in range(20)
    ]
    assert at_once(app, ratings) == [201] * 20
    assert store.customers["cust_1"].baselines["x5"].weight == 20
    assert len(store.history["cust_1"]) == 21  # the order, then each rating


def test_the_same_event_at_once_is_saved_once(
    slow_app: tuple[Flask, SlowStore],
) -> None:
    app, store = slow_app
    event = _event("order_placed", PLACED, 0)
    assert sorted(at_once(app, [event] * 5)) == [200, 200, 200, 200, 201]
    assert len(store.events) == 1
    assert store.customers["cust_1"].baselines["x3"].weight == 1


class BusyStore(MemoryStore):
    def handle(self, event: dict[str, Any], work: Work) -> Handled:
        raise Busy("too much contention")


def test_a_busy_store_asks_the_sender_to_try_again(reference_model: str) -> None:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    store = BusyStore()
    app.extensions["cfi_store"] = store
    response = app.test_client().post(
        "/api/v1/events", json=_event("order_placed", PLACED, 0)
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.get_json()["code"] == "service_unavailable"
    assert "send it again" in response.get_json()["message"]
    assert store.events == {}


def test_only_contention_counts_as_giving_up() -> None:
    aborted = Aborted("too much contention")
    try:
        raise ValueError("Failed to commit transaction in 10 attempts.") from aborted
    except ValueError as err:
        exhausted = err
    assert gave_up(aborted)
    assert gave_up(exhausted)
    assert not gave_up(ValueError("the event is about another customer"))
