"""T501: the rules service. Pub/Sub pushes each new score entry; the engine
decides, and the service answers 204, or 500 so Pub/Sub sends it again.
Anything that is not a change message is dropped, as no retry could fix it.
T505: Cloud Scheduler calls the daily sweep; 200 with how many it decided,
or 500 so Scheduler tries again.
"""

import base64
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from fitcore.sim_events import simulate_events
from flask import Flask

from cfi_api import create_app, rules
from cfi_api.engine import decide
from cfi_api.export import MemoryPublisher
from cfi_api.rules_service import create_rules_app, scoring
from cfi_api.store import MemoryStore

T0 = datetime(2026, 11, 28, 6, 0, tzinfo=UTC)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if name.startswith("CFI_")]:
        monkeypatch.delenv(name)


def envelope(row: dict[str, Any]) -> dict[str, Any]:
    """Pub/Sub's push body for one message holding row."""
    data = base64.b64encode(json.dumps(row).encode()).decode()
    return {
        "message": {"data": data, "messageId": "1", "attributes": {}},
        "subscription": "projects/demo-cfi/subscriptions/raw-changes-rules",
    }


@pytest.fixture
def store() -> MemoryStore:
    store = MemoryStore()
    rules.seed(store, T0)
    store.set_rule_status("fee-sensitivity", "dry_run", T0)
    # Fires on any lovemark order whose fees cost fit anything at all.
    path = "rules/fee-sensitivity"
    condition = {"quadrant": "lovemark", "min_fee_drag": 0.0}
    store.documents[path] = {**store.documents[path], "condition": condition}
    return store


@pytest.fixture
def rules_app(store: MemoryStore, monkeypatch: pytest.MonkeyPatch) -> Flask:
    _clean_env(monkeypatch)
    app = create_rules_app()
    app.extensions["cfi_store"] = store
    return app


def consent_event(customer_id: str, marketing: bool) -> dict[str, Any]:
    """customer_id says yes or no to marketing at sign-up, before any order."""
    return {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{customer_id}/consent")),
        "type": "consent_changed",
        "customer_id": customer_id,
        "occurred_at": "2025-12-31T00:00:00+00:00",
        "schema_version": 1,
        "payload": {"marketing": marketing, "source": "sign_up"},
    }


def changes(
    reference_model: str, store: MemoryStore, consents: tuple[bool, bool] = (True, True)
) -> list[dict[str, Any]]:
    """The rows two simulated customers' events send, posted through the API
    into store, each customer saying yes or no to marketing first (T610)."""
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    app.extensions["cfi_store"] = store
    store.has_outbox = True
    publisher = app.extensions["cfi_publisher"] = MemoryPublisher()
    client = app.test_client()
    answers = [consent_event(f"sim501-{n:06d}", yes) for n, yes in enumerate(consents)]
    for event in [*answers, *simulate_events(2, seed=501, days=14.0)]:
        assert client.post("/api/v1/events", json=event).status_code in (200, 201)
    return [json.loads(message) for message in publisher.sent]


def test_every_score_entry_pushed_is_decided_on_once_per_order(
    rules_app: Flask, store: MemoryStore, reference_model: str
) -> None:
    rows = changes(reference_model, store)
    scorings = [row for row in rows if row["collection"] == "scores"]
    orders = {json.loads(row["document"])["order_id"] for row in scorings}
    client = rules_app.test_client()
    for row in rows:
        assert client.post("/push", json=envelope(row)).status_code == 204
    actions = store.list_actions(1000)
    assert 0 < len(actions) <= len(orders)
    assert {a["action_id"].split("--")[1] for a in actions} <= orders
    # The same as deciding directly; pushing everything again adds nothing.
    assert all(decide(store, scoring(row), T0) == [] for row in scorings)
    for row in rows:
        client.post("/push", json=envelope(row))
    assert len(store.list_actions(1000)) == len(actions)


@pytest.mark.parametrize(
    ("consents", "expected"),
    [((False, True), {"sim501-000001"}), ((True, False), set())],
    ids=["the one who fires said yes", "the one who fires said no"],
)
def test_only_customers_with_marketing_consent_get_decisions(
    rules_app: Flask,
    store: MemoryStore,
    reference_model: str,
    consents: tuple[bool, bool],
    expected: set[str],
) -> None:
    # In these 14 days only sim501-000001's orders make the rule fire.
    rows = changes(reference_model, store, consents=consents)
    client = rules_app.test_client()
    for row in rows:
        assert client.post("/push", json=envelope(row)).status_code == 204
    decided = {action["customer_id"] for action in store.list_actions(1000)}
    assert decided == expected


def test_other_documents_are_acknowledged_and_left_alone(
    rules_app: Flask, store: MemoryStore
) -> None:
    row = {"collection": "orders", "path": "orders/ord_1", "document": "{}"}
    assert rules_app.test_client().post("/push", json=envelope(row)).status_code == 204
    assert store.list_actions(10) == []


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"message": {}},
        {"message": {"data": "not base64!"}},
        {"message": {"data": base64.b64encode(b"not json").decode()}},
        envelope({"path": "customers/c/scores/e"}),
        envelope({"collection": "scores", "path": "customers/c", "document": "{}"}),
        envelope({"collection": "scores", "path": "customers/c/scores/e"}),
        envelope(
            {"collection": "scores", "path": "customers/c/scores/e", "document": "x"}
        ),
    ],
    ids=[
        "no body",
        "empty",
        "no data",
        "bad base64",
        "not json",
        "no collection",
        "not a score path",
        "no document",
        "bad document",
    ],
)
def test_a_push_that_is_not_a_change_is_dropped_and_logged(
    rules_app: Flask, store: MemoryStore, body: Any, caplog: pytest.LogCaptureFixture
) -> None:
    client = rules_app.test_client()
    with caplog.at_level(logging.ERROR, logger="cfi_api.rules_service"):
        answer = (
            client.post("/push", json=body)
            if body is not None
            else client.post("/push")
        )
    assert answer.status_code == 204
    assert "Dropped a push" in caplog.text
    assert store.list_actions(10) == []


def test_when_deciding_fails_pub_sub_is_asked_to_send_it_again(
    rules_app: Flask, store: MemoryStore, reference_model: str
) -> None:
    row = next(
        r for r in changes(reference_model, store) if r["collection"] == "scores"
    )

    def broken() -> list[dict[str, Any]]:
        raise ConnectionError("Firestore is unreachable")

    store.list_rules = broken  # type: ignore[method-assign]
    assert rules_app.test_client().post("/push", json=envelope(row)).status_code == 500


def test_the_daily_sweep_decides_and_says_how_many(
    rules_app: Flask, store: MemoryStore, reference_model: str
) -> None:
    # Two customers who said yes, with no order since January (T505).
    changes(reference_model, store)
    store.set_rule_status("win-back", "dry_run", T0)
    path = "rules/win-back"
    store.documents[path] = {**store.documents[path], "condition": {"min_churn": 0.0}}
    client = rules_app.test_client()
    first = client.post("/sweep")
    assert (first.status_code, first.get_json()) == (200, {"decided": 2})
    assert {a["rule_id"] for a in store.list_actions(10)} == {"win-back"}
    again = client.post("/sweep")  # as when Scheduler tries again
    assert (again.status_code, again.get_json()) == (200, {"decided": 0})


def test_when_sweeping_fails_scheduler_is_asked_to_try_again(
    rules_app: Flask, store: MemoryStore
) -> None:
    def broken() -> list[dict[str, Any]]:
        raise ConnectionError("Firestore is unreachable")

    store.list_rules = broken  # type: ignore[method-assign]
    assert rules_app.test_client().post("/sweep").status_code == 500


def test_the_service_has_two_endpoints(rules_app: Flask) -> None:
    client = rules_app.test_client()
    assert client.get("/push").status_code == 405
    assert client.get("/sweep").status_code == 405
    assert client.post("/api/v1/events", json={}).status_code == 404


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("CFI_STORE", "nowhere")
    with pytest.raises(ValueError, match="STORE must be"):
        create_rules_app()
