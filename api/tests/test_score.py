"""T203: POST /api/v1/score answers as /events would, and saves nothing."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.store import MemoryStore

SCORE = "/api/v1/score"
EVENTS = "/api/v1/events"
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def event(kind: str, payload: dict[str, Any], minutes: int = 0) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": kind,
        "customer_id": "cust_1",
        "occurred_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


def placed() -> dict[str, Any]:
    payload = {
        "order_id": "ord_1",
        "restaurant_id": "rest_1",
        "basket_value": 50_000,
        "fees": 6_000,
        "promised_eta": (T0 + timedelta(minutes=30)).isoformat(),
        "promo_used": False,
    }
    return event("order_placed", payload)


def delivered(minutes: int, order_id: str = "ord_1") -> dict[str, Any]:
    return event("order_delivered", {"order_id": order_id, "courier_id": "c1"}, minutes)


def memory(app: Flask) -> MemoryStore:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return store


def watch_saves(app: Flask, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every save from now on, recorded (and still done)."""
    store = memory(app)
    calls: list[Any] = []
    save = store.save

    def recorded(*args: Any) -> None:
        calls.append(args)
        save(*args)

    monkeypatch.setattr(store, "save", recorded)
    return calls


def test_a_what_if_is_scored_and_nothing_is_saved(
    app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = client.post(EVENTS, json=placed()).get_json()["scores"]
    saves = watch_saves(app, monkeypatch)
    late = delivered(45)
    response = client.post(SCORE, json=late)
    assert response.status_code == 200
    body = response.get_json()
    assert (body["order_id"], body["duplicate"]) == ("ord_1", False)
    assert body["scores"]["fit"] < now["fit"]
    assert saves == []
    store = memory(app)
    assert late["event_id"] not in store.events
    assert set(store.orders["ord_1"].gaps) == {"x3"}
    customer = store.customers["cust_1"].scores
    assert customer is not None and customer.to_dict() == now


def test_the_answer_is_what_events_then_gives(client: FlaskClient) -> None:
    client.post(EVENTS, json=placed())
    late = delivered(45)
    what_if = client.post(SCORE, json=late)
    real = client.post(EVENTS, json=late)
    assert (what_if.status_code, real.status_code) == (200, 201)
    assert what_if.get_json() == real.get_json()


def test_the_same_what_if_can_be_scored_again(
    app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(EVENTS, json=placed())
    saves = watch_saves(app, monkeypatch)
    late = delivered(45)
    first = client.post(SCORE, json=late)
    again = client.post(SCORE, json=late)
    assert (first.status_code, again.status_code) == (200, 200)
    assert again.get_json() == first.get_json()
    assert saves == []


def test_an_event_already_received_gets_its_first_result(
    app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(EVENTS, json=placed())
    late = delivered(45)
    first = client.post(EVENTS, json=late).get_json()
    saves = watch_saves(app, monkeypatch)
    response = client.post(SCORE, json=late)
    assert response.status_code == 200
    assert response.get_json() == {**first, "duplicate": True}
    assert saves == []


def test_refusals_are_those_of_events(client: FlaskClient) -> None:
    first = placed()
    client.post(EVENTS, json=first)
    unknown_order = client.post(SCORE, json=delivered(45, order_id="ord_9"))
    assert unknown_order.status_code == 409
    assert unknown_order.get_json()["message"] == "Order 'ord_9' has not been placed."
    reused_id = client.post(
        SCORE, json={**delivered(45), "event_id": first["event_id"]}
    )
    assert (reused_id.status_code, reused_id.get_json()["code"]) == (409, "conflict")
    invalid = client.post(SCORE, json=event("order_issue", {"order_id": "ord_1"}))
    assert invalid.status_code == 422
    assert "issue_type" in invalid.get_json()["errors"]["json"]["payload"]


def test_without_a_model_a_what_if_is_refused() -> None:
    response = create_app().test_client().post(SCORE, json=placed())
    assert response.status_code == 503
    assert response.get_json()["code"] == "service_unavailable"


def test_the_openapi_document_describes_score(client: FlaskClient) -> None:
    spec = client.get("/api/v1/openapi.json").get_json()
    post = spec["paths"]["/api/v1/score"]["post"]
    body = post["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/Event"}
    assert {"200", "409", "422", "503"} <= set(post["responses"])
    assert "201" not in post["responses"]
