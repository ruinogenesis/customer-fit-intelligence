"""T202: POST /api/v1/events validates an event, stores it once and re-scores."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fitcore.features import INPUTS
from fitcore.model import LATENTS, SCORES, all_scores
from fitcore.scoring import score
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.store import MemoryStore

URL = "/api/v1/events"
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def memory(app: Flask) -> MemoryStore:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return store


def raw(
    kind: str, payload: dict[str, Any], minutes: int = 0, customer: str = "cust_1"
) -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": kind,
        "customer_id": customer,
        "occurred_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "schema_version": 1,
        "payload": payload,
    }


def placed(order_id: str = "ord_1", minutes: int = 0) -> dict[str, Any]:
    eta = T0 + timedelta(minutes=minutes + 30)
    payload = {
        "order_id": order_id,
        "restaurant_id": "rest_1",
        "basket_value": 50_000,
        "fees": 6_000,
        "promised_eta": eta.isoformat(),
        "promo_used": False,
    }
    return raw("order_placed", payload, minutes)


def delivered(order_id: str, minutes: int) -> dict[str, Any]:
    payload = {"order_id": order_id, "courier_id": "cour_1"}
    return raw("order_delivered", payload, minutes)


def test_a_valid_event_is_stored_and_scored(app: Flask, client: FlaskClient) -> None:
    event = placed()
    response = client.post(URL, json=event)
    assert response.status_code == 201
    body = response.get_json()
    params = app.extensions["cfi_model"].params
    assert body == {
        "event_id": event["event_id"],
        "customer_id": "cust_1",
        "duplicate": False,
        "order_id": "ord_1",
        "scores": score(params, {"x3": 0.0}).to_dict(),  # 6,000 / 50,000 = the prior
        "drivers": [],  # so nothing moved fit away from an order as expected
        "model_version": "t112-sim",
    }
    store = memory(app)
    assert set(store.events) == {event["event_id"]}
    assert store.customers["cust_1"].scores == score(params, {"x3": 0.0})
    assert [entry.order_id for entry in store.history["cust_1"]] == ["ord_1"]


def test_each_scoring_stores_its_split_and_answers_its_drivers(
    app: Flask, client: FlaskClient
) -> None:
    # T316: every score entry holds each input's share of each score, which
    # adds up to the score's distance from an order exactly as expected.
    answers = [
        client.post(URL, json=placed()).get_json(),
        client.post(URL, json=delivered("ord_1", 45)).get_json(),
    ]
    assert answers[1]["drivers"][0] == "x1"  # 15 minutes late
    entries = client.get("/api/v1/customers/cust_1/scores").get_json()["scores"]
    baseline = all_scores(
        app.extensions["cfi_model"].params, np.zeros((1, len(INPUTS)))
    )[0]
    for entry, answer in zip(reversed(entries), answers, strict=True):
        assert entry["drivers"] == answer["drivers"]
        shares = entry["contributions"]
        assert set(shares) == set(SCORES)  # JSON and Firestore keep no key order
        stored = [*(entry["scores"][name] for name in LATENTS), entry["churn_risk"]]
        for k, name in enumerate(SCORES):
            assert set(shares[name]) == set(INPUTS)
            total = sum(shares[name].values())
            assert total == pytest.approx(stored[k] - baseline[k], abs=1e-12)


def test_a_late_delivery_moves_the_customers_scores(
    app: Flask, client: FlaskClient
) -> None:
    before = client.post(URL, json=placed()).get_json()["scores"]
    after = client.post(URL, json=delivered("ord_1", 45)).get_json()["scores"]
    assert after["fit"] < before["fit"]
    assert after["churn_risk"] > before["churn_risk"]
    customer = memory(app).customers["cust_1"]
    assert customer.scores is not None
    assert customer.scores.to_dict() == after


def test_the_quadrant_follows_the_customers_scores(
    app: Flask, client: FlaskClient
) -> None:
    placed_scores = client.post(URL, json=placed()).get_json()["scores"]
    assert placed_scores["quadrant"] == "lovemark"
    client.post(URL, json=delivered("ord_1", 45))
    rating = raw("rating_submitted", {"order_id": "ord_1", "courier": 2, "food": 5}, 60)
    rated = client.post(URL, json=rating).get_json()["scores"]
    assert rated["respect"] < 0.5 <= rated["love"]
    assert rated["quadrant"] == "fad"
    customer = memory(app).customers["cust_1"].scores
    assert customer is not None and customer.quadrant == "fad"


def test_an_event_that_measures_nothing_is_stored_without_scores(
    app: Flask, client: FlaskClient
) -> None:
    event = raw("survey_response", {"item_id": "love", "value": 6})
    response = client.post(URL, json=event)
    assert response.status_code == 201
    body = response.get_json()
    assert (body["order_id"], body["scores"]) == (None, None)
    assert event["event_id"] in memory(app).events


def test_resending_an_event_returns_its_first_result(
    app: Flask, client: FlaskClient
) -> None:
    event = placed()
    first = client.post(URL, json=event)
    again = client.post(URL, json=event)
    assert again.status_code == 200
    assert again.get_json() == {**first.get_json(), "duplicate": True}
    store = memory(app)
    assert len(store.events) == 1
    assert len(store.history["cust_1"]) == 1


def test_an_event_id_used_for_another_event_is_a_conflict(
    app: Flask, client: FlaskClient
) -> None:
    event = placed()
    client.post(URL, json=event)
    response = client.post(URL, json={**placed("ord_2"), "event_id": event["event_id"]})
    assert response.status_code == 409
    assert response.get_json()["code"] == "conflict"
    assert set(memory(app).orders) == {"ord_1"}


def test_an_event_that_does_not_fit_is_a_conflict(
    app: Flask, client: FlaskClient
) -> None:
    response = client.post(URL, json=delivered("ord_9", 45))
    assert response.status_code == 409
    assert response.get_json() == {
        "code": "conflict",
        "message": "Order 'ord_9' has not been placed.",
    }
    assert memory(app).events == {}


def _invalid() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Name: (request body, the path to its error under errors.json)."""
    bad_issue = raw("order_issue", {"order_id": "ord_1", "issue_type": "rude"})
    return {
        "unknown type": ({**placed(), "type": "order_teleported"}, ("type",)),
        "missing customer": (
            {k: v for k, v in placed().items() if k != "customer_id"},
            ("customer_id",),
        ),
        "no timezone": (
            {**placed(), "occurred_at": "2026-11-14T12:00:00"},
            ("occurred_at",),
        ),
        "slash in an id": ({**placed(), "customer_id": "a/b/c"}, ("customer_id",)),
        "bad payload": (bad_issue, ("payload", "issue_type")),
        "not an object": ([placed()], ("_schema",)),
    }


@pytest.mark.parametrize("name", list(_invalid()))
def test_an_invalid_event_gets_a_clear_error(
    app: Flask, client: FlaskClient, name: str
) -> None:
    body, path = _invalid()[name]
    response = client.post(URL, json=body)
    assert response.status_code == 422
    error = response.get_json()
    assert error["code"] == "unprocessable_entity"
    assert error["message"] == "The request is not valid; see errors for the details."
    details = error["errors"]["json"]
    for key in path:
        details = details[key]
    assert details
    assert memory(app).events == {}


def test_without_a_model_events_are_refused() -> None:
    app = create_app()
    response = app.test_client().post(URL, json=placed())
    assert response.status_code == 503
    assert response.get_json()["code"] == "service_unavailable"
    assert memory(app).events == {}


def test_settings_come_from_cfi_environment_variables(
    monkeypatch: pytest.MonkeyPatch, reference_model: str
) -> None:
    monkeypatch.setenv("CFI_MODEL_PATH", reference_model)
    monkeypatch.setenv("CFI_MODEL_VERSION", "from-env")
    health = create_app().test_client().get("/api/v1/health").get_json()
    assert health["model_version"] == "from-env"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"MODEL_PATH": "model.json"}, "together"),
        ({"MODEL_VERSION": "v1"}, "together"),
        ({"STORE": "postgres"}, "STORE"),
    ],
)
def test_bad_settings_stop_the_app_from_starting(
    config: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_app(config)


def test_a_bad_model_file_stops_the_app_from_starting(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="model.json"):
        create_app({"MODEL_PATH": str(path), "MODEL_VERSION": "v1"})


def test_the_openapi_document_describes_events(client: FlaskClient) -> None:
    spec = client.get("/api/v1/openapi.json").get_json()
    post = spec["paths"]["/api/v1/events"]["post"]
    body = post["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/Event"}
    assert {"200", "201", "409", "422", "503"} <= set(post["responses"])
    event = spec["components"]["schemas"]["Event"]
    assert {"event_id", "type", "customer_id", "payload"} <= set(event["required"])
    quadrant = spec["components"]["schemas"]["Scores"]["properties"]["quadrant"]
    assert quadrant["enum"] == ["lovemark", "brand", "fad", "product"]
