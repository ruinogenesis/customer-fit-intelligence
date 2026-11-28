"""T610: marketing consent per customer. A consent_changed event sets the
customer's consent, the latest answer wins, and the customer's document and
GET /customers/{id} carry it; a customer never asked has none.
"""

import os
import uuid
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.store import MemoryStore

URL = "/api/v1/events"


@pytest.fixture
def app(reference_model: str, monkeypatch: pytest.MonkeyPatch) -> Flask:
    for name in [name for name in os.environ if name.startswith("CFI_")]:
        monkeypatch.delenv(name)
    return create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def consent(marketing: bool, at: str, source: str = "sign_up") -> dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "type": "consent_changed",
        "customer_id": "cust_1",
        "occurred_at": at,
        "schema_version": 1,
        "payload": {"marketing": marketing, "source": source},
    }


def test_a_consent_event_is_stored_and_scores_nothing(client: FlaskClient) -> None:
    answer = client.post(URL, json=consent(True, "2026-11-28T06:00:00+07:00"))
    assert answer.status_code == 201
    body = answer.get_json()
    assert (body["order_id"], body["scores"], body["drivers"]) == (None, None, None)
    customer = client.get("/api/v1/customers/cust_1").get_json()
    assert customer["consent"] == {
        "marketing": True,
        "at": "2026-11-27T23:00:00+00:00",
        "source": "sign_up",
    }


def test_withdrawing_wins_and_an_older_answer_arriving_late_does_not(
    client: FlaskClient,
) -> None:
    client.post(URL, json=consent(True, "2026-11-28T06:00:00+00:00"))
    client.post(URL, json=consent(False, "2026-11-28T08:00:00+00:00", "settings"))
    client.post(URL, json=consent(True, "2026-11-28T07:00:00+00:00"))
    customer = client.get("/api/v1/customers/cust_1").get_json()
    assert customer["consent"]["marketing"] is False
    assert customer["consent"]["source"] == "settings"


def test_a_customer_saved_before_t610_still_reads_and_takes_consent(
    app: Flask, client: FlaskClient
) -> None:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    client.post(URL, json=consent(True, "2026-11-28T06:00:00+00:00"))
    del store.documents["customers/cust_1"]["consent"]  # as saved before T610
    assert "consent" not in client.get("/api/v1/customers/cust_1").get_json()
    client.post(URL, json=consent(False, "2026-11-28T08:00:00+00:00", "support"))
    customer = client.get("/api/v1/customers/cust_1").get_json()
    assert customer["consent"]["source"] == "support"


def test_a_bad_consent_event_is_refused(client: FlaskClient) -> None:
    bad = consent(True, "2026-11-28T06:00:00+00:00")
    bad["payload"]["marketing"] = "yes"
    answer = client.post(URL, json=bad)
    assert answer.status_code == 422
    assert "marketing" in answer.get_json()["errors"]["json"]["payload"]
