"""T205: a write key on every request but a read, until T303's sign-in."""

import pytest
from fitcore.sim_events import simulate_events
from flask import Flask
from flask.testing import FlaskClient

from cfi_api import create_app
from cfi_api.store import MemoryStore
from cfi_api.write_key import HEADER, REFUSED

KEY = "a-long-random-key-from-secret-manager"


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app(
        {"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim", "WRITE_KEY": KEY}
    )


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


EVENT = simulate_events(1, seed=1, days=7.0)[0]


@pytest.mark.parametrize("headers", [{}, {HEADER: ""}, {HEADER: "wrong"}])
@pytest.mark.parametrize("path", ["/api/v1/events", "/api/v1/score"])
def test_a_post_without_the_key_is_refused_and_changes_nothing(
    app: Flask, client: FlaskClient, path: str, headers: dict[str, str]
) -> None:
    response = client.post(path, json=EVENT, headers=headers)
    assert response.status_code == 401
    assert response.get_json() == {"code": "unauthorized", "message": REFUSED}
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    assert store.find_event(EVENT["event_id"]) is None


def test_a_post_with_the_key_goes_through(client: FlaskClient) -> None:
    headers = {HEADER: KEY}
    assert client.post("/api/v1/score", json=EVENT, headers=headers).status_code == 200
    assert client.post("/api/v1/events", json=EVENT, headers=headers).status_code == 201
    assert client.post("/api/v1/events", json=EVENT, headers=headers).status_code == 200


def test_reads_need_no_key(client: FlaskClient) -> None:
    assert client.get("/api/v1/health").get_json()["model_version"] == "t112-sim"
    assert client.get("/api/v1/openapi.json").status_code == 200


def test_the_key_can_come_from_the_environment(
    reference_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFI_WRITE_KEY", KEY)
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    client = app.test_client()
    assert client.post("/api/v1/events", json=EVENT).status_code == 401
    assert (
        client.post("/api/v1/events", json=EVENT, headers={HEADER: KEY}).status_code
        == 201
    )


def test_without_a_key_nothing_is_checked(reference_model: str) -> None:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    assert app.test_client().post("/api/v1/events", json=EVENT).status_code == 201


def test_an_empty_key_stops_the_app_from_starting() -> None:
    with pytest.raises(ValueError, match="WRITE_KEY must not be empty"):
        create_app({"WRITE_KEY": ""})


def test_an_all_digit_key_from_the_environment_stops_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flask reads CFI_* values as JSON when it can: this one becomes an int.
    monkeypatch.setenv("CFI_WRITE_KEY", "12345678")
    with pytest.raises(TypeError, match="WRITE_KEY must be text"):
        create_app()


def test_the_openapi_document_asks_for_the_key_on_posts_only(
    client: FlaskClient,
) -> None:
    spec = client.get("/api/v1/openapi.json").get_json()
    scheme = spec["components"]["securitySchemes"]["writeKey"]
    assert scheme["type"] == "apiKey"
    assert (scheme["in"], scheme["name"]) == ("header", HEADER)
    for path in ("/api/v1/events", "/api/v1/score"):
        post = spec["paths"][path]["post"]
        assert post["security"] == [{"writeKey": []}]
        assert "401" in post["responses"]
    assert "security" not in spec["paths"]["/api/v1/health"]["get"]
