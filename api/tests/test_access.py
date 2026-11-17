"""T303: staff roles on the API, and the integration key on POST /events.

The tokens here are the Auth emulator's kind: unsigned, so the tests mint
their own and need no emulator running. firebase-admin accepts them only
while FIREBASE_AUTH_EMULATOR_HOST is set, and the app only for a demo-
project.
"""

import base64
import json
import time
from typing import Any

import pytest
from firebase_admin import credentials
from fitcore.documents import ROLES
from fitcore.sim_events import simulate_events
from flask import Flask
from flask.testing import FlaskClient
from google.auth.credentials import AnonymousCredentials

from cfi_api import create_app
from cfi_api.access import ACCESS, EMULATOR, KEY_HEADER, RANKS
from cfi_api.store import MemoryStore

KEY = "a-long-random-key-from-secret-manager"
PROJECT = "demo-cfi"
EVENTS = "/api/v1/events"
SCORE = "/api/v1/score"
EVENT = simulate_events(1, seed=1, days=7.0)[0]


def _part(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def token(role: object = None, project: str = PROJECT) -> str:
    """An emulator ID token for project, with role as its role claim if set."""
    now = int(time.time())
    claims = {
        "iss": f"https://securetoken.google.com/{project}",
        "aud": project,
        "sub": "staff-1",
        "iat": now,
        "exp": now + 3600,
        "auth_time": now,
    }
    if role is not None:
        claims["role"] = role
    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}."


def bearer(role: object = None, project: str = PROJECT) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(role, project)}"}


@pytest.fixture(autouse=True)
def emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EMULATOR, "127.0.0.1:9099")


@pytest.fixture
def app(reference_model: str) -> Flask:
    return create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "AUTH": "firebase",
            "PROJECT": PROJECT,
            "WRITE_KEY": KEY,
        }
    )


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def saved(app: Flask) -> bool:
    store = app.extensions["cfi_store"]
    assert isinstance(store, MemoryStore)
    return store.find_event(EVENT["event_id"]) is not None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {KEY_HEADER: ""},
        {KEY_HEADER: "wrong"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": f"Basic {KEY}"},
        bearer("admin", project="demo-other"),
    ],
    ids=["none", "empty key", "wrong key", "no token", "bad token", "basic", "other"],
)
@pytest.mark.parametrize("path", [EVENTS, SCORE])
def test_a_post_without_a_valid_token_is_refused_and_changes_nothing(
    app: Flask, client: FlaskClient, path: str, headers: dict[str, str]
) -> None:
    response = client.post(path, json=EVENT, headers=headers)
    assert response.status_code == 401
    assert response.get_json()["code"] == "unauthorized"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert not saved(app)


@pytest.mark.parametrize(
    ("path", "role", "status"),
    [
        (SCORE, None, 403),
        (SCORE, "viewer", 403),
        (SCORE, "analyst", 200),
        (SCORE, "admin", 200),
        (EVENTS, None, 403),
        (EVENTS, "viewer", 403),
        (EVENTS, "analyst", 403),
        (EVENTS, "admin", 201),
        (SCORE, "owner", 403),
        (SCORE, ["admin"], 403),
    ],
)
def test_a_role_lets_in_its_own_requests_and_those_below_it(
    client: FlaskClient, path: str, role: object, status: int
) -> None:
    response = client.post(path, json=EVENT, headers=bearer(role))
    assert response.status_code == status
    if status == 403:
        assert response.get_json()["code"] == "forbidden"


def test_an_unsigned_token_is_refused_outside_the_emulator(
    reference_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EMULATOR)
    # firebase-admin wants credentials before it checks a token, though it
    # uses none to check one: stand-ins, as this machine may have none.
    monkeypatch.setattr(
        credentials.ApplicationDefault,
        "get_credential",
        lambda self: AnonymousCredentials(),
    )
    app = create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "AUTH": "firebase",
            "PROJECT": PROJECT,
        }
    )
    response = app.test_client().post(SCORE, json=EVENT, headers=bearer("admin"))
    assert response.status_code == 401


def test_the_key_opens_post_events_only(app: Flask, client: FlaskClient) -> None:
    assert client.post(SCORE, json=EVENT, headers={KEY_HEADER: KEY}).status_code == 401
    assert client.post(EVENTS, json=EVENT, headers={KEY_HEADER: KEY}).status_code == 201
    assert client.post(EVENTS, json=EVENT, headers={KEY_HEADER: KEY}).status_code == 200
    assert saved(app)


def test_a_wrong_key_with_an_admin_token_still_gets_in(client: FlaskClient) -> None:
    headers = {KEY_HEADER: "wrong", **bearer("admin")}
    assert client.post(EVENTS, json=EVENT, headers=headers).status_code == 201


def test_without_a_write_key_events_need_an_admin(reference_model: str) -> None:
    app = create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "AUTH": "firebase",
            "PROJECT": PROJECT,
        }
    )
    client = app.test_client()
    # No key is set, so no header can match it, not even an empty one.
    assert client.post(EVENTS, json=EVENT, headers={KEY_HEADER: ""}).status_code == 401
    assert client.post(EVENTS, json=EVENT, headers=bearer("admin")).status_code == 201


def test_reads_and_options_need_nothing(client: FlaskClient) -> None:
    assert client.get("/api/v1/health").get_json()["model_version"] == "t112-sim"
    assert client.head("/api/v1/health").status_code == 200
    assert client.get("/api/v1/openapi.json").status_code == 200
    assert client.get("/api/v1/docs").status_code == 200
    assert client.options(EVENTS).status_code == 200


def test_the_dashboard_reads_need_a_staff_role(client: FlaskClient) -> None:
    assert client.post(EVENTS, json=EVENT, headers=bearer("admin")).status_code == 201
    customer = f"/api/v1/customers/{EVENT['customer_id']}"
    reads = ["/api/v1/customers", customer, f"{customer}/scores"]
    for path in [*reads, f"{customer}/orders", "/api/v1/matrix"]:
        assert client.get(path).status_code == 401
        assert client.get(path, headers=bearer()).status_code == 403  # no role
        assert client.get(path, headers=bearer("viewer")).status_code == 200


def test_a_path_nobody_serves_answers_404_or_405(client: FlaskClient) -> None:
    assert client.post("/api/v1/nothing-here").status_code == 404
    assert client.post("/api/v1/health").status_code == 405


def test_a_request_access_does_not_list_needs_an_admin(app: Flask) -> None:
    app.add_url_rule("/unlisted", "unlisted", lambda: "done", methods=["POST"])
    client = app.test_client()
    assert client.post("/unlisted", headers=bearer("analyst")).status_code == 403
    assert client.post("/unlisted", headers=bearer("admin")).status_code == 200


def test_access_lists_every_request_the_api_serves(app: Flask) -> None:
    served = {
        (rule.endpoint, method)
        for rule in app.url_map.iter_rules()
        for method in rule.methods or ()
        if method not in {"HEAD", "OPTIONS"}
    }
    assert served == set(ACCESS)


def test_every_role_has_a_rank() -> None:
    assert set(RANKS) == set(ROLES)


def test_settings_can_come_from_the_environment(
    reference_model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFI_AUTH", "firebase")
    monkeypatch.setenv("CFI_PROJECT", PROJECT)
    monkeypatch.setenv("CFI_WRITE_KEY", KEY)
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    client = app.test_client()
    assert client.post(EVENTS, json=EVENT).status_code == 401
    assert client.post(EVENTS, json=EVENT, headers={KEY_HEADER: KEY}).status_code == 201


def test_without_auth_nothing_is_checked(reference_model: str) -> None:
    app = create_app({"MODEL_PATH": reference_model, "MODEL_VERSION": "t112-sim"})
    assert app.test_client().post(EVENTS, json=EVENT).status_code == 201


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        ({"AUTH": "google"}, ValueError, "AUTH must be unset or 'firebase'"),
        ({"WRITE_KEY": KEY}, ValueError, "WRITE_KEY needs AUTH"),
        ({"AUTH": "firebase", "WRITE_KEY": ""}, ValueError, "must not be empty"),
        ({"AUTH": "firebase", "WRITE_KEY": 12345678}, TypeError, "must be text"),
        ({"AUTH": "firebase", "PROJECT": "cfi-dev-th26"}, ValueError, "demo-"),
        ({"AUTH": "firebase"}, ValueError, "demo-"),
    ],
)
def test_a_setting_that_would_check_too_little_stops_the_app(
    config: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        create_app(config)


def test_an_all_digit_key_from_the_environment_stops_the_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Flask reads CFI_* values as JSON when it can: this one becomes an int.
    monkeypatch.setenv("CFI_AUTH", "firebase")
    monkeypatch.setenv("CFI_PROJECT", PROJECT)
    monkeypatch.setenv("CFI_WRITE_KEY", "12345678")
    with pytest.raises(TypeError, match="WRITE_KEY must be text"):
        create_app()


def test_the_openapi_document_names_who_may_post(client: FlaskClient) -> None:
    spec = client.get("/api/v1/openapi.json").get_json()
    schemes = spec["components"]["securitySchemes"]
    assert (schemes["firebase"]["type"], schemes["firebase"]["scheme"]) == (
        "http",
        "bearer",
    )
    assert (schemes["writeKey"]["in"], schemes["writeKey"]["name"]) == (
        "header",
        KEY_HEADER,
    )
    events = spec["paths"][EVENTS]["post"]
    assert events["security"] == [{"firebase": []}, {"writeKey": []}]
    score = spec["paths"][SCORE]["post"]
    assert score["security"] == [{"firebase": []}]
    for post in (events, score):
        assert {"401", "403"} <= set(post["responses"])
    assert "security" not in spec["paths"]["/api/v1/health"]["get"]
