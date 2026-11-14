"""T206: Firestore rules. Staff with a role claim may read customers, nothing more.

These run against the Firestore emulator only; elsewhere they are skipped.
Each request carries an unsigned token built as the JS SDK's mockUserToken
builds it: the emulator accepts one, production never would. Documents are
seeded with the Python client, which the emulator lets past the rules.
"""

import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

import pytest
from google.cloud import firestore

PROJECT = "demo-cfi"

pytestmark = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)


def _encoded(part: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()


def token(**claims: Any) -> str:
    """An unsigned ID token for user "tester" with these extra claims."""
    payload = {
        "iss": f"https://securetoken.google.com/{PROJECT}",
        "aud": PROJECT,
        "iat": 0,
        "exp": 3600,
        "auth_time": 0,
        "sub": "tester",
        "user_id": "tester",
        "firebase": {"sign_in_provider": "custom", "identities": {}},
        **claims,
    }
    return ".".join([_encoded({"alg": "none", "type": "JWT"}), _encoded(payload), ""])


def status(
    method: str, path: str, bearer: str | None, body: dict[str, Any] | None = None
) -> int:
    """The HTTP status the emulator's REST API answers for one request."""
    host = os.environ["FIRESTORE_EMULATOR_HOST"]
    url = f"http://{host}/v1/projects/{PROJECT}/databases/(default)/documents/{path}"
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if bearer is not None:
        request.add_header("Authorization", f"Bearer {bearer}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            code: int = response.status
            return code
    except urllib.error.HTTPError as err:
        return err.code


@pytest.fixture
def seeded() -> dict[str, str]:
    """One customer with an order, an event and a history entry; their paths."""
    tag = uuid.uuid4().hex[:8]
    paths = {
        "customer": f"customers/cust_{tag}",
        "history": f"customers/cust_{tag}/scores/ev_{tag}",
        "order": f"orders/ord_{tag}",
        "event": f"events/ev_{tag}",
    }
    db = firestore.Client(project=PROJECT)
    for path in paths.values():
        db.document(path).set({"quadrant": "fad"})
    return paths


@pytest.mark.parametrize("role", ["admin", "analyst", "viewer"])
def test_staff_can_read_customers(seeded: dict[str, str], role: str) -> None:
    assert status("GET", seeded["customer"], token(role=role)) == 200
    assert status("GET", "customers", token(role=role)) == 200


@pytest.mark.parametrize(
    "bearer",
    [None, token(), token(role="customer"), token(role=["viewer"])],
    ids=["no sign-in", "no role", "another role", "role in a list"],
)
def test_others_cannot_read_customers(
    seeded: dict[str, str], bearer: str | None
) -> None:
    assert status("GET", seeded["customer"], bearer) == 403
    assert status("GET", "customers", bearer) == 403


def test_no_one_writes_through_the_rules(seeded: dict[str, str]) -> None:
    change = {"fields": {"quadrant": {"stringValue": "lovemark"}}}
    admin = token(role="admin")
    assert status("PATCH", seeded["customer"], admin, change) == 403
    assert status("DELETE", seeded["customer"], admin) == 403


@pytest.mark.parametrize("name", ["history", "order", "event"])
def test_everything_else_stays_locked(seeded: dict[str, str], name: str) -> None:
    assert status("GET", seeded[name], token(role="admin")) == 403
