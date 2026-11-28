"""T206 and T304: who may read what in Firestore, through the rules.

READERS is the table firestore.rules carries out, one row per collection
fitcore.documents knows. The tests run every collection against every kind
of caller in CALLERS, on the Firestore emulator only; elsewhere they are
skipped, except the check that READERS covers every collection.

Each request carries an unsigned token built as the JS SDK's mockUserToken
builds it: the emulator accepts one, production never would. Documents are
seeded with the Python client, which the emulator lets past the rules.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

import pytest
from fitcore.documents import COLLECTIONS
from google.cloud import firestore

PROJECT = "demo-cfi"

emulator = pytest.mark.skipif(
    "FIRESTORE_EMULATOR_HOST" not in os.environ,
    reason="needs the Firestore emulator: run via firebase emulators:exec",
)

STAFF = frozenset({"viewer", "analyst", "admin"})
# Who reads what the dashboard shows: staff, and guests (T411).
READS = STAFF | {"guest", "guest with a role"}

# Who may read each collection, a document or the whole listing. Every role
# reads what the dashboard shows; nobody reads what the API serves itself
# (rules, T406; models, T310), what nothing writes yet, or the outbox, which
# only the API uses (T306). Guests (T411) read what every role reads and no
# more, even with a role claim. Staff also read their own users/{uid}, which
# a test below checks on its own; guests have none.
READERS: dict[str, frozenset[str]] = {
    "customers": READS,
    "scores": READS,
    "orders": READS,
    "events": frozenset({"admin"}),
    "users": frozenset({"admin"}),
    "rules": frozenset(),
    "actions": frozenset(),
    "models": frozenset(),
    "outbox": frozenset(),
}

# Each kind of caller, by the claims its token adds; None is no sign-in.
CALLERS: dict[str, dict[str, Any] | None] = {
    "no sign-in": None,
    "no role": {},
    "another role": {"role": "customer"},
    "role in a list": {"role": ["admin"]},
    "viewer": {"role": "viewer"},
    "analyst": {"role": "analyst"},
    "admin": {"role": "admin"},
    "guest": {"firebase": {"sign_in_provider": "anonymous", "identities": {}}},
    "guest with a role": {
        "role": "admin",
        "firebase": {"sign_in_provider": "anonymous", "identities": {}},
    },
}


def _encoded(part: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()


def token(uid: str, **claims: Any) -> str:
    """An unsigned ID token for uid with these extra claims."""
    payload = {
        "iss": f"https://securetoken.google.com/{PROJECT}",
        "aud": PROJECT,
        "iat": 0,
        "exp": 3600,
        "auth_time": 0,
        "sub": uid,
        "user_id": uid,
        "firebase": {"sign_in_provider": "custom", "identities": {}},
        **claims,
    }
    return ".".join([_encoded({"alg": "none", "type": "JWT"}), _encoded(payload), ""])


def bearer(caller: str, uid: str = "tester") -> str | None:
    """The token caller sends as uid, or None if it is not signed in."""
    claims = CALLERS[caller]
    return None if claims is None else token(uid, **claims)


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


@pytest.fixture(scope="module")
def seeded() -> dict[str, str]:
    """One document in every collection, under new ids: its path, by collection."""
    tag = uuid.uuid4().hex[:8]
    paths = {
        name: re.sub(r"\{(\w+)\}", lambda m: f"{m.group(1)}_{tag}", collection.path)
        for name, collection in COLLECTIONS.items()
    }
    db = firestore.Client(project=PROJECT)
    for path in paths.values():
        db.document(path).set({"seeded": True})
    return paths


def test_readers_cover_every_collection() -> None:
    assert set(READERS) == set(COLLECTIONS)


@emulator
@pytest.mark.parametrize("caller", CALLERS)
@pytest.mark.parametrize("name", READERS)
def test_who_may_read_each_collection(
    seeded: dict[str, str], name: str, caller: str
) -> None:
    expected = 200 if caller in READERS[name] else 403
    document = seeded[name]
    listing = document.rsplit("/", 1)[0]
    assert status("GET", document, bearer(caller)) == expected
    assert status("GET", listing, bearer(caller)) == expected


@emulator
@pytest.mark.parametrize("caller", CALLERS)
def test_staff_read_their_own_user_document(caller: str) -> None:
    uid = f"staff_{uuid.uuid4().hex[:8]}"
    firestore.Client(project=PROJECT).document(f"users/{uid}").set({"seeded": True})
    expected = 200 if caller in STAFF else 403
    assert status("GET", f"users/{uid}", bearer(caller, uid)) == expected


@emulator
@pytest.mark.parametrize("name", READERS)
def test_no_one_writes_through_the_rules(seeded: dict[str, str], name: str) -> None:
    admin = bearer("admin")
    document = seeded[name]
    change = {"fields": {"seeded": {"booleanValue": False}}}
    assert status("PATCH", document, admin, change) == 403
    assert status("PATCH", f"{document}_new", admin, change) == 403
    assert status("DELETE", document, admin) == 403
