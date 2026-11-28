"""Who may call what (T303): staff sign-in, and the integration key for events.

The dev API is public: Firebase Hosting can only reach a Cloud Run service
that takes calls without Google credentials (T205). So with AUTH set to
"firebase", as on Cloud Run, the API checks each request against ACCESS:

- Staff send a Firebase ID token in the header Authorization: Bearer
  <token>. Its role claim, which python -m cfi_api.roles sets, must rank at
  least as high as the rule's role: viewer < analyst < admin.
- A guest of the dashboard (T411) signs in anonymously, where the project
  allows it (the Anonymous provider in Firebase Auth), and ranks as a
  viewer: it reads what every role reads, and nothing more.
- The ordering app may send events with the integration key in the header
  X-CFI-Key instead (WRITE_KEY, from Secret Manager), as it has no staff
  account. The key opens POST /events and nothing else.

No valid token (or key, where one is enough) gets 401; a valid token whose
role ranks too low gets 403. A request ACCESS does not list needs an admin.
With AUTH unset, as locally and in most tests, nothing is checked.
"""

import hmac
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from flask import Flask, request
from flask_smorest import abort

KEY_HEADER = "X-CFI-Key"

# How a role ranks: a rule for a role lets in that role and those above it.
RANKS = {"viewer": 1, "analyst": 2, "admin": 3}

# The OpenAPI security schemes, so the docs page can send them ("Authorize").
TOKEN = "firebase"
KEY = "writeKey"
SECURITY_SCHEMES: dict[str, dict[str, Any]] = {
    TOKEN: {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "A Firebase ID token of a staff account, whose role "
        "claim ranks high enough: viewer < analyst < admin.",
    },
    KEY: {
        "type": "apiKey",
        "in": "header",
        "name": KEY_HEADER,
        "description": "The integration key of the ordering app: it opens "
        "POST /events only.",
    },
}

# The environment variable that points firebase-admin at the Auth emulator,
# which accepts tokens without a signature.
EMULATOR = "FIREBASE_AUTH_EMULATOR_HOST"


@dataclass(frozen=True)
class Rule:
    """Who may make one kind of request.

    role is the lowest role that may, or None for anyone, signed in or not;
    key says whether the integration key is enough instead.
    """

    role: str | None = None
    key: bool = False

    @property
    def security(self) -> list[dict[str, list[str]]]:
        """The OpenAPI security of the request: any one of these will do."""
        options: list[dict[str, list[str]]] = []
        if self.role is not None:
            options.append({TOKEN: []})
            if self.key:
                options.append({KEY: []})
        return options


ANYONE = Rule()
ADMIN = Rule("admin")

# Every request the API serves, by (endpoint, method). HEAD is checked as
# GET, and OPTIONS is always open: it only lists the allowed methods.
ACCESS: dict[tuple[str, str], Rule] = {
    ("health.Health", "GET"): ANYONE,
    ("api-docs.openapi_json", "GET"): ANYONE,
    ("api-docs.openapi_swagger_ui", "GET"): ANYONE,
    ("events.Events", "POST"): Rule("admin", key=True),
    ("score.Score", "POST"): Rule("analyst"),
    # The dashboard's reads (T311): every staff role.
    ("customers.Customers", "GET"): Rule("viewer"),
    ("customers.Customer", "GET"): Rule("viewer"),
    ("customers.CustomerScores", "GET"): Rule("viewer"),
    ("customers.CustomerOrders", "GET"): Rule("viewer"),
    ("matrix.Matrix", "GET"): Rule("viewer"),
    ("drivers.Drivers", "GET"): Rule("viewer"),
    ("models.ModelList", "GET"): Rule("viewer"),
    ("models.ModelCard", "GET"): Rule("viewer"),
    ("models.LiveModel", "GET"): Rule("viewer"),
    ("overview.Overview", "GET"): Rule("viewer"),
    ("overview.Trend", "GET"): Rule("viewer"),
    ("automations.RuleList", "GET"): Rule("viewer"),
    ("models.Promote", "POST"): Rule("admin"),
}

# Checks an ID token: its claims, or None if it is not a valid token.
TokenCheck = Callable[[str], Mapping[str, Any] | None]


def firebase_token_check(project: str | None) -> TokenCheck:
    """Checks Firebase ID tokens issued for project (None: the environment's).

    Only the token is checked, offline once Google's public keys are
    cached: a role taken away or an account disabled still works until its
    token expires, within an hour. Raises ValueError if the Auth emulator is
    set and project is not a demo- project, as that would let in tokens
    nobody signed.
    """
    if os.environ.get(EMULATOR) and not (project or "").startswith("demo-"):
        raise ValueError(
            f"{EMULATOR} is set, so tokens would not be checked: "
            "use it only with a demo- project"
        )
    # Imported here, as the Firestore client is: most tests never need it.
    import firebase_admin
    from firebase_admin import auth

    options = {"projectId": project} if project else {}
    # A name of its own, so each app made by create_app has its own settings.
    firebase = firebase_admin.initialize_app(
        options=options, name=f"cfi-api-{uuid4().hex}"
    )

    def check(token: str) -> Mapping[str, Any] | None:
        try:
            claims: Mapping[str, Any] = auth.verify_id_token(token, app=firebase)
        except auth.InvalidIdTokenError:
            return None
        # Anything else, such as no project to check against, is a fault of
        # the server, not of the token: it fails the request with a 500.
        return claims

    return check


def require_access(
    app: Flask, auth: object, project: str | None, write_key: object
) -> None:
    """Check every request against ACCESS when auth is "firebase".

    Does nothing when auth is None. Raises ValueError for any other auth,
    for a write key without auth (it would check nothing) and for an empty
    key; raises TypeError for a key that is not text: Flask reads
    CFI_WRITE_KEY as JSON when it can, so an all-digit key would arrive as
    a number.
    """
    if write_key is not None:
        if not isinstance(write_key, str):
            raise TypeError("WRITE_KEY must be text, not a number")
        if not write_key:
            raise ValueError("WRITE_KEY must not be empty: leave it unset instead")
    if auth is None:
        if write_key is not None:
            raise ValueError("WRITE_KEY needs AUTH = 'firebase' to be checked")
        return
    if auth != "firebase":
        raise ValueError(f"AUTH must be unset or 'firebase', got {auth!r}")
    token_check = firebase_token_check(project)
    expected = None if write_key is None else write_key.encode()

    @app.before_request
    def check() -> None:
        # No endpoint: routing will answer 404 or 405, and run nothing.
        if request.endpoint is None or request.method == "OPTIONS":
            return
        method = "GET" if request.method == "HEAD" else request.method
        rule = ACCESS.get((request.endpoint, method), ADMIN)
        if rule.role is None:
            return
        if rule.key and expected is not None:
            given = request.headers.get(KEY_HEADER, "").encode()
            # A constant-time comparison: how long it takes says nothing
            # about how much of a guess was right.
            if hmac.compare_digest(given, expected):
                return
        header = request.authorization
        token = header.token if header is not None and header.type == "bearer" else None
        claims = token_check(token) if token else None
        if claims is None:
            needs = f" or a valid {KEY_HEADER} header" if rule.key else ""
            abort(
                401,
                message=f"This request needs a staff sign-in token{needs}.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif rank(role_of(claims)) < RANKS[rule.role]:
            abort(403, message=f"This request needs the {rule.role} role or higher.")


GUEST = "viewer"


def role_of(claims: Mapping[str, Any]) -> object:
    """The role a token carries: a guest's is GUEST, whatever else it says."""
    firebase = claims.get("firebase")
    if (
        isinstance(firebase, Mapping)
        and firebase.get("sign_in_provider") == "anonymous"
    ):
        return GUEST
    return claims.get("role")


def rank(role: object) -> int:
    """How role ranks; 0 for no role, or one this API does not know."""
    return RANKS.get(role, 0) if isinstance(role, str) else 0
