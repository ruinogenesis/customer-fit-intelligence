"""A shared write key on the API's POST endpoints, until T303's sign-in.

The dev API is public: Firebase Hosting can only reach a Cloud Run service
that takes calls without Google credentials (T205). Until T303 checks staff
roles, every request other than a read (GET, HEAD, OPTIONS) must carry the
header X-CFI-Key with the key the server holds as CFI_WRITE_KEY; on Cloud
Run that comes from Secret Manager. Reads, /health and the docs included,
stay open. A server without a key, such as a local one or the tests', checks
nothing.
"""

import hmac
from typing import Any

from flask import Flask, request
from flask_smorest import abort

HEADER = "X-CFI-Key"
REFUSED = f"This request needs a valid {HEADER} header."
READS = frozenset({"GET", "HEAD", "OPTIONS"})

# The OpenAPI security scheme the POST endpoints name, so the docs page can
# send the key ("Authorize").
SCHEME = "writeKey"
SECURITY_SCHEME: dict[str, Any] = {
    "type": "apiKey",
    "in": "header",
    "name": HEADER,
    "description": "Needed when the server has a write key (CFI_WRITE_KEY), "
    "as on Cloud Run, until T303's sign-in.",
}


def require_write_key(app: Flask, key: object) -> None:
    """Refuse every request but a read without the key, with 401.

    A key of None checks nothing. Raises ValueError for an empty key, which
    would let a caller without the header in, and TypeError for one that is
    not text: Flask reads CFI_WRITE_KEY as JSON when it can, so an
    all-digit key would arrive as a number.
    """
    if key is None:
        return
    if not isinstance(key, str):
        raise TypeError("WRITE_KEY must be text, not a number")
    if not key:
        raise ValueError("WRITE_KEY must not be empty: leave it unset instead")
    expected = key.encode()

    @app.before_request
    def check() -> None:
        if request.method in READS:
            return
        given = request.headers.get(HEADER, "").encode()
        # A constant-time comparison: how long it takes says nothing about
        # how much of a guess was right.
        if not hmac.compare_digest(given, expected):
            abort(401, message=REFUSED)
