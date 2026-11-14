"""Every error the API returns has one shape: {code, message}.

code is a short machine-readable name taken from the HTTP status, such as
not_found or method_not_allowed; message is a sentence for people. A request
that fails validation (400 or 422) also carries errors: the details, field
by field.
"""

from typing import Any

from flask_smorest import Api
from marshmallow import Schema, fields
from werkzeug.exceptions import HTTPException

# The message for a request that fails validation; errors holds the details.
INVALID = "The request is not valid; see errors for the details."


class ErrorSchema(Schema):
    """The body of every error response."""

    code = fields.String(required=True, metadata={"example": "not_found"})
    message = fields.String(required=True)
    errors = fields.Dict(metadata={"description": "Validation details (400, 422)"})


def error_code(error: HTTPException) -> str:
    """The HTTP status name in snake case: Not Found becomes not_found."""
    return (error.name or "error").lower().replace(" ", "_").replace("-", "_")


class CfiApi(Api):
    """flask-smorest's Api with errors shaped as {code, message}."""

    ERROR_SCHEMA = ErrorSchema

    def handle_http_exception(self, error: HTTPException) -> Any:
        data = getattr(error, "data", None) or {}
        body: dict[str, Any] = {
            "code": error_code(error),
            "message": data.get("message") or error.description or error.name,
        }
        if "messages" in data:
            body["errors"] = data["messages"]
            body["message"] = data.get("message") or INVALID
        headers = data.get("headers", {})
        return body, error.code, headers
