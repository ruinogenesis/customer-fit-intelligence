"""GET /models and POST /models/{version}/promote: the model registry (T310).

Every staff role sees the versions; only admins promote. Promoting checks
that the version's file loads before it goes live, and the next event on
every instance is scored by it (cfi_api.registry). Rolling back is
promoting an older version.
"""

from datetime import UTC, datetime
from typing import Any

from fitcore.documents import ModelDocument
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.registry import BadModel, Models, UnknownVersion, model_document, promote
from cfi_api.runtime import registry, store

blp = Blueprint("models", __name__, description="The model registry")

NO_REGISTRY = "This API has no model registry (REGISTRY is unset)"
NOT_ADMIN = "The caller is not an admin"


class ModelsSchema(Schema):
    models = fields.List(
        fields.Nested(ModelDocument),
        required=True,
        metadata={"description": "Every registered version, newest first"},
    )


def _registry() -> Models:
    models = registry()
    if models is None:
        abort(503, description=f"{NO_REGISTRY}.")
    return models


@blp.route("/models")
class ModelList(MethodView):
    @blp.response(200, ModelsSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(503, schema=ErrorSchema, description=NO_REGISTRY)
    @blp.doc(security=ACCESS["models.ModelList", "GET"].security)
    def get(self) -> Any:
        """Every model version, newest first, with its status and metrics."""
        _registry()
        return {"models": [model_document(e) for e in store().list_models()]}


@blp.route("/models/<string:version>/promote")
class Promote(MethodView):
    @blp.response(200, ModelDocument)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_ADMIN)
    @blp.alt_response(404, schema=ErrorSchema, description="No such version")
    @blp.alt_response(
        409, schema=ErrorSchema, description="Its file is missing or does not load"
    )
    @blp.alt_response(503, schema=ErrorSchema, description=NO_REGISTRY)
    @blp.doc(security=ACCESS["models.Promote", "POST"].security)
    def post(self, version: str) -> Any:
        """Make a version live; the live one before it is retired.

        Its file must load first, or nothing changes. Promoting the live
        version changes nothing; promoting an older one rolls back. The next
        event on every instance is scored by it.
        """
        models = _registry()
        try:
            entry = promote(store(), models, version, datetime.now(UTC))
        except UnknownVersion:
            abort(404, description=f"No model version {version}.")
        except BadModel as err:
            abort(409, description=f"Not promoted: {err}")
        return model_document(entry)
