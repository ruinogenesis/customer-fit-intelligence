"""GET /models, a version's card and POST /models/{version}/promote (T310).

Every staff role sees the versions and their model cards (T407); only
admins promote. GET /models/live (T410) answers the model events are scored
with now, and what it scores an order exactly as expected (every gap 0):
the dashboard marks the customers whose latest order measured nothing that
moves Love. Promoting checks
that the version's file loads before it goes live, and the next event on
every instance is scored by it (cfi_api.model_registry). Rolling back is
promoting an older version.
"""

from datetime import UTC, datetime
from typing import Any

from fitcore.documents import ModelDocument
from fitcore.model import SCORES
from fitcore.scoring import score
from flask import Response, abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.model_registry import (
    BadModel,
    Models,
    UnknownVersion,
    card_path,
    model_document,
    promote,
)
from cfi_api.runtime import live_model, registry, store

blp = Blueprint("models", __name__, description="The model registry")

NO_REGISTRY = "This API has no model registry (REGISTRY is unset)"
NOT_ADMIN = "The caller is not an admin"
NO_MODEL = "No model is loaded, or the live one will not load"


class ModelsSchema(Schema):
    models = fields.List(
        fields.Nested(ModelDocument),
        required=True,
        metadata={"description": "Every registered version, newest first"},
    )


class AsExpectedSchema(Schema):
    respect = fields.Float(required=True)
    love = fields.Float(required=True)
    convenience = fields.Float(required=True)
    trust = fields.Float(required=True)
    fit = fields.Float(required=True)
    churn_risk = fields.Float(required=True)


class LiveModelSchema(Schema):
    version = fields.String(
        required=True, metadata={"description": "The version events are scored with"}
    )
    as_expected = fields.Nested(
        AsExpectedSchema,
        required=True,
        metadata={
            "description": "What it scores an order exactly as expected (every gap "
            "0). A customer whose love is exactly this one's had nothing on their "
            "latest order that moves Love (food, personalization or recovery)"
        },
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


@blp.route("/models/live")
class LiveModel(MethodView):
    @blp.response(200, LiveModelSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(503, schema=ErrorSchema, description=NO_MODEL)
    @blp.doc(security=ACCESS["models.LiveModel", "GET"].security)
    def get(self) -> Any:
        """The model events are scored with now, and its as-expected scores.

        From the registry's live version, or the model file when the API has
        no registry.
        """
        try:
            model = live_model()
        except BadModel as err:
            abort(503, description=f"The live model cannot be loaded: {err}")
        if model is None:
            abort(503, description="No model is loaded.")
        expected = score(model.params, {})
        return {
            "version": model.version,
            "as_expected": {name: getattr(expected, name) for name in SCORES},
        }


@blp.route("/models/<string:version>/card")
class ModelCard(MethodView):
    @blp.doc(
        responses={
            "200": {
                "description": "The version's model_card.md",
                "content": {"text/markdown": {"schema": {"type": "string"}}},
            }
        }
    )
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(
        404, schema=ErrorSchema, description="No such version, or it has no card"
    )
    @blp.alt_response(503, schema=ErrorSchema, description=NO_REGISTRY)
    @blp.doc(security=ACCESS["models.ModelCard", "GET"].security)
    def get(self, version: str) -> Any:
        """A version's model card, as markdown (fitcore.card)."""
        models = _registry()
        entry = next((e for e in store().list_models() if e.version == version), None)
        if entry is None:
            abort(404, description=f"No model version {version}.")
        try:
            card = models.files.read(card_path(entry.path))
        except FileNotFoundError:
            abort(404, description=f"{version} has no model card.")
        return Response(card, content_type="text/markdown; charset=utf-8")


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
