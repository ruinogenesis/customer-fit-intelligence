"""GET /api/v1/health: is the API up, and which model does it serve?"""

from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

blp = Blueprint("health", __name__, description="Liveness and the live model")


class HealthSchema(Schema):
    status = fields.String(required=True, metadata={"example": "ok"})
    model_version = fields.String(
        required=True,
        allow_none=True,
        metadata={"description": "The live model's version; null until one is loaded"},
    )


@blp.route("/health")
class Health(MethodView):
    @blp.response(200, HealthSchema)
    def get(self) -> dict[str, str | None]:
        """Liveness check plus the live model version."""
        return {"status": "ok", "model_version": current_app.config["MODEL_VERSION"]}
