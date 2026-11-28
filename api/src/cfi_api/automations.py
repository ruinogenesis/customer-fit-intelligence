"""GET /rules (T406): the automation rules and their status.

Every staff role reads them, for the Automations screen. Until the rules
engine (T501) nothing changes a rule: python -m cfi_api.rules seed adds the
starter rules, each off. Apps cannot read rules/ in Firestore, so this is
the one way to see them.
"""

from typing import Any

from fitcore.documents import RuleDocument
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.runtime import store

blp = Blueprint("automations", __name__, description="The automation rules")


class RulesSchema(Schema):
    rules = fields.List(
        fields.Nested(RuleDocument),
        required=True,
        metadata={"description": "Every rule, by name"},
    )


@blp.route("/rules")
class RuleList(MethodView):
    @blp.response(200, RulesSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.doc(security=ACCESS["automations.RuleList", "GET"].security)
    def get(self) -> Any:
        """Every automation rule, by name, with its kind, status and holdout."""
        return {"rules": store().list_rules()}
