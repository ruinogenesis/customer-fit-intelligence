"""GET /matrix: how many customers sit in each quadrant now (T311).

Counted from each customer's stored quadrant, which the model that last
scored them placed (D5: lines at 0.5). Movement between two dates comes
with T313, from BigQuery.
"""

from typing import Any

from fitcore.model import QUADRANTS
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import Schema, fields

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.runtime import store

blp = Blueprint("matrix", __name__, description="The love matrix")


class QuadrantCountsSchema(Schema):
    lovemark = fields.Integer(required=True)
    brand = fields.Integer(required=True)
    fad = fields.Integer(required=True)
    product = fields.Integer(required=True)


class MatrixSchema(Schema):
    quadrants = fields.Nested(
        QuadrantCountsSchema,
        required=True,
        metadata={"description": "Customers per quadrant"},
    )
    unscored = fields.Integer(
        required=True, metadata={"description": "Customers with no scored order yet"}
    )
    customers = fields.Integer(required=True, metadata={"description": "All customers"})


@blp.route("/matrix")
class Matrix(MethodView):
    @blp.response(200, MatrixSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.doc(security=ACCESS["matrix.Matrix", "GET"].security)
    def get(self) -> Any:
        """Customers per quadrant now, plus those not scored yet."""
        counts = store().count_quadrants()
        return {
            "quadrants": {quadrant: counts[quadrant] for quadrant in QUADRANTS},
            "unscored": counts[None],
            "customers": sum(counts.values()),
        }
