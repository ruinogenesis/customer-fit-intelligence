"""GET /matrix: how many customers sit in each quadrant now (T311), and how
they moved between two days (T313).

The counts now come from each customer's stored quadrant, which the model
that last scored them placed (D5: lines at 0.5). The movement comes from
BigQuery's features.daily (cfi_api.history), which holds every customer's
quadrant at the end of each Bangkok day up to yesterday.
"""

from typing import Any

from fitcore.model import QUADRANTS
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import RAISE, Schema, ValidationError, fields, validates_schema

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.history import today
from cfi_api.runtime import history, store

blp = Blueprint("matrix", __name__, description="The love matrix")

UNSCORED = "unscored"
BAD_SPAN = "from and to are not a span of past days, or to has no daily snapshot yet"
NO_HISTORY = "This API reads no BigQuery history (HISTORY is unset)"


class MatrixArgs(Schema):
    first = fields.Date(
        data_key="from",
        load_default=None,
        metadata={"description": "Movement from the end of this Bangkok day"},
    )
    last = fields.Date(
        data_key="to",
        load_default=None,
        metadata={
            "description": "Movement to the end of this Bangkok day: before today"
        },
    )

    @validates_schema
    def _a_span_of_past_days(self, data: dict[str, Any], **kwargs: Any) -> None:
        first, last = data["first"], data["last"]
        if (first is None) != (last is None):
            raise ValidationError("Give from and to together, or neither.")
        if last is None:
            return
        if first >= last:
            raise ValidationError("from must come before to.", "from")
        if last >= today():
            raise ValidationError(
                "to must be before today in Bangkok: daily snapshots end yesterday.",
                "to",
            )


class QuadrantCountsSchema(Schema):
    lovemark = fields.Integer(required=True)
    brand = fields.Integer(required=True)
    fad = fields.Integer(required=True)
    product = fields.Integer(required=True)


class MovesSchema(Schema):
    """By quadrant on from: how many are in each quadrant on to."""

    lovemark = fields.Nested(QuadrantCountsSchema, required=True)
    brand = fields.Nested(QuadrantCountsSchema, required=True)
    fad = fields.Nested(QuadrantCountsSchema, required=True)
    product = fields.Nested(QuadrantCountsSchema, required=True)
    unscored = fields.Nested(
        QuadrantCountsSchema,
        required=True,
        metadata={"description": "Not scored yet on from"},
    )


class MovementSchema(Schema):
    first = fields.Date(data_key="from", required=True)
    last = fields.Date(data_key="to", required=True)
    moves = fields.Nested(
        MovesSchema,
        required=True,
        metadata={
            "description": "Customers by quadrant on from (or unscored), "
            "then by quadrant on to"
        },
    )
    customers = fields.Integer(
        required=True, metadata={"description": "Customers scored by the end of to"}
    )


class MatrixSchema(Schema):
    quadrants = fields.Nested(
        QuadrantCountsSchema,
        required=True,
        metadata={"description": "Customers per quadrant now"},
    )
    unscored = fields.Integer(
        required=True, metadata={"description": "Customers with no scored order yet"}
    )
    customers = fields.Integer(required=True, metadata={"description": "All customers"})
    movement = fields.Nested(
        MovementSchema,
        metadata={"description": "Only with from and to: from BigQuery, daily"},
    )


def _movement(first: Any, last: Any) -> dict[str, Any]:
    """The movement grid from the end of day first to the end of day last."""
    reader = history()
    if reader is None:
        abort(503, description=f"{NO_HISTORY}, so it shows no movement.")
    moves = reader.movement(first, last)
    if not moves:
        abort(
            422,
            description=f"No daily snapshot holds customers on {last}: each day is "
            "written at 02:00 Bangkok the next night, from the first scoring on.",
        )
    grid = {was: dict.fromkeys(QUADRANTS, 0) for was in (*QUADRANTS, UNSCORED)}
    for move in moves:
        grid[move.was or UNSCORED][move.now] += move.customers
    return {
        "first": first,
        "last": last,
        "moves": grid,
        "customers": sum(move.customers for move in moves),
    }


@blp.route("/matrix")
class Matrix(MethodView):
    @blp.arguments(MatrixArgs, location="query", unknown=RAISE, as_kwargs=True)
    @blp.response(200, MatrixSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(422, schema=ErrorSchema, description=BAD_SPAN)
    @blp.alt_response(503, schema=ErrorSchema, description=NO_HISTORY)
    @blp.doc(security=ACCESS["matrix.Matrix", "GET"].security)
    def get(self, first: Any, last: Any) -> Any:
        """Customers per quadrant now, plus those not scored yet.

        With from and to, also how customers moved between the end of those
        two Bangkok days, read from BigQuery's daily snapshots: by quadrant
        on from (or not scored yet), then by quadrant on to.
        """
        counts = store().count_quadrants()
        answer: dict[str, Any] = {
            "quadrants": {quadrant: counts[quadrant] for quadrant in QUADRANTS},
            "unscored": counts[None],
            "customers": sum(counts.values()),
        }
        if last is not None:
            answer["movement"] = _movement(first, last)
        return answer
