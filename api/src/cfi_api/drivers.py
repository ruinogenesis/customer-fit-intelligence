"""GET /drivers: which inputs pull fit down at each restaurant or courier
(T315).

Every scoring stores each input's contribution to fit (T316). This averages
them per restaurant or courier over the orders placed in a span of Bangkok
days, read from BigQuery (cfi_api.history), and compares each group with
all of those orders: an input pulls a group down when its average there is
below its average everywhere. Lateness drags nearly every courier, so a
plain average would name it first for all of them; the difference names
what is wrong at this courier.
"""

from datetime import timedelta
from typing import Any

from fitcore.features import INPUTS
from fitcore.scoring import DRIVERS as TOP
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import (
    RAISE,
    Schema,
    ValidationError,
    fields,
    validate,
    validates_schema,
)

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.history import Group, today
from cfi_api.runtime import history

blp = Blueprint("drivers", __name__, description="What pulls fit down, by group")

# Without from and to: the last WINDOW days, today included.
WINDOW = 28
BAD_ARGS = "by is missing or unknown, from is after to, or to is after today"
NO_HISTORY = "This API reads no BigQuery history (HISTORY is unset)"


class DriversArgs(Schema):
    by = fields.String(
        required=True,
        validate=validate.OneOf(["restaurant", "courier"]),
        metadata={"description": "Group orders by restaurant or by courier"},
    )
    first = fields.Date(
        data_key="from",
        load_default=None,
        metadata={
            "description": f"Orders placed from this Bangkok day; default {WINDOW - 1} "
            "days before to"
        },
    )
    last = fields.Date(
        data_key="to",
        load_default=None,
        metadata={"description": "Orders placed up to this Bangkok day; default today"},
    )
    min_orders = fields.Integer(
        load_default=5,
        validate=validate.Range(min=1),
        metadata={"description": "Leave out groups with fewer orders than this"},
    )
    limit = fields.Integer(
        load_default=50,
        validate=validate.Range(min=1, max=200),
        metadata={"description": "The most groups to answer, worst first"},
    )

    @validates_schema
    def _a_span_up_to_today(self, data: dict[str, Any], **kwargs: Any) -> None:
        last = data["last"] or today()
        first = data["first"] or last - timedelta(days=WINDOW - 1)
        if last > today():
            raise ValidationError("to must not be after today in Bangkok.", "to")
        if first > last:
            raise ValidationError("from must not be after to.", "from")


def _inputs(description: str) -> fields.Dict:
    return fields.Dict(
        keys=fields.String(validate=validate.OneOf(INPUTS)),
        values=fields.Float(),
        required=True,
        metadata={"description": description},
    )


class OverallSchema(Schema):
    orders = fields.Integer(
        required=True, metadata={"description": "Orders in the span, in any group"}
    )
    fit = fields.Float(
        required=True, allow_none=True, metadata={"description": "Their average fit"}
    )
    drag = _inputs("Each input's average contribution to their fit")


class GroupSchema(Schema):
    id = fields.String(
        required=True, metadata={"description": "The restaurant or courier"}
    )
    orders = fields.Integer(required=True)
    fit = fields.Float(
        required=True, metadata={"description": "Its orders' average fit"}
    )
    drag = _inputs("Each input's average contribution to its orders' fit")
    excess = _inputs("drag minus the overall drag: below 0, worse here than overall")
    drivers = fields.List(
        fields.String(),
        required=True,
        metadata={
            "description": f"Inputs worse here than overall, worst first (at most {TOP})"
        },
    )


class DriversSchema(Schema):
    by = fields.String(required=True)
    first = fields.Date(data_key="from", required=True)
    last = fields.Date(data_key="to", required=True)
    min_orders = fields.Integer(required=True)
    overall = fields.Nested(OverallSchema, required=True)
    groups = fields.List(
        fields.Nested(GroupSchema),
        required=True,
        metadata={
            "description": "Groups with at least min_orders orders, worst first: by "
            "their most negative excess, then by id"
        },
    )


def _overall(groups: list[Group]) -> dict[str, Any]:
    """All the groups' orders together."""
    orders = sum(group.orders for group in groups)
    if orders == 0:
        return {"orders": 0, "fit": None, "drag": dict.fromkeys(INPUTS, 0.0)}
    return {
        "orders": orders,
        "fit": sum(group.orders * group.fit for group in groups) / orders,
        "drag": {
            key: sum(group.orders * group.drag[i] for group in groups) / orders
            for i, key in enumerate(INPUTS)
        },
    }


def _group(group: Group, overall: dict[str, float]) -> dict[str, Any]:
    drag = dict(zip(INPUTS, group.drag, strict=True))
    excess = {key: drag[key] - overall[key] for key in INPUTS}
    worse = sorted((key for key in INPUTS if excess[key] < 0), key=lambda k: excess[k])
    return {
        "id": group.id,
        "orders": group.orders,
        "fit": group.fit,
        "drag": drag,
        "excess": excess,
        "drivers": worse[:TOP],
    }


@blp.route("/drivers")
class Drivers(MethodView):
    @blp.arguments(DriversArgs, location="query", unknown=RAISE, as_kwargs=True)
    @blp.response(200, DriversSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(422, schema=ErrorSchema, description=BAD_ARGS)
    @blp.alt_response(503, schema=ErrorSchema, description=NO_HISTORY)
    @blp.doc(security=ACCESS["drivers.Drivers", "GET"].security)
    def get(self, by: str, first: Any, last: Any, min_orders: int, limit: int) -> Any:
        """Which inputs pull fit down at each restaurant or courier.

        Averages each input's contribution to fit over each group's orders
        placed from from to to (Bangkok days; by default the last 28, today
        included), each order by its latest scoring with a split. A group's
        drivers are the inputs whose average there is below their average
        over every group's orders.
        """
        reader = history()
        if reader is None:
            abort(503, description=f"{NO_HISTORY}, so it cannot rank drivers.")
        last = last or today()
        first = first or last - timedelta(days=WINDOW - 1)
        found = reader.drivers(by, first, last)
        overall = _overall(found)
        groups = [
            _group(group, overall["drag"])
            for group in found
            if group.orders >= min_orders
        ]
        groups.sort(key=lambda g: (min(g["excess"].values()), g["id"]))
        return {
            "by": by,
            "first": first,
            "last": last,
            "min_orders": min_orders,
            "overall": overall,
            "groups": groups[:limit],
        }
