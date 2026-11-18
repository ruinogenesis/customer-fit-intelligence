"""GET /overview and GET /overview/trend: the Overview screen's numbers (T402).

/overview is now: every customer by quadrant, the scored customers' average
fit and churn risk, and how many are at or above a churn risk. Each number
is a Firestore aggregation, billed by the index entries it reads (one read
per 1,000), so the dashboard can ask every 30 seconds.

/overview/trend is the same averages and counts for each Bangkok day of a
span, read from BigQuery's features.daily (T308), which the 02:00 run writes
up to yesterday: a day's numbers change once, so the screen reads it once.
"""

from datetime import timedelta
from typing import Any

from fitcore.model import QUADRANTS
from flask import abort
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import RAISE, Schema, ValidationError, fields, validate, validates

from cfi_api.access import ACCESS
from cfi_api.customers import NOT_STAFF, UNSIGNED
from cfi_api.errors import ErrorSchema
from cfi_api.history import today
from cfi_api.matrix import NO_HISTORY, QuadrantCountsSchema
from cfi_api.runtime import history, store

blp = Blueprint("overview", __name__, description="The Overview screen")

AT_RISK = 0.1  # the default churn risk counted as at risk
DAYS = 28  # the default trend: four weeks, as /drivers' window
MOST_DAYS = 90


class OverviewArgs(Schema):
    at_risk = fields.Float(
        load_default=AT_RISK,
        validate=validate.Range(0, 1),
        metadata={
            "description": f"Count customers at or above this churn risk ({AT_RISK})"
        },
    )


class AtRiskSchema(Schema):
    threshold = fields.Float(
        required=True, metadata={"description": "The at_risk asked"}
    )
    customers = fields.Integer(
        required=True, metadata={"description": "Customers at or above it"}
    )


class OverviewSchema(Schema):
    customers = fields.Integer(required=True, metadata={"description": "All customers"})
    unscored = fields.Integer(
        required=True, metadata={"description": "Customers with no scored order yet"}
    )
    quadrants = fields.Nested(
        QuadrantCountsSchema,
        required=True,
        metadata={"description": "Customers per quadrant now"},
    )
    average_fit = fields.Float(
        required=True,
        allow_none=True,
        metadata={"description": "Over the scored customers; null before the first"},
    )
    average_churn_risk = fields.Float(
        required=True,
        allow_none=True,
        metadata={"description": "Over the scored customers; null before the first"},
    )
    at_risk = fields.Nested(AtRiskSchema, required=True)


class TrendArgs(Schema):
    days = fields.Integer(
        load_default=DAYS,
        validate=validate.Range(1, MOST_DAYS),
        metadata={
            "description": f"How many days, ending on to ({DAYS}, at most {MOST_DAYS})"
        },
    )
    last = fields.Date(
        data_key="to",
        load_default=None,
        metadata={"description": "The last Bangkok day (yesterday); before today"},
    )

    @validates("last")
    def _past(self, value: Any, **_: Any) -> None:
        if value is not None and value >= today():
            raise ValidationError("to must be a day before today (Bangkok).")


class DaySchema(Schema):
    day = fields.Date(required=True, metadata={"description": "A Bangkok day"})
    customers = fields.Integer(
        required=True, metadata={"description": "Customers scored by its end"}
    )
    average_fit = fields.Float(required=True)
    average_churn_risk = fields.Float(required=True)
    quadrants = fields.Nested(QuadrantCountsSchema, required=True)


class TrendSchema(Schema):
    first = fields.Date(data_key="from", required=True)
    last = fields.Date(data_key="to", required=True)
    days = fields.List(
        fields.Nested(DaySchema),
        required=True,
        metadata={"description": "Each day with a daily snapshot, oldest first"},
    )


@blp.route("/overview")
class Overview(MethodView):
    @blp.arguments(OverviewArgs, location="query", unknown=RAISE, as_kwargs=True)
    @blp.response(200, OverviewSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.doc(security=ACCESS["overview.Overview", "GET"].security)
    def get(self, at_risk: float) -> Any:
        """Customers by quadrant now, their averages, and how many are at risk."""
        now = store().overview(at_risk)
        return {
            "customers": sum(now.counts.values()),
            "unscored": now.counts[None],
            "quadrants": {quadrant: now.counts[quadrant] for quadrant in QUADRANTS},
            "average_fit": now.average_fit,
            "average_churn_risk": now.average_churn_risk,
            "at_risk": {"threshold": at_risk, "customers": now.at_risk},
        }


@blp.route("/overview/trend")
class Trend(MethodView):
    @blp.arguments(TrendArgs, location="query", unknown=RAISE, as_kwargs=True)
    @blp.response(200, TrendSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(503, schema=ErrorSchema, description=NO_HISTORY)
    @blp.doc(security=ACCESS["overview.Trend", "GET"].security)
    def get(self, days: int, last: Any) -> Any:
        """Each day's averages and quadrant counts, from BigQuery's daily snapshots.

        The span ends on to (yesterday by default) and is days long; a day
        before anyone's first scoring has no snapshot, so it is left out.
        """
        reader = history()
        if reader is None:
            abort(503, description=f"{NO_HISTORY}, so it shows no trend.")
        last = last or today() - timedelta(days=1)
        first = last - timedelta(days=days - 1)
        return {
            "first": first,
            "last": last,
            "days": [
                {
                    "day": day.day,
                    "customers": day.customers,
                    "average_fit": day.fit,
                    "average_churn_risk": day.churn_risk,
                    "quadrants": day.quadrants,
                }
                for day in reader.trend(first, last)
            ],
        }
