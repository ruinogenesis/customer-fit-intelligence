"""GET /customers and one customer's state, scores and orders (T311).

The dashboard's reads, for any staff role. Lists come a page at a time:
?limit= (default 50, at most 200) and, for the next page, the next_cursor of
the page before. A cursor is opaque to callers: inside, it is the sort key
of the previous page's last item, and the next page starts right after it.
So items added or changed elsewhere in the list don't shift the pages; an
item whose own sort key changes between two pages (a customer whose churn
risk moves) can be missed or shown twice.
"""

import base64
import binascii
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fitcore.documents import CustomerDocument, OrderDocument, ScoreEntryDocument
from fitcore.model import QUADRANTS
from fitcore.schemas import check_id
from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow import RAISE, Schema, ValidationError, fields, validate
from werkzeug.exceptions import NotFound

from cfi_api.access import ACCESS
from cfi_api.errors import ErrorSchema
from cfi_api.runtime import store
from cfi_api.store import CustomerKey, Page, TimeKey

blp = Blueprint(
    "customers", __name__, description="Customers, as the dashboard reads them"
)

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


# --- Cursors ---


def encode_cursor(key: CustomerKey | TimeKey) -> str:
    """A sort key as an opaque, URL-safe cursor."""
    values = [
        value.isoformat() if isinstance(value, datetime) else value for value in key
    ]
    return base64.urlsafe_b64encode(json.dumps(values).encode()).decode().rstrip("=")


def _customer_key(values: list[Any]) -> CustomerKey:
    churn_risk, customer_id = values
    if churn_risk is not None and (
        isinstance(churn_risk, bool) or not isinstance(churn_risk, int | float)
    ):
        raise TypeError("churn_risk")
    if not isinstance(customer_id, str):
        raise TypeError("customer_id")
    return (None if churn_risk is None else float(churn_risk)), customer_id


def _time_key(values: list[Any]) -> TimeKey:
    at, item_id = values
    if not isinstance(at, str) or not isinstance(item_id, str):
        raise TypeError("time key")
    when = datetime.fromisoformat(at)
    if when.tzinfo is None:
        raise ValueError("time key without a timezone")
    return when, item_id


class Cursor(fields.String):
    """A next_cursor from the same list, opened back into its sort key."""

    def __init__(self, parse: Callable[[list[Any]], Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parse = parse

    def _deserialize(self, value: Any, attr: Any, data: Any, **kwargs: Any) -> Any:
        text = super()._deserialize(value, attr, data, **kwargs)
        try:
            raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
            values = json.loads(raw)
            if not isinstance(values, list):
                raise TypeError("not a list")
            return self.parse(values)
        except (ValueError, TypeError, binascii.Error) as err:
            raise ValidationError("Not a next_cursor from this list.") from err


def _cursor(parse: Callable[[list[Any]], Any]) -> Cursor:
    return Cursor(
        parse,
        load_default=None,
        metadata={
            "description": "The next_cursor of the page before; none for the first"
        },
    )


def _limit() -> fields.Integer:
    return fields.Integer(
        load_default=DEFAULT_LIMIT,
        validate=validate.Range(1, MAX_LIMIT),
        metadata={"description": f"Items per page, 1 to {MAX_LIMIT}"},
    )


# --- Query arguments and answers ---


class CustomerListArgs(Schema):
    quadrant = fields.String(
        load_default=None,
        validate=validate.OneOf(QUADRANTS),
        metadata={"description": "Only customers in this quadrant"},
    )
    min_churn_risk = fields.Float(
        load_default=None,
        validate=validate.Range(0, 1),
        metadata={"description": "Only customers at or above this churn risk"},
    )
    limit = _limit()
    cursor = _cursor(_customer_key)


class TimeListArgs(Schema):
    limit = _limit()
    cursor = _cursor(_time_key)


class CustomerSummarySchema(CustomerDocument):
    """A customer as lists show them: all but the expectations."""

    class Meta(CustomerDocument.Meta):
        exclude = ("expectations",)


NEXT_CURSOR = {"description": "Pass as cursor for the next page; null on the last page"}


class CustomerPageSchema(Schema):
    customers = fields.List(fields.Nested(CustomerSummarySchema), required=True)
    next_cursor = fields.String(required=True, allow_none=True, metadata=NEXT_CURSOR)


class ScorePageSchema(Schema):
    scores = fields.List(fields.Nested(ScoreEntryDocument), required=True)
    next_cursor = fields.String(required=True, allow_none=True, metadata=NEXT_CURSOR)


class OrderPageSchema(Schema):
    orders = fields.List(fields.Nested(OrderDocument), required=True)
    next_cursor = fields.String(required=True, allow_none=True, metadata=NEXT_CURSOR)


def _answer(name: str, page: Page) -> dict[str, Any]:
    cursor = None if page.next is None else encode_cursor(page.next)
    return {name: page.documents, "next_cursor": cursor}


def _known_customer(customer_id: str) -> dict[str, Any]:
    """customers/{customer_id}; 404 if there is none or the id cannot be one."""
    try:
        check_id(customer_id)
    except ValidationError:
        found = None
    else:
        found = store().customer_document(customer_id)
    if found is None:
        raise NotFound(f"No customer {customer_id!r}.")
    return found


# --- Endpoints ---

UNSIGNED = "No valid sign-in token"
NOT_STAFF = "Only for staff: viewers, analysts and admins"
NOT_FOUND = "No such customer"


@blp.route("/customers")
class Customers(MethodView):
    @blp.arguments(CustomerListArgs, location="query", unknown=RAISE)
    @blp.response(200, CustomerPageSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.doc(security=ACCESS["customers.Customers", "GET"].security)
    def get(self, args: dict[str, Any]) -> Any:
        """List customers, riskiest first.

        Sorted by churn risk, highest first, then by customer id; customers
        not scored yet come last. Filter by quadrant, by a minimum churn
        risk, or both.
        """
        page = store().list_customers(
            quadrant=args["quadrant"],
            min_churn_risk=args["min_churn_risk"],
            limit=args["limit"],
            after=args["cursor"],
        )
        return _answer("customers", page)


@blp.route("/customers/<customer_id>")
class Customer(MethodView):
    @blp.response(200, CustomerDocument)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(404, schema=ErrorSchema, description=NOT_FOUND)
    @blp.doc(security=ACCESS["customers.Customer", "GET"].security)
    def get(self, customer_id: str) -> Any:
        """One customer: expectations, latest order and latest scores."""
        return _known_customer(customer_id)


@blp.route("/customers/<customer_id>/scores")
class CustomerScores(MethodView):
    @blp.arguments(TimeListArgs, location="query", unknown=RAISE)
    @blp.response(200, ScorePageSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(404, schema=ErrorSchema, description=NOT_FOUND)
    @blp.doc(security=ACCESS["customers.CustomerScores", "GET"].security)
    def get(self, args: dict[str, Any], customer_id: str) -> Any:
        """A customer's score history, newest first: one entry per scoring."""
        _known_customer(customer_id)
        page = store().list_scores(
            customer_id, limit=args["limit"], after=args["cursor"]
        )
        return _answer("scores", page)


@blp.route("/customers/<customer_id>/orders")
class CustomerOrders(MethodView):
    @blp.arguments(TimeListArgs, location="query", unknown=RAISE)
    @blp.response(200, OrderPageSchema)
    @blp.alt_response(401, schema=ErrorSchema, description=UNSIGNED)
    @blp.alt_response(403, schema=ErrorSchema, description=NOT_STAFF)
    @blp.alt_response(404, schema=ErrorSchema, description=NOT_FOUND)
    @blp.doc(security=ACCESS["customers.CustomerOrders", "GET"].security)
    def get(self, args: dict[str, Any], customer_id: str) -> Any:
        """A customer's orders, newest placed first, with their gaps so far."""
        _known_customer(customer_id)
        page = store().list_orders(
            customer_id, limit=args["limit"], after=args["cursor"]
        )
        return _answer("orders", page)
