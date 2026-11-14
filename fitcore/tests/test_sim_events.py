"""T207: simulated orders as v1 events, as fake customers' apps would send them."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from fitcore.features import INPUTS
from fitcore.model import ModelParams
from fitcore.schemas import load_event
from fitcore.scoring import Customer, Order, apply_event, order_id_for
from fitcore.sim_events import ISSUE_TYPES, RATING_RATE, simulate_events, stars
from fitcore.sim_orders import simulate_orders, simulate_profiles

MODEL = Path(__file__).resolve().parents[2] / "notebooks/results/t112_model.json"
CUSTOMERS, SEED, DAYS = 20, 3, 90.0
MINUTE = timedelta(minutes=1)


@cache
def events() -> tuple[dict[str, Any], ...]:
    """The run most tests share. Treat it as read-only."""
    return tuple(simulate_events(CUSTOMERS, SEED, DAYS))


def when(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(event["occurred_at"])


def by_order(run: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Each order's events by type. Every event names its order."""
    orders: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in run:
        payload = event["payload"]
        order_id = payload.get("order_id") or payload["session_id"].removesuffix(
            "-session"
        )
        assert event["type"] not in orders[order_id]
        orders[order_id][event["type"]] = event
    return orders


def test_every_event_is_a_valid_v1_event() -> None:
    for event in events():
        load_event(event)


def test_events_are_in_time_order() -> None:
    times = [when(event) for event in events()]
    assert times == sorted(times)


def test_the_same_arguments_always_give_the_same_events() -> None:
    assert simulate_events(CUSTOMERS, SEED, DAYS) == list(events())
    ids = [event["event_id"] for event in events()]
    assert len(set(ids)) == len(ids)


def test_runs_with_different_seeds_share_no_customer_and_no_event() -> None:
    other = simulate_events(CUSTOMERS, SEED + 1, DAYS)
    assert {e["customer_id"] for e in other}.isdisjoint(
        {e["customer_id"] for e in events()}
    )
    assert {e["event_id"] for e in other}.isdisjoint({e["event_id"] for e in events()})


def test_a_smaller_run_is_part_of_a_bigger_one() -> None:
    bigger = {event["event_id"]: event for event in events()}
    smaller = simulate_events(5, SEED, 30.0)
    assert smaller
    assert all(bigger[event["event_id"]] == event for event in smaller)


def test_each_order_reports_what_the_simulator_drew() -> None:
    profiles = simulate_profiles(CUSTOMERS, SEED)
    orders = simulate_orders(profiles, SEED, days=DAYS)
    position = {profile.customer_id: i for i, profile in enumerate(profiles)}
    promise = {profile.customer_id: profile.promise for profile in profiles}
    sent = by_order(events())
    assert len(sent) == len(orders)
    for row, sim_id in enumerate(orders.customer_ids):
        x = dict(zip(INPUTS, orders.measured[row], strict=True))
        number = orders.order_numbers[row]
        mine = sent[f"sim{SEED}-{position[sim_id]:06d}-{number:03d}"]
        placed = mine["order_placed"]["payload"]
        at = when(mine["order_placed"])
        assert at == orders.placed_at[row]
        eta = datetime.fromisoformat(placed["promised_eta"])
        assert (eta - at) / MINUTE == pytest.approx(promise[sim_id])
        basket = placed["basket_value"]
        assert placed["fees"] / basket == pytest.approx(x["x3"], abs=0.5 / basket)
        checkout = mine["app_session"]["payload"]["time_to_checkout_s"]
        assert checkout == round(x["x2"])
        delivered = when(mine["order_delivered"])
        assert (delivered - at) / MINUTE == pytest.approx(x["x1"])

        incident = orders.incidents[row]
        assert ("support_case" in mine) == (incident is not None)
        assert ("order_issue" in mine) == (incident in ISSUE_TYPES)
        if "order_issue" in mine:
            issue = mine["order_issue"]["payload"]["issue_type"]
            assert issue == ISSUE_TYPES[incident]
        if "support_case" in mine:
            assert "rating_submitted" in mine
            case = mine["support_case"]["payload"]
            assert case["satisfaction"] == stars(x["x8"])
        if "rating_submitted" in mine:
            rating = mine["rating_submitted"]["payload"]
            assert (rating["courier"], rating["food"]) == (
                stars(x["x5"]),
                stars(x["x6"]),
            )
            assert when(mine["rating_submitted"]) > delivered


def test_about_rating_rate_of_orders_without_an_incident_get_a_rating() -> None:
    calm = [
        sent
        for sent in by_order(simulate_events(100, SEED, 180.0)).values()
        if "support_case" not in sent
    ]
    rated = sum("rating_submitted" in sent for sent in calm)
    assert rated / len(calm) == pytest.approx(RATING_RATE, abs=0.03)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.2, 1), (1.49, 1), (1.5, 2), (4.3, 4), (4.5, 5), (5.7, 5)],
)
def test_stars_are_the_nearest_whole_star_from_1_to_5(
    value: float, expected: int
) -> None:
    assert stars(value) == expected


def test_replaying_them_scores_every_customer_and_moves_some_across_lines() -> None:
    params = ModelParams.from_json(MODEL.read_bytes())
    customers: dict[str, Customer] = {}
    orders: dict[str, Order] = {}
    quadrants: set[str] = set()
    for raw in events():
        event = load_event(raw)
        customer_id = event["customer_id"]
        customer = customers.get(customer_id) or Customer.new(customer_id)
        wanted = order_id_for(event, customer)
        order = None if wanted is None else orders.get(wanted)
        applied = apply_event(params, event, customer, order)
        customers[customer_id] = applied.customer
        if applied.order is not None:
            orders[applied.order.order_id] = applied.order
        if applied.scored is not None:
            quadrants.add(applied.scored.quadrant)
    assert len(customers) > 1
    assert all(customer.scores is not None for customer in customers.values())
    # Each session's checkout time went to its own order.
    assert all({"x1", "x2", "x3"} <= set(order.gaps) for order in orders.values())
    assert len(quadrants) >= 2
