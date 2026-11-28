"""T207: simulated orders as v1 events, as fake customers' apps would send them.

T315 adds planting: plant=True gives most of some incidents to a few
restaurants and couriers, and changes nothing else.
"""

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
from fitcore.sim_events import (
    CONSENT_LEAD,
    CONSENT_RATE,
    ISSUE_TYPES,
    PLANT_SHARE,
    PLANTED,
    RATING_RATE,
    WITHDRAW_RATE,
    simulate_events,
    stars,
)
from fitcore.sim_orders import START, simulate_orders, simulate_profiles

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


# --- Planting (T315) ---


def incidents(customers: int, seed: int, days: float) -> dict[str, str | None]:
    """Each simulated order's incident, by order id."""
    profiles = simulate_profiles(customers, seed)
    orders = simulate_orders(profiles, seed, start=START, days=days)
    position = {profile.customer_id: i for i, profile in enumerate(profiles)}
    return {
        f"sim{seed}-{position[sim_id]:06d}-{orders.order_numbers[row]:03d}": (
            orders.incidents[row]
        )
        for row, sim_id in enumerate(orders.customer_ids)
    }


SERVED = {"order_placed": "restaurant_id", "order_delivered": "courier_id"}


def test_planting_moves_only_who_served_an_incident_order() -> None:
    plain = simulate_events(CUSTOMERS, SEED, DAYS)
    planted = simulate_events(CUSTOMERS, SEED, DAYS, plant=True)
    assert simulate_events(CUSTOMERS, SEED, DAYS, plant=False) == plain
    assert [e["event_id"] for e in planted] == [e["event_id"] for e in plain]
    kinds = incidents(CUSTOMERS, SEED, DAYS)
    moved = 0
    for was, now in zip(plain, planted, strict=True):
        if was == now:
            continue
        field = SERVED[was["type"]]
        assert {**now["payload"], field: was["payload"][field]} == was["payload"]
        assert {**now, "payload": was["payload"]} == was
        assert kinds[was["payload"]["order_id"]] in PLANTED
        assert now["payload"][field] in PLANTED[kinds[was["payload"]["order_id"]]][1]
        moved += 1
    assert moved > 0


def test_planted_names_take_most_of_their_incidents() -> None:
    customers, days = 100, 180.0
    kinds = incidents(customers, SEED, days)
    served = {}
    for event in simulate_events(customers, SEED, days, plant=True):
        if event["type"] in SERVED:
            field = SERVED[event["type"]]
            served[event["payload"]["order_id"], field] = event["payload"][field]
    for incident, (field, names) in PLANTED.items():
        orders = [order for order, kind in kinds.items() if kind == incident]
        share = sum(served[order, field] in names for order in orders) / len(orders)
        assert share == pytest.approx(PLANT_SHARE, abs=0.1), incident
        calm = [order for order, kind in kinds.items() if kind is None]
        calm_share = sum(served[order, field] in names for order in calm) / len(calm)
        assert calm_share < 0.15, incident  # elsewhere they are like the rest


def test_a_smaller_planted_run_is_part_of_a_bigger_one() -> None:
    small = simulate_events(5, SEED, 30.0, plant=True)
    big = {e["event_id"]: e for e in simulate_events(10, SEED, 60.0, plant=True)}
    assert all(big[event["event_id"]] == event for event in small)


# --- Marketing consent (T610) ---


def test_consent_adds_only_consent_events() -> None:
    plain = simulate_events(CUSTOMERS, SEED, DAYS)
    with_consent = simulate_events(CUSTOMERS, SEED, DAYS, consent=True)
    assert simulate_events(CUSTOMERS, SEED, DAYS, consent=False) == plain
    others = [e for e in with_consent if e["type"] != "consent_changed"]
    assert others == plain
    assert len(with_consent) > len(plain)
    for event in with_consent:
        load_event(event)
    times = [when(event) for event in with_consent]
    assert times == sorted(times)


def test_every_customer_answers_at_sign_up_just_before_their_first_order() -> None:
    run = simulate_events(CUSTOMERS, SEED, DAYS, consent=True)
    first: dict[str, datetime] = {}
    for event in run:
        if event["type"] == "order_placed":
            first.setdefault(event["customer_id"], when(event))
    signed_up = {
        e["customer_id"]: when(e)
        for e in run
        if e["type"] == "consent_changed" and e["payload"]["source"] == "sign_up"
    }
    assert signed_up == {c: at - CONSENT_LEAD for c, at in first.items()}


def test_about_consent_rate_say_yes_and_some_of_them_withdraw() -> None:
    run = simulate_events(400, SEED, 180.0, consent=True)
    answers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in run:
        if event["type"] == "consent_changed":
            answers[event["customer_id"]].append(event["payload"])
    yes = [c for c, a in answers.items() if a[0]["marketing"]]
    assert len(yes) / len(answers) == pytest.approx(CONSENT_RATE, abs=0.06)
    withdrew = [c for c, a in answers.items() if len(a) == 2]
    assert set(withdrew) <= set(yes)
    assert all(
        answers[c][1] == {"marketing": False, "source": "settings"} for c in withdrew
    )
    # Withdrawals land within DAYS of sign-up, so within the run for about half.
    assert 0 < len(withdrew) / len(yes) <= WITHDRAW_RATE + 0.05


def test_a_smaller_run_with_consent_is_part_of_a_bigger_one() -> None:
    small = simulate_events(5, SEED, 30.0, consent=True)
    big = {e["event_id"]: e for e in simulate_events(10, SEED, 60.0, consent=True)}
    assert all(big[event["event_id"]] == event for event in small)


def test_replayed_consent_ends_on_each_customers_last_answer() -> None:
    params = ModelParams.from_json(MODEL.read_bytes())
    customers: dict[str, Customer] = {}
    orders: dict[str, Order] = {}
    last: dict[str, bool] = {}
    for raw in simulate_events(CUSTOMERS, SEED, DAYS, consent=True):
        event = load_event(raw)
        cid = event["customer_id"]
        customer = customers.get(cid) or Customer.new(cid)
        order_id = order_id_for(event, customer)
        applied = apply_event(params, event, customer, orders.get(order_id or ""))
        customers[cid] = applied.customer
        if applied.order is not None:
            orders[applied.order.order_id] = applied.order
        if event["type"] == "consent_changed":
            last[cid] = event["payload"]["marketing"]
    assert last
    for cid, marketing in last.items():
        consent = customers[cid].consent
        assert consent is not None and consent.marketing is marketing
