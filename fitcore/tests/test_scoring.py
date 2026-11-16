"""T202: what each event measures, and how the scores follow from it."""

import dataclasses
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fitcore import expectations
from fitcore.features import INPUTS
from fitcore.model import LATENTS, ModelParams, forward, outcomes, quadrant
from fitcore.schemas import load_event
from fitcore.scoring import (
    PRIORS,
    Applied,
    Customer,
    EventConflict,
    Issue,
    Order,
    Scores,
    apply_event,
    measures,
    order_id_for,
    score,
)
from fitcore.sim_orders import PRIORS as SIM_PRIORS
from fitcore.sim_orders import simulate_orders, simulate_profiles

MODEL = Path(__file__).resolve().parents[2] / "notebooks/results/t112_model.json"
PARAMS = ModelParams.from_json(MODEL.read_bytes())
T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
# What order_placed records about an order, for orders built by hand.
RECORD: dict[str, Any] = {
    "restaurant_id": "rest_1",
    "basket_value": 50_000,
    "fees": 6_000,
    "promo_used": False,
}


def event(
    kind: str, payload: dict[str, Any], at: datetime = T0, customer: str = "cust_1"
) -> dict[str, Any]:
    """A raw event, validated by fitcore.schemas as the API will."""
    return load_event(
        {
            "event_id": str(uuid.uuid4()),
            "type": kind,
            "customer_id": customer,
            "occurred_at": at.isoformat(),
            "schema_version": 1,
            "payload": payload,
        }
    )


def placed(
    order_id: str = "ord_1",
    at: datetime = T0,
    basket: int = 50_000,
    fees: int = 6_000,
    promise: int = 30,
    customer: str = "cust_1",
) -> dict[str, Any]:
    payload = {
        "order_id": order_id,
        "restaurant_id": "rest_1",
        "basket_value": basket,
        "fees": fees,
        "promised_eta": (at + promise * MINUTE).isoformat(),
        "promo_used": False,
    }
    return event("order_placed", payload, at, customer)


def delivered(order_id: str, at: datetime, customer: str = "cust_1") -> dict[str, Any]:
    payload = {"order_id": order_id, "courier_id": "cour_1"}
    return event("order_delivered", payload, at, customer)


def rated(order_id: str, at: datetime, **scores: int) -> dict[str, Any]:
    return event("rating_submitted", {"order_id": order_id, **scores}, at)


def session(seconds: int | None, at: datetime) -> dict[str, Any]:
    payload = {"session_id": "s1", "time_to_checkout_s": seconds, "steps": 4}
    return event("app_session", {**payload, "errors": 0}, at)


def case(at: datetime, **fields: Any) -> dict[str, Any]:
    payload = {"case_id": "case_1", "opened_at": at.isoformat(), "refund_amount": 0}
    return event("support_case", {**payload, **fields}, at)


class Replay:
    """The caller's part, done as the API does it, with dicts for storage."""

    def __init__(self) -> None:
        self.customers: dict[str, Customer] = {}
        self.orders: dict[str, Order] = {}

    def send(self, ev: dict[str, Any]) -> Applied:
        cid = ev["customer_id"]
        customer = self.customers.get(cid) or Customer.new(cid)
        order_id = order_id_for(ev, customer)
        order = self.orders.get(order_id) if order_id else None
        applied = apply_event(PARAMS, ev, customer, order)
        self.customers[cid] = applied.customer
        if applied.order is not None:
            self.orders[applied.order.order_id] = applied.order
        return applied


# --- What each event measures ---


def test_priors_are_the_simulators() -> None:
    assert dict(PRIORS) == dict(SIM_PRIORS)


@pytest.mark.parametrize(
    ("ev", "expected"),
    [
        (placed(basket=50_000, fees=6_000), {"x3": 0.12}),
        (placed(basket=0, fees=0), {}),
        (session(75, T0), {"x2": 75.0}),
        (session(None, T0), {}),
        (rated("ord_1", T0, food=4, courier=5, accuracy=3), {"x5": 5.0, "x6": 4.0}),
        (rated("ord_1", T0, overall=2), {}),
        (case(T0, satisfaction=2), {"x8": 2.0}),
        (case(T0), {}),
        (event("order_issue", {"order_id": "ord_1", "issue_type": "cold"}), {}),
        (event("survey_response", {"item_id": "love", "value": 6}), {}),
    ],
)
def test_measures(ev: dict[str, Any], expected: dict[str, float]) -> None:
    assert measures(ev) == expected


# --- Scores ---


def test_scores_follow_the_models_latent_order() -> None:
    names = [f.name for f in dataclasses.fields(Scores)]
    assert names == [*LATENTS, "churn_risk", "quadrant"]


def test_an_input_missing_from_the_gaps_counts_as_zero() -> None:
    row = np.zeros((1, len(INPUTS)))
    row[0, 0] = -10.0
    latents = forward(PARAMS, row)[0]
    churn = outcomes(PARAMS, latents[-1:]).churn[0]
    place = quadrant(PARAMS, latents[0], latents[1])
    expected = Scores(*latents.tolist(), churn_risk=float(churn), quadrant=place)
    assert score(PARAMS, {"x1": -10.0}) == expected


def test_score_rejects_an_unknown_input() -> None:
    with pytest.raises(ValueError, match="x9"):
        score(PARAMS, {"x9": 1.0})


def test_the_quadrant_follows_the_scores_and_the_models_lines() -> None:
    assert score(PARAMS, {}).quadrant == "lovemark"
    bad_delivery = score(PARAMS, {"x1": -15.0, "x5": -2.5, "x6": 0.7})
    assert bad_delivery.respect < 0.5 <= bad_delivery.love
    assert bad_delivery.quadrant == "fad"
    strict = dataclasses.replace(PARAMS, respect_line=0.95, love_line=0.95)
    assert score(strict, {}).quadrant == "product"
    with pytest.raises(ValueError, match="quadrant"):
        dataclasses.replace(score(PARAMS, {}), quadrant="cult")


# --- Applying events ---


def test_a_first_order_is_judged_against_the_priors() -> None:
    replay = Replay()
    applied = replay.send(placed(basket=50_000, fees=7_500))
    assert applied.order is not None
    assert applied.order.gaps == {"x3": pytest.approx(0.12 - 0.15)}
    assert applied.scored == score(PARAMS, applied.order.gaps)
    assert applied.customer.scores == applied.scored
    assert applied.customer.latest_order_id == "ord_1"
    assert applied.customer.baselines["x3"].weight == 1.0


def test_a_late_delivery_lowers_the_scores() -> None:
    replay = Replay()
    on_time = replay.send(placed(promise=30)).scored
    late = replay.send(delivered("ord_1", T0 + 45 * MINUTE))
    assert late.order is not None
    assert late.order.gaps["x1"] == pytest.approx(-15.0)
    assert on_time is not None and late.scored is not None
    assert late.scored.fit < on_time.fit
    assert late.scored.churn_risk > on_time.churn_risk


def test_ratings_are_judged_against_the_customers_own_habit() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0))
    first = replay.send(rated("ord_1", T0 + 60 * MINUTE, courier=5))
    assert first.order is not None
    assert first.order.gaps["x5"] == pytest.approx(5.0 - 4.5)
    habit = first.customer.baselines["x5"]
    at = T0 + 7 * 24 * 60 * MINUTE
    replay.send(placed("ord_2", at))
    second = replay.send(rated("ord_2", at, courier=5))
    assert second.order is not None
    usual = expectations.expected(habit, PRIORS["x5"], at)
    assert second.order.gaps["x5"] == pytest.approx(5.0 - usual)
    assert 4.5 < usual < 5.0


def test_each_order_is_scored_on_its_own_gaps() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0))
    replay.send(delivered("ord_1", T0 + 60 * MINUTE))
    applied = replay.send(placed("ord_2", T0 + 24 * 60 * MINUTE))
    assert applied.order is not None
    assert set(applied.order.gaps) == {"x3"}
    assert applied.customer.scores == score(PARAMS, applied.order.gaps)


def test_a_late_rating_for_an_older_order_leaves_the_current_scores() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0))
    now = replay.send(placed("ord_2", T0 + 24 * 60 * MINUTE)).customer.scores
    applied = replay.send(rated("ord_1", T0 + 25 * 60 * MINUTE, food=1))
    assert applied.order is not None and applied.order.order_id == "ord_1"
    assert applied.scored is not None
    assert applied.customer.scores == now


def test_an_older_order_arriving_late_does_not_become_the_latest() -> None:
    replay = Replay()
    now = replay.send(placed("ord_2", T0)).customer.scores
    applied = replay.send(placed("ord_1", T0 - 60 * MINUTE, fees=9_000))
    assert applied.scored is not None and applied.scored != now
    assert applied.customer.latest_order_id == "ord_2"
    assert applied.customer.scores == now


def test_a_session_counts_toward_the_latest_order() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0))
    applied = replay.send(session(80, T0 + MINUTE))
    assert applied.order is not None and applied.order.order_id == "ord_1"
    assert applied.order.gaps["x2"] == pytest.approx(90.0 - 80.0)


@pytest.mark.parametrize("before", [True, False])
def test_a_session_with_no_order_of_its_own_only_updates_the_habit(
    before: bool,
) -> None:
    replay = Replay()
    if before:
        replay.send(placed("ord_1", T0))
    applied = replay.send(session(80, T0 - MINUTE))
    assert applied.order is None
    assert applied.scored is None
    assert applied.customer.baselines["x2"].weight == 1.0


def test_a_support_case_without_an_order_counts_toward_the_latest() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0))
    applied = replay.send(case(T0 + 60 * MINUTE, satisfaction=2))
    assert applied.order is not None and applied.order.order_id == "ord_1"
    assert applied.order.gaps["x8"] == pytest.approx(2.0 - 4.0)


@pytest.mark.parametrize(
    "ev",
    [
        session(None, T0 + MINUTE),
        event("survey_response", {"item_id": "trust", "value": 2}),
    ],
)
def test_an_event_that_measures_nothing_changes_nothing(ev: dict[str, Any]) -> None:
    replay = Replay()
    before = replay.send(placed("ord_1", T0)).customer
    applied = replay.send(ev)
    assert applied == Applied(customer=before, order=None, scored=None)


@pytest.mark.parametrize(
    "ev",
    [
        rated("ord_1", T0 + MINUTE, overall=3),
        event("order_issue", {"order_id": "ord_1", "issue_type": "late"}),
    ],
)
def test_an_event_that_only_adds_to_the_record_changes_no_score(
    ev: dict[str, Any],
) -> None:
    replay = Replay()
    first = replay.send(placed("ord_1", T0))
    applied = replay.send(ev)
    assert first.order is not None and applied.order is not None
    assert (applied.customer, applied.scored) == (first.customer, None)
    assert applied.order.gaps == first.order.gaps
    assert applied.order != first.order


# --- The order's record ---


def test_an_order_keeps_its_record() -> None:
    replay = Replay()
    replay.send(placed("ord_1", T0, basket=42_000, fees=3_500))
    replay.send(rated("ord_1", T0 + 50 * MINUTE, food=2, overall=3))
    replay.send(event("order_issue", {"order_id": "ord_1", "issue_type": "cold"}))
    replay.send(delivered("ord_1", T0 + 40 * MINUTE))
    hour = 60 * MINUTE
    late = event("order_issue", {"order_id": "ord_1", "issue_type": "late"}, T0 + hour)
    replay.send(late)
    replay.send(rated("ord_1", T0 + 2 * hour, food=4, courier=5))
    order = replay.orders["ord_1"]
    assert (order.restaurant_id, order.basket_value, order.fees) == (
        "rest_1",
        42_000,
        3_500,
    )
    assert order.promo_used is False
    assert (order.courier_id, order.delivered_at) == ("cour_1", T0 + 40 * MINUTE)
    assert order.issues == (Issue("cold", T0), Issue("late", T0 + hour))
    assert order.ratings == {"food": 4, "overall": 3, "courier": 5}


def _conflicts() -> dict[str, tuple[list[dict[str, Any]], dict[str, Any], str]]:
    later = T0 + 60 * MINUTE
    return {
        "placed twice": ([placed()], placed(), "already placed"),
        "delivered, never placed": ([], delivered("ord_1", later), "not been placed"),
        "rated, never placed": ([], rated("ord_1", later, food=3), "not been placed"),
        "case names an unknown order": (
            [],
            case(later, order_id="ord_9", satisfaction=3),
            "not been placed",
        ),
        "another customer's order": (
            [placed()],
            delivered("ord_1", later, customer="cust_2"),
            "another customer",
        ),
        "delivered before placed": (
            [placed()],
            delivered("ord_1", T0 - MINUTE),
            "delivered before",
        ),
    }


@pytest.mark.parametrize("name", list(_conflicts()))
def test_events_that_do_not_fit_are_conflicts(name: str) -> None:
    history, ev, message = _conflicts()[name]
    replay = Replay()
    for earlier in history:
        replay.send(earlier)
    with pytest.raises(EventConflict, match=message):
        replay.send(ev)


def test_apply_event_checks_its_caller() -> None:
    ev = placed()
    with pytest.raises(ValueError, match="another customer"):
        apply_event(PARAMS, ev, Customer.new("cust_2"), None)
    order = Order("ord_7", "cust_1", T0, T0 + 30 * MINUTE, **RECORD)
    with pytest.raises(ValueError, match="ord_7"):
        apply_event(PARAMS, delivered("ord_1", T0), Customer.new("cust_1"), order)


def test_state_is_checked_and_read_only() -> None:
    customer = Customer.new("cust_1")
    order = Order("ord_1", "cust_1", T0, T0, {"x1": 1.0}, **RECORD, ratings={"food": 4})
    with pytest.raises(TypeError):
        customer.baselines["x2"] = expectations.Baseline()  # type: ignore[index]
    with pytest.raises(TypeError):
        order.gaps["x1"] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        order.ratings["food"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="baselines"):
        Customer("cust_1", {"x2": expectations.Baseline()})
    with pytest.raises(ValueError, match="together"):
        Customer("cust_1", customer.baselines, latest_order_id="ord_1")
    with pytest.raises(ValueError, match="x9"):
        Order("ord_1", "cust_1", T0, T0, {"x9": 1.0}, **RECORD)
    with pytest.raises(ValueError, match="taste"):
        Order("ord_1", "cust_1", T0, T0, **RECORD, ratings={"taste": 4})
    with pytest.raises(ValueError, match="together"):
        Order("ord_1", "cust_1", T0, T0, **RECORD, courier_id="cour_1")


# --- Serving matches training ---


def test_replaying_simulated_orders_reproduces_the_simulators_gaps() -> None:
    """Events built from simulated orders give the gaps the model trained on.

    These events skip the schema, so they can carry the simulator's exact
    (non-integer) values: fees with a basket of 1, seconds, ratings.
    """
    orders = simulate_orders(simulate_profiles(30, 2026), 2026)
    replay = Replay()
    checked = ("x1", "x2", "x3", "x5", "x6", "x8")
    for row in range(len(orders)):
        cid, at = orders.customer_ids[row], orders.placed_at[row]
        oid = f"{cid}-{orders.order_numbers[row]}"
        value = dict(zip(INPUTS, orders.measured[row].tolist(), strict=True))
        promise = float(orders.expected[row, 0])
        payloads = [
            (
                "order_placed",
                at,
                {
                    "order_id": oid,
                    "restaurant_id": "rest_1",
                    "basket_value": 1.0,
                    "fees": value["x3"],
                    "promised_eta": at + promise * MINUTE,
                    "promo_used": False,
                },
            ),
            ("app_session", at, {"time_to_checkout_s": value["x2"]}),
            (
                "order_delivered",
                at + value["x1"] * MINUTE,
                {"order_id": oid, "courier_id": "cour_1"},
            ),
            (
                "rating_submitted",
                at,
                {"order_id": oid, "courier": value["x5"], "food": value["x6"]},
            ),
            ("support_case", at, {"order_id": oid, "satisfaction": value["x8"]}),
        ]
        for kind, when, payload in payloads:
            ev = {"type": kind, "customer_id": cid, "occurred_at": when}
            replay.send({**ev, "payload": payload})
        gaps = replay.orders[oid].gaps
        for key in checked:
            truth = float(orders.gaps[row, INPUTS.index(key)])
            assert gaps[key] == pytest.approx(truth, abs=1e-6 if key == "x1" else 1e-12)
    assert len(replay.orders) == len(orders) > 300
