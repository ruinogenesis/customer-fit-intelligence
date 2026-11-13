import math
from functools import cache

import numpy as np
import pytest

from fitcore.features import INPUTS
from fitcore.sim_behaviour import BEHAVIOURS, Behaviour, simulate_behaviour
from fitcore.sim_customers import customer_table
from fitcore.sim_orders import Orders, simulate_orders, simulate_profiles


@cache
def simulated(n: int = 200, seed: int = 1) -> tuple[Orders, Behaviour]:
    orders = simulate_orders(simulate_profiles(n, seed=seed), seed=seed)
    return orders, simulate_behaviour(orders, seed=seed)


@cache
def table() -> dict[str, np.ndarray]:
    return customer_table(*simulated())


def rows_of(orders: Orders, customer: str) -> list[int]:
    return [i for i, c in enumerate(orders.customer_ids) if c == customer]


def test_one_row_per_customer_in_behaviour_order() -> None:
    orders, behaviour = simulated()
    t = table()
    assert tuple(t["customer_id"]) == behaviour.customer_ids
    assert t["orders"].sum() == len(orders)
    assert all(len(column) == len(behaviour.customer_ids) for column in t.values())


def test_gaps_are_each_customers_average() -> None:
    orders, _ = simulated()
    t = table()
    for r, customer in enumerate(t["customer_id"][:10]):
        mine = rows_of(orders, customer)
        for j, key in enumerate(INPUTS):
            assert t[key][r] == pytest.approx(orders.gaps[mine, j].mean())


def test_answers_are_each_customers_average_or_nan() -> None:
    orders, _ = simulated()
    t = table()
    for r, customer in enumerate(t["customer_id"]):
        for question, column in (("respect", "item_respect"), ("nps", "nps")):
            answers = [
                orders.survey_answers[i]
                for i in rows_of(orders, customer)
                if orders.survey_items[i] == question
            ]
            if answers:
                assert t[column][r] == pytest.approx(np.mean(answers))
            else:
                assert math.isnan(t[column][r])


def test_some_customers_never_answered_an_item() -> None:
    t = table()
    assert np.isnan(t["item_love"]).any()
    assert not np.isnan(t["item_love"]).all()


def test_rates_read_the_known_labels() -> None:
    orders, _ = simulated()
    t = table()
    for r, customer in enumerate(t["customer_id"][:20]):
        mine = rows_of(orders, customer)
        reorder = [orders.reorder_14d[i] for i in mine]
        churn = [orders.churn_30d[i] for i in mine]
        known_r = [x for x in reorder if x is not None]
        known_c = [x for x in churn if x is not None]
        if known_r:
            assert t["reorder"][r] == pytest.approx(np.mean(known_r))
        if known_c:
            assert t["stayed"][r] == pytest.approx(1.0 - np.mean(known_c))


def test_rates_stay_between_zero_and_one() -> None:
    t = table()
    for name in ("reorder", "stayed"):
        known = t[name][~np.isnan(t[name])]
        assert np.all((known >= 0.0) & (known <= 1.0))


def test_quick_checkout_flips_checkout_time() -> None:
    _, behaviour = simulated()
    t = table()
    assert "checkout_seconds" not in t
    seconds = behaviour.values["checkout_seconds"]
    assert np.allclose(t["quick_checkout"], -np.log(seconds))


def test_other_behaviours_and_true_means_are_carried_over() -> None:
    _, behaviour = simulated()
    t = table()
    for name in BEHAVIOURS:
        if name != "checkout_seconds":
            assert np.array_equal(t[name], behaviour.values[name])
    for latent, values in behaviour.true_means.items():
        assert np.array_equal(t[f"true_{latent}"], values)


def test_behaviour_must_match_the_orders() -> None:
    orders, _ = simulated()
    _, other_behaviour = simulated(n=50, seed=9)
    with pytest.raises(ValueError, match="exactly the customers"):
        customer_table(orders, other_behaviour)
