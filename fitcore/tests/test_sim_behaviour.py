from dataclasses import FrozenInstanceError, replace
from functools import cache

import numpy as np
import pytest

from fitcore.model import LAYER_1, LAYER_2
from fitcore.sim_behaviour import (
    BEHAVIOURS,
    DEFAULT_BEHAVIOUR,
    Behaviour,
    simulate_behaviour,
)
from fitcore.sim_orders import Orders, simulate_orders, simulate_profiles
from fitcore.sim_survey import QUESTIONS

LATENTS = (*LAYER_1, *LAYER_2)


@cache
def simulated(n: int = 300, seed: int = 1) -> tuple[Orders, Behaviour]:
    orders = simulate_orders(simulate_profiles(n, seed=seed), seed=seed)
    return orders, simulate_behaviour(orders, seed=seed)


def test_every_latent_has_two_or_three_indicators() -> None:
    for latent in LATENTS:
        items = sum(1 for q in QUESTIONS if q == latent)
        behaviours = sum(1 for anchors in BEHAVIOURS.values() if anchors == latent)
        assert 2 <= items + behaviours <= 3


def test_one_row_per_customer_in_order_of_first_order() -> None:
    orders, behaviour = simulated()
    assert behaviour.customer_ids == tuple(dict.fromkeys(orders.customer_ids))
    assert behaviour.orders.sum() == len(orders)


def test_true_means_are_each_customers_average_latent() -> None:
    orders, behaviour = simulated()
    ids = np.array(orders.customer_ids)
    for row, customer in enumerate(behaviour.customer_ids[:20]):
        for latent in LATENTS:
            mine = orders.latents[latent][ids == customer]
            assert behaviour.true_means[latent][row] == pytest.approx(mine.mean())


def test_behaviours_stay_in_range() -> None:
    v = simulated()[1].values
    assert np.all(v["favorites_saved"] >= 0)
    assert np.all(v["favorites_saved"] == np.round(v["favorites_saved"]))
    for share in ("promo_free_share", "new_restaurant_share"):
        assert np.all((v[share] >= 0.0) & (v[share] <= 1.0))
    assert np.all((v["average_rating"] >= 1.0) & (v["average_rating"] <= 5.0))
    assert set(np.unique(v["saved_payment"])) <= {0.0, 1.0}
    assert np.all(v["checkout_seconds"] > 0.0)


def test_each_behaviour_follows_its_latent_with_realistic_noise() -> None:
    _, behaviour = simulated(n=1_000)
    for name, latent in BEHAVIOURS.items():
        r = np.corrcoef(behaviour.values[name], behaviour.true_means[latent])[0, 1]
        # a shorter checkout means more convenient, so that one runs negative
        if name == "checkout_seconds":
            r = -r
        assert 0.35 < r < 0.8, name


def test_the_same_seed_gives_the_same_behaviour() -> None:
    orders, behaviour = simulated()
    again = simulate_behaviour(orders, seed=1)
    for name in BEHAVIOURS:
        assert np.array_equal(again.values[name], behaviour.values[name])


def test_a_different_seed_changes_the_noise_but_not_the_truth() -> None:
    orders, behaviour = simulated()
    other = simulate_behaviour(orders, seed=2)
    assert not np.array_equal(
        other.values["checkout_seconds"], behaviour.values["checkout_seconds"]
    )
    for latent in LATENTS:
        assert np.array_equal(other.true_means[latent], behaviour.true_means[latent])


def test_behaviour_arrays_are_read_only() -> None:
    _, behaviour = simulated()
    with pytest.raises(ValueError, match="read-only"):
        behaviour.values["average_rating"][0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        behaviour.true_means["love"][0] = 1.0


def test_behaviour_columns_are_ready_for_a_table() -> None:
    _, behaviour = simulated()
    cols = behaviour.columns()
    assert list(cols)[:2] == ["customer_id", "orders"]
    assert set(BEHAVIOURS) | {f"true_{k}" for k in LATENTS} <= set(cols)
    assert all(len(c) == len(behaviour.customer_ids) for c in cols.values())


def test_behaviour_settings_are_frozen_and_checked() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_BEHAVIOUR.rating = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="rating"):
        replace(DEFAULT_BEHAVIOUR, rating=-1.0)
