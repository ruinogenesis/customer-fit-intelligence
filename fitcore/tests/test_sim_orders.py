from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fitcore.features import INPUTS
from fitcore.sim_orders import (
    INCIDENTS,
    LATENESS_MEAN,
    MEASURES,
    PRIORS,
    ServiceProfile,
    draw_order,
    simulate_profiles,
)


def steady(profile: ServiceProfile) -> ServiceProfile:
    """The same customer, but nothing ever goes wrong."""
    return replace(profile, incident_rate=0.0)


def draws(profile: ServiceProfile, n: int, seed: int = 0) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [draw_order(profile, rng)[0] for _ in range(n)]


def test_every_input_has_a_measure() -> None:
    assert tuple(MEASURES) == INPUTS


def test_priors_are_the_population_usual_for_x2_to_x8() -> None:
    assert tuple(PRIORS) == INPUTS[1:]
    assert all(PRIORS[key] == MEASURES[key].usual for key in PRIORS)


def test_every_incident_hits_a_real_input() -> None:
    assert all(key in INPUTS for key, _ in INCIDENTS.values())


def test_the_same_seed_gives_the_same_profiles() -> None:
    assert simulate_profiles(50, seed=4) == simulate_profiles(50, seed=4)
    assert simulate_profiles(50, seed=4) != simulate_profiles(50, seed=5)


def test_profiles_are_frozen() -> None:
    profile = simulate_profiles(1, seed=1)[0]
    with pytest.raises(FrozenInstanceError):
        profile.incident_rate = 0.5  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.usual["x1"] = 0.0  # type: ignore[index]


def test_usual_values_stay_in_range() -> None:
    for profile in simulate_profiles(2_000, seed=1):
        for key, m in MEASURES.items():
            assert m.low <= profile.usual[key] <= m.high


def test_promise_is_usual_delivery_minus_lateness() -> None:
    profiles = simulate_profiles(5_000, seed=1)
    lateness = [p.usual["x1"] - p.promise for p in profiles]
    assert np.mean(lateness) == pytest.approx(LATENESS_MEAN, abs=0.2)


def test_incident_rates_vary_around_ten_percent() -> None:
    rates = np.array([p.incident_rate for p in simulate_profiles(5_000, seed=1)])
    assert np.all((rates > 0.0) & (rates < 1.0))
    assert rates.mean() == pytest.approx(0.1, abs=0.01)
    assert rates.std() > 0.03


def test_order_frequency_varies_around_a_week() -> None:
    days = [p.days_between_orders for p in simulate_profiles(5_000, seed=1)]
    assert np.median(days) == pytest.approx(7.0, rel=0.05)


def test_a_steady_order_is_the_usual_plus_noise() -> None:
    profile = steady(simulate_profiles(1, seed=1)[0])
    orders = draws(profile, 5_000)
    for key in ("x2", "x3"):
        values = np.array([order[key] for order in orders])
        assert values.mean() == pytest.approx(profile.usual[key], rel=0.01)
        assert values.std() == pytest.approx(MEASURES[key].noise, rel=0.1)


def test_a_steady_customer_never_has_an_incident() -> None:
    rng = np.random.default_rng(0)
    profile = steady(simulate_profiles(1, seed=1)[0])
    assert all(draw_order(profile, rng)[1] is None for _ in range(1_000))


def test_incidents_make_their_input_worse() -> None:
    profile = replace(simulate_profiles(1, seed=1)[0], incident_rate=1.0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        measured, incident = draw_order(profile, rng)
        assert incident in INCIDENTS
        key, effect = INCIDENTS[incident]
        # late (x1) means more minutes; the others mean lower values
        if effect > 0:
            assert measured[key] > profile.usual[key]
        else:
            assert measured[key] < profile.usual[key]


def test_incident_types_are_equally_likely() -> None:
    profile = replace(simulate_profiles(1, seed=1)[0], incident_rate=1.0)
    rng = np.random.default_rng(0)
    seen = [draw_order(profile, rng)[1] for _ in range(5_000)]
    for name in INCIDENTS:
        assert seen.count(name) / len(seen) == pytest.approx(0.2, abs=0.03)


def test_measurements_stay_in_range() -> None:
    profile = replace(simulate_profiles(1, seed=1)[0], incident_rate=0.5)
    for order in draws(profile, 2_000):
        for key, m in MEASURES.items():
            assert m.low <= order[key] <= m.high


def test_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        simulate_profiles(0, seed=1)
