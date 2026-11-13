from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from functools import cache
from types import MappingProxyType

import numpy as np
import pytest

from fitcore import expectations, sim_outcomes
from fitcore.features import INPUTS, input_gap
from fitcore.sim import true_latents
from fitcore.sim_orders import (
    CHURN_WINDOW,
    DAYS,
    INCIDENTS,
    LATENESS_MEAN,
    MEASURES,
    PRIORS,
    REORDER_WINDOW,
    START,
    Orders,
    ServiceProfile,
    draw_order,
    simulate_orders,
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


@cache
def history(n: int = 30, seed: int = 1) -> tuple[tuple[ServiceProfile, ...], Orders]:
    profiles = simulate_profiles(n, seed=seed)
    return profiles, simulate_orders(profiles, seed=seed)


def test_every_order_has_all_eight_gaps() -> None:
    _, orders = history()
    assert len(orders) > 0
    assert orders.gaps.shape == (len(orders), 8)
    assert np.all(np.isfinite(orders.gaps))


def test_orders_run_in_time_order_inside_the_window() -> None:
    _, orders = history()
    end = START + timedelta(days=DAYS)
    for i in range(len(orders)):
        assert START <= orders.placed_at[i] < end
        if orders.order_numbers[i] > 1:
            assert orders.customer_ids[i - 1] == orders.customer_ids[i]
            assert orders.order_numbers[i - 1] == orders.order_numbers[i] - 1
            assert orders.placed_at[i - 1] < orders.placed_at[i]


def test_a_first_order_is_judged_against_the_promise_and_the_priors() -> None:
    profiles, orders = history()
    promise = {p.customer_id: p.promise for p in profiles}
    for i in range(len(orders)):
        if orders.order_numbers[i] == 1:
            assert orders.expected[i, 0] == promise[orders.customer_ids[i]]
            assert orders.expected[i, 1:].tolist() == list(PRIORS.values())


def test_x1_is_always_judged_against_the_promise() -> None:
    profiles, orders = history()
    promise = {p.customer_id: p.promise for p in profiles}
    for i in range(len(orders)):
        assert orders.expected[i, 0] == promise[orders.customer_ids[i]]


def test_gaps_follow_fitcore_features() -> None:
    _, orders = history(n=5)
    for i in range(len(orders)):
        for j, key in enumerate(INPUTS):
            want = input_gap(key, orders.measured[i, j], orders.expected[i, j])
            assert orders.gaps[i, j] == want


def test_expectations_replay_with_fitcore_expectations() -> None:
    _, orders = history(n=3)
    for customer in set(orders.customer_ids):
        rows = [i for i in range(len(orders)) if orders.customer_ids[i] == customer]
        for j, key in enumerate(INPUTS[1:], start=1):
            baseline = expectations.Baseline()
            for i in rows:
                at = orders.placed_at[i]
                want = expectations.expected(baseline, PRIORS[key], at)
                assert orders.expected[i, j] == pytest.approx(want)
                baseline = expectations.update(baseline, orders.measured[i, j], at)


def test_expectations_adapt_to_a_customers_usual() -> None:
    profile = steady(simulate_profiles(1, seed=1)[0])
    usual = dict(profile.usual)
    usual["x2"] = 150.0  # a slow checkout, against a prior of 90 seconds
    slow = replace(profile, usual=MappingProxyType(usual), days_between_orders=3.0)
    x2 = simulate_orders([slow], seed=2).gaps[:, 1]
    assert x2[0] < -40.0  # the first order feels slow
    assert np.abs(x2[10:]).mean() < 0.5 * abs(x2[0])  # later, it is expected


def test_incidents_show_up_as_shortfalls() -> None:
    _, orders = history(n=100)
    for i, incident in enumerate(orders.incidents):
        if incident is not None:
            key, _ = INCIDENTS[incident]
            assert orders.gaps[i, INPUTS.index(key)] < 0.0


def test_true_latents_come_from_the_gaps() -> None:
    _, orders = history()
    for key, values in true_latents(orders.gaps).items():
        assert np.array_equal(orders.latents[key], values)


def test_incident_orders_have_lower_respect() -> None:
    _, orders = history(n=300)
    respect = orders.latents["respect"]
    hit = np.array([incident is not None for incident in orders.incidents])
    assert respect[hit].mean() < respect[~hit].mean() - 0.1


def test_customers_with_more_incidents_have_lower_respect() -> None:
    profiles, orders = history(n=1_000)
    ids = np.array(orders.customer_ids)
    respect = orders.latents["respect"]
    rates, means = [], []
    for p in profiles:
        mine = respect[ids == p.customer_id]
        if len(mine) > 0:
            rates.append(p.incident_rate)
            means.append(mine.mean())
    assert np.corrcoef(rates, means)[0, 1] < -0.1


def test_a_customers_history_does_not_depend_on_the_others() -> None:
    profiles = simulate_profiles(20, seed=1)
    alone = simulate_orders(profiles[:1], seed=9)
    together = simulate_orders(profiles, seed=9)
    n = len(alone)
    assert together.customer_ids[:n] == alone.customer_ids
    assert np.array_equal(together.gaps[:n], alone.gaps)


def test_the_same_seed_gives_the_same_orders() -> None:
    profiles = simulate_profiles(10, seed=1)
    a, b = simulate_orders(profiles, seed=3), simulate_orders(profiles, seed=3)
    assert a.placed_at == b.placed_at
    assert a.incidents == b.incidents
    assert np.array_equal(a.measured, b.measured)


def test_order_tables_are_read_only() -> None:
    _, orders = history(n=2)
    tables = (orders.measured, orders.expected, orders.gaps, orders.latents["love"])
    for table in tables:
        with pytest.raises(ValueError, match="read-only"):
            table[0] = 0.0


def test_order_columns_are_ready_for_a_table() -> None:
    _, orders = history(n=3)
    cols = orders.columns()
    assert list(cols)[:4] == ["customer_id", "order_number", "placed_at", "incident"]
    assert {"x1", "x1_measured", "x1_expected", "respect"} <= set(cols)
    assert all(len(column) == len(orders) for column in cols.values())
    assert set(cols["incident"]) <= {"none", *INCIDENTS}


def test_days_must_be_positive() -> None:
    with pytest.raises(ValueError, match="days"):
        simulate_orders(simulate_profiles(1, seed=1), seed=1, days=0.0)


def test_start_must_be_timezone_aware() -> None:
    naive = datetime(2026, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        simulate_orders(simulate_profiles(1, seed=1), seed=1, start=naive)


def next_is_same_customer(orders: Orders, i: int) -> bool:
    return i + 1 < len(orders) and orders.customer_ids[i + 1] == orders.customer_ids[i]


def known_rate(labels: tuple[bool | None, ...], keep: np.ndarray) -> float:
    known = [bool(x) for x, k in zip(labels, keep, strict=True) if k and x is not None]
    return sum(known) / len(known)


def test_every_order_has_a_true_fit() -> None:
    _, orders = history()
    assert orders.fit.shape == (len(orders),)
    for i in range(len(orders)):
        trust = orders.latents["trust"][i]
        convenience = orders.latents["convenience"][i]
        assert orders.fit[i] == pytest.approx(sim_outcomes.fit(trust, convenience))


def test_about_one_order_in_five_gets_an_nps_answer() -> None:
    _, orders = history(n=500)
    answered = [a for a in orders.nps if a is not None]
    assert len(answered) / len(orders) == pytest.approx(0.2, abs=0.02)
    assert set(answered) <= set(range(11))


def test_a_customer_who_stops_places_no_more_orders() -> None:
    _, orders = history(n=300)
    assert any(orders.stopped)
    for i, stopped in enumerate(orders.stopped):
        if stopped:
            assert not next_is_same_customer(orders, i)


def test_reorder_label_reads_the_history() -> None:
    _, orders = history(n=200)
    end = START + timedelta(days=DAYS)
    for i, label in enumerate(orders.reorder_14d):
        at = orders.placed_at[i]
        if next_is_same_customer(orders, i):
            assert label == (orders.placed_at[i + 1] - at <= REORDER_WINDOW)
        elif at + REORDER_WINDOW <= end:
            assert label is False
        else:
            assert label is None  # the window runs past the simulated period


def test_churn_label_reads_the_history() -> None:
    _, orders = history(n=200)
    end = START + timedelta(days=DAYS)
    for i, label in enumerate(orders.churn_30d):
        at = orders.placed_at[i]
        if next_is_same_customer(orders, i):
            assert label == (orders.placed_at[i + 1] - at > CHURN_WINDOW)
        elif at + CHURN_WINDOW <= end:
            assert label is True
        else:
            assert label is None  # the window runs past the simulated period


def test_outcomes_respond_to_fit_the_way_the_design_says() -> None:
    _, orders = history(n=1_000)
    high = orders.fit >= np.median(orders.fit)
    low = ~high
    # higher fit: more reorders within 14 days
    assert (
        known_rate(orders.reorder_14d, high)
        > known_rate(orders.reorder_14d, low) + 0.05
    )
    # higher fit: less churn within 30 days
    assert known_rate(orders.churn_30d, high) < known_rate(orders.churn_30d, low) - 0.02
    # higher fit: higher NPS answers
    nps = np.array([np.nan if a is None else a for a in orders.nps])
    assert np.nanmean(nps[high]) > np.nanmean(nps[low]) + 1.0


def test_outcomes_are_in_the_table() -> None:
    _, orders = history(n=3)
    cols = orders.columns()
    assert {"fit", "nps", "stopped", "reorder_14d", "churn_30d"} <= set(cols)
    with pytest.raises(ValueError, match="read-only"):
        orders.fit[0] = 0.0
