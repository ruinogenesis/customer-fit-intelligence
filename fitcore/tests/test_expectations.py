"""Tests for fitcore.expectations: EWMA baselines with a cold-start prior."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from fitcore.expectations import Baseline, expected, update

T0 = datetime(2026, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def test_new_baseline_is_empty() -> None:
    b = Baseline()
    assert b.mean == 0.0
    assert b.weight == 0.0
    assert b.last_seen is None


def test_first_observation_becomes_the_mean() -> None:
    b = update(Baseline(), 10.0, T0)
    assert b.mean == pytest.approx(10.0)
    assert b.weight == pytest.approx(1.0)
    assert b.last_seen == T0


def test_first_three_observations_average_evenly() -> None:
    # Same instant each time, so no forgetting gets in the way of the arithmetic
    b = update(Baseline(), 10.0, T0)
    b = update(b, 20.0, T0)
    assert b.mean == pytest.approx(15.0)
    b = update(b, 30.0, T0)
    assert b.mean == pytest.approx(20.0)
    assert b.weight == pytest.approx(3.0)


def test_fourth_observation_uses_alpha() -> None:
    b = Baseline(mean=20.0, weight=3.0, last_seen=T0)
    b = update(b, 40.0, T0)
    assert b.mean == pytest.approx(0.3 * 40.0 + 0.7 * 20.0)  # 26.0
    assert b.weight == pytest.approx(4.0)


def test_update_returns_a_new_baseline() -> None:
    before = Baseline(mean=20.0, weight=3.0, last_seen=T0)
    after = update(before, 40.0, T0 + DAY)
    assert after is not before
    assert before == Baseline(mean=20.0, weight=3.0, last_seen=T0)
    with pytest.raises(FrozenInstanceError):
        before.mean = 1.0  # type: ignore[misc]


def test_weight_halves_after_one_half_life() -> None:
    b = Baseline(mean=26.0, weight=4.0, last_seen=T0)
    b = update(b, 56.0, T0 + 60 * DAY)
    # 4 observations decay to 2, so alpha = max(0.3, 1/3) = 1/3
    assert b.weight == pytest.approx(3.0)
    assert b.mean == pytest.approx(36.0)


def test_new_customer_gets_the_prior() -> None:
    assert expected(Baseline(), 30.0, T0) == pytest.approx(30.0)


def test_returning_customer_blends_by_order_count() -> None:
    # s = w / (w + 3): one order trusts the habit 25%, three 50%, nine 75%
    one = Baseline(mean=10.0, weight=1.0, last_seen=T0)
    three = Baseline(mean=10.0, weight=3.0, last_seen=T0)
    nine = Baseline(mean=10.0, weight=9.0, last_seen=T0)
    assert expected(one, 30.0, T0) == pytest.approx(25.0)
    assert expected(three, 30.0, T0) == pytest.approx(20.0)
    assert expected(nine, 30.0, T0) == pytest.approx(15.0)


def test_lapsed_customer_drifts_back_toward_the_prior() -> None:
    b = Baseline(mean=10.0, weight=3.0, last_seen=T0)
    fresh = expected(b, 30.0, T0)
    after_60 = expected(b, 30.0, T0 + 60 * DAY)
    after_120 = expected(b, 30.0, T0 + 120 * DAY)
    # weight 3 -> 1.5 -> 0.75, so s = 1/2, then 1/3, then 1/5
    assert fresh == pytest.approx(20.0)
    assert after_60 == pytest.approx(10.0 / 3 + 20.0)
    assert after_120 == pytest.approx(26.0)
    assert fresh < after_60 < after_120 < 30.0


def test_k_and_half_life_are_parameters() -> None:
    one = Baseline(mean=10.0, weight=1.0, last_seen=T0)
    assert expected(one, 30.0, T0, k=1.0) == pytest.approx(20.0)
    lapsed = Baseline(mean=10.0, weight=3.0, last_seen=T0)
    got = expected(lapsed, 30.0, T0 + 30 * DAY, half_life_days=30.0)
    assert got == pytest.approx(10.0 / 3 + 20.0)
