import math

import pytest

from fitcore.features import (
    HIGHER_IS_BETTER,
    INPUTS,
    asymmetric,
    gap,
    input_gap,
    transform,
)


def test_inputs_are_x1_to_x8() -> None:
    assert INPUTS == ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8")


def test_every_input_has_a_direction() -> None:
    assert set(HIGHER_IS_BETTER) == set(INPUTS)


def test_time_and_fee_inputs_are_lower_is_better() -> None:
    # x1 delivery time, x2 checkout time, x3 fee-to-basket ratio
    lower = {key for key, higher in HIGHER_IS_BETTER.items() if not higher}
    assert lower == {"x1", "x2", "x3"}


def test_gap_is_positive_when_better_than_expected() -> None:
    assert gap(4.5, 4.0) == pytest.approx(0.5)
    assert gap(3.0, 4.0) == pytest.approx(-1.0)


def test_gap_flips_when_lower_is_better() -> None:
    # delivered in 25 min against a promised 30: 5 min early is +5
    assert gap(25.0, 30.0, higher_is_better=False) == pytest.approx(5.0)
    assert gap(40.0, 30.0, higher_is_better=False) == pytest.approx(-10.0)


def test_gap_is_zero_when_as_expected() -> None:
    assert gap(30.0, 30.0) == 0.0
    assert gap(30.0, 30.0, higher_is_better=False) == 0.0


def test_input_gap_uses_the_inputs_direction() -> None:
    # x1: delivery minutes, lower is better
    assert input_gap("x1", 25.0, 30.0) == pytest.approx(5.0)
    # x3: fee-to-basket ratio, lower is better
    assert input_gap("x3", 0.12, 0.10) == pytest.approx(-0.02)
    # x5: courier rating, higher is better
    assert input_gap("x5", 4.0, 4.5) == pytest.approx(-0.5)


def test_surplus_is_unchanged() -> None:
    assert asymmetric(2.0, kappa=1.5) == pytest.approx(2.0)


def test_shortfall_is_weighted_by_kappa() -> None:
    assert asymmetric(-2.0, kappa=1.5) == pytest.approx(-3.0)


def test_zero_gap_stays_zero() -> None:
    assert asymmetric(0.0, kappa=1.5) == 0.0


def test_kappa_one_is_symmetric() -> None:
    assert asymmetric(-2.0, kappa=1.0) == pytest.approx(-2.0)


def test_a_shortfall_hurts_more_than_an_equal_surplus_helps() -> None:
    assert abs(asymmetric(-1.0, kappa=2.0)) > asymmetric(1.0, kappa=2.0)


KAPPAS = {
    "x1": 2.0,
    "x2": 1.5,
    "x3": 1.5,
    "x4": 3.0,
    "x5": 1.5,
    "x6": 3.0,
    "x7": 1.2,
    "x8": 2.5,
}


def test_transform_uses_each_inputs_own_kappa() -> None:
    got = transform({"x1": -2.0, "x4": -2.0, "x6": 0.5}, KAPPAS)
    assert got == pytest.approx({"x1": -4.0, "x4": -6.0, "x6": 0.5})


def test_transform_returns_only_the_inputs_given() -> None:
    got = transform({"x5": -1.0}, KAPPAS)
    assert set(got) == {"x5"}


def test_transform_does_not_change_its_argument() -> None:
    gaps = {"x1": -2.0}
    transform(gaps, KAPPAS)
    assert gaps == {"x1": -2.0}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_gap_rejects_values_that_are_not_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        gap(bad, 1.0)
    with pytest.raises(ValueError, match="finite"):
        gap(1.0, bad)


def test_input_gap_rejects_an_unknown_input() -> None:
    with pytest.raises(ValueError, match="unknown input 'x9'"):
        input_gap("x9", 1.0, 1.0)


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, math.nan, math.inf])
def test_kappa_must_be_finite_and_at_least_one(bad: float) -> None:
    with pytest.raises(ValueError, match="kappa"):
        asymmetric(-1.0, kappa=bad)


def test_kappa_is_checked_even_for_a_surplus() -> None:
    with pytest.raises(ValueError, match="kappa"):
        asymmetric(1.0, kappa=0.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_asymmetric_rejects_gaps_that_are_not_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        asymmetric(bad, kappa=2.0)


def test_transform_rejects_an_unknown_input() -> None:
    with pytest.raises(ValueError, match="unknown input 'x9'"):
        transform({"x9": 1.0}, {**KAPPAS, "x9": 2.0})


def test_transform_needs_a_kappa_for_every_gap() -> None:
    with pytest.raises(ValueError, match="no kappa for 'x5'"):
        transform({"x5": -1.0}, {"x1": 2.0})
