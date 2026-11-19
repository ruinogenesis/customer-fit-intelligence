import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fitcore.sim_outcomes import (
    DEFAULT_OUTCOMES,
    FIT_MIDPOINT,
    churn_chance,
    fit,
    nps_answer,
    nps_chances,
    wait_factor,
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_fit_follows_the_doc_formula() -> None:
    o = DEFAULT_OUTCOMES
    want = sigmoid(o.w_fit[0] * 0.6 + o.w_fit[1] * 0.5 + o.b_fit)
    assert fit(0.6, 0.5) == pytest.approx(want)


def test_an_interaction_adds_trust_times_convenience() -> None:
    o = replace(DEFAULT_OUTCOMES, w_interaction=3.0)
    want = sigmoid(o.w_fit[0] * 0.6 + o.w_fit[1] * 0.5 + 3.0 * 0.6 * 0.5 + o.b_fit)
    assert fit(0.6, 0.5, o) == pytest.approx(want)


def test_by_default_there_is_no_interaction() -> None:
    assert DEFAULT_OUTCOMES.w_interaction == 0.0


def test_the_interaction_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(DEFAULT_OUTCOMES, w_interaction=math.inf)


def test_fit_rises_with_trust_and_with_convenience() -> None:
    assert fit(0.3, 0.5) < fit(0.6, 0.5) < fit(0.9, 0.5)
    assert fit(0.6, 0.2) < fit(0.6, 0.5) < fit(0.6, 0.8)


def test_fit_stays_between_zero_and_one() -> None:
    for trust in (0.0, 0.5, 1.0):
        for convenience in (0.0, 0.5, 1.0):
            assert 0.0 < fit(trust, convenience) < 1.0


def test_higher_fit_means_a_shorter_wait() -> None:
    assert wait_factor(FIT_MIDPOINT) == pytest.approx(1.0)
    assert wait_factor(0.8) < 1.0 < wait_factor(0.2)


def test_with_no_pace_fit_does_not_change_the_wait() -> None:
    calm = replace(DEFAULT_OUTCOMES, pace=0.0)
    assert wait_factor(0.1, calm) == wait_factor(0.9, calm) == 1.0


def test_lower_fit_means_more_churn() -> None:
    assert churn_chance(FIT_MIDPOINT) == pytest.approx(
        sigmoid(DEFAULT_OUTCOMES.churn_base)
    )
    assert churn_chance(0.8) < churn_chance(0.5) < churn_chance(0.2)


def test_nps_chances_cover_0_to_10_and_sum_to_one() -> None:
    for f in (0.1, 0.5, 0.9):
        chances = nps_chances(f)
        assert len(chances) == 11
        assert all(p >= 0.0 for p in chances)
        assert sum(chances) == pytest.approx(1.0)


def test_higher_fit_shifts_nps_up() -> None:
    def mean_answer(f: float) -> float:
        return sum(k * p for k, p in enumerate(nps_chances(f)))

    assert mean_answer(0.3) < mean_answer(0.55) < mean_answer(0.8)


def test_a_typical_customer_gives_a_typical_nps_mix() -> None:
    chances = nps_chances(0.55)
    detractors, passives, promoters = (
        sum(chances[:7]),
        sum(chances[7:9]),
        sum(chances[9:]),
    )
    assert detractors == pytest.approx(0.20, abs=0.03)
    assert passives == pytest.approx(0.35, abs=0.03)
    assert promoters == pytest.approx(0.45, abs=0.03)


def test_nps_answers_follow_their_chances() -> None:
    rng = np.random.default_rng(0)
    answers = [nps_answer(0.55, rng) for _ in range(20_000)]
    assert set(answers) <= set(range(11))
    for k, p in enumerate(nps_chances(0.55)):
        assert answers.count(k) / len(answers) == pytest.approx(p, abs=0.01)


def test_outcome_settings_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_OUTCOMES.pace = 1.0  # type: ignore[misc]


def test_fit_weights_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="w_fit"):
        replace(DEFAULT_OUTCOMES, w_fit=(5.0, -1.0))


def test_slopes_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="zero or more"):
        replace(DEFAULT_OUTCOMES, churn_slope=-1.0)


def test_survey_rate_must_be_a_share() -> None:
    with pytest.raises(ValueError, match="survey_rate"):
        replace(DEFAULT_OUTCOMES, survey_rate=1.5)


def test_nps_cuts_must_be_ten_and_increasing() -> None:
    with pytest.raises(ValueError, match="nps_cuts"):
        replace(DEFAULT_OUTCOMES, nps_cuts=(0.0, 1.0))
    with pytest.raises(ValueError, match="nps_cuts"):
        replace(DEFAULT_OUTCOMES, nps_cuts=(0.0,) * 10)


def test_settings_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(DEFAULT_OUTCOMES, pace=math.nan)
