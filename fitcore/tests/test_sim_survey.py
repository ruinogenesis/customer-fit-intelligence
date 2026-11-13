import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fitcore.model import LAYER_1, LAYER_2
from fitcore.sim_survey import (
    DEFAULT_ITEMS,
    ITEM_TEXT,
    QUESTIONS,
    item_answer,
    item_chances,
)


def mean_answer(latent: float) -> float:
    return sum(k * p for k, p in enumerate(item_chances(latent), start=1))


def test_the_rotation_is_nps_then_one_item_per_latent() -> None:
    assert QUESTIONS[0] == "nps"
    assert set(QUESTIONS[1:]) == set(LAYER_1) | set(LAYER_2)
    assert len(set(QUESTIONS)) == len(QUESTIONS)


def test_every_latent_item_has_the_doc_wording() -> None:
    assert set(ITEM_TEXT) == set(QUESTIONS[1:])
    assert ITEM_TEXT["love"] == "I'd miss this app if it closed"


def test_item_chances_cover_1_to_7_and_sum_to_one() -> None:
    for latent in (0.1, 0.5, 0.9):
        chances = item_chances(latent)
        assert len(chances) == 7
        assert all(p >= 0.0 for p in chances)
        assert sum(chances) == pytest.approx(1.0)


def test_a_higher_latent_gives_higher_answers() -> None:
    assert mean_answer(0.3) < mean_answer(0.6) < mean_answer(0.9)


def test_typical_answers_lean_positive() -> None:
    assert 4.5 < mean_answer(0.6) < 6.0


def test_item_answers_follow_their_chances() -> None:
    rng = np.random.default_rng(0)
    answers = [item_answer(0.6, rng) for _ in range(20_000)]
    assert set(answers) <= set(range(1, 8))
    for k, p in enumerate(item_chances(0.6), start=1):
        assert answers.count(k) / len(answers) == pytest.approx(p, abs=0.01)


def test_item_settings_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_ITEMS.slope = 1.0  # type: ignore[misc]


def test_slope_must_be_finite_and_not_negative() -> None:
    for bad in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="slope"):
            replace(DEFAULT_ITEMS, slope=bad)


def test_cuts_must_be_six_and_increasing() -> None:
    with pytest.raises(ValueError, match="cuts"):
        replace(DEFAULT_ITEMS, cuts=(1.0, 2.0))
    with pytest.raises(ValueError, match="cuts"):
        replace(DEFAULT_ITEMS, cuts=(1.0, 2.0, 2.0, 3.0, 4.0, 5.0))
