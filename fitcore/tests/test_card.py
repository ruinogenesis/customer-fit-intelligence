"""T113: the model card, its metrics and its markdown."""

import dataclasses
import itertools
import math
from datetime import date

import numpy as np
import pytest

from fitcore.card import (
    Comparison,
    DataWindow,
    ModelCard,
    auc,
    compare_churn,
    log_loss,
    outcome_results,
    weight_changes,
)
from fitcore.dataset import TrainingSet, from_simulation
from fitcore.model import PATHS, THEORY_MASKS, ModelParams, forward, outcomes
from fitcore.sim import DEFAULT_TRUTH
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.sim_outcomes import DEFAULT_OUTCOMES


def _params() -> ModelParams:
    """The simulator's own model (a kappa per input), with made-up heads."""
    w_trust, w_convenience = DEFAULT_OUTCOMES.w_fit
    return ModelParams(
        gap_scale=DEFAULT_TRUTH.scale,
        kappa=DEFAULT_TRUTH.kappa,
        masks=THEORY_MASKS,
        weights=(DEFAULT_TRUTH.w1, DEFAULT_TRUTH.w2, [[w_convenience, w_trust]]),
        biases=(DEFAULT_TRUTH.b1, DEFAULT_TRUTH.b2, [DEFAULT_OUTCOMES.b_fit]),
        reorder_intercept=0.5,
        reorder_slope=2.0,
        churn_intercept=-1.0,
        churn_slope=3.0,
        nps_slope=DEFAULT_OUTCOMES.nps_slope,
        nps_cuts=DEFAULT_OUTCOMES.nps_cuts,
    )


def test_log_loss_by_hand_and_skipping_unknown_labels() -> None:
    want = -(math.log(0.8) + math.log(0.7)) / 2
    assert log_loss([0.8, 0.3], [1.0, 0.0]) == pytest.approx(want, rel=1e-15)
    assert log_loss([0.8, 0.999], [1.0, math.nan]) == pytest.approx(-math.log(0.8))


@pytest.mark.parametrize(
    ("chance", "label"),
    [
        ([0.0, 0.5], [1.0, 0.0]),  # a chance of exactly 0
        ([0.5, 1.0], [1.0, 0.0]),  # a chance of exactly 1
        ([0.5, 0.5], [math.nan, math.nan]),  # no known label
        ([0.5, 0.5], [1.0, 2.0]),  # a label that is not 0 or 1
        ([0.5, 0.5, 0.5], [1.0, 0.0]),  # lengths differ
    ],
)
def test_log_loss_rejects_bad_input(chance: list[float], label: list[float]) -> None:
    with pytest.raises(ValueError):
        log_loss(chance, label)


def test_auc_at_its_extremes() -> None:
    label = [0.0, 0.0, 1.0, 1.0]
    assert auc([0.1, 0.2, 0.8, 0.9], label) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], label) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], label) == 0.5


def test_auc_counts_every_pair() -> None:
    rng = np.random.default_rng(2026)
    score = np.round(rng.random(60), 1)  # rounding makes ties
    label = (rng.random(60) < 0.4).astype(float)
    label[::7] = math.nan
    known = ~np.isnan(label)
    ones, zeros = score[known & (label == 1)], score[known & (label == 0)]
    pairs = [
        1.0 if a > b else 0.5 if a == b else 0.0
        for a, b in itertools.product(ones, zeros)
    ]
    assert auc(score, label) == pytest.approx(np.mean(pairs), rel=1e-12)


def test_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        auc([0.2, 0.8], [1.0, 1.0])


@pytest.fixture(scope="module")
def small() -> TrainingSet:
    orders = simulate_orders(simulate_profiles(60, 2026), 2026)
    return from_simulation(orders, simulate_behaviour(orders, 2026))


def test_outcome_results_score_with_the_numpy_pass(small: TrainingSet) -> None:
    params = _params()
    base = {"reorder": 0.4, "churn": 0.2, "nps": 2.0}
    reorder, churn, nps = outcome_results(params, small, base)
    heads = outcomes(params, forward(params, small.gaps)[:, -1])
    known = ~np.isnan(small.reorder)
    assert reorder.outcome == "reorder 14d"
    assert reorder.known == known.sum()
    assert reorder.log_loss == log_loss(heads.reorder, small.reorder)
    assert reorder.auc == auc(heads.reorder, small.reorder)
    assert reorder.predicted == pytest.approx(heads.reorder[known].mean())
    assert reorder.observed == pytest.approx(small.reorder[known].mean())
    assert (reorder.intercept_only, churn.intercept_only) == (0.4, 0.2)
    assert churn.log_loss == log_loss(heads.churn, small.churn)
    assert nps.outcome == "NPS"
    assert nps.known == np.sum(~np.isnan(small.nps))
    assert nps.auc is None
    assert nps.intercept_only == 2.0
    assert 0.0 <= nps.predicted <= 10.0


def test_a_constant_baseline_ranks_no_better_than_chance(small: TrainingSet) -> None:
    params = _params()
    rate = np.nanmean(small.churn)
    comparison = compare_churn(
        params, small, "the average rate", np.full(len(small), rate)
    )
    assert comparison.baseline == "the average rate"
    assert comparison.known == np.sum(~np.isnan(small.churn))
    assert comparison.baseline_auc == 0.5
    model = outcomes(params, forward(params, small.gaps)[:, -1]).churn
    assert comparison.model_log_loss == log_loss(model, small.churn)
    with pytest.raises(ValueError):
        compare_churn(params, small, "too short", np.full(3, rate))


@pytest.mark.parametrize(
    ("model_log_loss", "model_auc", "wins"),
    [(0.1, 0.8, True), (0.3, 0.8, False), (0.1, 0.6, False), (0.2, 0.7, False)],
)
def test_the_model_must_beat_the_baseline_on_both(
    model_log_loss: float, model_auc: float, wins: bool
) -> None:
    comparison = Comparison("a baseline", 100, model_log_loss, 0.2, model_auc, 0.7)
    assert comparison.model_wins is wins


def test_weight_changes_list_every_path_and_kappa() -> None:
    params = _params()
    weights = np.array(params.weights[1])
    weights[1, 0] = 9.0  # trust <- respect
    changed = dataclasses.replace(
        params, weights=(params.weights[0], weights, params.weights[2])
    )
    rows = weight_changes(changed, params)
    assert len(rows) == len(PATHS) + 8
    moved = [row for row in rows if row[1] != row[2]]
    assert moved == [("trust <- respect", DEFAULT_TRUTH.w2[1][0], 9.0)]
    assert rows[-1][0] == "kappa x8"


def _card(**changes: object) -> ModelCard:
    window = DataWindow(
        "simulated, seed 2026",
        date(2026, 1, 1),
        date(2026, 6, 29),
        1600,
        35707,
        400,
        8520,
    )
    card = ModelCard(
        title="test_model.json",
        created=date(2026, 11, 14),
        params=_params(),
        window=window,
        results=(),
    )
    return dataclasses.replace(card, **changes)


def test_the_card_holds_the_data_window_masks_and_changes(small: TrainingSet) -> None:
    params = _params()
    results = outcome_results(params, small, {"reorder": 0.4, "churn": 0.2, "nps": 2.0})
    comparison = Comparison("order frequency alone", 120, 0.15, 0.16, 0.79, 0.63)
    text = _card(
        results=results,
        churn_baseline=comparison,
        recovery={"respect": 0.985, "love": 0.983},
        changes=("One shared kappa.",),
        limits=("Simulated data only.",),
    ).to_markdown()
    assert text.startswith("# Model card: test_model.json\n")
    assert text.endswith("\n")
    assert "orders 2026-01-01 to 2026-06-29" in text
    assert "Training: 1,600 customers, 35,707 orders." in text
    assert "Held out: 400 customers, 8,520 orders." in text
    for heading in (
        "Held-out metrics",
        "Paths: masks and weights",
        "Changes from the previous version",
        "Limits and intended use",
    ):
        assert f"\n## {heading}\n" in text
    for result in results:
        assert (
            f"| {result.outcome} | {result.known:,} | {result.log_loss:.3f} |" in text
        )
    assert (
        "log loss 0.150 vs 0.160, AUC 0.790 vs 0.630. Target (beat it on both): met."
        in text
    )
    assert "respect 0.985, love 0.983." in text
    for path in PATHS:
        assert f"| {path.name} | yes |" in text
    assert "kappa per input: 2.000, 1.500" in text
    assert "- First version: there is no previous model.\n- One shared kappa.\n" in text
    assert "- Simulated data only.\n" in text
    assert not any(line != line.rstrip() for line in text.splitlines())


def test_a_card_lists_what_changed_from_the_previous_model() -> None:
    previous = _params()
    biases = (np.array(previous.biases[0]) + 0.1, *previous.biases[1:])
    kappa = np.full(8, 1.6)
    params = dataclasses.replace(previous, kappa=kappa, biases=biases)
    text = _card(
        params=params, previous=previous, changes=("Retrained.",)
    ).to_markdown()
    assert "First version" not in text
    assert "kappa, shared by all eight inputs: 1.600." in text
    assert "| Weight | Before | Now |" in text
    assert "| kappa x1 | 2.000 | 1.600 |" in text
    assert "| respect <- x1 |" not in text.split("| Weight | Before | Now |")[1]


def test_a_card_leaves_out_the_parts_it_was_not_given() -> None:
    text = _card().to_markdown()
    assert "Churn 30d against" not in text
    assert "Recovery" not in text
    assert "## Limits and intended use" not in text
    assert "## Changes from the previous version" in text
