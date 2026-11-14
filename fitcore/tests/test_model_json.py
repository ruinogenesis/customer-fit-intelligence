"""T112: model.json and the numpy forward pass, checked against PyTorch."""

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from fitcore.dataset import TrainingSet, from_simulation
from fitcore.features import INPUTS
from fitcore.model import (
    LATENTS,
    MODEL_FORMAT,
    PATHS,
    THEORY_MASKS,
    ModelParams,
    forward,
    outcomes,
)
from fitcore.sim import DEFAULT_TRUTH, simulate_customers, true_latents
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.sim_outcomes import DEFAULT_OUTCOMES, fit, nps_chances
from fitcore.train import DTYPE, FitModel, export_params, fit_model, without_path


def _truth() -> ModelParams:
    """The simulator's own model: a kappa per input and its true weights.

    Layer 3 is the simulator's Fit = sigmoid(5 trust + 3 convenience - 4.3),
    and NPS uses its slope and cuts. Reorder and churn are made up: the
    simulator draws those from the order history instead.
    """
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


def test_the_numpy_pass_reproduces_the_simulators_truth() -> None:
    gaps = simulate_customers(1000, 2026).gaps
    got = forward(_truth(), gaps)
    assert got.shape == (1000, len(LATENTS))
    want = true_latents(gaps)
    for column, name in enumerate(LATENTS[:4]):
        np.testing.assert_allclose(got[:, column], want[name], rtol=0.0, atol=1e-12)
    true_fit = [
        fit(trust, convenience)
        for trust, convenience in zip(want["trust"], want["convenience"], strict=True)
    ]
    np.testing.assert_allclose(got[:, 4], true_fit, rtol=0.0, atol=1e-12)


def test_nps_chances_match_the_simulators() -> None:
    fits = np.array([0.1, 0.55, 0.9])
    chances = outcomes(_truth(), fits).nps
    assert chances.shape == (3, 11)
    want = [nps_chances(f) for f in fits]
    np.testing.assert_allclose(chances, want, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(chances.sum(axis=1), 1.0, rtol=1e-12)


def test_reorder_rises_and_churn_falls_with_fit() -> None:
    heads = outcomes(_truth(), np.array([0.2, 0.8]))
    assert heads.reorder[1] > heads.reorder[0]
    assert heads.churn[1] < heads.churn[0]
    assert heads.reorder[0] == pytest.approx(1.0 / (1.0 + math.exp(-0.9)))


@pytest.fixture(scope="module")
def small() -> TrainingSet:
    orders = simulate_orders(simulate_profiles(60, 2026), 2026)
    return from_simulation(orders, simulate_behaviour(orders, 2026))


@pytest.fixture(scope="module", params=[True, False], ids=["shared", "per_input"])
def trained(request: pytest.FixtureRequest, small: TrainingSet) -> FitModel:
    model = FitModel.for_training_set(small, shared_kappa=request.param)
    fit_model(model, small, rounds=5)
    return model


def test_numpy_scores_match_pytorch_on_1000_customers(trained: FitModel) -> None:
    gaps = simulate_customers(1000, 2026).gaps
    params = ModelParams.from_json(export_params(trained).to_json())
    with torch.no_grad():
        latents = trained.network(torch.tensor(gaps, dtype=DTYPE))
        reorder, churn, eta = trained.heads(latents[:, -1])
        at_most = torch.sigmoid(trained.heads.cuts[None, :] - eta[:, None])
    got = forward(params, gaps)
    np.testing.assert_allclose(got, latents.numpy(), rtol=0.0, atol=1e-12)
    heads = outcomes(params, got[:, -1])
    want_reorder = torch.sigmoid(reorder).numpy()
    np.testing.assert_allclose(heads.reorder, want_reorder, rtol=0.0, atol=1e-12)
    want_churn = torch.sigmoid(churn).numpy()
    np.testing.assert_allclose(heads.churn, want_churn, rtol=0.0, atol=1e-12)
    cumulative = np.cumsum(heads.nps, axis=1)[:, :10]
    np.testing.assert_allclose(cumulative, at_most.numpy(), rtol=0.0, atol=1e-12)


def test_export_holds_the_effective_weights_masks_and_kappa(trained: FitModel) -> None:
    params = export_params(trained)
    network = trained.network
    layers = zip(network.layers, params.masks, params.weights, strict=True)
    for layer, mask, weight in layers:
        np.testing.assert_array_equal(mask, layer.mask.numpy())
        np.testing.assert_array_equal(weight, layer.weight.detach().numpy())
    np.testing.assert_array_equal(params.kappa, network.gaps.kappa.detach().numpy())
    np.testing.assert_array_equal(params.gap_scale, network.gaps.scale.numpy())
    np.testing.assert_array_equal(params.nps_cuts, trained.heads.cuts.detach().numpy())


def test_a_path_switched_off_exports_as_exactly_zero(small: TrainingSet) -> None:
    path = PATHS[0]  # respect <- x1
    params = export_params(without_path(FitModel.for_training_set(small), path))
    row, column = path.position()
    assert params.masks[0][row, column] == 0.0
    assert params.weights[0][row, column] == 0.0


def test_model_json_reads_back_exactly(trained: FitModel) -> None:
    params = export_params(trained)
    text = params.to_json()
    assert text.endswith("}\n")
    again = ModelParams.from_json(text.encode())
    assert again.to_json() == text
    np.testing.assert_array_equal(again.weights[1], params.weights[1])
    assert again.churn_slope == params.churn_slope


def test_model_json_names_its_numbers() -> None:
    document = json.loads(_truth().to_json())
    assert document["format"] == MODEL_FORMAT
    assert document["inputs"] == list(INPUTS)
    assert document["latents"] == list(LATENTS)
    layers = document["layers"]
    assert [layer["outputs"] for layer in layers] == [
        ["respect", "love"],
        ["convenience", "trust"],
        ["fit"],
    ]
    assert layers[2]["sources"] == ["convenience", "trust"]
    assert layers[0]["mask"][0] == [1, 0, 0, 1, 1, 1, 0, 0]
    assert set(document["heads"]) == {"reorder", "churn", "nps"}
    assert len(document["heads"]["nps"]["cuts"]) == 10


def test_params_are_read_only() -> None:
    params = _truth()
    with pytest.raises(ValueError, match="read-only"):
        params.kappa[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        params.weights[0][0, 0] = 1.0


def _bad_numbers() -> dict[str, dict[str, object]]:
    truth = _truth()
    negative = np.array(truth.weights[0])
    negative[0, 0] = -0.1
    off_mask = np.array(truth.weights[0])
    off_mask[0, 1] = 0.5  # x2 does not feed respect
    extra_path = np.array(truth.masks[0])
    extra_path[1, 0] = 1.0  # love <- x1 is not in the theory
    return {
        "kappa below 1": {"kappa": np.full(8, 0.9)},
        "kappa not finite": {"kappa": np.full(8, math.nan)},
        "scale of zero": {"gap_scale": np.zeros(8)},
        "seven scales": {"gap_scale": np.ones(7)},
        "negative weight": {"weights": (negative, *truth.weights[1:])},
        "weight off its mask": {"weights": (off_mask, *truth.weights[1:])},
        "mask adds a path": {"masks": (extra_path, *truth.masks[1:])},
        "two biases for fit": {"biases": (*truth.biases[:2], [0.0, 0.0])},
        "negative churn slope": {"churn_slope": -1.0},
        "infinite intercept": {"reorder_intercept": math.inf},
        "cuts that fall": {"nps_cuts": truth.nps_cuts[::-1]},
        "nine cuts": {"nps_cuts": truth.nps_cuts[:9]},
    }


BAD_NUMBERS = _bad_numbers()


@pytest.mark.parametrize("case", list(BAD_NUMBERS))
def test_bad_numbers_are_rejected(case: str) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_truth(), **BAD_NUMBERS[case])


def test_a_head_number_must_be_a_number() -> None:
    with pytest.raises(TypeError):
        dataclasses.replace(_truth(), nps_slope="6")


@pytest.mark.parametrize(
    ("where", "value"),
    [
        (("format",), 2),
        (("inputs", 0), "speed"),
        (("layers", 1, "sources", 0), "love"),
        (("kappa",), [1.5] * 7),
        (("layers", 0, "weight"), {"respect": 1.0}),
        (("heads", "nps", "slope"), "6"),
        (("heads", "churn"), {}),
    ],
)
def test_a_wrong_model_json_is_rejected(
    where: tuple[str | int, ...], value: object
) -> None:
    document = json.loads(_truth().to_json())
    target = document
    for key in where[:-1]:
        target = target[key]
    target[where[-1]] = value
    with pytest.raises(ValueError):
        ModelParams.from_json(json.dumps(document))


@pytest.mark.parametrize("text", ["{", "[]", "{}"])
def test_text_that_is_not_model_json_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        ModelParams.from_json(text)


@pytest.mark.parametrize(
    "gaps", [np.zeros((3, 7)), np.zeros(8), np.full((2, 8), math.nan)]
)
def test_forward_rejects_bad_gaps(gaps: np.ndarray) -> None:
    with pytest.raises(ValueError):
        forward(_truth(), gaps)


def test_outcomes_need_one_fit_per_row() -> None:
    with pytest.raises(ValueError):
        outcomes(_truth(), np.zeros((2, 1)))
