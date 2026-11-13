import dataclasses
import math
from itertools import pairwise

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fitcore.dataset import TrainingSet, from_simulation, gap_scale, split_by_customer
from fitcore.features import asymmetric
from fitcore.model import LATENTS
from fitcore.sim import DEFAULT_TRUTH, TrueModel, simulate_customers, true_latents
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import (
    DTYPE,
    TERMS,
    Anchors,
    AsymmetricGaps,
    Batch,
    FitModel,
    FitNetwork,
    OutcomeHeads,
    PositiveMaskedLinear,
    TrainingResult,
    evaluate,
    fit_model,
    intercept_only_losses,
    inverse_softplus,
    loss_terms,
    ordinal_nll,
)

SCALE = np.asarray(DEFAULT_TRUTH.scale)


def _gaps(n: int, seed: int = 2026) -> torch.Tensor:
    return torch.tensor(simulate_customers(n, seed).gaps, dtype=DTYPE)


def _layers(net: FitNetwork) -> list[PositiveMaskedLinear]:
    return [net.layer1, net.layer2, net.layer3]


def _load_truth(net: FitNetwork, truth: TrueModel) -> None:
    """Set the network's kappas, weights and biases to the simulator's truth.

    The truth has its own kappa per input, so net needs shared_kappa=False.
    """
    with torch.no_grad():
        kappa = torch.tensor(truth.kappa, dtype=DTYPE)
        net.gaps.theta.copy_(inverse_softplus(kappa - 1.0))
        for layer, w, b in (
            (net.layer1, truth.w1, truth.b1),
            (net.layer2, truth.w2, truth.b2),
        ):
            weight = torch.tensor(w, dtype=DTYPE)
            on_mask = layer.mask > 0.0
            layer.raw.copy_(inverse_softplus(torch.where(on_mask, weight, 1.0)))
            layer.bias.copy_(torch.tensor(b, dtype=DTYPE))


def test_output_has_one_column_per_latent_between_0_and_1() -> None:
    out = FitNetwork(SCALE)(_gaps(500))
    assert out.shape == (500, len(LATENTS))
    assert out.dtype == torch.float64
    assert bool(torch.all((out > 0.0) & (out < 1.0)))


def test_weights_are_positive_on_the_mask_and_exactly_zero_off_it() -> None:
    for layer in _layers(FitNetwork(SCALE)):
        on_mask = layer.mask > 0.0
        assert bool(torch.all(layer.weight[on_mask] > 0.0))
        assert bool(torch.all(layer.weight[~on_mask] == 0.0))


def test_no_gradient_reaches_a_weight_off_the_mask() -> None:
    net = FitNetwork(SCALE)
    net(_gaps(200)).sum().backward()
    for layer in _layers(net):
        on_mask = layer.mask > 0.0
        assert layer.raw.grad is not None
        assert bool(torch.all(layer.raw.grad[~on_mask] == 0.0))
        assert bool(torch.all(layer.raw.grad[on_mask] != 0.0))


def test_constraints_hold_at_every_step_of_adversarial_training() -> None:
    # Targets that fall as gaps improve: the data wants every weight negative.
    gaps = _gaps(500)
    truth_net = FitNetwork(SCALE, shared_kappa=False)
    _load_truth(truth_net, DEFAULT_TRUTH)
    with torch.no_grad():
        target = 1.0 - truth_net(gaps)

    net = FitNetwork(SCALE)
    optimiser = torch.optim.Adam(net.parameters(), lr=0.05)
    smallest_weight = math.inf
    for _ in range(400):
        optimiser.zero_grad()
        F.mse_loss(net(gaps), target).backward()
        optimiser.step()
        for layer in _layers(net):
            on_mask = layer.mask > 0.0
            assert bool(torch.all(layer.weight[on_mask] >= 0.0))
            assert bool(torch.all(layer.weight[~on_mask] == 0.0))
            smallest_weight = min(smallest_weight, layer.weight[on_mask].min().item())
        assert bool(torch.all(net.gaps.kappa >= 1.0))

    # The pressure was real: training drove some weights almost to zero.
    assert smallest_weight < 0.01


def test_kappa_starts_above_1_and_matches_fitcore_features() -> None:
    layer = AsymmetricGaps(SCALE)
    kappa = layer.kappa.detach().numpy()
    assert np.all(kappa > 1.0)
    np.testing.assert_allclose(kappa, 1.5, rtol=1e-15)

    sizes = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    gaps = sizes[:, None] * SCALE[None, :]
    got = layer(torch.tensor(gaps, dtype=DTYPE)).detach().numpy()
    want = np.array(
        [
            [asymmetric(g, k) / s for g, k, s in zip(row, kappa, SCALE, strict=True)]
            for row in gaps
        ]
    )
    np.testing.assert_allclose(got, want, rtol=1e-15)


def test_the_simulators_true_weights_reproduce_its_latents_exactly() -> None:
    population = simulate_customers(1000, 2026)
    net = FitNetwork(DEFAULT_TRUTH.scale, shared_kappa=False)
    _load_truth(net, DEFAULT_TRUTH)
    with torch.no_grad():
        out = net(torch.tensor(population.gaps, dtype=DTYPE)).numpy()
    expected = true_latents(population.gaps)
    for column, name in enumerate(LATENTS[:4]):
        np.testing.assert_allclose(out[:, column], expected[name], rtol=0.0, atol=1e-12)


def test_scale_is_a_fixed_buffer_not_a_trained_parameter() -> None:
    net = FitNetwork(SCALE)
    assert "gaps.scale" in dict(net.named_buffers())
    assert "gaps.scale" not in dict(net.named_parameters())


@pytest.mark.parametrize(
    "scale",
    [
        np.ones(7),
        np.zeros(8),
        -np.ones(8),
        np.array([1.0] * 7 + [math.nan]),
        np.array([1.0] * 7 + [math.inf]),
    ],
)
def test_a_bad_scale_is_rejected(scale: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FitNetwork(scale)


def test_a_starting_kappa_of_1_or_less_is_rejected() -> None:
    with pytest.raises(ValueError):
        AsymmetricGaps(SCALE, init_kappa=1.0)


@pytest.mark.parametrize("y", [1e-6, 0.3, 1.0, 2.5, 15.0])
def test_inverse_softplus_undoes_softplus(y: float) -> None:
    value = torch.tensor(y, dtype=DTYPE)
    assert F.softplus(inverse_softplus(value)).item() == pytest.approx(y, rel=1e-12)


# --- T109: heads, anchors, loss and training ---


def _simulated(customers: int) -> TrainingSet:
    orders = simulate_orders(simulate_profiles(customers, 2026), 2026)
    return from_simulation(orders, simulate_behaviour(orders, 2026))


@pytest.mark.parametrize("eta", [-20.0, -1.0, 0.0, 2.5, 30.0])
def test_ordinal_chances_sum_to_1_and_follow_the_cuts(eta: float) -> None:
    cuts = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=DTYPE)
    answers = torch.arange(5, dtype=DTYPE)
    etas = torch.full((5,), eta, dtype=DTYPE)
    chances = torch.exp(-ordinal_nll(etas, cuts, answers))
    assert chances.sum().item() == pytest.approx(1.0, abs=1e-12)
    # sigmoid(upper) - sigmoid(lower), written so that nothing cancels
    upper, lower = 0.5 - eta, 0.0 - eta
    want = math.exp(lower) * math.expm1(upper - lower)
    want /= (1.0 + math.exp(upper)) * (1.0 + math.exp(lower))
    assert chances[2].item() == pytest.approx(want, rel=1e-12)


def test_ordinal_gradients_stay_finite_far_into_the_tails() -> None:
    cuts = torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    eta = torch.tensor([-60.0, -60.0, 60.0, 60.0], dtype=DTYPE, requires_grad=True)
    answers = torch.tensor([0.0, 3.0, 0.0, 3.0], dtype=DTYPE)
    nll = ordinal_nll(eta, cuts, answers)
    nll.sum().backward()
    assert eta.grad is not None
    assert bool(torch.all(torch.isfinite(nll)))
    assert bool(torch.all(torch.isfinite(eta.grad)))


def test_heads_point_the_way_theory_says_whatever_their_raw_values() -> None:
    heads = OutcomeHeads()
    with torch.no_grad():
        heads.raw_slope.copy_(torch.tensor([-30.0, -5.0, 3.0], dtype=DTYPE))
        heads.raw_steps.copy_(torch.linspace(-30.0, 4.0, 9, dtype=DTYPE))
    assert bool(torch.all(heads.slope > 0.0))
    assert bool(torch.all(torch.diff(heads.cuts) > 0.0))
    reorder, churn, nps = heads(torch.tensor([0.2, 0.8], dtype=DTYPE))
    assert reorder[1] > reorder[0]
    assert churn[1] < churn[0]
    assert nps[1] > nps[0]


def test_item_thresholds_are_fixed_at_sevenths() -> None:
    anchors = Anchors(np.zeros(6), np.ones(6))
    want = torch.arange(1, 7, dtype=DTYPE) / 7.0
    assert torch.equal(anchors.thresholds, want)
    assert "thresholds" in dict(anchors.named_buffers())
    assert "thresholds" not in dict(anchors.named_parameters())


@pytest.mark.parametrize(
    ("mean", "sd"),
    [
        (np.zeros(5), np.ones(5)),
        (np.zeros(6), np.zeros(6)),
        (np.zeros(6), -np.ones(6)),
        (np.full(6, math.nan), np.ones(6)),
    ],
)
def test_bad_behaviour_stats_are_rejected(mean: np.ndarray, sd: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Anchors(mean, sd)


@pytest.fixture(scope="module")
def small() -> TrainingSet:
    return _simulated(60)


def test_the_gap_scale_comes_from_the_training_set(small: TrainingSet) -> None:
    model = FitModel.for_training_set(small)
    scale = model.network.gaps.scale.numpy()
    np.testing.assert_allclose(scale, gap_scale(small.gaps), rtol=1e-15)
    assert not np.allclose(scale, DEFAULT_TRUTH.scale)


def test_a_term_averages_only_the_rows_whose_value_is_known(
    small: TrainingSet,
) -> None:
    model = FitModel.for_training_set(small)
    batch = Batch.from_training_set(small)
    with torch.no_grad():
        terms = loss_terms(model, batch)
        reorder, _, _ = model.heads(model.network(batch.gaps)[:, -1])
    known = ~torch.isnan(batch.reorder)
    assert 0 < int(known.sum()) < len(small)
    want = F.binary_cross_entropy_with_logits(reorder[known], batch.reorder[known])
    assert terms["reorder"].item() == pytest.approx(want.item(), rel=1e-14)
    assert tuple(terms) == TERMS


def test_missing_answers_add_nothing(small: TrainingSet) -> None:
    model = FitModel.for_training_set(small)
    nothing = np.full(len(small), math.nan)
    no_nps = dataclasses.replace(small, nps=nothing)
    with torch.no_grad():
        before = loss_terms(model, Batch.from_training_set(small))
        after = loss_terms(model, Batch.from_training_set(no_nps))
    assert after["nps"].item() == 0.0
    for name in TERMS:
        if name != "nps":
            assert after[name].item() == before[name].item()


def test_the_same_data_gives_exactly_the_same_model(small: TrainingSet) -> None:
    models = [FitModel.for_training_set(small) for _ in range(2)]
    results = [fit_model(model, small, rounds=3) for model in models]
    assert results[0].losses == results[1].losses
    first, second = (model.state_dict() for model in models)
    for name, value in first.items():
        assert torch.equal(value, second[name]), name


@pytest.fixture(scope="module")
def trained() -> tuple[FitModel, TrainingResult, TrainingSet, TrainingSet]:
    train, test = split_by_customer(_simulated(300), 0.2, seed=2026)
    model = FitModel.for_training_set(train)
    return model, fit_model(model, train), train, test


Trained = tuple[FitModel, TrainingResult, TrainingSet, TrainingSet]


def test_training_converges_and_the_loss_never_rises(trained: Trained) -> None:
    _, result, _, _ = trained
    assert result.converged
    losses = result.losses
    assert all(b <= a + 1e-12 for a, b in pairwise(losses))
    assert losses[-1] < losses[0]


def test_every_head_beats_intercept_only_on_held_out_customers(
    trained: Trained,
) -> None:
    model, _, train, test = trained
    held_out = evaluate(model, test)
    baseline = intercept_only_losses(train, test)
    for head in ("reorder", "churn", "nps"):
        assert held_out[head] < baseline[head], head


def test_every_head_slope_is_clearly_above_zero(trained: Trained) -> None:
    model, _, _, _ = trained
    assert bool(torch.all(model.heads.slope > 1.0))


def test_masks_signs_and_kappa_still_hold_after_training(trained: Trained) -> None:
    model, _, _, _ = trained
    for layer in _layers(model.network):
        on_mask = layer.mask > 0.0
        assert bool(torch.all(layer.weight[on_mask] > 0.0))
        assert bool(torch.all(layer.weight[~on_mask] == 0.0))
    assert bool(torch.all(model.network.gaps.kappa > 1.0))


def test_recovered_latents_follow_the_truth_on_held_out_customers(
    trained: Trained,
) -> None:
    model, _, _, test = trained
    with torch.no_grad():
        latents = model.network(torch.tensor(test.gaps, dtype=DTYPE)).numpy()
    assert test.truth is not None
    for column, name in enumerate(LATENTS):
        r = np.corrcoef(latents[:, column], test.truth[:, column])[0, 1]
        assert r > 0.9, name
