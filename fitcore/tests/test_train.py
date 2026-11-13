import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fitcore.features import asymmetric
from fitcore.model import LATENTS
from fitcore.sim import DEFAULT_TRUTH, TrueModel, simulate_customers, true_latents
from fitcore.train import (
    DTYPE,
    AsymmetricGaps,
    FitNetwork,
    PositiveMaskedLinear,
    inverse_softplus,
)

SCALE = np.asarray(DEFAULT_TRUTH.scale)


def _gaps(n: int, seed: int = 2026) -> torch.Tensor:
    return torch.tensor(simulate_customers(n, seed).gaps, dtype=DTYPE)


def _layers(net: FitNetwork) -> list[PositiveMaskedLinear]:
    return [net.layer1, net.layer2, net.layer3]


def _load_truth(net: FitNetwork, truth: TrueModel) -> None:
    """Set the network's kappas, weights and biases to the simulator's truth."""
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
    truth_net = FitNetwork(SCALE)
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
    net = FitNetwork(DEFAULT_TRUTH.scale)
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
