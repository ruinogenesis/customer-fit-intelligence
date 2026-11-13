"""T114: one kappa shared by all eight inputs, still reported per input."""

import numpy as np
import pytest
import torch

from fitcore.dataset import TrainingSet, from_simulation
from fitcore.features import INPUTS
from fitcore.model import PATHS
from fitcore.sim import DEFAULT_TRUTH, simulate_customers
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import (
    DTYPE,
    AsymmetricGaps,
    FitModel,
    FitNetwork,
    TrainingResult,
    fit_model,
    without_path,
)

SCALE = np.asarray(DEFAULT_TRUTH.scale)


def test_one_theta_is_learned_and_kappa_still_has_eight_values() -> None:
    layer = AsymmetricGaps(SCALE)
    assert layer.shared
    assert tuple(layer.theta.shape) == (1,)
    kappa = layer.kappa.detach().numpy()
    assert kappa.shape == (len(INPUTS),)
    np.testing.assert_allclose(kappa, 1.5, rtol=1e-15)


def test_shared_false_learns_one_theta_per_input() -> None:
    layer = AsymmetricGaps(SCALE, shared=False)
    assert not layer.shared
    assert tuple(layer.theta.shape) == (len(INPUTS),)


def test_the_shared_theta_gets_the_sum_of_the_per_input_gradients() -> None:
    # Both layers start at the same kappa, so they compute the same thing.
    # By the chain rule, the one shared theta collects all eight gradients.
    gaps = torch.tensor(simulate_customers(200, 2026).gaps, dtype=DTYPE)
    shared, separate = AsymmetricGaps(SCALE), AsymmetricGaps(SCALE, shared=False)
    for layer in (shared, separate):
        layer(gaps).sum().backward()
    assert shared.theta.grad is not None and separate.theta.grad is not None
    assert bool(torch.all(separate.theta.grad != 0.0))
    total = separate.theta.grad.sum().item()
    assert shared.theta.grad.item() == pytest.approx(total, rel=1e-12)


@pytest.fixture(scope="module")
def small() -> TrainingSet:
    orders = simulate_orders(simulate_profiles(60, 2026), 2026)
    return from_simulation(orders, simulate_behaviour(orders, 2026))


def test_the_choice_passes_from_the_model_to_its_network(small: TrainingSet) -> None:
    assert FitNetwork(SCALE).gaps.shared
    assert not FitNetwork(SCALE, shared_kappa=False).gaps.shared
    assert FitModel.for_training_set(small).network.gaps.shared
    per_input = FitModel.for_training_set(small, shared_kappa=False)
    assert not per_input.network.gaps.shared
    assert not without_path(per_input, PATHS[0]).network.gaps.shared


@pytest.fixture(scope="module")
def trained(small: TrainingSet) -> tuple[FitModel, TrainingResult]:
    model = FitModel.for_training_set(small)
    return model, fit_model(model, small)


def test_training_learns_one_kappa_and_reports_eight_equal_values(
    trained: tuple[FitModel, TrainingResult],
) -> None:
    model, result = trained
    assert result.converged
    gaps = model.network.gaps
    assert tuple(gaps.theta.shape) == (1,)
    kappa = gaps.kappa.detach().numpy()
    assert kappa.shape == (len(INPUTS),)
    assert np.all(kappa == kappa[0])
    assert kappa[0] > 1.0
    assert abs(kappa[0] - 1.5) > 0.01  # training moved it from its start
