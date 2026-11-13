import numpy as np
import pytest
import torch

from fitcore.dataset import TrainingSet, from_simulation
from fitcore.model import (
    LATENTS,
    PATHS,
    THEORY_MASKS,
    Path,
    check_masks,
    masks_without,
)
from fitcore.sim import DEFAULT_TRUTH, simulate_customers
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import DTYPE, FitModel, FitNetwork, fit_model, without_path

SCALE = np.asarray(DEFAULT_TRUTH.scale)


def _paths_in_use(masks: tuple[np.ndarray, ...]) -> int:
    return int(sum(mask.sum() for mask in masks))


def test_paths_list_every_theory_path_once_in_mask_order() -> None:
    assert len(PATHS) == 15
    assert len(set(PATHS)) == 15
    assert [sum(p.layer == layer for p in PATHS) for layer in (1, 2, 3)] == [7, 6, 2]
    for path in PATHS:
        row, column = path.position()
        assert THEORY_MASKS[path.layer - 1][row, column] == 1.0
        assert path.output in LATENTS
    assert PATHS[0].name == "respect <- x1"
    assert PATHS[-1].name == "fit <- trust"


def test_masks_without_switches_off_exactly_the_paths_given() -> None:
    path = PATHS[7]
    assert path.name == "convenience <- respect"
    masks = masks_without([path])
    assert _paths_in_use(masks) == 14
    row, column = path.position()
    assert masks[1][row, column] == 0.0
    assert _paths_in_use(masks_without(PATHS)) == 0


def test_a_path_the_theory_does_not_have_is_rejected() -> None:
    with pytest.raises(ValueError):
        masks_without([Path(1, "love", "x1")])


@pytest.mark.parametrize("change", ["add", "shape", "value", "count"])
def test_masks_may_only_remove_paths(change: str) -> None:
    masks = [np.array(mask) for mask in THEORY_MASKS]
    if change == "add":
        masks[0][1, 0] = 1.0  # love <- x1: not in the theory
    elif change == "shape":
        masks[2] = np.ones((1, 3))
    elif change == "value":
        masks[1][0, 0] = 0.5
    else:
        masks = masks[:2]
    with pytest.raises(ValueError):
        check_masks(masks)


def test_checked_masks_are_read_only_copies() -> None:
    masks = check_masks(THEORY_MASKS)
    assert masks[0] is not THEORY_MASKS[0]
    with pytest.raises(ValueError):
        masks[0][0, 0] = 0.0


def test_a_network_without_a_path_keeps_it_at_exactly_zero() -> None:
    path = PATHS[0]  # respect <- x1
    net = FitNetwork(SCALE, masks_without([path]))
    gaps = torch.tensor(simulate_customers(200, 2026).gaps, dtype=DTYPE)
    net(gaps).sum().backward()
    row, column = path.position()
    layer = net.layers[0]
    assert layer.weight[row, column].item() == 0.0
    assert layer.raw.grad is not None
    assert layer.raw.grad[row, column].item() == 0.0
    assert _paths_in_use(net.masks()) == 14


def test_a_network_cannot_add_a_path() -> None:
    masks = [np.array(mask) for mask in THEORY_MASKS]
    masks[0][1, 0] = 1.0
    with pytest.raises(ValueError):
        FitNetwork(SCALE, masks)


@pytest.fixture(scope="module")
def small() -> TrainingSet:
    orders = simulate_orders(simulate_profiles(60, 2026), 2026)
    return from_simulation(orders, simulate_behaviour(orders, 2026))


def test_without_path_copies_the_model_and_switches_one_path_off(
    small: TrainingSet,
) -> None:
    model = FitModel.for_training_set(small)
    fit_model(model, small, rounds=2)
    path = PATHS[7]  # convenience <- respect
    ablated = without_path(model, path)
    row, column = path.position()
    assert ablated.network.layer2.weight[row, column].item() == 0.0
    assert model.network.layer2.weight[row, column].item() > 0.0
    trained, copied = model.state_dict(), ablated.state_dict()
    for name, value in trained.items():
        if name != "network.layer2.mask":
            assert torch.equal(value, copied[name]), name
    with pytest.raises(ValueError):
        without_path(ablated, path)


def test_retraining_without_a_path_keeps_it_off(small: TrainingSet) -> None:
    model = without_path(FitModel.for_training_set(small), PATHS[-1])  # fit <- trust
    result = fit_model(model, small, rounds=5)
    assert model.network.layer3.weight[0, 1].item() == 0.0
    assert model.network.masks()[2].tolist() == [[1.0, 0.0]]
    assert result.losses[-1] < result.losses[0]


def test_a_model_can_start_without_paths(small: TrainingSet) -> None:
    masks = masks_without(PATHS[:2])
    model = FitModel.for_training_set(small, masks)
    assert _paths_in_use(model.network.masks()) == 13
