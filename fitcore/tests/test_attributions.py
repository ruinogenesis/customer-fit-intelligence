"""T115: each score's distance from an as-expected order, split exactly
across the eight inputs (fitcore.model.attributions).

Checked on simulated customers with T112's reference model: the parts add
up to the change, the split is the Shapley value by its own definition
(every order of adding the inputs, counted here one by one), and an input
that changes nothing gets exactly nothing.
"""

import dataclasses
import itertools
import math
import pathlib

import numpy as np
import pytest

from fitcore.features import INPUTS
from fitcore.model import (
    LATENTS,
    PATHS,
    SCORES,
    ModelParams,
    Path,
    all_scores,
    attributions,
    masks_without,
)
from fitcore.sim import simulate_customers

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "notebooks/results/t112_model.json"
PARAMS = ModelParams.from_json(MODEL.read_bytes())
ZERO = np.zeros((1, len(INPUTS)))


def gaps(n: int, seed: int, size: float = 1.0) -> np.ndarray:
    """n simulated customers' gaps (every input measured), times size."""
    return simulate_customers(n, seed).gaps * size


def change(params: ModelParams, g: np.ndarray) -> np.ndarray:
    """Each score minus its value for an order exactly as expected."""
    return all_scores(params, g) - all_scores(params, ZERO)


def reaches(params: ModelParams) -> np.ndarray:
    """6 x 8: True where an input has a path to a score in params' masks."""
    found = {name: {name} for name in INPUTS}
    for path in PATHS:
        if params.masks[path.layer - 1][path.position()]:
            found.setdefault(path.output, set()).update(found.get(path.source, set()))
    found["churn_risk"] = found["fit"]
    return np.array([[name in found[score] for name in INPUTS] for score in SCORES])


# --- The parts add up ---


@pytest.mark.parametrize("size", [1.0, 5.0], ids=["typical", "five times larger"])
def test_the_parts_add_up_to_each_scores_change(size: float) -> None:
    g = gaps(1_000, seed=115, size=size)
    found = attributions(PARAMS, g)
    assert found.shape == (1_000, len(SCORES), len(INPUTS))
    assert np.abs(found.sum(axis=2) - change(PARAMS, g)).max() < 1e-12
    # Larger gaps push the sigmoids to their ends, where the inputs interact
    # most: the scores must still have moved a long way.
    assert np.abs(change(PARAMS, g)).max() > 0.5


def test_it_is_the_shapley_value_by_its_definition() -> None:
    # Every one of the 8! orders of adding the inputs, one at a time: an
    # input's share is what adding it changed, averaged over the orders.
    g = gaps(3, seed=1150)
    value = {}
    for coalition in itertools.product((0, 1), repeat=len(INPUTS)):
        value[coalition] = all_scores(PARAMS, g * np.array(coalition))
    want = np.zeros((len(g), len(SCORES), len(INPUTS)))
    for order in itertools.permutations(range(len(INPUTS))):
        present = [0] * len(INPUTS)
        for i in order:
            before = value[tuple(present)]
            present[i] = 1
            want[:, :, i] += value[tuple(present)] - before
    want /= math.factorial(len(INPUTS))
    # 40,320 orders summed one at a time: the reference rounds a little.
    assert np.abs(attributions(PARAMS, g) - want).max() < 1e-12


# --- What gets exactly nothing ---


def test_an_input_at_zero_gets_exactly_nothing() -> None:
    g = gaps(200, seed=1151)
    g[::2, INPUTS.index("x4")] = 0.0  # as unmeasured inputs are scored
    g[1::2, INPUTS.index("x1")] = 0.0
    found = attributions(PARAMS, g)
    assert np.all(found[::2, :, INPUTS.index("x4")] == 0.0)
    assert np.all(found[1::2, :, INPUTS.index("x1")] == 0.0)


def test_an_input_with_no_path_to_a_score_gets_exactly_nothing() -> None:
    found = attributions(PARAMS, gaps(200, seed=1152))
    off = ~reaches(PARAMS)
    assert off[SCORES.index("love"), INPUTS.index("x1")]  # Love <- x6, x7, x8 only
    assert np.all(found[:, off] == 0.0)
    assert np.all(found[:, ~off] != 0.0)


def test_a_switched_off_path_gets_exactly_nothing() -> None:
    # Without love <- x6, x6 reaches Love by no path, but still reaches
    # Trust and fit through Respect.
    path = Path(1, "love", "x6")
    masks = masks_without([path])
    weights = tuple(w * m for w, m in zip(PARAMS.weights, masks, strict=True))
    params = dataclasses.replace(PARAMS, masks=masks, weights=weights)
    found = attributions(params, gaps(100, seed=1153))
    x6 = INPUTS.index("x6")
    assert np.all(found[:, SCORES.index("love"), x6] == 0.0)
    assert np.all(found[:, SCORES.index("trust"), x6] != 0.0)
    assert np.all(found[:, ~reaches(params)] == 0.0)


def test_one_input_alone_gets_the_whole_change() -> None:
    g = np.zeros((len(INPUTS), len(INPUTS)))
    np.fill_diagonal(g, gaps(1, seed=1154)[0])
    found = attributions(PARAMS, g)
    for i in range(len(INPUTS)):
        others = np.delete(found[i], i, axis=1)
        assert np.all(others == 0.0)
        assert np.allclose(found[i, :, i], change(PARAMS, g[i : i + 1])[0], atol=1e-15)


# --- Use ---


def test_two_orders_subtract_to_explain_their_change() -> None:
    first, second = gaps(2, seed=1155)
    found = attributions(PARAMS, np.stack([first, second]))
    moved = all_scores(PARAMS, second[None])[0] - all_scores(PARAMS, first[None])[0]
    assert np.abs((found[1] - found[0]).sum(axis=1) - moved).max() < 1e-12


def test_a_late_order_is_explained_by_lateness() -> None:
    # 15 minutes late and nothing else: x1 takes the whole drop in fit (as
    # T202 measured, 0.74 to 0.25) and raises churn risk.
    late = np.zeros((1, len(INPUTS)))
    late[0, INPUTS.index("x1")] = -15.0
    found = attributions(PARAMS, late)[0]
    fit, churn = SCORES.index("fit"), SCORES.index("churn_risk")
    assert found[fit, INPUTS.index("x1")] == pytest.approx(-0.49, abs=0.01)
    assert found[churn, INPUTS.index("x1")] > 0.05


def test_long_lists_come_out_as_one_order_at_a_time() -> None:
    # attributions() works through long lists in chunks of orders.
    g = gaps(1_100, seed=1156)
    found = attributions(PARAMS, g)
    for row in (0, 511, 512, 1_099):
        assert np.array_equal(found[row], attributions(PARAMS, g[row : row + 1])[0])


def test_no_orders_and_bad_gaps() -> None:
    assert attributions(PARAMS, np.zeros((0, len(INPUTS)))).shape == (0, 6, 8)
    with pytest.raises(ValueError, match="shape"):
        attributions(PARAMS, np.zeros(len(INPUTS)))
    with pytest.raises(ValueError, match="finite"):
        attributions(PARAMS, np.full((1, len(INPUTS)), np.nan))


def test_scores_are_the_latents_then_churn_risk() -> None:
    assert SCORES == (*LATENTS, "churn_risk")
    g = gaps(10, seed=1157)
    assert all_scores(PARAMS, g).shape == (10, len(SCORES))
