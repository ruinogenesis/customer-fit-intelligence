import numpy as np
import pytest

from fitcore import model, sim


def test_masks_have_the_doc_shapes_and_fifteen_paths() -> None:
    assert model.MASK_1.shape == (len(model.LAYER_1), 8)
    assert model.MASK_2.shape == (len(model.LAYER_2), len(model.LAYER_2_INPUTS))
    assert model.MASK_3.shape == (len(model.LAYER_3), len(model.LAYER_3_INPUTS))
    paths = [int(m.sum()) for m in (model.MASK_1, model.MASK_2, model.MASK_3)]
    assert paths == [7, 6, 2]


def test_latents_are_listed_in_network_order() -> None:
    assert model.LATENTS == ("respect", "love", "convenience", "trust", "fit")


def test_masks_are_read_only() -> None:
    with pytest.raises(ValueError):
        model.MASK_1[0, 1] = 1.0


def test_the_simulator_uses_the_same_masks() -> None:
    assert sim.MASK_1 is model.MASK_1
    assert sim.MASK_2 is model.MASK_2


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 0], [0, 1], [1, 1]],  # wrong shape
        [[1, 2], [0, 1]],  # not 0 or 1
        [[1, 0.5], [0, 1]],  # not 0 or 1
    ],
)
def test_a_bad_mask_is_rejected(rows: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        model._mask(np.array(rows), (2, 2))
