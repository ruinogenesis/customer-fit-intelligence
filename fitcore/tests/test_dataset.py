import math

import numpy as np
import pytest

from fitcore.dataset import (
    ANCHOR_BEHAVIOURS,
    ITEM_LATENTS,
    TrainingSet,
    from_simulation,
    gap_scale,
    split_by_customer,
)
from fitcore.model import LATENTS
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import Orders, simulate_orders, simulate_profiles


@pytest.fixture(scope="module")
def orders() -> Orders:
    return simulate_orders(simulate_profiles(60, 2026), 2026)


@pytest.fixture(scope="module")
def data(orders: Orders) -> TrainingSet:
    return from_simulation(orders, simulate_behaviour(orders, 2026))


def test_one_row_per_order_and_one_behaviour_row_per_customer(
    orders: Orders, data: TrainingSet
) -> None:
    assert len(data) == len(orders)
    assert len(data.customer_ids) == len(set(orders.customer_ids))
    assert data.gaps.shape == (len(orders), 8)
    assert data.items.shape == (len(orders), len(ITEM_LATENTS))
    assert data.behaviours.shape == (len(data.customer_ids), len(ANCHOR_BEHAVIOURS))
    names = [data.customer_ids[i] for i in data.customer]
    assert names == list(orders.customer_ids)


def test_labels_answers_and_truth_line_up_with_the_orders(
    orders: Orders, data: TrainingSet
) -> None:
    for row in range(len(orders)):
        for got, want in (
            (data.reorder[row], orders.reorder_14d[row]),
            (data.churn[row], orders.churn_30d[row]),
            (data.nps[row], orders.nps[row]),
        ):
            assert math.isnan(got) if want is None else got == float(want)
        asked = orders.survey_items[row]
        for column, latent in enumerate(ITEM_LATENTS):
            if asked == latent:
                assert data.items[row, column] == orders.survey_answers[row]
            else:
                assert math.isnan(data.items[row, column])
    assert data.truth is not None
    for column, name in enumerate(LATENTS[:4]):
        np.testing.assert_array_equal(data.truth[:, column], orders.latents[name])
    np.testing.assert_array_equal(data.truth[:, 4], orders.fit)


def test_every_order_has_at_most_one_answer(data: TrainingSet) -> None:
    answers = (~np.isnan(data.items)).sum(axis=1) + ~np.isnan(data.nps)
    assert answers.max() == 1
    assert answers.sum() > 0


def test_behaviours_are_on_the_documented_scales(orders: Orders) -> None:
    behaviour = simulate_behaviour(orders, 2026)
    data = from_simulation(orders, behaviour)
    column = list(ANCHOR_BEHAVIOURS).index
    np.testing.assert_allclose(
        data.behaviours[:, column("log_favorites")],
        np.log1p(behaviour.values["favorites_saved"]),
    )
    np.testing.assert_allclose(
        data.behaviours[:, column("quick_checkout")],
        -np.log(behaviour.values["checkout_seconds"]),
    )


def test_arrays_are_read_only(data: TrainingSet) -> None:
    with pytest.raises(ValueError):
        data.gaps[0, 0] = 1.0
    with pytest.raises(ValueError):
        data.customer[0] = 1


def test_gap_scale_is_the_root_mean_square_without_centring() -> None:
    gaps = np.array([[3.0] * 8, [-4.0] * 8])
    np.testing.assert_allclose(gap_scale(gaps), math.sqrt(12.5))
    # A constant gap is not centred away: its scale is its size.
    np.testing.assert_allclose(gap_scale(np.full((5, 8), -2.0)), 2.0)


@pytest.mark.parametrize(
    "gaps",
    [
        np.ones((0, 8)),
        np.ones((3, 7)),
        np.ones(8),
        np.array([[1.0] * 7 + [math.nan]]),
        np.array([[1.0] * 7 + [0.0]]),
    ],
)
def test_gap_scale_rejects_bad_gaps(gaps: np.ndarray) -> None:
    with pytest.raises(ValueError):
        gap_scale(gaps)


def test_split_keeps_each_customer_whole_on_one_side(data: TrainingSet) -> None:
    train, test = split_by_customer(data, 0.2, seed=7)
    assert len(test.customer_ids) == round(0.2 * len(data.customer_ids))
    assert set(train.customer_ids).isdisjoint(test.customer_ids)
    assert set(train.customer_ids) | set(test.customer_ids) == set(data.customer_ids)
    assert len(train) + len(test) == len(data)
    for part in (train, test):
        old = np.array([data.customer_ids.index(c) for c in part.customer_ids])
        np.testing.assert_array_equal(part.behaviours, data.behaviours[old])
        rows = np.isin(data.customer, old)
        np.testing.assert_array_equal(part.gaps, data.gaps[rows])
        np.testing.assert_array_equal(part.items, data.items[rows])
        assert part.truth is not None and data.truth is not None
        np.testing.assert_array_equal(part.truth, data.truth[rows])
        names = [part.customer_ids[i] for i in part.customer]
        assert names == [data.customer_ids[i] for i in data.customer[rows]]


def test_split_depends_only_on_the_seed(data: TrainingSet) -> None:
    first = split_by_customer(data, 0.2, seed=7)[1].customer_ids
    assert split_by_customer(data, 0.2, seed=7)[1].customer_ids == first
    assert split_by_customer(data, 0.2, seed=8)[1].customer_ids != first


@pytest.mark.parametrize("share", [0.0, 0.001, 0.999, 1.0])
def test_split_needs_customers_on_both_sides(data: TrainingSet, share: float) -> None:
    with pytest.raises(ValueError):
        split_by_customer(data, share, seed=7)


def _tiny(**changes: object) -> TrainingSet:
    fields: dict[str, object] = {
        "customer_ids": ("a", "b"),
        "customer": np.array([0, 1, 1]),
        "gaps": np.zeros((3, 8)),
        "reorder": np.array([1.0, 0.0, math.nan]),
        "churn": np.array([0.0, math.nan, 1.0]),
        "nps": np.array([math.nan, 10.0, 0.0]),
        "items": np.full((3, 4), math.nan),
        "behaviours": np.zeros((2, 6)),
    }
    fields.update(changes)
    return TrainingSet(**fields)


def test_a_valid_tiny_set_is_accepted() -> None:
    assert len(_tiny()) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"customer": np.array([0, 0, 0])},  # customer b has no orders
        {"customer": np.array([0, 1, 2])},  # no customer 2
        {"gaps": np.zeros((3, 7))},
        {"gaps": np.array([[math.inf] * 8] * 3)},
        {"reorder": np.array([1.0, 0.5, 0.0])},
        {"nps": np.array([11.0, 0.0, 0.0])},
        {"items": np.full((3, 4), 0.0)},
        {"items": np.full((3, 4), 7.5)},
        {"behaviours": np.full((2, 6), math.nan)},
        {"truth": np.zeros((3, 4))},
    ],
)
def test_a_bad_set_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _tiny(**changes)
