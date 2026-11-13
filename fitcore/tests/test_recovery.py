import math

import numpy as np
import pytest

from fitcore.model import LATENTS
from fitcore.recovery import (
    RECOVERY_MEASURES,
    average_ranks,
    customer_means,
    kappa_summary,
    latent_recovery,
    pearson,
    spearman,
)
from fitcore.sim import DEFAULT_TRUTH


def _random(n: int, k: int, seed: int = 2026) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, k))


def test_pearson_matches_numpy_column_by_column() -> None:
    a, b = _random(200, 3), _random(200, 3, seed=7)
    b[:, 0] += a[:, 0]
    want = [np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(3)]
    np.testing.assert_allclose(pearson(a, b), want, rtol=1e-12)


def test_pearson_is_1_or_minus_1_on_a_straight_line() -> None:
    a = _random(50, 1)
    np.testing.assert_allclose(pearson(a, 3.0 * a + 1.0), [1.0], rtol=1e-12)
    np.testing.assert_allclose(pearson(a, -0.5 * a), [-1.0], rtol=1e-12)


def test_tied_values_share_their_average_rank() -> None:
    ranks = average_ranks(np.array([[10.0], [20.0], [20.0], [5.0]]))
    np.testing.assert_array_equal(ranks[:, 0], [1.0, 2.5, 2.5, 0.0])


def test_spearman_ignores_the_shape_of_a_rising_relation() -> None:
    a = _random(300, 2)
    bent = np.exp(3.0 * a)  # only ever rises with a, but far from a line
    assert np.all(pearson(a, bent) < 0.9)
    np.testing.assert_allclose(spearman(a, bent), [1.0, 1.0], rtol=1e-12)


def test_customer_means_average_each_customers_rows() -> None:
    values = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    means = customer_means(values, np.array([0, 0, 1]))
    np.testing.assert_array_equal(means, [[2.0, 20.0], [5.0, 50.0]])


@pytest.mark.parametrize(
    "customer",
    [
        np.array([0, 2, 2]),  # customer 1 has no rows
        np.array([0, -1, 1]),
        np.array([0, 1]),
        np.array([0.0, 1.0, 1.0]),
    ],
)
def test_customer_means_rejects_bad_customers(customer: np.ndarray) -> None:
    with pytest.raises(ValueError):
        customer_means(np.ones((3, 2)), customer)


def _orders(customers: int = 40, per_customer: int = 5) -> tuple[np.ndarray, ...]:
    """True latents that differ mostly between customers, a little between orders."""
    rng = np.random.default_rng(2026)
    customer = np.repeat(np.arange(customers), per_customer)
    level = rng.uniform(0.2, 0.8, size=(customers, 5))[customer]
    truth = level + rng.normal(0.0, 0.03, size=level.shape)
    return truth, customer


def test_a_perfect_prediction_recovers_every_latent() -> None:
    truth, customer = _orders()
    result = latent_recovery(truth, truth, customer)
    assert tuple(result) == RECOVERY_MEASURES
    for measure in RECOVERY_MEASURES:
        assert result[measure].shape == (len(LATENTS),)
        np.testing.assert_allclose(result[measure], 1.0, rtol=1e-12)


def test_a_stretched_scale_costs_pearson_but_not_spearman() -> None:
    truth, customer = _orders()
    stretched = truth**4  # a different scale, same order
    result = latent_recovery(stretched, truth, customer)
    assert np.all(result["pearson"] < 0.99)
    np.testing.assert_allclose(result["spearman"], 1.0, rtol=1e-12)


def test_customer_columns_compare_customer_averages() -> None:
    truth, customer = _orders()
    noisy = truth + np.random.default_rng(1).normal(0.0, 0.2, size=truth.shape)
    result = latent_recovery(noisy, truth, customer)
    want = pearson(customer_means(noisy, customer), customer_means(truth, customer))
    np.testing.assert_allclose(result["customer_pearson"], want, rtol=1e-12)
    # Averaging five orders cancels much of the noise.
    assert np.all(result["customer_pearson"] > result["pearson"])


@pytest.mark.parametrize(
    ("predicted", "truth"),
    [
        (np.ones((10, 4)), np.ones((10, 4))),
        (np.ones((10, 5)), np.ones((9, 5))),
        (np.full((10, 5), math.nan), np.ones((10, 5))),
    ],
)
def test_latent_recovery_rejects_bad_shapes_and_values(
    predicted: np.ndarray, truth: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        latent_recovery(predicted, truth, np.arange(10))


def test_a_constant_column_has_no_correlation() -> None:
    a = _random(20, 2)
    b = a.copy()
    b[:, 1] = 0.5
    with pytest.raises(ValueError):
        pearson(a, b)


def test_kappa_summary_spreads_runs_around_the_truth() -> None:
    truth = DEFAULT_TRUTH.kappa
    runs = np.vstack([truth * 0.9, truth, truth * 1.4])
    summary = kappa_summary(runs, truth)
    np.testing.assert_array_equal(summary["truth"], truth)
    np.testing.assert_allclose(summary["min"], truth * 0.9)
    np.testing.assert_allclose(summary["max"], truth * 1.4)
    np.testing.assert_allclose(summary["mean_over_truth"], 1.1, rtol=1e-12)


@pytest.mark.parametrize(
    "runs",
    [np.ones((2, 7)), np.full((2, 8), 0.9), np.full((2, 8), math.inf), np.ones((0, 8))],
)
def test_kappa_summary_rejects_bad_runs(runs: np.ndarray) -> None:
    with pytest.raises(ValueError):
        kappa_summary(runs, DEFAULT_TRUTH.kappa)
