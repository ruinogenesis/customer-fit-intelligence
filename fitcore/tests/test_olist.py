"""T805: reading Olist's orders, and the cumulative logit and kappa fitted to them."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pd = pytest.importorskip("pandas")

from fitcore.olist import (
    FILES,
    LR_95,
    asymmetric,
    fit_ordinal,
    kappa_interval,
    load,
    ordinal_gradient,
    ordinal_nll,
    profile_kappa,
)


def ordinal_answers(eta: np.ndarray, cuts: np.ndarray, seed: int) -> np.ndarray:
    """Answers 0..len(cuts) from a logistic latent: the cumulative logit's own model."""
    latent = eta + np.random.default_rng(seed).logistic(size=len(eta))
    return np.searchsorted(cuts, latent)


def test_asymmetric_is_cfis_g():
    assert asymmetric([2.0, 0.0, -2.0], 3.0).tolist() == [2.0, 0.0, -6.0]


def test_ordinal_nll_is_minus_the_log_of_the_answers_band():
    nll = ordinal_nll([1.0], [-1.0, 0.5], [[0.0], [0.0], [0.0]], [0, 1, 2])
    sig = lambda t: 1 / (1 + np.exp(-t))
    expected = -np.log([sig(-1.0), sig(0.5) - sig(-1.0), 1 - sig(0.5)])
    assert nll == pytest.approx(expected)


def test_fit_ordinal_recovers_the_slopes_and_cuts_it_was_simulated_with():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20_000, 2))
    beta, cuts = np.array([1.2, -0.5]), np.array([-2.0, -0.5, 0.4, 1.8])
    y = ordinal_answers(x @ beta, cuts, seed=6)
    fit = fit_ordinal(x, y, 5)
    assert fit.beta == pytest.approx(beta, abs=0.05)
    assert fit.cuts == pytest.approx(cuts, abs=0.07)
    assert fit.nll == pytest.approx(ordinal_nll(fit.beta, fit.cuts, x, y).sum())


def test_ordinal_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 2))
    y = ordinal_answers(x @ np.array([0.8, -0.4]), np.array([-1.0, 0.0, 1.2]), seed=4)
    beta, cuts = np.array([0.5, -0.2]), np.array([-0.8, 0.1, 1.0])
    by_beta, by_cuts = ordinal_gradient(beta, cuts, x, y)
    step = 1e-6

    def total(b, c):
        return ordinal_nll(b, c, x, y).sum()

    for i in range(2):
        e = np.zeros(2)
        e[i] = step
        numeric = (total(beta + e, cuts) - total(beta - e, cuts)) / (2 * step)
        assert by_beta[i] == pytest.approx(numeric, rel=1e-5)
    for j in range(3):
        e = np.zeros(3)
        e[j] = step
        numeric = (total(beta, cuts + e) - total(beta, cuts - e)) / (2 * step)
        assert by_cuts[j] == pytest.approx(numeric, rel=1e-5)


def test_fit_ordinal_stops_at_the_optimum_on_a_large_sample():
    # Sums over many rows defeat a finite-difference search (it stopped with
    # a gradient near 1e-3 here); with the exact gradient it goes all the way,
    # so a profile over kappa has no jags.
    rng = np.random.default_rng(7)
    n = 80_000
    x = np.column_stack(
        [asymmetric(rng.normal(8.0, 9.0, n), 7.0) / 12.0, rng.normal(size=n)]
    )
    y = ordinal_answers(x @ np.array([0.9, -0.5]), np.array([-2.5, -1.8, -1.2, 0.2]), 8)
    fit = fit_ordinal(x, y, 5)
    by_beta, by_cuts = ordinal_gradient(fit.beta, fit.cuts, x, y)
    assert np.abs(np.concatenate([by_beta, by_cuts])).max() < 1e-4


def test_with_no_inputs_the_cuts_are_the_answer_shares():
    y = np.array([0] * 10 + [1] * 30 + [2] * 60)
    fit = fit_ordinal(np.empty((100, 0)), y, 3)
    assert 1 / (1 + np.exp(-fit.cuts)) == pytest.approx([0.1, 0.4], abs=1e-4)


def test_fit_ordinal_refuses_answers_out_of_range():
    with pytest.raises(ValueError, match="0 to 2"):
        fit_ordinal(np.zeros((2, 1)), [0, 3], 3)


def test_profile_kappa_finds_the_asymmetry_it_was_simulated_with():
    rng = np.random.default_rng(8)
    gaps = rng.normal(1.0, 2.0, 15_000)
    y = ordinal_answers(
        0.8 * asymmetric(gaps, 3.0), np.array([-3.0, -1.5, 0.0, 1.5]), seed=9
    )
    profile = profile_kappa(
        gaps, 1.0, np.empty((len(y), 0)), y, 5, [1.0, 2.0, 2.5, 3.0, 3.5, 5.0]
    )
    best, low, high = kappa_interval(profile)
    assert best == 3.0
    assert low <= 3.0 <= high
    assert low > 1.0  # symmetric is ruled out


def test_kappa_interval_keeps_kappas_within_the_likelihood_ratio_bound():
    profile = [
        (1.0, 10.0),
        (2.0, 5.0),
        (3.0, 5.0 + LR_95 - 1e-9),
        (4.0, 5.0 + LR_95 + 1e-9),
    ]
    assert kappa_interval(profile) == (2.0, 2.0, 3.0)


def write(folder: Path, name: str, frame) -> None:
    frame.to_csv(folder / FILES[name], index=False)


def test_load_keeps_delivered_orders_with_their_latest_review(tmp_path: Path):
    write(
        tmp_path,
        "orders",
        pd.DataFrame(
            {
                "order_id": ["o1", "o2", "o3"],
                "customer_id": ["c1", "c2", "c3"],
                "order_status": ["delivered", "delivered", "canceled"],
                "order_purchase_timestamp": [
                    "2017-01-01 10:00",
                    "2017-01-02 10:00",
                    "2017-01-03",
                ],
                "order_approved_at": ["", "", ""],
                "order_delivered_carrier_date": ["", "", ""],
                "order_delivered_customer_date": [
                    "2017-01-05 10:00",
                    "2017-01-12 22:00",
                    "",
                ],
                "order_estimated_delivery_date": [
                    "2017-01-08",
                    "2017-01-10",
                    "2017-01-09",
                ],
            }
        ),
    )
    write(
        tmp_path,
        "items",
        pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2"],
                "order_item_id": [1, 2, 1],
                "product_id": ["p", "q", "r"],
                "seller_id": ["s", "s", "t"],
                "shipping_limit_date": ["", "", ""],
                "price": [60.0, 40.0, 50.0],
                "freight_value": [10.0, 10.0, 25.0],
            }
        ),
    )
    write(
        tmp_path,
        "reviews",
        pd.DataFrame(
            {
                "review_id": ["r1", "r2", "r3"],
                "order_id": ["o1", "o1", "o3"],
                "review_score": [2, 5, 1],
                "review_comment_title": ["", "", ""],
                "review_comment_message": ["", "", ""],
                "review_creation_date": ["2017-01-06", "2017-01-07", "2017-01-04"],
                "review_answer_timestamp": [
                    "2017-01-06 09:00",
                    "2017-01-07 12:00",
                    "2017-01-04 08:00",
                ],
            }
        ),
    )
    write(
        tmp_path,
        "customers",
        pd.DataFrame(
            {
                "customer_id": ["c1", "c2", "c3"],
                "customer_unique_id": ["ann", "ann", "bob"],
                "customer_zip_code_prefix": [1, 2, 3],
                "customer_city": ["x", "y", "z"],
                "customer_state": ["SP", "SP", "RJ"],
            }
        ),
    )
    out = load(tmp_path)
    assert out["order_id"].tolist() == ["o1", "o2"]  # the canceled order is gone
    assert out["customer"].tolist() == ["ann", "ann"]
    assert out["gap_days"].tolist() == pytest.approx([2 + 14 / 24, -(2 + 22 / 24)])
    assert out["days"].tolist() == pytest.approx([4.0, 10.5])
    assert out["price"].tolist() == [100.0, 50.0]
    assert out["fee"].tolist() == pytest.approx([0.2, 0.5])
    assert out["review"].iloc[0] == 5.0  # the later of two reviews
    assert np.isnan(out["review"].iloc[1])
    assert out["reviewed"].iloc[0] == pd.Timestamp("2017-01-07 12:00")
    assert pd.isna(out["reviewed"].iloc[1])


def test_load_names_the_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="olist_orders_dataset.csv"):
        load(tmp_path)
