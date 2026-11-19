"""T802: the paper's baselines and scores, checked against their definitions."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fitcore.baselines import (
    BgNbd,
    auc,
    bgnbd_log_likelihood,
    brier,
    carry_forward,
    ece,
    event_mask,
    fit_bgnbd,
    history,
    log_loss,
    p_alive,
    p_no_order,
)
from fitcore.features import INPUTS

pytest.importorskip("scipy")

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def days(*offsets: float) -> list[datetime]:
    return [T0 + timedelta(days=d) for d in offsets]


# --- history ---


def test_history_reads_only_earlier_orders_and_ranks_rows_by_time():
    customers = ["b", "a", "a", "b", "a"]
    placed = days(3, 10, 0, 0, 4)
    h = history(customers, placed)
    assert h.number.tolist() == [2, 3, 1, 1, 2]
    assert h.days_since_first.tolist() == [3, 10, 0, 0, 4]
    assert np.isnan(h.days_since_previous[2]) and np.isnan(h.days_since_previous[3])
    assert h.days_since_previous[[0, 1, 4]].tolist() == [3, 6, 4]
    # T113's rate: a first order reads as about weekly (30 / 7 a month).
    assert h.log_rate[2] == pytest.approx(np.log(30 / 7))
    assert h.log_rate[1] == pytest.approx(np.log(3 / 17 * 30))


def test_history_needs_one_time_per_order():
    with pytest.raises(ValueError, match="same length"):
        history(["a"], [])


# --- scores ---


def test_auc_is_the_chance_a_positive_outranks_a_negative():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    rng = np.random.default_rng(1)
    p = rng.integers(0, 5, 300) / 4  # many ties
    y = (rng.random(300) < p * 0.8).astype(float)
    pos, neg = p[y == 1], p[y == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).mean()
    assert auc(p, y) == pytest.approx(pairs)


def test_scores_leave_out_unknown_labels():
    p = [0.9, 0.1, 0.5]
    y = [1.0, 0.0, np.nan]
    assert auc(p, y) == 1.0
    assert brier(p, y) == pytest.approx((0.01 + 0.01) / 2)
    assert log_loss(p, y) == pytest.approx(-np.log(0.9))


def test_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        auc([0.2, 0.3], [1, 1])


def test_ece_is_zero_when_every_bin_is_calibrated_and_the_gap_when_not():
    p = np.array([0.15] * 20 + [0.85] * 20)
    y = np.array([1] * 3 + [0] * 17 + [1] * 17 + [0] * 3, dtype=float)
    assert ece(p, y) == pytest.approx(0.0)
    assert ece([0.9, 0.9], [0, 0]) == pytest.approx(0.9)


# --- BG/NBD ---


def simulate_bgnbd(truth: BgNbd, n: int, T: float, seed: int):
    """Customers from BG/NBD itself: x, t_x and T after T days."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(truth.r, 1 / truth.alpha, n)
    p = rng.beta(truth.a, truth.b, n)
    x = np.zeros(n)
    t_x = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, np.full(n, T)


def test_bgnbd_recovers_the_parameters_it_was_simulated_with():
    truth = BgNbd(r=2.0, alpha=14.0, a=0.6, b=4.0)
    x, t_x, T = simulate_bgnbd(truth, 4000, 180.0, seed=7)
    found = fit_bgnbd(x, t_x, T)
    assert found.r / found.alpha == pytest.approx(
        truth.r / truth.alpha, rel=0.1
    )  # mean rate
    assert found.a / (found.a + found.b) == pytest.approx(
        truth.a / (truth.a + truth.b), rel=0.15
    )
    fitted = bgnbd_log_likelihood(found, x, t_x, T).sum()
    assert fitted >= bgnbd_log_likelihood(truth, x, t_x, T).sum()


def test_bgnbd_likelihood_of_a_one_time_buyer_is_its_closed_form():
    params = BgNbd(r=1.5, alpha=10.0, a=0.5, b=3.0)
    T = 30.0
    expected = (10.0 / (10.0 + T)) ** 1.5  # B(a, b)/B(a, b) = 1, no purchase in T
    assert np.exp(bgnbd_log_likelihood(params, [0], [0], [T]))[0] == pytest.approx(
        expected
    )


def test_p_alive_and_p_no_order_follow_the_model():
    params = BgNbd(r=1.5, alpha=10.0, a=0.5, b=3.0)
    # A first order: nobody drops out before buying again.
    assert p_alive(params, [0], [0], [0])[0] == 1.0
    # Right after a repeat order, alive unless that order was the last.
    assert p_alive(params, [3], [20], [20])[0] == pytest.approx(
        (3 + 3 - 1) / (0.5 + 3 + 3 - 1)
    )
    # Long silence makes death more likely.
    assert p_alive(params, [3], [20], [120])[0] < p_alive(params, [3], [20], [30])[0]
    # No time, no order; the longer the horizon, the likelier an order, until
    # only the chance of having left remains.
    assert p_no_order(params, [3], [20], [20], 0.0)[0] == pytest.approx(1.0)
    alive = p_alive(params, [3], [20], [20])[0]
    assert p_no_order(params, [3], [20], [20], 1e9)[0] == pytest.approx(
        1 - alive, abs=1e-6
    )
    assert (
        p_no_order(params, [3], [20], [20], 14.0)[0]
        > p_no_order(params, [3], [20], [20], 30.0)[0]
    )


# --- what events measure ---


def test_event_mask_follows_v1_events():
    incidents = [None] * 2000 + ["late"] * 50
    mask = event_mask(incidents, 0.3, seed=3)
    col = {key: INPUTS.index(key) for key in INPUTS}
    assert mask[:, [col["x1"], col["x2"], col["x3"]]].all()
    assert not mask[:, [col["x4"], col["x7"]]].any()
    assert mask[
        2000:, [col["x5"], col["x6"], col["x8"]]
    ].all()  # incidents: rated, with a case
    assert not mask[:2000, col["x8"]].any()
    assert (
        mask[:, col["x5"]] == mask[:, col["x6"]]
    ).all()  # one rating gives both scores
    assert mask[:2000, col["x5"]].mean() == pytest.approx(0.3, abs=0.03)
    assert (event_mask(incidents, 0.3, seed=3) == mask).all()


def test_carry_forward_keeps_each_customers_last_measured_gap():
    gaps = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
    mask = np.array([[True, False], [False, True], [False, False], [True, False]])
    out = carry_forward(gaps, mask, ["a", "a", "a", "b"], [1, 2, 3, 1])
    assert out.tolist() == [[1.0, 0.0], [1.0, 6.0], [1.0, 6.0], [4.0, 0.0]]
    with pytest.raises(ValueError, match="same rows"):
        carry_forward(gaps, mask[:2], ["a"] * 4, [1, 2, 3, 4])
