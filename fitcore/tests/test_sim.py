import math
from dataclasses import replace

import numpy as np
import pytest

from fitcore.features import INPUTS, transform
from fitcore.sim import (
    DEFAULT_TRUTH,
    MASK_1,
    MASK_2,
    apply_kappa,
    simulate_customers,
    true_latents,
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def no_gaps() -> np.ndarray:
    return np.zeros((1, len(INPUTS)))


def one_gap(key: str, typical_sizes: float) -> np.ndarray:
    """No gaps except `key`, set to a number of typical gap sizes."""
    gaps = no_gaps()
    i = INPUTS.index(key)
    gaps[0, i] = typical_sizes * DEFAULT_TRUTH.scale[i]
    return gaps


def test_layer_1_mask_matches_the_doc() -> None:
    # rows: respect, love; columns: x1..x8
    assert MASK_1.tolist() == [
        [1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ]


def test_layer_2_mask_matches_the_doc() -> None:
    # rows: convenience, trust; columns: respect, love, x1, x2, x3
    assert MASK_2.tolist() == [
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ]


def test_default_truth_uses_every_path_and_only_those() -> None:
    assert np.array_equal(DEFAULT_TRUTH.w1 > 0, MASK_1 == 1)
    assert np.array_equal(DEFAULT_TRUTH.w2 > 0, MASK_2 == 1)


def test_default_truth_has_valid_kappas_and_scales() -> None:
    assert np.all(DEFAULT_TRUTH.kappa >= 1.0)
    assert np.all(DEFAULT_TRUTH.scale > 0.0)


def test_no_gaps_give_the_bias_levels() -> None:
    t = DEFAULT_TRUTH
    respect = sigmoid(t.b1[0])
    love = sigmoid(t.b1[1])
    got = true_latents(no_gaps())
    assert got["respect"][0] == pytest.approx(respect)
    assert got["love"][0] == pytest.approx(love)
    assert got["convenience"][0] == pytest.approx(
        sigmoid(t.w2[0, 0] * respect + t.b2[0])
    )
    assert got["trust"][0] == pytest.approx(
        sigmoid(t.w2[1, 0] * respect + t.w2[1, 1] * love + t.b2[1])
    )


def test_latents_stay_between_zero_and_one() -> None:
    rng = np.random.default_rng(0)
    gaps = rng.normal(0.0, 3.0, size=(1000, 8)) * DEFAULT_TRUTH.scale
    for values in true_latents(gaps).values():
        assert values.shape == (1000,)
        assert np.all((values > 0.0) & (values < 1.0))


def test_x7_moves_love_but_not_respect() -> None:
    before, after = true_latents(no_gaps()), true_latents(one_gap("x7", -2.0))
    assert after["respect"][0] == before["respect"][0]
    assert after["love"][0] < before["love"][0]


def test_x2_moves_only_convenience() -> None:
    before, after = true_latents(no_gaps()), true_latents(one_gap("x2", -2.0))
    assert after["respect"][0] == before["respect"][0]
    assert after["love"][0] == before["love"][0]
    assert after["trust"][0] == before["trust"][0]
    assert after["convenience"][0] < before["convenience"][0]


def test_a_shortfall_hurts_respect_more_than_an_equal_surplus_helps() -> None:
    base = true_latents(no_gaps())["respect"][0]
    loss = base - true_latents(one_gap("x4", -1.0))["respect"][0]
    gain = true_latents(one_gap("x4", 1.0))["respect"][0] - base
    assert loss > gain > 0.0


def test_apply_kappa_matches_fitcore_features() -> None:
    gaps = np.array([-5.0, 10.0, -0.01, 0.05, -0.7, 0.3, -0.1, 2.0])
    kappas = dict(zip(INPUTS, DEFAULT_TRUTH.kappa.tolist(), strict=True))
    want = transform(dict(zip(INPUTS, gaps.tolist(), strict=True)), kappas)
    got = apply_kappa(gaps, DEFAULT_TRUTH.kappa)
    assert got.tolist() == pytest.approx([want[key] for key in INPUTS])


def test_true_model_rejects_a_negative_weight() -> None:
    w1 = DEFAULT_TRUTH.w1.copy()
    w1[0, 0] = -0.1
    with pytest.raises(ValueError, match="w1 must be zero or more"):
        replace(DEFAULT_TRUTH, w1=w1)


def test_true_model_rejects_a_weight_outside_the_mask() -> None:
    w1 = DEFAULT_TRUTH.w1.copy()
    w1[0, 1] = 0.5  # x2 does not feed respect
    with pytest.raises(ValueError, match="outside its mask"):
        replace(DEFAULT_TRUTH, w1=w1)


def test_true_model_rejects_kappa_below_one() -> None:
    kappa = DEFAULT_TRUTH.kappa.copy()
    kappa[0] = 0.9
    with pytest.raises(ValueError, match="kappa"):
        replace(DEFAULT_TRUTH, kappa=kappa)


def test_true_model_arrays_are_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        DEFAULT_TRUTH.w1[0, 0] = 9.0


def test_gaps_need_eight_columns() -> None:
    with pytest.raises(ValueError, match="shape"):
        true_latents(np.zeros((3, 7)))


def test_gaps_must_be_finite() -> None:
    gaps = no_gaps()
    gaps[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        true_latents(gaps)


def test_population_rows_line_up() -> None:
    pop = simulate_customers(500, seed=1)
    assert len(pop.customer_ids) == 500
    assert pop.gaps.shape == (500, 8)
    assert set(pop.latents) == {"respect", "love", "convenience", "trust"}
    for values in pop.latents.values():
        assert values.shape == (500,)


def test_customer_ids_are_unique_and_marked_as_simulated() -> None:
    ids = simulate_customers(500, seed=1).customer_ids
    assert len(set(ids)) == 500
    assert ids[0] == "sim-000000"
    assert ids[-1] == "sim-000499"


def test_the_same_seed_gives_the_same_population() -> None:
    a, b = simulate_customers(200, seed=7), simulate_customers(200, seed=7)
    assert a.customer_ids == b.customer_ids
    assert np.array_equal(a.gaps, b.gaps)
    for key in a.latents:
        assert np.array_equal(a.latents[key], b.latents[key])


def test_a_different_seed_gives_different_gaps() -> None:
    a, b = simulate_customers(200, seed=7), simulate_customers(200, seed=8)
    assert not np.array_equal(a.gaps, b.gaps)


def test_stored_latents_are_the_truth_applied_to_the_stored_gaps() -> None:
    truth = replace(DEFAULT_TRUTH, b1=np.array([0.0, 0.0]))
    pop = simulate_customers(300, seed=3, truth=truth)
    assert pop.truth is truth
    for key, values in true_latents(pop.gaps, truth).items():
        assert np.array_equal(pop.latents[key], values)


def test_gaps_are_one_typical_size_wide_in_their_own_units() -> None:
    gaps = simulate_customers(20_000, seed=1).gaps
    scale = DEFAULT_TRUTH.scale
    assert np.allclose(gaps.std(axis=0), scale, rtol=0.05)
    assert np.all(np.abs(gaps.mean(axis=0)) < 0.05 * scale)


def test_true_latents_spread_without_piling_up_at_zero_or_one() -> None:
    latents = simulate_customers(20_000, seed=1).latents
    for values in latents.values():
        assert 0.15 < values.std() < 0.3
        assert np.quantile(values, 0.05) > 0.05
        assert np.quantile(values, 0.95) < 0.95


def test_population_arrays_are_read_only() -> None:
    pop = simulate_customers(10, seed=1)
    with pytest.raises(ValueError, match="read-only"):
        pop.gaps[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        pop.latents["love"][0] = 1.0


def test_columns_are_ready_for_a_table() -> None:
    cols = simulate_customers(50, seed=1).columns()
    latents = ["respect", "love", "convenience", "trust"]
    assert list(cols) == ["customer_id", *INPUTS, *latents]
    assert all(len(column) == 50 for column in cols.values())
    assert cols["customer_id"][0] == "sim-000000"


def test_n_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        simulate_customers(0, seed=1)
