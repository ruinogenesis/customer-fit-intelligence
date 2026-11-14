# %% [markdown]
# # T113: model card v1
#
# The card for notebooks/results/t112_model.json (T112): T109's population
# (2,000 simulated customers, seed 2026, split 80/20 by customer), scored
# with the numpy forward pass, exactly as the API will score. It reports the
# data window, held-out metrics, the doc's success target (beat order
# frequency alone at predicting 30-day churn), recovery against the
# simulator's truth, the masks and weights, the changes since the design and
# the limits.
#
# Saves notebooks/results/t112_model_card.md next to the model.

# %%
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.card import DataWindow, ModelCard, compare_churn, outcome_results
from fitcore.dataset import from_simulation, split_by_customer
from fitcore.model import LATENTS, ModelParams, forward
from fitcore.recovery import pearson
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import intercept_only_losses

SEED = 2026
CREATED = date(2026, 11, 14)  # when t112_model.json was trained (T112)


def repo_root() -> Path:
    """The repository's top folder: the nearest folder upwards with a .git."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError("run this from inside the repository")


# %% The model, and T109's data split as in training (about 5 seconds)
results_dir = repo_root() / "notebooks" / "results"
params = ModelParams.from_json((results_dir / "t112_model.json").read_text())
orders = simulate_orders(simulate_profiles(2000, SEED), SEED)
data = from_simulation(orders, simulate_behaviour(orders, SEED))
train, test = split_by_customer(data, 0.2, seed=SEED)
in_test = np.isin(np.array(data.customer_ids)[data.customer], test.customer_ids)
assert np.array_equal(data.gaps[in_test], test.gaps)
print(f"held out: {len(test.customer_ids)} customers, {len(test.gaps):,} orders")

# %% The baseline: order frequency alone
# Orders so far per 30 days, with a 7-day head start so that a first order
# reads as about weekly, the simulator's typical pace. Only the history up to
# each order is used. A logistic regression on its log is fitted on the
# training customers and judged on the held-out ones.
history = pd.DataFrame(
    {
        "customer": orders.customer_ids,
        "number": orders.order_numbers,
        "at": pd.to_datetime(list(orders.placed_at)),
    }
)
first = history.groupby("customer")["at"].transform("min")
days = ((history["at"] - first).dt.total_seconds() / 86400).to_numpy()
log_rate = np.log(history["number"].to_numpy() / (days + 7.0) * 30.0)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Intercept and slope of a one-input logistic regression (Newton's method)."""
    design = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-design @ w))
        hessian = design.T @ (design * (p * (1.0 - p))[:, None])
        w -= np.linalg.solve(hessian, design.T @ (p - y))
    return w


known = ~np.isnan(data.churn) & ~in_test
w = fit_logistic(log_rate[known], data.churn[known])
baseline_chance = 1.0 / (1.0 + np.exp(-(w[0] + w[1] * log_rate[in_test])))
print(f"baseline: logit = {w[0]:.3f} {w[1]:+.3f} x log(orders per 30 days)")

# %% What changed since the design, and the limits
CHANGES = (
    (
        "Since the design: one kappa shared by all eight inputs (T114); a kappa"
        " per input did not earn its place (T111)."
    ),
    (
        "All 15 theory paths kept: each improved held-out loss in 5 of 5"
        " simulated populations (T111)."
    ),
    (
        "Gaps are divided by their root mean square on the training set, with"
        " no centring."
    ),
    (
        "Each latent's survey item is its marker, with thresholds fixed at k/7,"
        " which sets the latents' scale."
    ),
    "An L2 penalty of 1e-4 on every slope.",
)
LIMITS = (
    (
        "Trained and tested on simulated data only. The numbers show that the"
        " model can learn the simulator's mechanics, not that it works for real"
        " customers; retrain on real data (T701) before any decision about them."
    ),
    (
        "The simulator drives churn from fit, so beating order frequency here is"
        " partly built in; the real test of the doc's target needs real orders."
    ),
    (
        "The shared kappa sits below the simulator's true average of 1.96, so"
        " shortfalls are understated (T114)."
    ),
    (
        "Latents spread about twice as wide as the truth and sit around 0.55 to"
        " 0.8, so fixed 0.5 quadrant midlines would put most customers in the"
        " upper quadrants (D5 is open)."
    ),
    (
        "Per-input contributions to each score (attributions) are not available"
        " yet (T115)."
    ),
    (
        "Acting on these scores automatically is profiling under PDPA and GDPR;"
        " automations need consent and guardrails (T610, T502)."
    ),
)
print(f"{len(CHANGES)} changes, {len(LIMITS)} limits")

# %% The card
recovery = pearson(forward(params, test.gaps), test.truth)
card = ModelCard(
    title="t112_model.json",
    created=CREATED,
    params=params,
    window=DataWindow(
        source=f"simulated, seed {SEED}",
        first_order=min(orders.placed_at).date(),
        last_order=max(orders.placed_at).date(),
        train_customers=len(train.customer_ids),
        train_orders=len(train.gaps),
        test_customers=len(test.customer_ids),
        test_orders=len(test.gaps),
    ),
    results=outcome_results(params, test, intercept_only_losses(train, test)),
    churn_baseline=compare_churn(
        params, test, "order frequency alone", baseline_chance
    ),
    recovery=dict(zip(LATENTS, recovery, strict=True)),
    changes=CHANGES,
    limits=LIMITS,
)
markdown = card.to_markdown()
out = results_dir / "t112_model_card.md"
out.write_text(markdown)
print(f"saved {out.relative_to(repo_root())}: {len(markdown.splitlines())} lines")
print(markdown)
