# %% [markdown]
# # T803: how much data does CFI need?
#
# A theory-constrained model should need less data than one that must find
# its own structure. Here the default world of T802 (seeds 2026 to 2028,
# 2,000 customers each, split 80/20 by customer) is trained again on a
# random 100, 200, 400 and 800 of its 1,600 training customers, and on all
# of them, and judged on the same 400 held-out customers each time.
#
# For each size: churn and reorder AUC for CFI, logistic regression and the
# tuned unconstrained network (as in T802), and how closely CFI's Respect,
# Love and Fit follow the truth (Pearson per order). Saves
# notebooks/results/t803_small_data.csv. About ten minutes on two CPU
# cores.

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.baselines import auc, log_loss
from fitcore.dataset import TrainingSet, from_simulation, split_by_customer
from fitcore.model import forward, outcomes
from fitcore.recovery import pearson
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import FitModel, export_params, fit_model
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (2026, 2027, 2028)
CUSTOMERS = 2000
TEST_SHARE = 0.2
SIZES = (100, 200, 400, 800, 1600)  # training customers
FIT = 4  # Fit's column in the latents and in the simulator's truth
RECOVERED = {"respect": 0, "love": 1, "fit": FIT}
NETWORK_PENALTIES = (0.1, 1.0, 10.0)


# %% Helpers (as in t803_robustness.py)
def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


RESULTS = repo_root() / "notebooks" / "results"


def held_out(data: TrainingSet, seed: int) -> np.ndarray:
    """Which rows split_by_customer puts in test: the same draw, as a row mask."""
    count = len(data.customer_ids)
    chosen = np.random.default_rng(seed).permutation(count)[: round(TEST_SHARE * count)]
    return np.isin(data.customer, chosen)


def network(alpha: float, seed: int):
    """Two hidden layers of 16, fitted by L-BFGS like CFI, with an L2 penalty alpha."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 16),
            solver="lbfgs",
            alpha=alpha,
            max_iter=2000,
            random_state=seed,
        ),
    )


def tuned_network_chance(seed, x_train, y_train, customer_train, x_test) -> np.ndarray:
    """The unconstrained network, its penalty picked on a fifth of the training
    customers (by held-out log loss), then refitted on them all."""
    known = ~np.isnan(y_train)
    x, y, who = x_train[known], y_train[known].astype(int), customer_train[known]
    people = np.unique(who)
    inner = np.isin(
        who, np.random.default_rng(seed).permutation(people)[: len(people) // 5]
    )
    losses = {}
    for alpha in NETWORK_PENALTIES:
        p = network(alpha, seed).fit(x[~inner], y[~inner]).predict_proba(x[inner])[:, 1]
        losses[alpha] = log_loss(p, y[inner])
    best = min(losses, key=losses.__getitem__)
    return network(best, seed).fit(x, y).predict_proba(x_test)[:, 1]


def logistic_chance(x_train, y_train, x_test) -> np.ndarray:
    known = ~np.isnan(y_train)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train[known], y_train[known].astype(int))
    return model.predict_proba(x_test)[:, 1]


# %% Train on fewer and fewer customers
rows: list[dict] = []
started = time.perf_counter()
for seed in SEEDS:
    orders = simulate_orders(simulate_profiles(CUSTOMERS, seed), seed)
    data = from_simulation(orders, simulate_behaviour(orders, seed))
    full_train, _ = split_by_customer(data, TEST_SHARE, seed=seed)
    in_test = held_out(data, seed)
    test_gaps = data.gaps[in_test]
    true_latents = np.asarray(data.truth)[in_test]
    labels = {"churn 30d": data.churn[in_test], "reorder 14d": data.reorder[in_test]}
    for size in SIZES:
        keep = np.zeros(len(full_train.customer_ids), dtype=bool)
        keep[np.random.default_rng([seed, size]).permutation(len(keep))[:size]] = True
        train = full_train.select(keep)
        model = FitModel.for_training_set(train)
        if not fit_model(model, train).converged:
            raise RuntimeError(f"CFI did not converge: {size} customers, seed {seed}")
        params = export_params(model)
        latents = forward(params, test_gaps)
        heads = outcomes(params, latents[:, FIT])
        cfi = {"churn 30d": heads.churn, "reorder 14d": heads.reorder}
        base = {"seed": seed, "customers": size, "orders": len(train.customer)}
        for name, column in RECOVERED.items():
            r = pearson(latents[:, [column]], true_latents[:, [column]])
            rows.append(
                {
                    **base,
                    "measure": f"recovery {name}",
                    "model": "CFI (15 paths)",
                    "value": float(r[0]),
                }
            )
        train_labels = {"churn 30d": train.churn, "reorder 14d": train.reorder}
        for outcome, y_test in labels.items():
            y_train = train_labels[outcome]
            chance = {
                "CFI (15 paths)": cfi[outcome],
                "logistic regression": logistic_chance(train.gaps, y_train, test_gaps),
                "unconstrained network": tuned_network_chance(
                    seed, train.gaps, y_train, train.customer, test_gaps
                ),
            }
            for model_name, p in chance.items():
                rows.append(
                    {
                        **base,
                        "measure": f"auc {outcome}",
                        "model": model_name,
                        "value": auc(p, y_test),
                    }
                )
        print(f"seed {seed}, {size} customers: {time.perf_counter() - started:.0f} s")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(rows)
results.to_csv(RESULTS / "t803_small_data.csv", index=False, float_format="%.6f")

# %% Summary: mean over the three populations
pd.set_option("display.width", 160)
print(
    results.pivot_table(
        index=["measure", "model"],
        columns="customers",
        values="value",
        sort=False,
    ).round(3)
)
