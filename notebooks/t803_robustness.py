# %% [markdown]
# # T803: what happens when the world breaks CFI's theory
#
# T110 and T802 judged CFI in a world built on its own theory. Here the
# simulator's truth breaks that theory one way at a time, and every model is
# trained and judged as in T802 (three populations, seeds 2026 to 2028,
# 2,000 customers each, split 80/20 by customer):
#
# - baseline: the default truth, for reference.
# - missing paths: two connections the masks forbid, Love from app ease
#   (x2, weight 0.3) and Trust from fee load (x3, weight 0.3).
# - flipped sign: courier rating (x5) lowers Respect (weight -0.25 instead
#   of +0.25); CFI can only learn positive weights.
# - interaction: Fit also rises with Trust times Convenience (weight 3, the
#   Fit bias lowered so the average Fit stays where it was); CFI's Fit adds
#   its two inputs.
# - weak anchors: survey answers follow their latent a third as strongly
#   (item slope 4 instead of 12, cut points scaled to match).
# - noisy labels: a tenth of the training customers' churn and reorder
#   labels flipped; the held-out labels stay true.
#
# For each: churn and reorder AUC on the held-out customers for CFI and the
# three models that see the same eight gaps (logistic regression, gradient
# boosting, the tuned unconstrained network), the true Fit as a ceiling, and
# how closely CFI's Respect, Love and Fit follow the truth (Pearson per
# order). Saves notebooks/results/t803_robustness.csv. About half an hour on
# two CPU cores.

# %%
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fitcore.baselines import auc, log_loss
from fitcore.dataset import TrainingSet, from_simulation, split_by_customer
from fitcore.model import forward, outcomes
from fitcore.recovery import pearson
from fitcore.sim import DEFAULT_TRUTH, TrueModel
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.sim_outcomes import DEFAULT_OUTCOMES, TrueOutcomes
from fitcore.sim_survey import DEFAULT_ITEMS, TrueItems
from fitcore.train import FitModel, export_params, fit_model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (2026, 2027, 2028)
CUSTOMERS = 2000
TEST_SHARE = 0.2
FIT = 4  # Fit's column in the latents and in the simulator's truth
RECOVERED = {"respect": 0, "love": 1, "fit": FIT}
NETWORK_PENALTIES = (0.1, 1.0, 10.0)
FLIP_SHARE = 0.1
INTERACTION = 3.0
WEAK_SLOPE = 4.0


# %% Helpers
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


def fitted_chance(model, x_train, y_train, x_test) -> np.ndarray:
    """model fitted on the known training labels; its chance of 1 on test."""
    known = ~np.isnan(y_train)
    model.fit(x_train[known], y_train[known].astype(int))
    return model.predict_proba(x_test)[:, 1]


def flip_labels(y: np.ndarray, share: float, seed: int) -> np.ndarray:
    """y with a share of its known labels flipped, chosen at random."""
    flipped = y.copy()
    known = np.flatnonzero(~np.isnan(y))
    chosen = np.random.default_rng(seed).choice(
        known, round(share * len(known)), replace=False
    )
    flipped[chosen] = 1.0 - flipped[chosen]
    return flipped


# %% The broken worlds
missing_w1 = DEFAULT_TRUTH.w1.copy()
missing_w1[1, 1] = 0.3  # Love <- x2 app ease
missing_w2 = DEFAULT_TRUTH.w2.copy()
missing_w2[1, 4] = 0.3  # Trust <- x3 fee load (columns: respect, love, x1, x2, x3)
flipped_w1 = DEFAULT_TRUTH.w1.copy()
flipped_w1[0, 4] = -DEFAULT_TRUTH.w1[0, 4]  # Respect <- x5 courier rating
weak_items = TrueItems(
    slope=WEAK_SLOPE,
    cuts=tuple(c * WEAK_SLOPE / DEFAULT_ITEMS.slope for c in DEFAULT_ITEMS.cuts),
)


def interaction_outcomes() -> TrueOutcomes:
    """Fit gains Trust x Convenience; the bias drops by its baseline average
    so the average Fit stays near the default world's."""
    orders = simulate_orders(simulate_profiles(CUSTOMERS, SEEDS[0]), SEEDS[0])
    truth = np.asarray(
        from_simulation(orders, simulate_behaviour(orders, SEEDS[0])).truth
    )
    product = float(np.mean(truth[:, 2] * truth[:, 3]))  # convenience x trust
    return dataclasses.replace(
        DEFAULT_OUTCOMES,
        w_interaction=INTERACTION,
        b_fit=DEFAULT_OUTCOMES.b_fit - INTERACTION * product,
    )


SCENARIOS: dict[str, dict] = {
    "baseline": {},
    "missing paths": {
        "truth": dataclasses.replace(
            DEFAULT_TRUTH, w1=missing_w1, w2=missing_w2, relaxed=True
        )
    },
    "flipped sign": {
        "truth": dataclasses.replace(DEFAULT_TRUTH, w1=flipped_w1, relaxed=True)
    },
    "interaction": {"outcomes": interaction_outcomes()},
    "weak anchors": {"items": weak_items},
    "noisy labels": {"flip": FLIP_SHARE},
}


# %% Run every world and population
rows: list[dict] = []
started = time.perf_counter()
for scenario, change in SCENARIOS.items():
    truth: TrueModel = change.get("truth", DEFAULT_TRUTH)
    true_outcomes: TrueOutcomes = change.get("outcomes", DEFAULT_OUTCOMES)
    items: TrueItems = change.get("items", DEFAULT_ITEMS)
    for seed in SEEDS:
        orders = simulate_orders(
            simulate_profiles(CUSTOMERS, seed),
            seed,
            truth=truth,
            outcomes=true_outcomes,
            items=items,
        )
        data = from_simulation(orders, simulate_behaviour(orders, seed))
        if "flip" in change:
            # flip before the split, then keep the true labels for test
            noisy = dataclasses.replace(
                data,
                churn=flip_labels(data.churn, change["flip"], seed),
                reorder=flip_labels(data.reorder, change["flip"], seed + 1),
            )
            train, _ = split_by_customer(noisy, TEST_SHARE, seed=seed)
        else:
            train, _ = split_by_customer(data, TEST_SHARE, seed=seed)
        in_test = held_out(data, seed)
        gaps = data.gaps
        true_latents = np.asarray(data.truth)

        model = FitModel.for_training_set(train)
        result = fit_model(model, train)
        if not result.converged:
            raise RuntimeError(f"CFI did not converge: {scenario}, seed {seed}")
        params = export_params(model)
        latents = forward(params, gaps)
        heads = outcomes(params, latents[in_test, FIT])
        cfi = {"churn 30d": heads.churn, "reorder 14d": heads.reorder}

        base = {"scenario": scenario, "seed": seed}
        for name, column in RECOVERED.items():
            r = pearson(
                latents[in_test][:, [column]], true_latents[in_test][:, [column]]
            )
            rows.append({**base, "measure": f"recovery {name}", "value": float(r[0])})
        rows.append({**base, "measure": "kappa", "value": float(params.kappa[0])})
        rows.append(
            {
                **base,
                "measure": "mean true fit",
                "value": float(true_latents[:, FIT].mean()),
            }
        )

        train_labels = {"churn 30d": train.churn, "reorder 14d": train.reorder}
        test_labels = {
            "churn 30d": data.churn[in_test],
            "reorder 14d": data.reorder[in_test],
        }
        true_fit = true_latents[:, [FIT]]
        for outcome, y_test in test_labels.items():
            y_train = train_labels[outcome]
            chance = {
                "CFI (15 paths)": cfi[outcome],
                "logistic regression": fitted_chance(
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                    gaps[~in_test],
                    y_train,
                    gaps[in_test],
                ),
                "gradient boosting": fitted_chance(
                    HistGradientBoostingClassifier(random_state=seed),
                    gaps[~in_test],
                    y_train,
                    gaps[in_test],
                ),
                "unconstrained network": tuned_network_chance(
                    seed, gaps[~in_test], y_train, train.customer, gaps[in_test]
                ),
                "true Fit (ceiling)": fitted_chance(
                    LogisticRegression(), true_fit[~in_test], y_train, true_fit[in_test]
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
        print(f"{scenario}, seed {seed}: {time.perf_counter() - started:.0f} s so far")

# %% Save
RESULTS.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame(rows)[["scenario", "seed", "measure", "model", "value"]]
results.to_csv(RESULTS / "t803_robustness.csv", index=False, float_format="%.6f")

# %% Summary: mean over the three populations
pd.set_option("display.width", 160)
aucs = results[results["measure"].str.startswith("auc")]
print(
    aucs.pivot_table(
        index=["measure", "model"],
        columns="scenario",
        values="value",
        sort=False,
    )[list(SCENARIOS)].round(3)
)
other = results[~results["measure"].str.startswith("auc")]
print(
    other.pivot_table(index="measure", columns="scenario", values="value", sort=False)[
        list(SCENARIOS)
    ].round(3)
)
