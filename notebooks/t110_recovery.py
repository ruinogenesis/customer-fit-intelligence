# %% [markdown]
# # T110: does the model recover the simulator's true states?
#
# Five simulated populations (seeds 2026 to 2030, 2,000 customers each), each
# split 80/20 by customer and trained as in T109. On the held-out customers,
# the predicted latents are compared with the simulator's truth: Pearson and
# Spearman correlations, per order and per customer average. Then kappa
# against the truth across the seeds, and one run with the simulator's own
# item cut points as the marker thresholds, to show what the k/7 marker costs.
#
# Phase 1 gate: recovered Love and Respect correlate 0.9 or more with the
# truth. Saves notebooks/results/t110_recovery.csv.

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fitcore.dataset import TrainingSet, from_simulation, split_by_customer
from fitcore.features import INPUTS
from fitcore.model import LATENTS
from fitcore.recovery import kappa_summary, latent_recovery
from fitcore.sim import DEFAULT_TRUTH
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.sim_survey import DEFAULT_ITEMS
from fitcore.train import DTYPE, FitModel, fit_model

SEEDS = (2026, 2027, 2028, 2029, 2030)
CUSTOMERS = 2000


def repo_root() -> Path:
    """The repository's top folder: the nearest folder upwards with a .git."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError("run this from inside the repository")


def population(seed: int) -> tuple[TrainingSet, TrainingSet]:
    """One simulated population, split 80/20 by customer."""
    orders = simulate_orders(simulate_profiles(CUSTOMERS, seed), seed)
    data = from_simulation(orders, simulate_behaviour(orders, seed))
    return split_by_customer(data, 0.2, seed=seed)


def train_and_compare(
    seed: int, thresholds: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Train on one population; recovery on its held-out customers, and kappa.

    thresholds: marker thresholds to use instead of k/7 (the oracle run).
    """
    train, test = population(seed)
    model = FitModel.for_training_set(train)
    if thresholds is not None:
        with torch.no_grad():
            model.anchors.thresholds.copy_(torch.tensor(thresholds, dtype=DTYPE))
    start = time.perf_counter()
    result = fit_model(model, train)
    assert result.converged, f"seed {seed} did not converge"
    with torch.no_grad():
        predicted = model.network(torch.tensor(test.gaps, dtype=DTYPE)).numpy()
    assert test.truth is not None
    recovery = latent_recovery(predicted, test.truth, test.customer)
    print(
        f"seed {seed}: {len(result.losses) - 1} rounds, "
        f"{time.perf_counter() - start:.1f} s, "
        f"Love {recovery['pearson'][1]:.3f}, Respect {recovery['pearson'][0]:.3f}"
    )
    return recovery, network_kappa(model)


def network_kappa(model: FitModel) -> np.ndarray:
    return model.network.gaps.kappa.detach().numpy()


def as_rows(
    marker: str, seed: int, recovery: dict[str, np.ndarray], kappa: np.ndarray
) -> list[tuple[str, int, str, str, float]]:
    """One row per number, in the layout of t110_recovery.csv."""
    rows = [
        (marker, seed, latent, measure, float(value))
        for measure, values in recovery.items()
        for latent, value in zip(LATENTS, values, strict=True)
    ]
    rows += [
        (marker, seed, name, "kappa", float(value))
        for name, value in zip(INPUTS, kappa, strict=True)
    ]
    return rows


COLUMNS = ["marker", "seed", "name", "measure", "value"]


# %% Train on every seed (about 1.5 to 2 minutes)
rows = []
kappas = []
for seed in SEEDS:
    recovery, kappa = train_and_compare(seed)
    kappas.append(kappa)
    rows += as_rows("k/7", seed, recovery, kappa)

# %% Latent recovery: the worst seed for each measure
results = pd.DataFrame(rows, columns=COLUMNS)
latents = results[results["measure"] != "kappa"]
worst = latents.pivot_table(
    index="name", columns="measure", values="value", aggfunc="min", sort=False
)
gate = bool(worst.loc[["love", "respect"], "pearson"].min() >= 0.9)
print(f"Gate (Love and Respect Pearson >= 0.9 in every seed): {gate}")
worst.round(3)

# %% Pearson per order, seed by seed
latents[latents["measure"] == "pearson"].pivot_table(
    index="name", columns="seed", values="value", sort=False
).round(3)

# %% kappa against the truth, across the seeds
kappa_table = pd.DataFrame(
    kappa_summary(np.array(kappas), DEFAULT_TRUTH.kappa), index=list(INPUTS)
)
kappa_table.round(2)

# %% Oracle: the simulator's own item cut points as the marker thresholds
oracle_thresholds = np.array(DEFAULT_ITEMS.cuts) / DEFAULT_ITEMS.slope
print("k/7 thresholds:      ", (np.arange(1, 7) / 7).round(3))
print("simulator's cuts:    ", oracle_thresholds.round(3))
oracle, oracle_kappa = train_and_compare(SEEDS[0], oracle_thresholds)
oracle_rows = as_rows("simulator cuts", SEEDS[0], oracle, oracle_kappa)
same_seed = results[(results["seed"] == SEEDS[0]) & (results["measure"] != "kappa")]
compare = same_seed.pivot_table(
    index="name", columns="measure", values="value", sort=False
)[["pearson", "spearman"]]
compare["oracle_pearson"] = oracle["pearson"]
compare["oracle_spearman"] = oracle["spearman"]
compare.round(3)

# %% Save every number to notebooks/results/t110_recovery.csv
truth_rows = [
    ("truth", None, name, "kappa", float(value))
    for name, value in zip(INPUTS, DEFAULT_TRUTH.kappa, strict=True)
]
table = pd.DataFrame(rows + oracle_rows + truth_rows, columns=COLUMNS)
table["seed"] = table["seed"].astype("Int64")
out = repo_root() / "notebooks" / "results" / "t110_recovery.csv"
out.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(out, index=False, float_format="%.6g")
print(f"saved {len(table)} rows to {out.relative_to(repo_root())}")
