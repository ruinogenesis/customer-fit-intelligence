# %% [markdown]
# # T111: which paths earn their place?
#
# The five populations of T110 (seeds 2026 to 2030, 2,000 customers each,
# split 80/20 by customer). In each, train the full model, then retrain it
# without one path at a time, starting from the full model (a warm start), and
# compare the held-out data loss: heads + items + behaviours, no L2. A path
# earns its place if dropping it makes the held-out loss worse in every seed.
#
# Two kappa choices are tested the same way. "kappa per input" is compared
# with one shared kappa for all eight inputs; "asymmetry" with no asymmetry at
# all (every kappa fixed at 1). Each earns its place if the full model beats
# the simpler version in every seed.
#
# Since T114 the model shares one kappa by default. This notebook still
# trains its full model with a kappa per input (shared_kappa=False), as when
# its results were saved, so the comparison above stays meaningful.
#
# Saves notebooks/results/t111_ablation.csv.

# %%
import copy
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from fitcore.dataset import TrainingSet, from_simulation, split_by_customer
from fitcore.model import PATHS
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import DTYPE, FitModel, evaluate, fit_model, without_path
from torch import nn

SEEDS = (2026, 2027, 2028, 2029, 2030)
CUSTOMERS = 2000
COLUMNS = ["seed", "variant", "measure", "value"]


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


def held_out(model: FitModel, test: TrainingSet) -> dict[str, float]:
    """Each held-out data term, and their total."""
    terms = evaluate(model, test)
    return terms | {"total": sum(terms.values())}


def shared_kappa(model: FitModel) -> FitModel:
    """A copy whose eight kappas are one, starting at the mean of theirs."""
    variant = copy.deepcopy(model)
    gaps = variant.network.gaps
    gaps.theta = nn.Parameter(gaps.theta.detach().mean().reshape(1))
    return variant


def no_asymmetry(model: FitModel) -> FitModel:
    """A copy with every kappa fixed at 1: a shortfall counts like a surplus."""
    variant = copy.deepcopy(model)
    theta = torch.full((1,), -50.0, dtype=DTYPE)  # softplus(-50) is about 2e-22
    variant.network.gaps.theta = nn.Parameter(theta, requires_grad=False)
    return variant


def drop(path_index: int) -> Callable[[FitModel], FitModel]:
    return lambda model: without_path(model, PATHS[path_index])


VARIANTS: dict[str, Callable[[FitModel], FitModel]] = {
    **{path.name: drop(i) for i, path in enumerate(PATHS)},
    "kappa per input (vs one shared)": shared_kappa,
    "asymmetry (vs kappa = 1)": no_asymmetry,
}

# %% Every variant in every seed (about 6 to 10 minutes)
rows = []
for seed in SEEDS:
    start = time.perf_counter()
    train, test = population(seed)
    full = FitModel.for_training_set(train, shared_kappa=False)
    assert fit_model(full, train).converged, f"seed {seed}: full model"
    base = held_out(full, test)
    rows += [(seed, "full model", measure, value) for measure, value in base.items()]
    for name, make in VARIANTS.items():
        variant = make(full)
        assert fit_model(variant, train).converged, f"seed {seed}: {name}"
        losses = held_out(variant, test)
        rows += [(seed, name, measure, value) for measure, value in losses.items()]
    print(
        f"seed {seed}: {len(VARIANTS) + 1} models in {time.perf_counter() - start:.0f} s"
    )

# %% How much worse the held-out total gets without each path or kappa choice
results = pd.DataFrame(rows, columns=COLUMNS)
full_loss = results[results["variant"] == "full model"].set_index(["seed", "measure"])
changes = results[results["variant"] != "full model"].copy()
changes["increase"] = (
    changes["value"]
    - full_loss.loc[
        list(zip(changes["seed"], changes["measure"], strict=True)), "value"
    ].to_numpy()
)
total = changes[changes["measure"] == "total"].groupby("variant", sort=False)
table = total["increase"].agg(["min", "mean", "max"])
table["worse_in"] = total["increase"].apply(lambda d: int((d > 0).sum()))
table["earns_its_place"] = table["worse_in"] == len(SEEDS)
kept = int(table.loc[[path.name for path in PATHS], "earns_its_place"].sum())
print(f"paths that earn their place (worse in all {len(SEEDS)} seeds): {kept} of 15")
table.round(4)

# %% Where the increase shows up: mean increase per held-out term
changes[changes["measure"] != "total"].pivot_table(
    index="variant", columns="measure", values="increase", sort=False
)[["reorder", "churn", "nps", "items", "behaviours"]].round(4)

# %% Save every held-out loss to notebooks/results/t111_ablation.csv
out = repo_root() / "notebooks" / "results" / "t111_ablation.csv"
out.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(out, index=False, float_format="%.6g")
print(f"saved {len(results)} rows to {out.relative_to(repo_root())}")
