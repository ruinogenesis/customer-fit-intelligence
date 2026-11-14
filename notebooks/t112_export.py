# %% [markdown]
# # T112: export model.json and score with numpy
#
# T109's population (2,000 simulated customers, seed 2026, split 80/20 by
# customer), trained with one shared kappa (T114). The trained numbers are
# exported to model.json, read back, and scored with the numpy forward pass
# in fitcore.model, which the API will use without PyTorch.
#
# Done when: numpy scores match PyTorch scores on 1,000 simulated customers.
# Saves notebooks/results/t112_model.json, the reference model for the API
# (T202) and the golden replay test (T309).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fitcore.dataset import from_simulation, split_by_customer
from fitcore.model import LATENTS, PATHS, ModelParams, forward, outcomes
from fitcore.sim import simulate_customers
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import DTYPE, FitModel, export_params, fit_model

SEED = 2026


def repo_root() -> Path:
    """The repository's top folder: the nearest folder upwards with a .git."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise FileNotFoundError("run this from inside the repository")


# %% Train as in T109 (about 10 seconds)
orders = simulate_orders(simulate_profiles(2000, SEED), SEED)
data = from_simulation(orders, simulate_behaviour(orders, SEED))
train, test = split_by_customer(data, 0.2, seed=SEED)
model = FitModel.for_training_set(train)
result = fit_model(model, train)
print(f"converged: {result.converged} after {len(result.losses) - 1} rounds")

# %% Export, save, and read the file back
out = repo_root() / "notebooks" / "results" / "t112_model.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(export_params(model).to_json())
params = ModelParams.from_json(out.read_text())
print(f"saved {out.relative_to(repo_root())}: {out.stat().st_size:,} bytes")
print(f"kappa (shared): {params.kappa[0]:.4f}")

# %% Done when: numpy matches PyTorch on 1,000 simulated customers
gaps = simulate_customers(1000, SEED).gaps
with torch.no_grad():
    latents = model.network(torch.tensor(gaps, dtype=DTYPE))
    reorder, churn, _ = model.heads(latents[:, -1])
scores = forward(params, gaps)
heads = outcomes(params, scores[:, -1])
differences = pd.Series(
    {
        **{
            name: np.abs(scores[:, i] - latents[:, i].numpy()).max()
            for i, name in enumerate(LATENTS)
        },
        "reorder": np.abs(heads.reorder - torch.sigmoid(reorder).numpy()).max(),
        "churn": np.abs(heads.churn - torch.sigmoid(churn).numpy()).max(),
    },
    name="max |numpy - torch|",
)
print(f"done when met: {bool(differences.max() < 1e-12)}")
differences.to_frame()

# %% What was learned: the weight on each path
weights = pd.Series(
    {path.name: params.weights[path.layer - 1][path.position()] for path in PATHS},
    name="weight",
)
weights.round(3).to_frame()

# %% Do the heads' chances match the held-out customers?
held_out = forward(params, test.gaps)
held_heads = outcomes(params, held_out[:, -1])
answers = np.arange(11)
calibration = pd.DataFrame(
    {
        "predicted": [
            held_heads.reorder[~np.isnan(test.reorder)].mean(),
            held_heads.churn[~np.isnan(test.churn)].mean(),
            (held_heads.nps @ answers)[~np.isnan(test.nps)].mean(),
        ],
        "observed": [
            np.nanmean(test.reorder),
            np.nanmean(test.churn),
            np.nanmean(test.nps),
        ],
    },
    index=["reorder rate", "churn rate", "mean NPS"],
)
calibration.round(3)
