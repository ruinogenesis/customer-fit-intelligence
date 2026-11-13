# %% [markdown]
# # T109: training the v2 model on simulated data
#
# 2,000 simulated customers (seed 2026), one row per order, split 80/20 by
# customer. The model learns from gaps, outcome labels, survey answers and
# behaviours only; the simulator's truth is used at the end to check it.
#
# Done when: training converges, and every head (reorder, churn, NPS) beats
# an intercept-only guess on the held-out customers.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fitcore.dataset import from_simulation, split_by_customer
from fitcore.features import INPUTS
from fitcore.model import LATENTS
from fitcore.sim_behaviour import simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.train import (
    DTYPE,
    FitModel,
    evaluate,
    fit_model,
    intercept_only_losses,
)

SEED = 2026

# %% Simulate, then split by customer
orders = simulate_orders(simulate_profiles(2000, SEED), SEED)
data = from_simulation(orders, simulate_behaviour(orders, SEED))
train, test = split_by_customer(data, 0.2, seed=SEED)
print(f"{len(data):,} orders from {len(data.customer_ids):,} customers")
print(f"train: {len(train):,} orders, {len(train.customer_ids):,} customers")
print(f"test:  {len(test):,} orders, {len(test.customer_ids):,} customers")

# %% Train
model = FitModel.for_training_set(train)
start = time.perf_counter()
result = fit_model(model, train)
seconds = time.perf_counter() - start
rounds = len(result.losses) - 1
print(f"converged: {result.converged} after {rounds} rounds, {seconds:.1f} s")
print(f"loss: {result.losses[0]:.6f} -> {result.losses[-1]:.6f}")

# %% Loss curve
losses = np.array(result.losses)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
left.plot(losses, marker=".")
left.set(title="Training loss", xlabel="LBFGS round", ylabel="loss")
right.semilogy(losses[:-1] - losses[-1], marker=".")
right.set(title="Distance from the final loss", xlabel="LBFGS round")
fig.tight_layout()
plt.show()

# %% Held-out heads vs intercept-only
held_out = evaluate(model, test)
baseline = intercept_only_losses(train, test)
heads = pd.DataFrame(
    {
        "held_out": {head: held_out[head] for head in baseline},
        "intercept_only": baseline,
    }
)
heads["beats_baseline"] = heads["held_out"] < heads["intercept_only"]
done = result.converged and bool(heads["beats_baseline"].all())
print(f"done when met: {done}")
heads.round(4)

# %% What was learned: kappa per input, head and anchor slopes
network = model.network
kappa = pd.Series(network.gaps.kappa.detach().numpy(), index=INPUTS, name="kappa")
print(kappa.round(2).to_string())
print("head slopes (reorder, churn, NPS):", model.heads.slope.detach().numpy().round(2))
print("item slopes:", model.anchors.item_slope.detach().numpy().round(2))
print("behaviour slopes:", model.anchors.behaviour_slope.detach().numpy().round(2))

# %% A first look at recovery (T110 does this properly)
with torch.no_grad():
    latents = network(torch.tensor(test.gaps, dtype=DTYPE)).numpy()
truth = test.truth
assert truth is not None
recovery = pd.DataFrame(
    {
        "corr_with_truth": [
            np.corrcoef(latents[:, i], truth[:, i])[0, 1] for i in range(len(LATENTS))
        ],
        "mean": latents.mean(axis=0),
        "sd": latents.std(axis=0),
        "true_mean": truth.mean(axis=0),
        "true_sd": truth.std(axis=0),
    },
    index=list(LATENTS),
)
recovery.round(3)
