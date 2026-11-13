# %% [markdown]
# # T104: anchoring indicators and survey answers
#
# Simulates 2,000 customers, then checks how well each anchoring indicator
# follows the true latent it is meant to measure: the four 1-to-7 survey
# items (answered after some orders, in rotation with NPS) and six
# behaviours (one value per customer). Stores the orders and behaviours in
# `data/sim/`. In VS Code, click "Run Cell" above each `# %%` (Ctrl+Enter on
# the last cell); or run the whole file with
# `python notebooks/t104_indicators.py`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitcore.sim_behaviour import BEHAVIOURS, simulate_behaviour
from fitcore.sim_orders import simulate_orders, simulate_profiles
from fitcore.sim_survey import ITEM_TEXT, QUESTIONS

N = 2_000
SEED = 2026
LATENTS = ["love", "respect", "trust", "convenience"]


def repo_root() -> Path:
    """The folder that holds .git, searching up from where this runs."""
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / ".git").exists():
            return folder
    raise RuntimeError("run this inside the customer-fit-intelligence repo")


# %%
simulated = simulate_orders(simulate_profiles(N, seed=SEED), seed=SEED)
orders = pd.DataFrame(simulated.columns())
behaviour = pd.DataFrame(simulate_behaviour(simulated, seed=SEED).columns())
print(f"{len(orders):,} orders from {len(behaviour):,} customers")
orders["survey_item"].value_counts().reindex(QUESTIONS).to_frame("answers")

# %% [markdown]
# ## How well each indicator follows its latent
#
# T104's "done when": each latent has 2 to 3 indicators with realistic noise.
# Survey answers are compared with the true latent at that order; behaviours
# with the customer's average true latent. A shorter checkout means more
# convenience, so its correlation is shown with the sign flipped.

# %%
rows = []
for latent in LATENTS:
    answered = orders[orders["survey_item"] == latent]
    r = np.corrcoef(answered["survey_answer"].astype(int), answered[latent])[0, 1]
    rows.append((latent, f'survey: "{ITEM_TEXT[latent]}"', r, len(answered)))
for name, latent in BEHAVIOURS.items():
    r = np.corrcoef(behaviour[name], behaviour[f"true_{latent}"])[0, 1]
    if name == "checkout_seconds":
        r = -r
    rows.append((latent, name, r, len(behaviour)))
anchors = pd.DataFrame(rows, columns=["latent", "indicator", "correlation", "n"])
anchors.round(2)

# %%
counts = anchors.groupby("latent", sort=False).size()
print("indicators per latent:", counts.to_dict())
assert counts.between(2, 3).all()

fig, ax = plt.subplots(figsize=(8, 4))
labels = [f"{a.latent}: {a.indicator[:32]}" for a in anchors.itertuples()]
ax.barh(labels, anchors["correlation"])
ax.axvspan(0.4, 0.7, alpha=0.15, label="realistic range")
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("correlation with the true latent")
ax.legend(loc="lower right")
fig.suptitle(f"Anchoring indicators: {N:,} simulated customers, seed {SEED}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Store them

# %%
folder = repo_root() / "data" / "sim"
folder.mkdir(parents=True, exist_ok=True)
orders.to_csv(folder / "t104_orders.csv", index=False)
behaviour.to_csv(folder / "t104_behaviour.csv", index=False)
print(f"Saved {len(orders):,} orders and {len(behaviour):,} customers to {folder}")

# %%
# Read the files back: the same survey answers and behaviours, exactly.
again = pd.read_csv(folder / "t104_orders.csv")
assert (again["survey_item"].isna() == orders["survey_item"].isna()).all()
asked = orders["survey_item"].notna()
assert (again.loc[asked, "survey_answer"] == orders.loc[asked, "survey_answer"]).all()
back = pd.read_csv(folder / "t104_behaviour.csv")
assert (back[list(BEHAVIOURS)] == behaviour[list(BEHAVIOURS)]).all().all()
print("Stored survey answers and behaviours match the simulation exactly.")
