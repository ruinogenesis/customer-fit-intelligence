# Where the paper's numbers come from

Every table, figure and number in `main.tex` and `SUPPLEMENT.md` comes from a
file in this repository. Result files are in `notebooks/results/`, each
written by the notebook with the same task number from fixed seeds, so
rerunning a notebook writes the same file. The one exception is the
unconstrained network, whose AUC can differ between machines by up to 0.003;
the paper quotes the committed files.

| In the paper | Source | Written by |
| --- | --- | --- |
| III Data: event types | `fitcore/src/fitcore/schemas.py` (schema v1) | |
| III Data: expectations, gaps (equations 1 and 2), Table I | `fitcore/src/fitcore/expectations.py`, `fitcore/src/fitcore/features.py` | |
| III Data: simulated populations (2,000 customers each, seeds 2026 to 2045), the simulator's settings (supplement, section 2) | `fitcore/src/fitcore/sim*.py` | each notebook, from its seeds |
| III Data: labels and their windows | `fitcore/src/fitcore/sim_orders.py`; the share of labels from open windows in `t802_facts.csv` | `notebooks/t802_baselines.py` |
| IV Model: paths, parameter counts, heads, loss (equation 5), training, Fig. 1 | `fitcore/src/fitcore/model.py`, `fitcore/src/fitcore/train.py` | |
| IV Model: exact explanations over 256 coalitions | `fitcore/src/fitcore/scoring.py` | |
| V-A Setup: comparators and their settings | `notebooks/t802_baselines.py`, `fitcore/src/fitcore/baselines.py` | |
| V-B Prediction, Table II (AUC over 20 populations, paired differences) | `t802_baselines.csv`, `t802_paired.csv` | `notebooks/t802_baselines.py` |
| V-B labels from closed windows only (0.799) | the `auc_closed` columns of `t802_baselines.csv` and `t802_paired.csv` | `notebooks/t802_baselines.py` |
| V-B calibration, base rates, the churn head's ceiling, the riskiest 1%, the NPS head | the `ece_mass` column of `t802_baselines.csv`; `t802_facts.csv`; `t802_calibration.csv` (supplement) | `notebooks/t802_baselines.py` |
| V-C recovery (0.975 to 0.988), with equal and random weights | `t110_recovery.csv` (markers `k/7`, `equal weights`, `random weights`) | `notebooks/t110_recovery.py` |
| V-C weights against the simulator's, per unit of raw gap (1.8 to 4.0) | `t112_model.json` (weights and gap scales) against `DEFAULT_TRUTH` in `fitcore/src/fitcore/sim.py` | arithmetic: (learned weight / learned scale) / (true weight / true scale) |
| Table III weights and kappa 1.60 | `t112_model.json`, the model live on dev | `notebooks/t112_export.py` |
| Table III change in loss, total and heads, per path, for asymmetry and for a kappa per input | `t111_ablation.csv` (the shared-kappa CFI, retrained without each path or choice; heads are the reorder and churn measures) | `notebooks/t111_ablation.py` |
| V-C learned kappa 1.55 | the `kappa` rows of `t110_recovery.csv` | `notebooks/t110_recovery.py` |
| V-D Robustness, Table IV | `t803_robustness.csv` | `notebooks/t803_robustness.py` |
| V-D Small data, Fig. 2, the outcome-only CFI | `t803_small_data.csv` | `notebooks/t803_small_data.py`; drawn by `figures.py` |
| V-E one order explained, Fig. 3 | `t112_model.json`, scored and explained by `fitcore.scoring` | `figures.py` |
| V-E drivers, Fig. 4 (with standard errors and the stricter hit); 670,639 scorings whose splits add up exactly | `t804_drivers.csv` | `notebooks/t804_drivers.py`; Fig. 4 drawn by `figures.py` |
| V-F what events measure: the deployed and retrained CFI and the comparators | the `event-measured gaps` rows of `t802_baselines.csv` and `t802_paired.csv` | `notebooks/t802_baselines.py` |
| V-F the matrix, Table V, quadrant changes between orders, Love as expected (0.82) | `t802_display.csv`, `t802_facts.csv` | `notebooks/t802_baselines.py` |
| V-G Real orders: models, held-out gains, repeat buying | `t805_models.csv`, `t805_repeat.csv` | `notebooks/t805_olist.py`, with the Olist files in `notebooks/data/olist/` |
| V-G kappa's interval (6.3 to 7.7; fees 0.07 to 0.27) | `t805_kappa_profile.csv` | `notebooks/t805_olist.py`; interpolated between grid points by `figures.py`, which also draws `fig_kappa.pdf` for the supplement |
| V-G reviews written before and after delivery, Fig. 5, kappa on those reviews | `t805_review_by_gap.csv`, `t805_variants.csv` | `notebooks/t805_olist.py`; Fig. 5 drawn by `figures.py` |
| VI Pipeline, Fig. 6, the outbox re-send | `api/src/cfi_api/events.py`, `api/src/cfi_api/firestore_store.py`, `api/src/cfi_api/export.py` | |
| VI Models: the golden replay of 795 events | `t309_events.jsonl`, `t309_golden.json` | `notebooks/t309_golden.py` |
| VI Dashboard (223 customers, 1,633 orders, 122 on the band); supplement section 11 | `shot_matrix.png`, `shot_drivers.png`: the dev dashboard on 2026-11-19, signed in as admin, Light theme | screenshots |
| VI Work, latency and cost | `t806_system.csv` (Cloud Run's own times are from its request log); 3.6 events per order is its bytes per order over bytes per event | `notebooks/t806_system.py`, against dev |
| VI Cost per 1,000 events | the per-event reads, writes, deletes and bytes in `t806_system.csv`, priced at Google Cloud's list rates | arithmetic in the text |
| VII Discussion, VIII Conclusion | the rows above | |
| Supplement tables | the files named in each section, printed without rounding beyond what they show | |
