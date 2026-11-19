# Where the paper's numbers come from

Every table, figure and number in `main.tex` comes from a file in this
repository. Result files are in `notebooks/results/`, each written by the
notebook with the same task number from fixed seeds, so rerunning a notebook
writes the same file. The one exception is the unconstrained network, whose
AUC can differ between machines by up to 0.003; the paper quotes the committed
files.

| In the paper | Source | Written by |
| --- | --- | --- |
| III Data: event types | `fitcore/src/fitcore/schemas.py` (schema v1) | |
| III Data: expectations, gaps (equations 1 and 2), Table I | `fitcore/src/fitcore/expectations.py`, `fitcore/src/fitcore/features.py` | |
| III Data: simulated populations (2,000 customers each, seeds 2026 to 2030) | `fitcore/src/fitcore/sim*.py` | each notebook, from its seeds |
| IV Model: paths, parameter counts, heads, training, Fig. 1 | `fitcore/src/fitcore/model.py`, `fitcore/src/fitcore/train.py` | |
| IV Model: exact explanations over 256 coalitions | `fitcore/src/fitcore/scoring.py` | |
| V-B Prediction, Table II | `t802_baselines.csv` | `notebooks/t802_baselines.py` |
| V-B calibration | `t802_calibration.csv`, and the `ece` column of `t802_baselines.csv` | `notebooks/t802_baselines.py` |
| V-C recovery (r 0.973 to 0.990), learned and true kappa | `t110_recovery.csv` | `notebooks/t110_recovery.py` |
| V-C structural equation model, 15 positive paths | `t107_sem_paths.csv` (one population of 5,000 customers, seed 2026) | `notebooks/t107_sem.py` |
| Table III weights and kappa 1.60 | `t112_model.json`, the model live on dev | `notebooks/t112_export.py` |
| Table III change in loss per path, for asymmetry and for a kappa per input | `t111_ablation.csv` (the shared-kappa CFI, retrained without each path or choice) | `notebooks/t111_ablation.py` |
| V-D Robustness, Table IV | `t803_robustness.csv` | `notebooks/t803_robustness.py` |
| V-D Small data, Fig. 2 | `t803_small_data.csv` | `notebooks/t803_small_data.py`; drawn by `figures.py` |
| V-E one order explained, Fig. 3 | `t112_model.json`, scored and explained by `fitcore.scoring` | `figures.py` |
| V-E drivers, Fig. 4; 670,639 scorings whose splits add up exactly | `t804_drivers.csv` | `notebooks/t804_drivers.py`; Fig. 4 drawn by `figures.py` |
| V-F Measurement and the matrix, Table V | `t802_display.csv` | `notebooks/t802_baselines.py` |
| V-G Real orders: models, held-out gains, repeat buying | `t805_models.csv`, `t805_repeat.csv` | `notebooks/t805_olist.py`, with the Olist files in `notebooks/data/olist/` |
| V-G Fig. 5, reviews by days early or late | `t805_review_by_gap.csv` | `notebooks/t805_olist.py`; drawn by `figures.py` |
| V-G Fig. 6, kappa's profile likelihood and its interval (6.3 to 7.7; 0.07 to 0.27) | `t805_kappa_profile.csv` | `notebooks/t805_olist.py`; the interval interpolated between grid points and drawn by `figures.py` |
| VI Pipeline, Fig. 7, Algorithm 1 | `api/src/cfi_api/events.py`, `api/src/cfi_api/firestore_store.py`, `api/src/cfi_api/export.py` | |
| VI Models: the golden replay of 795 events | `t309_events.jsonl`, `t309_golden.json` | `notebooks/t309_golden.py` |
| VI Dashboard, Fig. 8 (223 customers, 1,633 orders, 122 on the band) | `shot_matrix.png`, `shot_drivers.png`: the dev dashboard on 2026-11-19, signed in as admin, Light theme | screenshots; `figures.py` cuts them into `fig_dash_matrix.png` and `fig_dash_drivers.png` |
| VI Work per event, latency, Table VI | `t806_system.csv` (Cloud Run's own times are from its request log) | `notebooks/t806_system.py`, against dev |
| VI Cost per 1,000 events | the per-event reads, writes, deletes and bytes in `t806_system.csv`, priced at Google Cloud's list rates | arithmetic in the text |
| VII Discussion, VIII Conclusion | the rows above | |
