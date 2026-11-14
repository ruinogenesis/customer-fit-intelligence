# Model card: t112_model.json

Created 2026-11-14. Data: simulated, seed 2026, orders 2026-01-01 to 2026-06-29.
Training: 1,600 customers, 35,707 orders. Held out: 400 customers, 8,520 orders.

## Held-out metrics

| Outcome | Known | Log loss | Intercept only | AUC | Predicted | Observed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reorder 14d | 8,342 | 0.339 | 0.370 | 0.711 | 0.880 | 0.879 |
| churn 30d | 8,302 | 0.145 | 0.168 | 0.789 | 0.041 | 0.040 |
| NPS | 349 | 1.788 | 1.953 |  | 7.736 | 7.765 |

Churn 30d against order frequency alone (8,302 orders): log loss 0.145 vs 0.163, AUC 0.789 vs 0.631. Target (beat it on both): met.

Recovery (Pearson with the simulator's true latents): respect 0.985, love 0.983, convenience 0.988, trust 0.976, fit 0.987.

## Paths: masks and weights

| Path | In use | Weight |
| --- | --- | ---: |
| respect <- x1 | yes | 0.516 |
| respect <- x4 | yes | 0.396 |
| respect <- x5 | yes | 0.271 |
| respect <- x6 | yes | 0.425 |
| love <- x6 | yes | 0.628 |
| love <- x7 | yes | 0.384 |
| love <- x8 | yes | 0.570 |
| convenience <- respect | yes | 2.717 |
| convenience <- x1 | yes | 0.470 |
| convenience <- x2 | yes | 0.390 |
| convenience <- x3 | yes | 0.564 |
| trust <- respect | yes | 3.998 |
| trust <- love | yes | 2.769 |
| fit <- convenience | yes | 2.087 |
| fit <- trust | yes | 3.679 |

kappa, shared by all eight inputs: 1.604.

## Changes from the previous version

- First version: there is no previous model.
- Since the design: one kappa shared by all eight inputs (T114); a kappa per input did not earn its place (T111).
- All 15 theory paths kept: each improved held-out loss in 5 of 5 simulated populations (T111).
- Gaps are divided by their root mean square on the training set, with no centring.
- Each latent's survey item is its marker, with thresholds fixed at k/7, which sets the latents' scale.
- An L2 penalty of 1e-4 on every slope.

## Limits and intended use

- Trained and tested on simulated data only. The numbers show that the model can learn the simulator's mechanics, not that it works for real customers; retrain on real data (T701) before any decision about them.
- The simulator drives churn from fit, so beating order frequency here is partly built in; the real test of the doc's target needs real orders.
- The shared kappa sits below the simulator's true average of 1.96, so shortfalls are understated (T114).
- Latents spread about twice as wide as the truth and sit around 0.55 to 0.8, so fixed 0.5 quadrant midlines would put most customers in the upper quadrants (D5 is open).
- Per-input contributions to each score (attributions) are not available yet (T115).
- Acting on these scores automatically is profiling under PDPA and GDPR; automations need consent and guardrails (T610, T502).
