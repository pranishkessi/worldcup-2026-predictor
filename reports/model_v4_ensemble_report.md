# Model v4: probability-focused ensemble

Model v4 blends the first baseline logistic-regression probabilities with Model v3 calibrated random-forest probabilities. The ensemble weight was selected only on the 2022 validation split, using validation log loss.

## Ensemble formula

```text
P_v4 = (1 - w) * P_first_baseline + w * P_model_v3_calibrated
```

Selected weight: `w = 0.814` for Model v3 and `0.186` for the first baseline.

## Validation metrics

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| first_baseline_logistic | 0.5212 | 0.4961 | 0.9776 | 0.5788 | 0.0350 |
| model_v3_calibrated_rf | 0.5212 | 0.4917 | 0.9595 | 0.5694 | 0.0562 |
| model_v4_probability_ensemble | 0.5170 | 0.4879 | 0.9586 | 0.5688 | 0.0622 |

## Test metrics

| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| first_baseline_logistic | 0.5712 | 0.5269 | 0.8844 | 0.5213 | 0.0597 |
| model_v3_calibrated_rf | 0.5766 | 0.5342 | 0.8860 | 0.5229 | 0.0535 |
| model_v4_probability_ensemble | 0.5738 | 0.5301 | 0.8819 | 0.5204 | 0.0636 |

## Verdict

- Compared with Model v3, Model v4 changes test log loss by `-0.0041` and Brier by `-0.0026`.

- Compared with the first baseline, Model v4 changes test log loss by `-0.0025` and accuracy by `+0.0026`.

Lower log loss, Brier, and ECE are better; higher accuracy and F1 are better.
