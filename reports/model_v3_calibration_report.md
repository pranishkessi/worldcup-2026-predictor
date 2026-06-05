# Model v3: Calibrated Model v2

## Summary

Model v3 applies post-hoc single-temperature calibration to the Model v2 `random_forest_shallow` probabilities.
The temperature was fitted only on the 2022 validation split, then evaluated on the 2023–2026 test split.

Single-temperature scaling preserves each row's probability ordering, so the predicted W/D/L class remains unchanged while the probability distribution is softened or sharpened.

## Calibration setting

- Method: temperature scaling on class probabilities
- Fitted temperature: `0.932393`
- Class order: `['H', 'D', 'A']`
- Calibration data: validation split only

## Validation result

| Metric | V2 uncalibrated | V3 calibrated | Change |
|---|---:|---:|---:|
| Accuracy | 0.5212 | 0.5212 | 0.0000 |
| Macro F1 | 0.4917 | 0.4917 | 0.0000 |
| Log loss | 0.9600 | 0.9595 | -0.0005 |
| Brier score | 0.5702 | 0.5694 | -0.0008 |
| ECE, 10-bin | 0.0586 | 0.0562 | -0.0025 |
| Avg confidence | 0.5105 | 0.5213 | 0.0108 |

## Test result

| Metric | V2 uncalibrated | V3 calibrated | Change |
|---|---:|---:|---:|
| Accuracy | 0.5766 | 0.5766 | 0.0000 |
| Balanced accuracy | 0.5357 | 0.5357 | 0.0000 |
| Macro F1 | 0.5342 | 0.5342 | 0.0000 |
| Weighted F1 | 0.5792 | 0.5792 | 0.0000 |
| Log loss | 0.8884 | 0.8860 | -0.0024 |
| Brier score | 0.5244 | 0.5229 | -0.0015 |
| ECE, 10-bin | 0.0538 | 0.0535 | -0.0003 |
| Avg confidence | 0.5449 | 0.5569 | 0.0120 |

## Verdict

Model v3 keeps Model v2's hard-class performance because temperature scaling does not change the argmax prediction.
Its value is judged by probability metrics: log loss, Brier score, and reliability/ECE.

See:

- `models/v3_calibration_metrics.csv`
- `models/baseline_v2_v3_calibration_comparison.csv`
- `models/v3_reliability_bins.csv`
