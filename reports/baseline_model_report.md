# First Baseline Model Report

## Goal
Train a first time-safe baseline model for international football match result prediction.

Target classes:

- `H` = home team win
- `D` = draw
- `A` = away team win

## Input dataset

`data/processed/matches_with_elo_fifa_form_confed.csv`

The feature set uses only pre-match information: Elo, FIFA ranking, recent form, venue/tournament flags, and confederation-strength features.

## Time split

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2014-01-01 to 2021-12-31 | 7,279 |
| Validation | 2022-01-01 to 2022-12-31 | 969 |
| Test | 2023-01-01 to 2026-06-02 | 3,491 |

No random split was used. This protects against future-data leakage.

## Models trained

- Dummy most-frequent baseline
- Multinomial logistic regression

The selected model is the model with the best validation log loss:

**logistic_regression**

## Best model performance

| Split | Accuracy | Macro F1 | Log loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| Validation | 0.5212 | 0.4961 | 0.9776 | 0.5788 |
| Test | 0.5712 | 0.5269 | 0.8844 | 0.5213 |

Dummy most-frequent test accuracy: **0.4718**

## Important caution

This is a first baseline, not the final model. It is useful because it gives us a clean benchmark using the strongest available structured features. Next improvements should focus on:

1. adding tournament-experience features,
2. adding date-safe head-to-head features,
3. building a Poisson expected-goals model,
4. calibrating final probabilities,
5. evaluating separately by tournament type and confederation.

## Output files

- `data/modeling/model_dataset_baseline.csv`
- `models/baseline_metrics.csv`
- `models/baseline_validation_test_predictions.csv`
- `models/baseline_test_predictions.csv`
- `models/baseline_feature_importance.csv`
- `models/baseline_confusion_matrix_validation.csv`
- `models/baseline_confusion_matrix_test.csv`
- `models/baseline_classification_reports.json`
- `models/baseline_model_best.joblib`
- `baseline_model_metadata.json`
