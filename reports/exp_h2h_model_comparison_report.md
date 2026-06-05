# Tournament Experience + Head-to-Head Model Comparison

## Goal
Improve the first baseline by adding two date-safe feature families:

1. tournament-experience features,
2. head-to-head features.

Input dataset: `data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv`

The same past-to-future split is used as the first baseline:

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2014-01-01 to 2021-12-31 | 7,279 |
| Validation | 2022-01-01 to 2022-12-31 | 969 |
| Test | 2023-01-01 to 2026-06-02 | 3,491 |

No random split was used.

## Validation comparison
| Metric | Baseline | Exp+H2H | Change | Improved? |
|---|---:|---:|---:|:---:|
| accuracy | 0.5212 | 0.4788 | -0.0423 | no |
| macro_f1 | 0.4961 | 0.4655 | -0.0305 | no |
| log_loss | 0.9776 | 1.0121 | +0.0344 | no |
| multiclass_brier | 0.5788 | 0.6003 | +0.0215 | no |
| ranked_probability_score | 0.1301 | 0.1331 | +0.0031 | no |

## Test comparison
| Metric | Baseline | Exp+H2H | Change | Improved? |
|---|---:|---:|---:|:---:|
| accuracy | 0.5712 | 0.5486 | -0.0226 | no |
| macro_f1 | 0.5269 | 0.5270 | +0.0002 | yes |
| log_loss | 0.8844 | 0.9202 | +0.0358 | no |
| multiclass_brier | 0.5213 | 0.5456 | +0.0243 | no |
| ranked_probability_score | 0.1139 | 0.1181 | +0.0042 | no |



## Verdict

On the held-out test period, accuracy changed from **0.5712** to **0.5486** and log loss changed from **0.8844** to **0.9202**.

Because log loss is the most important probability-quality metric here, the new features are considered not an improvement on log loss on the test period.

## Top new-feature coefficient signals

| class   | feature                                              |   importance |   abs_importance |
|:--------|:-----------------------------------------------------|-------------:|-----------------:|
| A       | num__h2h_home_team_points_per_match_last5_filled     |     0.559862 |         0.559862 |
| H       | num__exp_points_per_match_prior_diff_filled          |    -0.434313 |         0.434313 |
| H       | num__h2h_home_team_points_per_match_last5_filled     |    -0.397445 |         0.397445 |
| H       | num__away_exp_points_per_match_prior_filled          |    -0.376151 |         0.376151 |
| A       | num__h2h_home_team_win_rate_last5_filled             |    -0.348316 |         0.348316 |
| A       | num__exp_points_per_match_prior_diff_filled          |     0.321785 |         0.321785 |
| H       | num__h2h_home_team_win_rate_last5_filled             |     0.31831  |         0.31831  |
| A       | num__exp_avg_tournament_importance_prior_diff_filled |     0.289218 |         0.289218 |
| H       | num__exp_goal_diff_per_match_prior_diff_filled       |    -0.273367 |         0.273367 |
| H       | num__home_exp_points_per_match_prior_filled          |     0.264722 |         0.264722 |
| A       | num__away_exp_avg_tournament_importance_prior_filled |     0.253165 |         0.253165 |
| A       | num__away_exp_points_per_match_prior_filled          |     0.239117 |         0.239117 |
| A       | num__home_exp_points_per_match_prior_filled          |    -0.236454 |         0.236454 |
| H       | num__home_exp_goal_diff_per_match_prior_filled       |     0.218584 |         0.218584 |
| A       | num__home_exp_avg_tournament_importance_prior_filled |    -0.215667 |         0.215667 |
| H       | num__exp_avg_tournament_importance_prior_diff_filled |    -0.195413 |         0.195413 |
| A       | num__h2h_goal_diff_per_match_last5_filled            |    -0.191707 |         0.191707 |
| A       | num__exp_years_since_first_match_diff_filled         |    -0.177477 |         0.177477 |
| H       | num__away_exp_goal_diff_per_match_prior_filled       |    -0.175654 |         0.175654 |
| H       | num__home_exp_avg_tournament_importance_prior_filled |     0.166406 |         0.166406 |

## Output files

- `data/modeling/model_dataset_exp_h2h.csv`
- `models/exp_h2h_metrics.csv`
- `models/baseline_vs_exp_h2h_comparison.csv`
- `models/exp_h2h_validation_test_predictions.csv`
- `models/exp_h2h_test_predictions.csv`
- `models/exp_h2h_feature_importance.csv`
- `models/exp_h2h_confusion_matrix_validation.csv`
- `models/exp_h2h_confusion_matrix_test.csv`
- `models/exp_h2h_classification_reports.json`
- `models/exp_h2h_model_best.joblib`
- `exp_h2h_model_metadata.json`
