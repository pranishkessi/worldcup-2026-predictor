# Model v2: Feature Selection + Stronger Classifiers

## Goal
Train a stronger second milestone using a compact, selected feature set and several model families. Selection is based on **validation log loss**.

## Time-safe split

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2014-01-01 to 2021-12-31 | 7,279 |
| Validation | 2022-01-01 to 2022-12-31 | 969 |
| Test | 2023-01-01 to 2026-06-02 | 3,491 |

## Candidate ranking on validation

| model                 |   accuracy |   macro_f1 |   log_loss |   multiclass_brier |   ranked_probability_score |
|:----------------------|-----------:|-----------:|-----------:|-------------------:|---------------------------:|
| random_forest_shallow |     0.5212 |     0.4917 |     0.9600 |             0.5702 |                     0.1284 |
| logistic_kbest_35     |     0.5046 |     0.4700 |     0.9924 |             0.5859 |                     0.1317 |
| logistic_l2_curated   |     0.4954 |     0.4755 |     0.9987 |             0.5907 |                     0.1326 |
| dummy_most_frequent   |     0.4974 |     0.2215 |    17.3585 |             1.0052 |                     0.2594 |

## Candidate ranking on test

| model                 |   accuracy |   macro_f1 |   log_loss |   multiclass_brier |   ranked_probability_score |
|:----------------------|-----------:|-----------:|-----------:|-------------------:|---------------------------:|
| random_forest_shallow |     0.5766 |     0.5342 |     0.8884 |             0.5244 |                     0.1151 |
| logistic_kbest_35     |     0.5755 |     0.5284 |     0.8959 |             0.5272 |                     0.1154 |
| logistic_l2_curated   |     0.5689 |     0.5308 |     0.8975 |             0.5291 |                     0.1159 |
| dummy_most_frequent   |     0.4718 |     0.2137 |    18.2439 |             1.0564 |                     0.2758 |

## Best selected model

`random_forest_shallow`

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.5212 | 0.5766 |
| Macro F1 | 0.4917 | 0.5342 |
| Log loss | 0.9600 | 0.8884 |
| Multiclass Brier | 0.5702 | 0.5244 |
| Ranked probability score | 0.1284 | 0.1151 |

## Family comparison: validation

| model_family                  | model                       |   accuracy |   macro_f1 |   log_loss |   multiclass_brier |   ranked_probability_score |
|:------------------------------|:----------------------------|-----------:|-----------:|-----------:|-------------------:|---------------------------:|
| model_v2_best                 | random_forest_shallow       |     0.5212 |     0.4917 |     0.9600 |             0.5702 |                     0.1284 |
| previous_baseline             | logistic_regression         |     0.5212 |     0.4961 |     0.9776 |             0.5788 |                     0.1301 |
| exp_h2h_logistic_all_features | logistic_regression_exp_h2h |     0.4788 |     0.4655 |     1.0121 |             0.6003 |                     0.1331 |

## Family comparison: test

| model_family                  | model                       |   accuracy |   macro_f1 |   log_loss |   multiclass_brier |   ranked_probability_score |
|:------------------------------|:----------------------------|-----------:|-----------:|-----------:|-------------------:|---------------------------:|
| previous_baseline             | logistic_regression         |     0.5712 |     0.5269 |     0.8844 |             0.5213 |                     0.1139 |
| model_v2_best                 | random_forest_shallow       |     0.5766 |     0.5342 |     0.8884 |             0.5244 |                     0.1151 |
| exp_h2h_logistic_all_features | logistic_regression_exp_h2h |     0.5486 |     0.5270 |     0.9202 |             0.5456 |                     0.1181 |

## Verdict

Compared with the first baseline, Model v2 test log loss changed from **0.8844** to **0.8884** (+0.0040), and test accuracy changed from **0.5712** to **0.5766** (+0.0054).

A lower log loss is better for probability prediction. A higher accuracy is better for hard W/D/L classification.

## Top model feature signals

| model                 | feature                                     |   importance |   abs_importance |
|:----------------------|:--------------------------------------------|-------------:|-----------------:|
| random_forest_shallow | elo_prob_home_win_proxy                     |       0.1517 |           0.1517 |
| random_forest_shallow | elo_diff_pre                                |       0.1230 |           0.1230 |
| random_forest_shallow | elo_diff                                    |       0.1204 |           0.1204 |
| random_forest_shallow | fifa_rank_diff_filled                       |       0.0772 |           0.0772 |
| random_forest_shallow | fifa_points_diff_filled                     |       0.0557 |           0.0557 |
| random_forest_shallow | exp_goal_diff_per_match_prior_diff_filled   |       0.0408 |           0.0408 |
| random_forest_shallow | exp_points_per_match_prior_diff_filled      |       0.0352 |           0.0352 |
| random_forest_shallow | h2h_goal_diff_per_match_prior_filled        |       0.0322 |           0.0322 |
| random_forest_shallow | h2h_goal_diff_per_match_last5_filled        |       0.0226 |           0.0226 |
| random_forest_shallow | avg_goal_diff_delta_last10                  |       0.0218 |           0.0218 |
| random_forest_shallow | h2h_home_team_points_per_match_prior_filled |       0.0209 |           0.0209 |
| random_forest_shallow | exp_major_matches_prior_diff                |       0.0203 |           0.0203 |
| random_forest_shallow | weighted_goal_diff_delta_last10             |       0.0201 |           0.0201 |
| random_forest_shallow | exp_total_matches_prior_diff                |       0.0163 |           0.0163 |
| random_forest_shallow | weighted_goal_diff_delta_last5              |       0.0159 |           0.0159 |
| random_forest_shallow | avg_goal_diff_delta_last5                   |       0.0147 |           0.0147 |
| random_forest_shallow | exp_high_importance_matches_prior_diff      |       0.0140 |           0.0140 |
| random_forest_shallow | h2h_goals_against_per_match_prior_filled    |       0.0140 |           0.0140 |
| random_forest_shallow | exp_world_cup_matches_prior_diff            |       0.0119 |           0.0119 |
| random_forest_shallow | exp_continental_matches_prior_diff          |       0.0110 |           0.0110 |
| random_forest_shallow | h2h_home_team_points_per_match_last5_filled |       0.0099 |           0.0099 |
| random_forest_shallow | exp_years_since_first_match_diff_filled     |       0.0099 |           0.0099 |
| random_forest_shallow | exp_qualifier_matches_prior_diff            |       0.0092 |           0.0092 |
| random_forest_shallow | weighted_form_points_diff_last10            |       0.0092 |           0.0092 |
| random_forest_shallow | h2h_goals_for_per_match_prior_filled        |       0.0090 |           0.0090 |

## Top numeric screening signals

| feature                                     |   f_score |   p_value |
|:--------------------------------------------|----------:|----------:|
| elo_prob_home_win_proxy                     | 1492.3942 |    0.0000 |
| elo_diff                                    | 1320.6941 |    0.0000 |
| elo_diff_pre                                | 1320.6941 |    0.0000 |
| fifa_rank_diff_filled                       |  737.8321 |    0.0000 |
| fifa_points_diff_filled                     |  710.1368 |    0.0000 |
| exp_points_per_match_prior_diff_filled      |  705.2814 |    0.0000 |
| exp_goal_diff_per_match_prior_diff_filled   |  588.5221 |    0.0000 |
| weighted_goal_diff_delta_last10             |  581.9013 |    0.0000 |
| avg_goal_diff_delta_last10                  |  577.3163 |    0.0000 |
| h2h_goal_diff_per_match_prior_filled        |  568.5996 |    0.0000 |
| h2h_goal_diff_per_match_last5_filled        |  554.4000 |    0.0000 |
| exp_total_matches_prior_diff                |  489.2799 |    0.0000 |
| avg_goal_diff_delta_last5                   |  478.0888 |    0.0000 |
| form_points_per_match_diff_last10           |  468.4013 |    0.0000 |
| weighted_goal_diff_delta_last5              |  467.6505 |    0.0000 |
| h2h_home_team_points_per_match_prior_filled |  446.1916 |    0.0000 |
| weighted_form_points_diff_last10            |  442.2868 |    0.0000 |
| exp_major_matches_prior_diff                |  442.2150 |    0.0000 |
| exp_high_importance_matches_prior_diff      |  442.2150 |    0.0000 |
| h2h_home_team_points_per_match_last5_filled |  423.9456 |    0.0000 |
| h2h_home_team_win_rate_last5_filled         |  411.5360 |    0.0000 |
| exp_world_cup_matches_prior_diff            |  394.1048 |    0.0000 |
| win_rate_diff_last10                        |  370.0719 |    0.0000 |
| form_points_per_match_diff_last5            |  335.6120 |    0.0000 |
| h2h_goals_against_per_match_prior_filled    |  311.8315 |    0.0000 |

## Output files

- `data/modeling/model_dataset_v2_curated.csv`
- `models/v2_model_metrics.csv`
- `models/v2_model_family_metrics_comparison.csv`
- `models/baseline_vs_v2_comparison.csv`
- `models/v2_validation_test_predictions.csv`
- `models/v2_test_predictions.csv`
- `models/v2_feature_importance.csv`
- `models/v2_numeric_feature_screening.csv`
- `models/v2_confusion_matrix_validation.csv`
- `models/v2_confusion_matrix_test.csv`
- `models/v2_classification_reports.json`
- `models/v2_model_best.joblib`
- `models/v2_model_best_train_plus_validation.joblib`
- `model_v2_metadata.json`
