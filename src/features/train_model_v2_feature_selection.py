#!/usr/bin/env python3
"""Model v2: feature selection + stronger classifiers for international W/D/L prediction."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss as skl_log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "matches_with_elo_fifa_form_confed_exp_h2h.csv"
BASELINE_METRICS = ROOT / "models" / "baseline_metrics.csv"
EXP_H2H_METRICS = ROOT / "models" / "exp_h2h_metrics.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (MODELING_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "result"
CLASSES = ["H", "D", "A"]
ID_COLUMNS = ["match_id", "date", "home_team", "away_team", "home_score", "away_score", "result", "tournament", "city", "country", "neutral"]
CATEGORICAL_FEATURES = ["home_confederation", "away_confederation", "confederation_pair", "host_confederation"]

# Curated compact signals: keep directional differences and rates, avoid many duplicate raw counts.
CURATED_NUMERIC = [
    # Core strength
    "elo_diff", "elo_diff_pre", "elo_prob_home_win_proxy",
    "fifa_rank_diff_filled", "fifa_points_diff_filled",
    # Match context
    "neutral", "tournament_importance", "is_friendly", "is_qualifier", "is_world_cup", "is_continental",
    # Recent form
    "form_points_per_match_diff_last5", "avg_goal_diff_delta_last5", "win_rate_diff_last5",
    "weighted_form_points_diff_last5", "weighted_goal_diff_delta_last5",
    "form_points_per_match_diff_last10", "avg_goal_diff_delta_last10", "win_rate_diff_last10",
    "weighted_form_points_diff_last10", "weighted_goal_diff_delta_last10",
    "home_days_since_last_match_last5", "away_days_since_last_match_last5",
    # Confederation
    "same_confederation", "home_confed_matches_host_confed", "away_confed_matches_host_confed",
    "confed_points_per_match_diff_prior_filled", "confed_goal_diff_per_match_diff_prior_filled",
    "confed_avg_elo_diff_prior_filled", "confed_inter_points_per_match_diff_prior_filled",
    "confed_inter_goal_diff_per_match_diff_prior_filled",
    # Tournament experience - compact differences/rates only
    "exp_total_matches_prior_diff", "exp_same_tournament_matches_prior_diff", "exp_same_family_matches_prior_diff",
    "exp_world_cup_matches_prior_diff", "exp_continental_matches_prior_diff", "exp_qualifier_matches_prior_diff",
    "exp_nations_league_matches_prior_diff", "exp_major_matches_prior_diff",
    "exp_major_points_per_match_prior_diff_filled", "exp_major_goal_diff_per_match_prior_diff_filled",
    "exp_points_per_match_prior_diff_filled", "exp_goal_diff_per_match_prior_diff_filled",
    "exp_avg_tournament_importance_prior_diff_filled", "exp_high_importance_matches_prior_diff",
    "exp_years_since_first_match_diff_filled", "exp_major_recency_advantage_filled",
    "both_teams_have_major_tournament_history",
    # H2H - compact signals only
    "h2h_matches_prior", "h2h_home_team_points_per_match_prior_filled",
    "h2h_goal_diff_per_match_prior_filled", "h2h_goals_for_per_match_prior_filled", "h2h_goals_against_per_match_prior_filled",
    "h2h_matches_last5", "h2h_home_team_points_per_match_last5_filled", "h2h_goal_diff_per_match_last5_filled",
    "h2h_home_team_win_rate_last5_filled", "h2h_days_since_last_meeting_filled",
    "h2h_same_tournament_matches_prior", "h2h_same_family_matches_prior", "has_h2h_history", "has_h2h_last5",
]


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros_like(proba, dtype=float)
    for i, y in enumerate(y_true):
        y_onehot[i, idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    vals = []
    for y, p in zip(y_true, proba):
        obs = np.zeros(len(classes)); obs[idx[y]] = 1.0
        vals.append(np.mean((np.cumsum(p) - np.cumsum(obs)) ** 2))
    return float(np.mean(vals))


def align_proba(model, X: pd.DataFrame, desired_classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(X)
    model_classes = list(model.classes_)
    aligned = np.zeros((raw.shape[0], len(desired_classes)), dtype=float)
    for j, c in enumerate(desired_classes):
        if c in model_classes:
            aligned[:, j] = raw[:, model_classes.index(c)]
    sums = aligned.sum(axis=1)
    aligned[sums == 0] = 1.0 / len(desired_classes)
    return aligned / aligned.sum(axis=1, keepdims=True)


def safe_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(skl_log_loss(y_true, proba, labels=CLASSES))


def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    pred = model.predict(X)
    proba = align_proba(model, X, CLASSES)
    return {
        "model": name,
        "split": split,
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "log_loss": safe_log_loss(y.to_numpy(), proba),
        "multiclass_brier": multiclass_brier(y.to_numpy(), proba, CLASSES),
        "ranked_probability_score": ranked_probability_score(y.to_numpy(), proba, CLASSES),
    }


def main():
    df = pd.read_csv(INPUT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df[TARGET].isin(CLASSES)].copy()

    bool_cols = [c for c in CURATED_NUMERIC if c in df.columns and (df[c].dtype == bool or c.startswith("is_") or c in {
        "neutral", "same_confederation", "home_confed_matches_host_confed", "away_confed_matches_host_confed",
        "both_teams_have_major_tournament_history", "has_h2h_history", "has_h2h_last5"
    })]
    for c in bool_cols:
        df[c] = df[c].astype(int)

    numeric_features = [c for c in CURATED_NUMERIC if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    missing = sorted(set(CURATED_NUMERIC) - set(numeric_features))

    selected_cols = list(dict.fromkeys([c for c in ID_COLUMNS if c in df.columns] + numeric_features + categorical_features))
    model_df = df[selected_cols].copy()
    model_df["split"] = np.select(
        [model_df["date"] < pd.Timestamp("2022-01-01"),
         (model_df["date"] >= pd.Timestamp("2022-01-01")) & (model_df["date"] < pd.Timestamp("2023-01-01")),
         model_df["date"] >= pd.Timestamp("2023-01-01")],
        ["train", "validation", "test"], default="unused")
    model_df.to_csv(MODELING_DIR / "model_dataset_v2_curated.csv", index=False)

    train_df = model_df[model_df.split == "train"].copy()
    val_df = model_df[model_df.split == "validation"].copy()
    test_df = model_df[model_df.split == "test"].copy()
    X_train, y_train = train_df[numeric_features + categorical_features], train_df[TARGET]
    X_val, y_val = val_df[numeric_features + categorical_features], val_df[TARGET]
    X_test, y_test = test_df[numeric_features + categorical_features], test_df[TARGET]

    # Preprocessors
    linear_preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical_features),
    ], remainder="drop")

    tree_ohe_preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical_features),
    ], remainder="drop")

    tree_ordinal_preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_features),
    ], remainder="drop")

    # Candidate models. Selection is based on validation log loss only.
    candidates = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_l2_curated": Pipeline([
            ("preprocess", linear_preprocess),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", C=0.5, class_weight="balanced")),
        ]),
        "logistic_l1_select": Pipeline([
            ("preprocess", linear_preprocess),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("model", LogisticRegression(max_iter=3000, solver="saga", penalty="l1", C=0.08, class_weight="balanced", n_jobs=-1)),
        ]),
        "logistic_kbest_35": Pipeline([
            ("preprocess", linear_preprocess),
            ("select", SelectKBest(score_func=f_classif, k=35)),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", C=0.7, class_weight="balanced")),
        ]),
        "random_forest_curated": Pipeline([
            ("preprocess", tree_ohe_preprocess),
            ("model", RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=25, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ]),
        "extra_trees_curated": Pipeline([
            ("preprocess", tree_ohe_preprocess),
            ("model", ExtraTreesClassifier(n_estimators=600, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient_boosting_curated": Pipeline([
            ("preprocess", tree_ordinal_preprocess),
            ("model", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=12, l2_regularization=0.35, min_samples_leaf=45, random_state=42)),
        ]),
    }

    metrics = []
    for name, model in candidates.items():
        model_path = MODELS_DIR / f"v2_model_{name}.joblib"
        if model_path.exists():
            print(f"Loading existing {name}...")
            model = joblib.load(model_path)
            candidates[name] = model
        else:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            candidates[name] = model
            joblib.dump(model, model_path)
        for split_name, X, y in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
            metrics.append(evaluate(name, model, X, y, split_name))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(MODELS_DIR / "v2_model_metrics.csv", index=False)

    val_metrics = metrics_df[metrics_df.split == "validation"].copy()
    val_metrics = val_metrics[val_metrics.model != "dummy_most_frequent"]
    best_name = val_metrics.sort_values(["log_loss", "multiclass_brier", "macro_f1"], ascending=[True, True, False]).iloc[0]["model"]
    best_model = candidates[best_name]
    joblib.dump(best_model, MODELS_DIR / "v2_model_best.joblib")

    # Optional deployment model: refit the chosen model on train+validation, held-out test untouched for evaluation already above.
    trainval_df = model_df[model_df.split.isin(["train", "validation"])].copy()
    X_trainval = trainval_df[numeric_features + categorical_features]
    y_trainval = trainval_df[TARGET]
    # recreate clone via joblib roundtrip avoids importing clone? use sklearn clone
    from sklearn.base import clone
    deployment_model = clone(candidates[best_name])
    deployment_model.fit(X_trainval, y_trainval)
    joblib.dump(deployment_model, MODELS_DIR / "v2_model_best_train_plus_validation.joblib")

    # Predictions for validation/test from the fair selected model trained on training split only.
    pred_parts = []
    for split_name, split_df, X, y in [("validation", val_df, X_val, y_val), ("test", test_df, X_test, y_test)]:
        pred = best_model.predict(X)
        proba = align_proba(best_model, X, CLASSES)
        tmp = split_df[["match_id", "date", "home_team", "away_team", "home_score", "away_score", "result", "tournament", "neutral", "split"]].copy()
        tmp["model"] = best_name
        tmp["predicted_result"] = pred
        tmp["prob_home_win"] = proba[:, 0]
        tmp["prob_draw"] = proba[:, 1]
        tmp["prob_away_win"] = proba[:, 2]
        tmp["confidence"] = proba.max(axis=1)
        tmp["correct"] = (tmp.predicted_result == tmp.result).astype(int)
        pred_parts.append(tmp)
    preds = pd.concat(pred_parts, ignore_index=True)
    preds.to_csv(MODELS_DIR / "v2_validation_test_predictions.csv", index=False)
    preds[preds.split == "test"].to_csv(MODELS_DIR / "v2_test_predictions.csv", index=False)

    # Classification diagnostics for best model.
    reports = {}
    for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        pred = best_model.predict(X)
        pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES), index=[f"actual_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(MODELS_DIR / f"v2_confusion_matrix_{split_name}.csv")
        reports[split_name] = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    (MODELS_DIR / "v2_classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    # Feature importance for best model.
    feature_importance_path = MODELS_DIR / "v2_feature_importance.csv"
    imp_df = pd.DataFrame()
    try:
        pre = best_model.named_steps.get("preprocess")
        feature_names = pre.get_feature_names_out()
        if "select" in best_model.named_steps:
            selector = best_model.named_steps["select"]
            if hasattr(selector, "get_support"):
                support = selector.get_support()
                feature_names = feature_names[support]
        estimator = best_model.named_steps.get("model")
        if hasattr(estimator, "coef_"):
            rows = []
            for class_idx, cls in enumerate(estimator.classes_):
                for feat, val in zip(feature_names, estimator.coef_[class_idx]):
                    rows.append({"model": best_name, "class": cls, "feature": feat, "importance": float(val), "abs_importance": abs(float(val))})
            imp_df = pd.DataFrame(rows).sort_values("abs_importance", ascending=False)
        elif hasattr(estimator, "feature_importances_"):
            imp_df = pd.DataFrame({"model": best_name, "feature": feature_names, "importance": estimator.feature_importances_})
            imp_df["abs_importance"] = imp_df["importance"].abs()
            imp_df = imp_df.sort_values("abs_importance", ascending=False)
    except Exception as e:
        imp_df = pd.DataFrame([{"error": repr(e)}])
    imp_df.to_csv(feature_importance_path, index=False)

    # Compare against previous baseline and augmented model.
    comparisons = []
    previous_sources = []
    if BASELINE_METRICS.exists():
        b = pd.read_csv(BASELINE_METRICS)
        b = b[b.model.eq("logistic_regression")].copy()
        b["model_family"] = "previous_baseline"
        previous_sources.append(b)
    if EXP_H2H_METRICS.exists():
        e = pd.read_csv(EXP_H2H_METRICS)
        e = e[e.model.eq("logistic_regression_exp_h2h")].copy()
        e["model_family"] = "exp_h2h_logistic_all_features"
        previous_sources.append(e)
    v2 = metrics_df[metrics_df.model.eq(best_name)].copy()
    v2["model_family"] = "model_v2_best"
    previous_sources.append(v2)
    all_comp = pd.concat(previous_sources, ignore_index=True)
    all_comp.to_csv(MODELS_DIR / "v2_model_family_metrics_comparison.csv", index=False)

    # Per-metric delta from previous baseline.
    baseline = all_comp[all_comp.model_family.eq("previous_baseline")]
    v2_rows = all_comp[all_comp.model_family.eq("model_v2_best")]
    for split in ["validation", "test"]:
        if not baseline[baseline.split.eq(split)].empty and not v2_rows[v2_rows.split.eq(split)].empty:
            br = baseline[baseline.split.eq(split)].iloc[0]
            nr = v2_rows[v2_rows.split.eq(split)].iloc[0]
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]:
                direction = "higher_better" if metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"] else "lower_better"
                change = float(nr[metric] - br[metric])
                improved = change > 0 if direction == "higher_better" else change < 0
                comparisons.append({"split": split, "metric": metric, "baseline": float(br[metric]), "model_v2": float(nr[metric]), "change": change, "direction": direction, "improved": bool(improved)})
    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(MODELS_DIR / "baseline_vs_v2_comparison.csv", index=False)

    # Compact feature screening table for report.
    screen_rows = []
    # F scores on numeric features only for readable screening.
    X_num = train_df[numeric_features].copy()
    X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=numeric_features)
    f_vals, p_vals = f_classif(X_num, y_train)
    for feat, score, p in zip(numeric_features, f_vals, p_vals):
        screen_rows.append({"feature": feat, "f_score": float(score) if np.isfinite(score) else None, "p_value": float(p) if np.isfinite(p) else None})
    screen_df = pd.DataFrame(screen_rows).sort_values("f_score", ascending=False)
    screen_df.to_csv(MODELS_DIR / "v2_numeric_feature_screening.csv", index=False)

    # Metadata/report.
    best_val = metrics_df[(metrics_df.model.eq(best_name)) & (metrics_df.split.eq("validation"))].iloc[0]
    best_test = metrics_df[(metrics_df.model.eq(best_name)) & (metrics_df.split.eq("test"))].iloc[0]
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT.relative_to(ROOT)),
        "modeling_dataset": "data/modeling/model_dataset_v2_curated.csv",
        "date_splits": {"train": "2014-01-01 to 2021-12-31", "validation": "2022-01-01 to 2022-12-31", "test": "2023-01-01 to 2026-06-02"},
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "test": int(len(test_df)), "total": int(len(model_df))},
        "candidate_models": list(candidates.keys()),
        "selection_metric": "validation log_loss, then validation multiclass_brier, then validation macro_f1",
        "best_model": best_name,
        "numeric_features_used": numeric_features,
        "categorical_features_used": categorical_features,
        "missing_curated_features": missing,
        "best_validation": {k: float(best_val[k]) for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]},
        "best_test": {k: float(best_test[k]) for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]},
        "fair_model_note": "v2_model_best.joblib is trained on train split only and evaluated on validation/test. v2_model_best_train_plus_validation.joblib is for later deployment use after model selection.",
        "output_files": {
            "metrics": "models/v2_model_metrics.csv",
            "family_comparison": "models/v2_model_family_metrics_comparison.csv",
            "baseline_vs_v2": "models/baseline_vs_v2_comparison.csv",
            "validation_test_predictions": "models/v2_validation_test_predictions.csv",
            "test_predictions": "models/v2_test_predictions.csv",
            "feature_importance": "models/v2_feature_importance.csv",
            "feature_screening": "models/v2_numeric_feature_screening.csv",
            "best_model": "models/v2_model_best.joblib",
            "deployment_model": "models/v2_model_best_train_plus_validation.joblib",
        },
    }
    (ROOT / "model_v2_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def md_metric_table(dfm: pd.DataFrame, split: str) -> str:
        keep = dfm[dfm.split.eq(split)][["model_family", "model", "accuracy", "macro_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]].copy()
        keep = keep.sort_values("log_loss")
        return keep.to_markdown(index=False, floatfmt=".4f")

    candidate_val = metrics_df[metrics_df.split.eq("validation")][["model", "accuracy", "macro_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]].sort_values("log_loss")
    candidate_test = metrics_df[metrics_df.split.eq("test")][["model", "accuracy", "macro_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]].sort_values("log_loss")
    top_features_md = ""
    if not imp_df.empty and "feature" in imp_df.columns:
        top_features_md = imp_df.head(25).to_markdown(index=False, floatfmt=".4f")
    top_screen_md = screen_df.head(25).to_markdown(index=False, floatfmt=".4f")

    verdict_line = ""
    if not comp_df.empty:
        test_log = comp_df[(comp_df.split.eq("test")) & (comp_df.metric.eq("log_loss"))].iloc[0]
        test_acc = comp_df[(comp_df.split.eq("test")) & (comp_df.metric.eq("accuracy"))].iloc[0]
        verdict_line = f"Compared with the first baseline, Model v2 test log loss changed from **{test_log.baseline:.4f}** to **{test_log.model_v2:.4f}** ({test_log.change:+.4f}), and test accuracy changed from **{test_acc.baseline:.4f}** to **{test_acc.model_v2:.4f}** ({test_acc.change:+.4f})."

    report = f"""# Model v2: Feature Selection + Stronger Classifiers

## Goal
Build a second modelling milestone using a compact curated feature set plus stronger classifiers. The selection criterion is validation **log loss**, because probability quality is more important than only hard-result accuracy.

Input dataset: `data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv`

## Time-safe split

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2014-01-01 to 2021-12-31 | {len(train_df):,} |
| Validation | 2022-01-01 to 2022-12-31 | {len(val_df):,} |
| Test | 2023-01-01 to 2026-06-02 | {len(test_df):,} |

No random split was used. The best model was selected using validation log loss only.

## Candidate model ranking on validation

{candidate_val.to_markdown(index=False, floatfmt='.4f')}

## Candidate model ranking on test

{candidate_test.to_markdown(index=False, floatfmt='.4f')}

## Best selected model

`{best_name}`

Validation metrics:

| Metric | Value |
|---|---:|
| Accuracy | {best_val['accuracy']:.4f} |
| Macro F1 | {best_val['macro_f1']:.4f} |
| Log loss | {best_val['log_loss']:.4f} |
| Multiclass Brier | {best_val['multiclass_brier']:.4f} |
| Ranked probability score | {best_val['ranked_probability_score']:.4f} |

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | {best_test['accuracy']:.4f} |
| Macro F1 | {best_test['macro_f1']:.4f} |
| Log loss | {best_test['log_loss']:.4f} |
| Multiclass Brier | {best_test['multiclass_brier']:.4f} |
| Ranked probability score | {best_test['ranked_probability_score']:.4f} |

## Family comparison: validation

{md_metric_table(all_comp, 'validation')}

## Family comparison: test

{md_metric_table(all_comp, 'test')}

## Verdict

{verdict_line}

A lower log loss means better probability estimates. A higher accuracy means better hard W/D/L classifications.

## Top numeric feature-screening signals

{top_screen_md}

## Best-model feature importance / coefficients

{top_features_md if top_features_md else 'Feature importance was not available for this estimator.'}

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
"""
    (REPORTS_DIR / "model_v2_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "best_model": best_name,
        "rows": meta["rows"],
        "validation": meta["best_validation"],
        "test": meta["best_test"],
        "baseline_vs_v2_test": comp_df[comp_df.split.eq("test")].to_dict(orient="records"),
        "candidate_validation": candidate_val.to_dict(orient="records"),
        "report": "reports/model_v2_report.md",
    }, indent=2))


if __name__ == "__main__":
    main()
