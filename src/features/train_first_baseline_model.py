#!/usr/bin/env python3
"""Train first time-safe baseline model for international match result prediction.

Input:  data/processed/matches_with_elo_fifa_form_confed.csv
Output: data/modeling/model_dataset_baseline.csv
        models/baseline_model_logistic_regression.joblib
        models/baseline_model_gradient_boosting.joblib
        models/baseline_metrics.csv
        models/baseline_test_predictions.csv
        models/baseline_feature_importance.csv
        baseline_model_metadata.json
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "matches_with_elo_fifa_form_confed.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (MODELING_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "result"
CLASSES = ["H", "D", "A"]

NUMERIC_FEATURES = [
    # Core team-strength features
    "elo_diff",
    "elo_diff_pre",
    "elo_prob_home_win_proxy",
    "fifa_rank_diff_filled",
    "fifa_points_diff_filled",
    # Recent-form features
    "form_points_per_match_diff_last5",
    "avg_goal_diff_delta_last5",
    "win_rate_diff_last5",
    "weighted_form_points_diff_last5",
    "weighted_goal_diff_delta_last5",
    "form_points_per_match_diff_last10",
    "avg_goal_diff_delta_last10",
    "win_rate_diff_last10",
    "weighted_form_points_diff_last10",
    "weighted_goal_diff_delta_last10",
    "home_days_since_last_match_last5",
    "away_days_since_last_match_last5",
    # Tournament/venue features
    "neutral",
    "tournament_importance",
    "is_friendly",
    "is_qualifier",
    "is_world_cup",
    "is_continental",
    # Confederation features
    "same_confederation",
    "home_confed_matches_host_confed",
    "away_confed_matches_host_confed",
    "confed_points_per_match_diff_prior_filled",
    "confed_goal_diff_per_match_diff_prior_filled",
    "confed_avg_elo_diff_prior_filled",
    "confed_inter_points_per_match_diff_prior_filled",
    "confed_inter_goal_diff_per_match_diff_prior_filled",
]

CATEGORICAL_FEATURES = [
    "home_confederation",
    "away_confederation",
    "confederation_pair",
    "host_confederation",
]

ID_COLUMNS = [
    "match_id", "date", "home_team", "away_team", "home_score", "away_score", "result",
    "tournament", "city", "country", "neutral",
]


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros_like(proba, dtype=float)
    for row_i, y in enumerate(y_true):
        y_onehot[row_i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    # Ordered as home win, draw, away win. Lower is better.
    class_to_idx = {c: i for i, c in enumerate(classes)}
    scores = []
    for y, p in zip(y_true, proba):
        obs = np.zeros(len(classes))
        obs[class_to_idx[y]] = 1.0
        scores.append(np.mean((np.cumsum(p) - np.cumsum(obs)) ** 2))
    return float(np.mean(scores))


def align_proba(model, X: pd.DataFrame, desired_classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(X)
    model_classes = list(model.classes_)
    aligned = np.zeros((raw.shape[0], len(desired_classes)), dtype=float)
    for j, c in enumerate(desired_classes):
        if c in model_classes:
            aligned[:, j] = raw[:, model_classes.index(c)]
    row_sums = aligned.sum(axis=1)
    # Should not happen if all classes seen in training, but guard against it.
    aligned[row_sums == 0, :] = 1.0 / len(desired_classes)
    row_sums = aligned.sum(axis=1)
    aligned = aligned / row_sums[:, None]
    return aligned


def manual_log_loss(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    eps = 1e-15
    p = np.clip(proba, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx[y] for y in y_true])
    return float(-np.mean(np.log(p[np.arange(len(y_true)), idx])))


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, split: str) -> dict:
    y_pred = model.predict(X)
    proba = align_proba(model, X, CLASSES)
    return {
        "model": name,
        "split": split,
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y, y_pred, average="weighted")),
        "log_loss": manual_log_loss(y.to_numpy(), proba, CLASSES),
        "multiclass_brier": multiclass_brier(y.to_numpy(), proba, CLASSES),
        "ranked_probability_score": ranked_probability_score(y.to_numpy(), proba, CLASSES),
    }


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df[TARGET].isin(CLASSES)].copy()

    # Ensure booleans are numeric-friendly and keep only selected modelling columns.
    for col in ["neutral", "is_friendly", "is_qualifier", "is_world_cup", "is_continental", "same_confederation", "home_confed_matches_host_confed", "away_confed_matches_host_confed"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    selected_cols = list(dict.fromkeys([c for c in ID_COLUMNS if c in df.columns] + NUMERIC_FEATURES + CATEGORICAL_FEATURES))
    model_df = df[selected_cols].copy()
    model_df["split"] = np.select(
        [
            model_df["date"] < pd.Timestamp("2022-01-01"),
            (model_df["date"] >= pd.Timestamp("2022-01-01")) & (model_df["date"] < pd.Timestamp("2023-01-01")),
            model_df["date"] >= pd.Timestamp("2023-01-01"),
        ],
        ["train", "validation", "test"],
        default="unused",
    )

    modeling_path = MODELING_DIR / "model_dataset_baseline.csv"
    model_df.to_csv(modeling_path, index=False)

    train_df = model_df[model_df["split"] == "train"].copy()
    val_df = model_df[model_df["split"] == "validation"].copy()
    test_df = model_df[model_df["split"] == "test"].copy()

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]),
    }

    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        for split_name, X, y in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
            metrics.append(evaluate_model(name, model, X, y, split_name))
        joblib.dump(model, MODELS_DIR / f"baseline_model_{name}.joblib")

    metrics_df = pd.DataFrame(metrics).sort_values(["split", "log_loss", "accuracy"])
    metrics_path = MODELS_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Select best model by validation log loss among non-dummy models.
    val_metrics = metrics_df[(metrics_df["split"] == "validation") & (metrics_df["model"] != "dummy_most_frequent")]
    best_model_name = val_metrics.sort_values("log_loss").iloc[0]["model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "baseline_model_best.joblib")

    # Predictions for validation + test.
    pred_parts = []
    for split_name, split_df, X, y in [("validation", val_df, X_val, y_val), ("test", test_df, X_test, y_test)]:
        pred = best_model.predict(X)
        proba = align_proba(best_model, X, CLASSES)
        tmp = split_df[["match_id", "date", "home_team", "away_team", "home_score", "away_score", "result", "tournament", "neutral", "split"]].copy()
        tmp["predicted_result"] = pred
        tmp["prob_home_win"] = proba[:, 0]
        tmp["prob_draw"] = proba[:, 1]
        tmp["prob_away_win"] = proba[:, 2]
        tmp["confidence"] = proba.max(axis=1)
        tmp["correct"] = (tmp["predicted_result"] == tmp["result"]).astype(int)
        pred_parts.append(tmp)
    predictions = pd.concat(pred_parts, ignore_index=True)
    predictions_path = MODELS_DIR / "baseline_validation_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # Separate test predictions for convenience.
    test_predictions_path = MODELS_DIR / "baseline_test_predictions.csv"
    predictions[predictions["split"] == "test"].to_csv(test_predictions_path, index=False)

    # Confusion matrices and classification reports for best model.
    reports = {}
    for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = best_model.predict(X)
        cm = pd.DataFrame(confusion_matrix(y, y_pred, labels=CLASSES), index=[f"actual_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
        cm.to_csv(MODELS_DIR / f"baseline_confusion_matrix_{split_name}.csv")
        reports[split_name] = classification_report(y, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    with open(MODELS_DIR / "baseline_classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    # Feature importance / coefficients.
    importance_records = []
    if isinstance(best_model, Pipeline) and best_model_name == "logistic_regression":
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        coefs = best_model.named_steps["model"].coef_
        for class_idx, cls in enumerate(best_model.named_steps["model"].classes_):
            for feat, val in zip(feature_names, coefs[class_idx]):
                importance_records.append({"model": best_model_name, "class": cls, "feature": feat, "importance": float(val), "abs_importance": float(abs(val))})
    elif isinstance(best_model, Pipeline) and hasattr(best_model.named_steps["model"], "feature_importances_"):
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        imps = best_model.named_steps["model"].feature_importances_
        for feat, val in zip(feature_names, imps):
            importance_records.append({"model": best_model_name, "class": "all", "feature": feat, "importance": float(val), "abs_importance": float(abs(val))})
    else:
        # For HistGradientBoosting no native importances; compute simple univariate correlations for numeric columns as lightweight proxy.
        y_code = y_train.map({"H": 1, "D": 0, "A": -1}).astype(float)
        for col in NUMERIC_FEATURES:
            series = pd.to_numeric(X_train[col], errors="coerce").fillna(pd.to_numeric(X_train[col], errors="coerce").median())
            corr = float(np.corrcoef(series, y_code)[0, 1]) if series.std() > 0 else 0.0
            importance_records.append({"model": best_model_name, "class": "ordinal_proxy_H_to_A", "feature": col, "importance": corr, "abs_importance": abs(corr)})

    importance_df = pd.DataFrame(importance_records).sort_values("abs_importance", ascending=False)
    importance_path = MODELS_DIR / "baseline_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # Metadata and markdown report.
    summary_metrics = metrics_df.pivot_table(index="model", columns="split", values=["accuracy", "log_loss", "macro_f1", "multiclass_brier"], aggfunc="first")
    class_counts = model_df.groupby(["split", TARGET]).size().unstack(fill_value=0).reindex(index=["train", "validation", "test"])
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT.relative_to(ROOT)),
        "modeling_dataset": str(modeling_path.relative_to(ROOT)),
        "target": TARGET,
        "class_order": CLASSES,
        "date_splits": {
            "train": "2014-01-01 to 2021-12-31",
            "validation": "2022-01-01 to 2022-12-31",
            "test": "2023-01-01 to 2026-06-02",
        },
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "test": int(len(test_df)), "total": int(len(model_df))},
        "class_counts": {idx: {col: int(val) for col, val in row.items()} for idx, row in class_counts.iterrows()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "models_trained": list(models.keys()),
        "best_model_by_validation_log_loss": str(best_model_name),
        "output_files": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "predictions_validation_test": str(predictions_path.relative_to(ROOT)),
            "predictions_test": str(test_predictions_path.relative_to(ROOT)),
            "feature_importance": str(importance_path.relative_to(ROOT)),
            "best_model": str((MODELS_DIR / "baseline_model_best.joblib").relative_to(ROOT)),
        },
    }
    metadata_path = ROOT / "baseline_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    best_test = metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == "test")].iloc[0]
    best_val = metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == "validation")].iloc[0]
    dummy_test = metrics_df[(metrics_df["model"] == "dummy_most_frequent") & (metrics_df["split"] == "test")].iloc[0]

    report = f"""# First Baseline Model Report

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
| Train | 2014-01-01 to 2021-12-31 | {len(train_df):,} |
| Validation | 2022-01-01 to 2022-12-31 | {len(val_df):,} |
| Test | 2023-01-01 to 2026-06-02 | {len(test_df):,} |

No random split was used. This protects against future-data leakage.

## Models trained

- Dummy most-frequent baseline
- Multinomial logistic regression

The selected model is the model with the best validation log loss:

**{best_model_name}**

## Best model performance

| Split | Accuracy | Macro F1 | Log loss | Multiclass Brier |
|---|---:|---:|---:|---:|
| Validation | {best_val['accuracy']:.4f} | {best_val['macro_f1']:.4f} | {best_val['log_loss']:.4f} | {best_val['multiclass_brier']:.4f} |
| Test | {best_test['accuracy']:.4f} | {best_test['macro_f1']:.4f} | {best_test['log_loss']:.4f} | {best_test['multiclass_brier']:.4f} |

Dummy most-frequent test accuracy: **{dummy_test['accuracy']:.4f}**

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
"""
    report_path = REPORTS_DIR / "baseline_model_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "best_model": best_model_name,
        "rows": metadata["rows"],
        "validation": {k: float(best_val[k]) for k in ["accuracy", "macro_f1", "log_loss", "multiclass_brier"]},
        "test": {k: float(best_test[k]) for k in ["accuracy", "macro_f1", "log_loss", "multiclass_brier"]},
        "outputs": metadata["output_files"],
    }, indent=2))

if __name__ == "__main__":
    main()
