#!/usr/bin/env python3
"""Train and compare baseline+experience+H2H model against previous baseline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "matches_with_elo_fifa_form_confed_exp_h2h.csv"
BASELINE_METRICS = ROOT / "models" / "baseline_metrics.csv"
MODELING_DIR = ROOT / "data" / "modeling"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (MODELING_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "result"
CLASSES = ["H", "D", "A"]

BASE_NUMERIC_FEATURES = [
    "elo_diff", "elo_diff_pre", "elo_prob_home_win_proxy",
    "fifa_rank_diff_filled", "fifa_points_diff_filled",
    "form_points_per_match_diff_last5", "avg_goal_diff_delta_last5", "win_rate_diff_last5",
    "weighted_form_points_diff_last5", "weighted_goal_diff_delta_last5",
    "form_points_per_match_diff_last10", "avg_goal_diff_delta_last10", "win_rate_diff_last10",
    "weighted_form_points_diff_last10", "weighted_goal_diff_delta_last10",
    "home_days_since_last_match_last5", "away_days_since_last_match_last5",
    "neutral", "tournament_importance", "is_friendly", "is_qualifier", "is_world_cup", "is_continental",
    "same_confederation", "home_confed_matches_host_confed", "away_confed_matches_host_confed",
    "confed_points_per_match_diff_prior_filled", "confed_goal_diff_per_match_diff_prior_filled",
    "confed_avg_elo_diff_prior_filled", "confed_inter_points_per_match_diff_prior_filled",
    "confed_inter_goal_diff_per_match_diff_prior_filled",
]
NEW_NUMERIC_FEATURES = [
    # Tournament experience: counts and differences
    "home_exp_total_matches_prior", "away_exp_total_matches_prior", "exp_total_matches_prior_diff",
    "home_exp_same_tournament_matches_prior", "away_exp_same_tournament_matches_prior", "exp_same_tournament_matches_prior_diff",
    "home_exp_same_family_matches_prior", "away_exp_same_family_matches_prior", "exp_same_family_matches_prior_diff",
    "home_exp_world_cup_matches_prior", "away_exp_world_cup_matches_prior", "exp_world_cup_matches_prior_diff",
    "home_exp_continental_matches_prior", "away_exp_continental_matches_prior", "exp_continental_matches_prior_diff",
    "home_exp_qualifier_matches_prior", "away_exp_qualifier_matches_prior", "exp_qualifier_matches_prior_diff",
    "home_exp_nations_league_matches_prior", "away_exp_nations_league_matches_prior", "exp_nations_league_matches_prior_diff",
    "home_exp_major_matches_prior", "away_exp_major_matches_prior", "exp_major_matches_prior_diff",
    "home_exp_major_points_per_match_prior_filled", "away_exp_major_points_per_match_prior_filled", "exp_major_points_per_match_prior_diff_filled",
    "home_exp_major_goal_diff_per_match_prior_filled", "away_exp_major_goal_diff_per_match_prior_filled", "exp_major_goal_diff_per_match_prior_diff_filled",
    "home_exp_points_per_match_prior_filled", "away_exp_points_per_match_prior_filled", "exp_points_per_match_prior_diff_filled",
    "home_exp_goal_diff_per_match_prior_filled", "away_exp_goal_diff_per_match_prior_filled", "exp_goal_diff_per_match_prior_diff_filled",
    "home_exp_avg_tournament_importance_prior_filled", "away_exp_avg_tournament_importance_prior_filled", "exp_avg_tournament_importance_prior_diff_filled",
    "home_exp_high_importance_matches_prior", "away_exp_high_importance_matches_prior", "exp_high_importance_matches_prior_diff",
    "home_exp_years_since_first_match_filled", "away_exp_years_since_first_match_filled", "exp_years_since_first_match_diff_filled",
    "exp_major_recency_advantage_filled", "both_teams_have_major_tournament_history",
    # Head-to-head
    "h2h_matches_prior", "h2h_home_team_wins_prior", "h2h_draws_prior", "h2h_home_team_losses_prior",
    "h2h_home_team_points_per_match_prior_filled", "h2h_goal_diff_per_match_prior_filled",
    "h2h_goals_for_per_match_prior_filled", "h2h_goals_against_per_match_prior_filled",
    "h2h_matches_last5", "h2h_home_team_points_per_match_last5_filled", "h2h_goal_diff_per_match_last5_filled",
    "h2h_home_team_win_rate_last5_filled", "h2h_days_since_last_meeting_filled",
    "h2h_same_tournament_matches_prior", "h2h_same_family_matches_prior", "has_h2h_history", "has_h2h_last5",
]
CATEGORICAL_FEATURES = ["home_confederation", "away_confederation", "confederation_pair", "host_confederation"]
ID_COLUMNS = ["match_id", "date", "home_team", "away_team", "home_score", "away_score", "result", "tournament", "city", "country", "neutral"]


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros_like(proba, dtype=float)
    for i, y in enumerate(y_true):
        y_onehot[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
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
    sums = aligned.sum(axis=1)
    aligned[sums == 0] = 1.0 / len(desired_classes)
    return aligned / aligned.sum(axis=1, keepdims=True)


def manual_log_loss(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    eps = 1e-15
    p = np.clip(proba, eps, 1-eps)
    p = p / p.sum(axis=1, keepdims=True)
    idx = np.array([{c:i for i,c in enumerate(classes)}[y] for y in y_true])
    return float(-np.mean(np.log(p[np.arange(len(y_true)), idx])))


def evaluate(name, model, X, y, split):
    pred = model.predict(X)
    proba = align_proba(model, X, CLASSES)
    return {
        "model": name, "split": split, "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "log_loss": manual_log_loss(y.to_numpy(), proba, CLASSES),
        "multiclass_brier": multiclass_brier(y.to_numpy(), proba, CLASSES),
        "ranked_probability_score": ranked_probability_score(y.to_numpy(), proba, CLASSES),
    }


def main():
    df = pd.read_csv(INPUT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df[TARGET].isin(CLASSES)].copy()
    bool_cols = ["neutral", "is_friendly", "is_qualifier", "is_world_cup", "is_continental", "same_confederation", "home_confed_matches_host_confed", "away_confed_matches_host_confed", "both_teams_have_major_tournament_history", "has_h2h_history", "has_h2h_last5"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    features = [c for c in BASE_NUMERIC_FEATURES + NEW_NUMERIC_FEATURES if c in df.columns]
    missing = sorted(set(BASE_NUMERIC_FEATURES + NEW_NUMERIC_FEATURES) - set(features))
    if missing:
        print("Warning missing features:", missing)
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    selected_cols = list(dict.fromkeys([c for c in ID_COLUMNS if c in df.columns] + features + cats))
    model_df = df[selected_cols].copy()
    model_df["split"] = np.select(
        [model_df["date"] < pd.Timestamp("2022-01-01"),
         (model_df["date"] >= pd.Timestamp("2022-01-01")) & (model_df["date"] < pd.Timestamp("2023-01-01")),
         model_df["date"] >= pd.Timestamp("2023-01-01")],
        ["train", "validation", "test"], default="unused")
    model_path = MODELING_DIR / "model_dataset_exp_h2h.csv"
    model_df.to_csv(model_path, index=False)
    train_df = model_df[model_df.split == "train"].copy()
    val_df = model_df[model_df.split == "validation"].copy()
    test_df = model_df[model_df.split == "test"].copy()
    X_train, y_train = train_df[features + cats], train_df[TARGET]
    X_val, y_val = val_df[features + cats], val_df[TARGET]
    X_test, y_test = test_df[features + cats], test_df[TARGET]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cats),
    ], remainder="drop")
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_exp_h2h": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=0.7)),
        ])
    }
    metrics=[]
    for name, model in models.items():
        model.fit(X_train, y_train)
        for split_name, X, y in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
            metrics.append(evaluate(name, model, X, y, split_name))
        joblib.dump(model, MODELS_DIR / f"exp_h2h_model_{name}.joblib")
    metrics_df = pd.DataFrame(metrics)
    metrics_path = MODELS_DIR / "exp_h2h_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    best_name = "logistic_regression_exp_h2h"
    best = models[best_name]
    joblib.dump(best, MODELS_DIR / "exp_h2h_model_best.joblib")
    # predictions
    pred_parts=[]
    for split_name, split_df, X, y in [("validation", val_df, X_val, y_val), ("test", test_df, X_test, y_test)]:
        pred=best.predict(X); proba=align_proba(best, X, CLASSES)
        tmp=split_df[["match_id","date","home_team","away_team","home_score","away_score","result","tournament","neutral","split"]].copy()
        tmp["predicted_result"]=pred
        tmp["prob_home_win"]=proba[:,0]; tmp["prob_draw"]=proba[:,1]; tmp["prob_away_win"]=proba[:,2]
        tmp["confidence"]=proba.max(axis=1); tmp["correct"]=(tmp.predicted_result==tmp.result).astype(int)
        pred_parts.append(tmp)
    preds=pd.concat(pred_parts, ignore_index=True)
    preds.to_csv(MODELS_DIR / "exp_h2h_validation_test_predictions.csv", index=False)
    preds[preds.split=="test"].to_csv(MODELS_DIR / "exp_h2h_test_predictions.csv", index=False)
    # confusion/reports
    reports={}
    for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        pred=best.predict(X)
        pd.DataFrame(confusion_matrix(y,pred,labels=CLASSES), index=[f"actual_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]).to_csv(MODELS_DIR / f"exp_h2h_confusion_matrix_{split_name}.csv")
        reports[split_name]=classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0)
    (MODELS_DIR / "exp_h2h_classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    # feature coefficients
    feature_names=best.named_steps["preprocess"].get_feature_names_out()
    coefs=best.named_steps["model"].coef_
    imps=[]
    for class_idx, cls in enumerate(best.named_steps["model"].classes_):
        for feat, val in zip(feature_names, coefs[class_idx]):
            imps.append({"model": best_name, "class": cls, "feature": feat, "importance": float(val), "abs_importance": abs(float(val))})
    imp_df=pd.DataFrame(imps).sort_values("abs_importance", ascending=False)
    imp_df.to_csv(MODELS_DIR / "exp_h2h_feature_importance.csv", index=False)
    # compare with previous baseline
    baseline = pd.read_csv(BASELINE_METRICS)
    base_lr = baseline[baseline["model"].eq("logistic_regression")].copy()
    new_lr = metrics_df[metrics_df["model"].eq(best_name)].copy()
    rows=[]
    for split in ["validation", "test"]:
        b=base_lr[base_lr.split==split].iloc[0]
        n=new_lr[new_lr.split==split].iloc[0]
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier", "ranked_probability_score"]:
            direction = "higher_better" if metric in ["accuracy","balanced_accuracy","macro_f1","weighted_f1"] else "lower_better"
            change = n[metric] - b[metric]
            improved = bool(change > 0) if direction == "higher_better" else bool(change < 0)
            rows.append({"split": split, "metric": metric, "baseline": float(b[metric]), "exp_h2h_model": float(n[metric]), "change": float(change), "direction": direction, "improved": improved})
    comp=pd.DataFrame(rows)
    comp_path=MODELS_DIR / "baseline_vs_exp_h2h_comparison.csv"
    comp.to_csv(comp_path, index=False)
    # metadata/report
    val=metrics_df[(metrics_df.model==best_name)&(metrics_df.split=="validation")].iloc[0]
    test=metrics_df[(metrics_df.model==best_name)&(metrics_df.split=="test")].iloc[0]
    base_test=base_lr[base_lr.split=="test"].iloc[0]
    comp_test=comp[comp.split=="test"]
    meta={
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT.relative_to(ROOT)),
        "modeling_dataset": str(model_path.relative_to(ROOT)),
        "date_splits": {"train":"2014-01-01 to 2021-12-31", "validation":"2022", "test":"2023-01-01 to 2026-06-02"},
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "test": int(len(test_df)), "total": int(len(model_df))},
        "base_numeric_features": BASE_NUMERIC_FEATURES,
        "new_numeric_features": NEW_NUMERIC_FEATURES,
        "categorical_features": cats,
        "best_model": best_name,
        "validation": {k: float(val[k]) for k in ["accuracy","macro_f1","log_loss","multiclass_brier"]},
        "test": {k: float(test[k]) for k in ["accuracy","macro_f1","log_loss","multiclass_brier"]},
        "comparison_file": str(comp_path.relative_to(ROOT)),
        "output_files": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "predictions": "models/exp_h2h_validation_test_predictions.csv",
            "test_predictions": "models/exp_h2h_test_predictions.csv",
            "feature_importance": "models/exp_h2h_feature_importance.csv",
            "best_model": "models/exp_h2h_model_best.joblib",
            "comparison": str(comp_path.relative_to(ROOT)),
        }
    }
    (ROOT / "exp_h2h_model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    def fmt_metric(split, metric):
        r=comp[(comp.split==split)&(comp.metric==metric)].iloc[0]
        return r.baseline, r.exp_h2h_model, r.change, r.improved
    lines=[]
    for split in ["validation","test"]:
        lines.append(f"## {split.title()} comparison\n")
        lines.append("| Metric | Baseline | Exp+H2H | Change | Improved? |\n|---|---:|---:|---:|:---:|\n")
        for metric in ["accuracy","macro_f1","log_loss","multiclass_brier","ranked_probability_score"]:
            r=comp[(comp.split==split)&(comp.metric==metric)].iloc[0]
            lines.append(f"| {metric} | {r.baseline:.4f} | {r.exp_h2h_model:.4f} | {r.change:+.4f} | {'yes' if r.improved else 'no'} |\n")
        lines.append("\n")
    top_new=imp_df[imp_df.feature.str.contains("exp_|h2h_", regex=True)].head(20)
    report = f"""# Tournament Experience + Head-to-Head Model Comparison

## Goal
Improve the first baseline by adding two date-safe feature families:

1. tournament-experience features,
2. head-to-head features.

Input dataset: `data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv`

The same past-to-future split is used as the first baseline:

| Split | Date range | Rows |
|---|---:|---:|
| Train | 2014-01-01 to 2021-12-31 | {len(train_df):,} |
| Validation | 2022-01-01 to 2022-12-31 | {len(val_df):,} |
| Test | 2023-01-01 to 2026-06-02 | {len(test_df):,} |

No random split was used.

{''.join(lines)}

## Verdict

On the held-out test period, accuracy changed from **{base_test['accuracy']:.4f}** to **{test['accuracy']:.4f}** and log loss changed from **{base_test['log_loss']:.4f}** to **{test['log_loss']:.4f}**.

Because log loss is the most important probability-quality metric here, the new features are considered {'an improvement' if comp_test[comp_test.metric.eq('log_loss')].iloc[0].improved else 'not an improvement on log loss'} on the test period.

## Top new-feature coefficient signals

{top_new[['class','feature','importance','abs_importance']].to_markdown(index=False)}

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
"""
    report_path=REPORTS_DIR / "exp_h2h_model_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status":"ok",
        "rows": meta["rows"],
        "validation": meta["validation"],
        "test": meta["test"],
        "comparison_test": comp[comp.split=="test"].to_dict(orient="records"),
        "report": str(report_path),
    }, indent=2))

if __name__ == "__main__":
    main()
