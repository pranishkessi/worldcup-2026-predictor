#!/usr/bin/env python3
"""Model v3: post-hoc temperature calibration for Model v2 probabilities.

Uses the 2022 validation split to fit a single temperature parameter on
Model v2 predicted probabilities. Single-temperature scaling preserves the
class ranking/argmax for every match, so hard-class accuracy remains the same
while probability sharpness/calibration can improve.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
SRC = ROOT / 'src' / 'features'
CLASSES = ['H', 'D', 'A']
PROB_COLS = ['prob_home_win', 'prob_draw', 'prob_away_win']
CAL_PROB_COLS = ['cal_prob_home_win', 'cal_prob_draw', 'cal_prob_away_win']
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}



def custom_log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    idx = np.array([LABEL_TO_IDX[y] for y in y_true])
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), idx])))

def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), [LABEL_TO_IDX[y] for y in y_true]] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    # Ordinal order H-D-A is not a natural football ordering, but keep same metric as previous reports.
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), [LABEL_TO_IDX[y] for y in y_true]] = 1.0
    return float(np.mean(np.sum((np.cumsum(probs, axis=1) - np.cumsum(y_onehot, axis=1)) ** 2, axis=1) / (probs.shape[1] - 1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    pred_idx = probs.argmax(axis=1)
    pred_labels = np.array(CLASSES)[pred_idx]
    conf = probs.max(axis=1)
    correct = (pred_labels == y_true).astype(float)
    rows = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            acc = avg_conf = gap = np.nan
        else:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = abs(acc - avg_conf)
            ece += (count / len(y_true)) * gap
        rows.append({
            'bin': i + 1,
            'confidence_lower': lo,
            'confidence_upper': hi,
            'count': count,
            'accuracy': acc,
            'avg_confidence': avg_conf,
            'abs_gap': gap,
        })
    return float(ece), pd.DataFrame(rows)


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0)
    logits = np.log(probs)
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_for_temperature(temp: float, probs: np.ndarray, y_true: np.ndarray) -> float:
    if temp <= 0 or not np.isfinite(temp):
        return float('inf')
    cal = apply_temperature(probs, temp)
    return custom_log_loss(y_true, cal)


def golden_section_search(func, low: float, high: float, tol: float = 1e-5, max_iter: int = 200) -> tuple[float, float]:
    gr = (math.sqrt(5) + 1) / 2
    c = high - (high - low) / gr
    d = low + (high - low) / gr
    fc, fd = func(c), func(d)
    for _ in range(max_iter):
        if abs(high - low) < tol:
            break
        if fc < fd:
            high = d
            d = c
            fd = fc
            c = high - (high - low) / gr
            fc = func(c)
        else:
            low = c
            c = d
            fc = fd
            d = low + (high - low) / gr
            fd = func(d)
    best = (low + high) / 2
    return best, func(best)


def metrics_for(df: pd.DataFrame, prob_cols: list[str], model_name: str, split: str) -> dict:
    y = df['result'].to_numpy()
    probs = df[prob_cols].to_numpy(float)
    pred = np.array(CLASSES)[probs.argmax(axis=1)]
    ece, _ = expected_calibration_error(y, probs)
    return {
        'model': model_name,
        'split': split,
        'rows': len(df),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'weighted_f1': float(f1_score(y, pred, average='weighted')),
        'log_loss': custom_log_loss(y, probs),
        'multiclass_brier': multiclass_brier(y, probs),
        'ranked_probability_score': ranked_probability_score(y, probs),
        'ece_10bin': ece,
        'avg_confidence': float(probs.max(axis=1).mean()),
    }


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(MODELS / 'v2_validation_test_predictions.csv')
    # Keep only best v2 model rows if older files have multiple model outputs.
    if 'model' in preds.columns:
        preds = preds[preds['model'].astype(str).str.contains('random_forest_shallow|v2_best|random_forest', case=False, regex=True, na=False) | (preds['model'] == 'random_forest_shallow')].copy() if preds['model'].nunique() > 1 else preds.copy()
    val = preds[preds['split'] == 'validation'].copy()
    test = preds[preds['split'] == 'test'].copy()
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError('Need validation and test rows in v2_validation_test_predictions.csv')
    p_val = val[PROB_COLS].to_numpy(float)
    y_val = val['result'].to_numpy()

    # Search a broad range. T > 1 softens overconfident probabilities; T < 1 sharpens.
    best_t, best_val_nll = golden_section_search(lambda t: nll_for_temperature(t, p_val, y_val), 0.20, 5.00)

    out_frames = []
    for split_name, part in [('validation', val), ('test', test)]:
        cal_probs = apply_temperature(part[PROB_COLS].to_numpy(float), best_t)
        out = part.copy()
        for c, values in zip(CAL_PROB_COLS, cal_probs.T):
            out[c] = values
        out['v3_predicted_result'] = np.array(CLASSES)[cal_probs.argmax(axis=1)]
        out['v3_confidence'] = cal_probs.max(axis=1)
        out['v3_correct'] = (out['v3_predicted_result'] == out['result']).astype(int)
        out['temperature'] = best_t
        out_frames.append(out)
    v3_preds = pd.concat(out_frames, ignore_index=True)
    v3_preds.to_csv(MODELS / 'v3_calibrated_validation_test_predictions.csv', index=False)
    v3_preds[v3_preds['split'] == 'test'].to_csv(MODELS / 'v3_calibrated_test_predictions.csv', index=False)

    # Metrics: v2 uncalibrated vs v3 calibrated.
    rows = []
    for split_name in ['validation', 'test']:
        part = v3_preds[v3_preds['split'] == split_name]
        rows.append(metrics_for(part, PROB_COLS, 'v2_random_forest_shallow_uncalibrated', split_name))
        rows.append(metrics_for(part, CAL_PROB_COLS, 'v3_temperature_calibrated_rf', split_name))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODELS / 'v3_calibration_metrics.csv', index=False)

    # Compare v3 to first baseline and v2 using existing baseline/v2 summaries when available.
    comparison_rows = []
    try:
        b = pd.read_csv(MODELS / 'baseline_vs_v2_comparison.csv')
        # It contains first baseline and v2 comparison; use test rows if format matches.
    except Exception:
        b = None
    v2_test = metrics[(metrics['model'] == 'v2_random_forest_shallow_uncalibrated') & (metrics['split'] == 'test')].iloc[0]
    v3_test = metrics[(metrics['model'] == 'v3_temperature_calibrated_rf') & (metrics['split'] == 'test')].iloc[0]
    # Previous first-baseline values from current baseline metrics file.
    baseline_metrics = pd.read_csv(MODELS / 'baseline_metrics.csv')
    if 'model' in baseline_metrics.columns:
        # Select best non-dummy test row by log_loss if multiple exist.
        bm = baseline_metrics[(baseline_metrics['split'] == 'test') & (~baseline_metrics['model'].astype(str).str.contains('dummy', case=False, na=False))].copy()
        bm = bm.sort_values('log_loss').iloc[0]
        comparison_targets = [('first_baseline', bm), ('model_v2_uncalibrated', v2_test), ('model_v3_calibrated', v3_test)]
    else:
        comparison_targets = [('model_v2_uncalibrated', v2_test), ('model_v3_calibrated', v3_test)]
    for name, s in comparison_targets:
        comparison_rows.append({
            'model': name,
            'test_accuracy': float(s['accuracy']),
            'test_balanced_accuracy': float(s.get('balanced_accuracy', np.nan)),
            'test_macro_f1': float(s['macro_f1']),
            'test_weighted_f1': float(s.get('weighted_f1', np.nan)),
            'test_log_loss': float(s['log_loss']),
            'test_multiclass_brier': float(s['multiclass_brier']),
            'test_ranked_probability_score': float(s.get('ranked_probability_score', np.nan)),
            'test_ece_10bin': float(s.get('ece_10bin', np.nan)),
            'test_avg_confidence': float(s.get('avg_confidence', np.nan)),
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(MODELS / 'baseline_v2_v3_calibration_comparison.csv', index=False)

    # Reliability bin tables.
    rel_tables = []
    for split_name in ['validation', 'test']:
        part = v3_preds[v3_preds['split'] == split_name]
        for model_name, cols in [('v2_uncalibrated', PROB_COLS), ('v3_calibrated', CAL_PROB_COLS)]:
            ece, tbl = expected_calibration_error(part['result'].to_numpy(), part[cols].to_numpy(float))
            tbl.insert(0, 'split', split_name)
            tbl.insert(1, 'model', model_name)
            tbl['ece_10bin'] = ece
            rel_tables.append(tbl)
    reliability = pd.concat(rel_tables, ignore_index=True)
    reliability.to_csv(MODELS / 'v3_reliability_bins.csv', index=False)

    # Confusion matrices: should match v2 because temp scaling preserves argmax.
    for split_name in ['validation', 'test']:
        part = v3_preds[v3_preds['split'] == split_name]
        cm = confusion_matrix(part['result'], part['v3_predicted_result'], labels=CLASSES)
        pd.DataFrame(cm, index=[f'actual_{c}' for c in CLASSES], columns=[f'pred_{c}' for c in CLASSES]).to_csv(MODELS / f'v3_confusion_matrix_{split_name}.csv')

    metadata = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': 'Model v3 = temperature-calibrated Model v2 random_forest_shallow',
        'calibration_method': 'single-temperature scaling on predicted class probabilities',
        'temperature': best_t,
        'validation_log_loss_after_temperature_fit': best_val_nll,
        'calibration_split': 'validation: 2022-01-01 to 2022-12-31',
        'test_split': 'test: 2023-01-01 to 2026-06-02',
        'class_order': CLASSES,
        'hard_prediction_preservation': 'Single-temperature scaling preserves probability ordering and therefore preserves argmax predictions for every row.',
        'input_predictions': str(MODELS / 'v2_validation_test_predictions.csv'),
        'outputs': [
            str(MODELS / 'v3_calibrated_validation_test_predictions.csv'),
            str(MODELS / 'v3_calibrated_test_predictions.csv'),
            str(MODELS / 'v3_calibration_metrics.csv'),
            str(MODELS / 'baseline_v2_v3_calibration_comparison.csv'),
            str(MODELS / 'v3_reliability_bins.csv'),
        ],
    }
    with open(ROOT / 'model_v3_calibration_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Report.
    val_v2 = metrics[(metrics['model'] == 'v2_random_forest_shallow_uncalibrated') & (metrics['split'] == 'validation')].iloc[0]
    val_v3 = metrics[(metrics['model'] == 'v3_temperature_calibrated_rf') & (metrics['split'] == 'validation')].iloc[0]
    test_v2 = v2_test
    test_v3 = v3_test
    def fmt(x): return f'{float(x):.4f}'
    report = f"""# Model v3: Calibrated Model v2

## Summary

Model v3 applies post-hoc single-temperature calibration to the Model v2 `random_forest_shallow` probabilities.
The temperature was fitted only on the 2022 validation split, then evaluated on the 2023–2026 test split.

Single-temperature scaling preserves each row's probability ordering, so the predicted W/D/L class remains unchanged while the probability distribution is softened or sharpened.

## Calibration setting

- Method: temperature scaling on class probabilities
- Fitted temperature: `{best_t:.6f}`
- Class order: `{CLASSES}`
- Calibration data: validation split only

## Validation result

| Metric | V2 uncalibrated | V3 calibrated | Change |
|---|---:|---:|---:|
| Accuracy | {fmt(val_v2['accuracy'])} | {fmt(val_v3['accuracy'])} | {fmt(val_v3['accuracy'] - val_v2['accuracy'])} |
| Macro F1 | {fmt(val_v2['macro_f1'])} | {fmt(val_v3['macro_f1'])} | {fmt(val_v3['macro_f1'] - val_v2['macro_f1'])} |
| Log loss | {fmt(val_v2['log_loss'])} | {fmt(val_v3['log_loss'])} | {fmt(val_v3['log_loss'] - val_v2['log_loss'])} |
| Brier score | {fmt(val_v2['multiclass_brier'])} | {fmt(val_v3['multiclass_brier'])} | {fmt(val_v3['multiclass_brier'] - val_v2['multiclass_brier'])} |
| ECE, 10-bin | {fmt(val_v2['ece_10bin'])} | {fmt(val_v3['ece_10bin'])} | {fmt(val_v3['ece_10bin'] - val_v2['ece_10bin'])} |
| Avg confidence | {fmt(val_v2['avg_confidence'])} | {fmt(val_v3['avg_confidence'])} | {fmt(val_v3['avg_confidence'] - val_v2['avg_confidence'])} |

## Test result

| Metric | V2 uncalibrated | V3 calibrated | Change |
|---|---:|---:|---:|
| Accuracy | {fmt(test_v2['accuracy'])} | {fmt(test_v3['accuracy'])} | {fmt(test_v3['accuracy'] - test_v2['accuracy'])} |
| Balanced accuracy | {fmt(test_v2['balanced_accuracy'])} | {fmt(test_v3['balanced_accuracy'])} | {fmt(test_v3['balanced_accuracy'] - test_v2['balanced_accuracy'])} |
| Macro F1 | {fmt(test_v2['macro_f1'])} | {fmt(test_v3['macro_f1'])} | {fmt(test_v3['macro_f1'] - test_v2['macro_f1'])} |
| Weighted F1 | {fmt(test_v2['weighted_f1'])} | {fmt(test_v3['weighted_f1'])} | {fmt(test_v3['weighted_f1'] - test_v2['weighted_f1'])} |
| Log loss | {fmt(test_v2['log_loss'])} | {fmt(test_v3['log_loss'])} | {fmt(test_v3['log_loss'] - test_v2['log_loss'])} |
| Brier score | {fmt(test_v2['multiclass_brier'])} | {fmt(test_v3['multiclass_brier'])} | {fmt(test_v3['multiclass_brier'] - test_v2['multiclass_brier'])} |
| ECE, 10-bin | {fmt(test_v2['ece_10bin'])} | {fmt(test_v3['ece_10bin'])} | {fmt(test_v3['ece_10bin'] - test_v2['ece_10bin'])} |
| Avg confidence | {fmt(test_v2['avg_confidence'])} | {fmt(test_v3['avg_confidence'])} | {fmt(test_v3['avg_confidence'] - test_v2['avg_confidence'])} |

## Verdict

Model v3 keeps Model v2's hard-class performance because temperature scaling does not change the argmax prediction.
Its value is judged by probability metrics: log loss, Brier score, and reliability/ECE.

See:

- `models/v3_calibration_metrics.csv`
- `models/baseline_v2_v3_calibration_comparison.csv`
- `models/v3_reliability_bins.csv`
"""
    (REPORTS / 'model_v3_calibration_report.md').write_text(report)

    print('temperature', best_t)
    print(metrics)
    print(comparison)

if __name__ == '__main__':
    main()
