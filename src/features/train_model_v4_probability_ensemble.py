from __future__ import annotations
import json, math, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT/'models'
REPORTS = ROOT/'reports'
MODELS.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

LABELS = ['H','D','A']
BASE_PROB_COLS = ['prob_home_win','prob_draw','prob_away_win']
V3_PROB_COLS = ['cal_prob_home_win','cal_prob_draw','cal_prob_away_win']

base = pd.read_csv(MODELS/'baseline_validation_test_predictions.csv')
v3 = pd.read_csv(MODELS/'v3_calibrated_validation_test_predictions.csv')

# Align by match_id and split; keep baseline metadata and both probability sets.
keep_v3 = ['match_id','split'] + V3_PROB_COLS + ['v3_predicted_result','v3_confidence','v3_correct']
df = base.merge(v3[keep_v3], on=['match_id','split'], how='inner', validate='one_to_one')
assert len(df) == len(base) == len(v3), (len(df), len(base), len(v3))

# Numeric safety.
for c in BASE_PROB_COLS + V3_PROB_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce')
if df[BASE_PROB_COLS + V3_PROB_COLS].isna().any().any():
    raise ValueError('Missing probabilities found after merge')

def normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    return arr / arr.sum(axis=1, keepdims=True)

def y_indices(y):
    mapping = {lab:i for i, lab in enumerate(LABELS)}
    return np.array([mapping[v] for v in y])

def multiclass_brier(y_true, probs):
    yi = y_indices(y_true)
    y_one = np.eye(len(LABELS))[yi]
    return float(np.mean(np.sum((probs - y_one) ** 2, axis=1)))

def ordered_log_loss(y_true, probs):
    probs = normalize_probs(probs)
    yi = y_indices(y_true)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(yi)), yi], 1e-15, 1.0))))

def ece_score(y_true, probs, n_bins=10):
    yi = y_indices(y_true)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == yi).astype(float)
    ece = 0.0
    bins = []
    edges = np.linspace(0, 1, n_bins+1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({'bin': i+1, 'lower': lo, 'upper': hi, 'count': 0, 'avg_confidence': np.nan, 'accuracy': np.nan, 'gap': np.nan})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (n / len(y_true)) * gap
        bins.append({'bin': i+1, 'lower': lo, 'upper': hi, 'count': n, 'avg_confidence': avg_conf, 'accuracy': acc, 'gap': gap})
    return float(ece), pd.DataFrame(bins)

def metrics_for(y_true, probs, name, split):
    probs = normalize_probs(probs)
    pred_idx = probs.argmax(axis=1)
    pred = np.array(LABELS)[pred_idx]
    ece, _ = ece_score(y_true, probs, n_bins=10)
    return {
        'model': name,
        'split': split,
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'macro_f1': float(f1_score(y_true, pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, pred, average='weighted')),
        'log_loss': ordered_log_loss(y_true, probs),
        'multiclass_brier': multiclass_brier(y_true, probs),
        'ece_10_bin': ece,
    }

val = df[df['split']=='validation'].copy()
test = df[df['split']=='test'].copy()

P_base_val = normalize_probs(val[BASE_PROB_COLS].to_numpy())
P_v3_val = normalize_probs(val[V3_PROB_COLS].to_numpy())
P_base_test = normalize_probs(test[BASE_PROB_COLS].to_numpy())
P_v3_test = normalize_probs(test[V3_PROB_COLS].to_numpy())

# Tune convex blend: P_v4 = (1-w)*baseline + w*v3, w in [0,1].
# Dense grid is deterministic and enough for one-dimensional validation optimization.
grid = np.round(np.linspace(0, 1, 1001), 3)
rows = []
for w in grid:
    P = normalize_probs((1-w)*P_base_val + w*P_v3_val)
    rows.append({'v3_weight': float(w), 'baseline_weight': float(1-w), 'validation_log_loss': ordered_log_loss(val['result'], P), 'validation_brier': multiclass_brier(val['result'], P)})
grid_df = pd.DataFrame(rows)
# primary: log loss; tie-break: brier, then closer to v3 for stronger class prediction
best_row = grid_df.sort_values(['validation_log_loss','validation_brier','v3_weight'], ascending=[True, True, False]).iloc[0]
w_best = float(best_row['v3_weight'])

# Also evaluate nearby simpler weights for readability.
for w in [0, .25, .5, .75, 1.0, w_best]:
    pass

P_v4_val = normalize_probs((1-w_best)*P_base_val + w_best*P_v3_val)
P_v4_test = normalize_probs((1-w_best)*P_base_test + w_best*P_v3_test)

metric_rows = []
for split_name, y, Pb, Pv3, Pv4 in [
    ('validation', val['result'], P_base_val, P_v3_val, P_v4_val),
    ('test', test['result'], P_base_test, P_v3_test, P_v4_test),
]:
    metric_rows.append(metrics_for(y, Pb, 'first_baseline_logistic', split_name))
    metric_rows.append(metrics_for(y, Pv3, 'model_v3_calibrated_rf', split_name))
    metric_rows.append(metrics_for(y, Pv4, 'model_v4_probability_ensemble', split_name))
metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(MODELS/'v4_ensemble_metrics.csv', index=False)
grid_df.to_csv(MODELS/'v4_ensemble_weight_search.csv', index=False)

# Comparison table on test vs baseline and v3.
test_metrics = metrics_df[metrics_df['split']=='test'].set_index('model')
comp_rows = []
for m in ['accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss','multiclass_brier','ece_10_bin']:
    b = test_metrics.loc['first_baseline_logistic', m]
    v3m = test_metrics.loc['model_v3_calibrated_rf', m]
    v4m = test_metrics.loc['model_v4_probability_ensemble', m]
    lower_better = m in ['log_loss','multiclass_brier','ece_10_bin']
    comp_rows.append({
        'metric': m,
        'first_baseline': b,
        'model_v3': v3m,
        'model_v4': v4m,
        'v4_minus_baseline': v4m - b,
        'v4_minus_v3': v4m - v3m,
        'v4_improves_vs_baseline': (v4m < b if lower_better else v4m > b),
        'v4_improves_vs_v3': (v4m < v3m if lower_better else v4m > v3m),
        'lower_is_better': lower_better,
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(MODELS/'baseline_v3_v4_ensemble_comparison.csv', index=False)

# Predictions output.
def add_v4_preds(part, P_v4):
    out = part.copy()
    out['v4_prob_home_win'] = P_v4[:,0]
    out['v4_prob_draw'] = P_v4[:,1]
    out['v4_prob_away_win'] = P_v4[:,2]
    pred = np.array(LABELS)[P_v4.argmax(axis=1)]
    out['v4_predicted_result'] = pred
    out['v4_confidence'] = P_v4.max(axis=1)
    out['v4_correct'] = (out['result'].to_numpy() == pred).astype(int)
    out['v4_v3_weight'] = w_best
    out['v4_baseline_weight'] = 1 - w_best
    # keep cleaner columns
    cols = ['match_id','date','home_team','away_team','home_score','away_score','result','tournament','neutral','split',
            'prob_home_win','prob_draw','prob_away_win','predicted_result','confidence','correct',
            'cal_prob_home_win','cal_prob_draw','cal_prob_away_win','v3_predicted_result','v3_confidence','v3_correct',
            'v4_prob_home_win','v4_prob_draw','v4_prob_away_win','v4_predicted_result','v4_confidence','v4_correct','v4_baseline_weight','v4_v3_weight']
    return out[cols]
val_out = add_v4_preds(val, P_v4_val)
test_out = add_v4_preds(test, P_v4_test)
pred_all = pd.concat([val_out, test_out], ignore_index=True)
pred_all.to_csv(MODELS/'v4_ensemble_validation_test_predictions.csv', index=False)
test_out.to_csv(MODELS/'v4_ensemble_test_predictions.csv', index=False)

# Reliability bins for v4 test.
ece, bins = ece_score(test['result'], P_v4_test, n_bins=10)
bins.insert(0, 'model', 'model_v4_probability_ensemble')
bins.insert(1, 'split', 'test')
bins.to_csv(MODELS/'v4_reliability_bins.csv', index=False)

# Confusion matrices and classification report.
cm = confusion_matrix(test['result'], test_out['v4_predicted_result'], labels=LABELS)
pd.DataFrame(cm, index=[f'actual_{x}' for x in LABELS], columns=[f'pred_{x}' for x in LABELS]).to_csv(MODELS/'v4_confusion_matrix_test.csv')
with open(MODELS/'v4_classification_report_test.json','w') as f:
    json.dump(classification_report(test['result'], test_out['v4_predicted_result'], labels=LABELS, output_dict=True), f, indent=2)

# Persist simple ensemble metadata/calibrator object.
ensemble_obj = {
    'type': 'convex_probability_blend',
    'formula': 'P_v4 = (1 - w) * P_first_baseline + w * P_model_v3_calibrated',
    'v3_weight': w_best,
    'baseline_weight': 1 - w_best,
    'labels': LABELS,
    'selected_by': 'minimum validation log_loss on 2022 validation split',
}
joblib.dump(ensemble_obj, MODELS/'v4_probability_ensemble.joblib')

metadata = {
    'model_name': 'Model v4 probability-focused ensemble',
    'created_from': ['first baseline logistic probabilities', 'Model v3 calibrated random forest probabilities'],
    'blend_formula': ensemble_obj['formula'],
    'selected_v3_weight': w_best,
    'selected_baseline_weight': 1-w_best,
    'validation_objective': 'log_loss',
    'validation_best_log_loss': float(best_row['validation_log_loss']),
    'validation_best_brier': float(best_row['validation_brier']),
    'splits': {'train':'2014-01-01 to 2021-12-31','validation':'2022-01-01 to 2022-12-31','test':'2023-01-01 to 2026-06-02'},
    'rows': {'validation': int(len(val)), 'test': int(len(test)), 'total_eval_rows': int(len(df))},
    'metrics_file': str(MODELS/'v4_ensemble_metrics.csv'),
    'comparison_file': str(MODELS/'baseline_v3_v4_ensemble_comparison.csv'),
}
with open(ROOT/'model_v4_ensemble_metadata.json','w') as f:
    json.dump(metadata, f, indent=2)

# Report markdown.
def fmt(x): return f'{x:.4f}'
mt = metrics_df.pivot(index='model', columns='split')
val_m = metrics_df[metrics_df.split=='validation'].set_index('model')
tst_m = metrics_df[metrics_df.split=='test'].set_index('model')
report = []
report.append('# Model v4: probability-focused ensemble\n')
report.append('Model v4 blends the first baseline logistic-regression probabilities with Model v3 calibrated random-forest probabilities. The ensemble weight was selected only on the 2022 validation split, using validation log loss.\n')
report.append('## Ensemble formula\n')
report.append('```text\nP_v4 = (1 - w) * P_first_baseline + w * P_model_v3_calibrated\n```\n')
report.append(f'Selected weight: `w = {w_best:.3f}` for Model v3 and `{1-w_best:.3f}` for the first baseline.\n')
report.append('## Validation metrics\n')
report.append('| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |\n|---|---:|---:|---:|---:|---:|')
for name in ['first_baseline_logistic','model_v3_calibrated_rf','model_v4_probability_ensemble']:
    r=val_m.loc[name]
    report.append(f"| {name} | {fmt(r.accuracy)} | {fmt(r.macro_f1)} | {fmt(r.log_loss)} | {fmt(r.multiclass_brier)} | {fmt(r.ece_10_bin)} |")
report.append('\n## Test metrics\n')
report.append('| Model | Accuracy | Macro F1 | Log loss | Brier | ECE |\n|---|---:|---:|---:|---:|---:|')
for name in ['first_baseline_logistic','model_v3_calibrated_rf','model_v4_probability_ensemble']:
    r=tst_m.loc[name]
    report.append(f"| {name} | {fmt(r.accuracy)} | {fmt(r.macro_f1)} | {fmt(r.log_loss)} | {fmt(r.multiclass_brier)} | {fmt(r.ece_10_bin)} |")
report.append('\n## Verdict\n')
r4=tst_m.loc['model_v4_probability_ensemble']; rb=tst_m.loc['first_baseline_logistic']; r3=tst_m.loc['model_v3_calibrated_rf']
report.append(f'- Compared with Model v3, Model v4 changes test log loss by `{r4.log_loss-r3.log_loss:+.4f}` and Brier by `{r4.multiclass_brier-r3.multiclass_brier:+.4f}`.\n')
report.append(f'- Compared with the first baseline, Model v4 changes test log loss by `{r4.log_loss-rb.log_loss:+.4f}` and accuracy by `{r4.accuracy-rb.accuracy:+.4f}`.\n')
report.append('Lower log loss, Brier, and ECE are better; higher accuracy and F1 are better.\n')
(REPORTS/'model_v4_ensemble_report.md').write_text('\n'.join(report), encoding='utf-8')

# Make zip package.
zip_path = ROOT.parent / 'datacamp_worldcup_predictor_model_v4_ensemble.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        if p.is_file():
            z.write(p, p.relative_to(ROOT.parent))

print(json.dumps({
    'w_best': w_best,
    'metrics': metrics_df.to_dict(orient='records'),
    'zip': str(zip_path),
}, indent=2))
