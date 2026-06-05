from pathlib import Path
from collections import defaultdict, deque
import json
import math
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'data/processed/matches_with_elo_fifa.csv'
FULL = ROOT/'data/raw/results_full.csv'
OUT = ROOT/'data/processed/matches_with_elo_fifa_form.csv'
COVERAGE = ROOT/'data/processed/recent_form_coverage.csv'
META = ROOT/'recent_form_metadata.json'
README = ROOT/'README_recent_form.md'
SCRIPT = ROOT/'src/features/build_recent_form_features.py'
ZIP_OUT = ROOT.parent / 'datacamp_worldcup_predictor_with_elo_fifa_form.zip'

for p in [ROOT/'data/processed', ROOT/'src/features']:
    p.mkdir(parents=True, exist_ok=True)

def normalize_bool(x):
    if isinstance(x, bool): return x
    if pd.isna(x): return False
    return str(x).strip().lower() in {'true','1','yes'}

def make_key(row):
    return (
        str(row['date'])[:10],
        str(row['home_team']), str(row['away_team']),
        int(row['home_score']), int(row['away_score']),
        str(row['tournament']), str(row.get('city','')), str(row.get('country','')),
        bool(normalize_bool(row.get('neutral', False)))
    )

def result_points(gf, ga):
    if gf > ga: return 3, 'W'
    if gf == ga: return 1, 'D'
    return 0, 'L'

def stats_from_history(hist, as_of_date, n):
    recent = list(hist)[-n:]
    available = len(recent)
    d = {}
    prefix = f'last{n}'
    d[f'matches_available_{prefix}'] = available
    if available == 0:
        d.update({
            f'form_points_{prefix}': 0,
            f'form_points_per_match_{prefix}': np.nan,
            f'goals_for_{prefix}': 0,
            f'goals_against_{prefix}': 0,
            f'avg_goals_for_{prefix}': np.nan,
            f'avg_goals_against_{prefix}': np.nan,
            f'goal_diff_sum_{prefix}': 0,
            f'avg_goal_diff_{prefix}': np.nan,
            f'wins_{prefix}': 0,
            f'draws_{prefix}': 0,
            f'losses_{prefix}': 0,
            f'win_rate_{prefix}': np.nan,
            f'draw_rate_{prefix}': np.nan,
            f'loss_rate_{prefix}': np.nan,
            f'days_since_last_match_{prefix}': np.nan,
            f'weighted_form_points_{prefix}': 0.0,
            f'weighted_goal_diff_{prefix}': 0.0,
        })
        return d
    pts = sum(m['points'] for m in recent)
    gf = sum(m['gf'] for m in recent)
    ga = sum(m['ga'] for m in recent)
    gd = sum(m['gd'] for m in recent)
    wins = sum(1 for m in recent if m['result'] == 'W')
    draws = sum(1 for m in recent if m['result'] == 'D')
    losses = sum(1 for m in recent if m['result'] == 'L')
    last_date = recent[-1]['date']
    days = (as_of_date - last_date).days
    # Recency weights: oldest to newest increases linearly, newest gets weight n.
    weights = np.arange(n - available + 1, n + 1, dtype=float)
    weights = weights / weights.sum()
    w_pts = float(sum(w * m['points'] for w, m in zip(weights, recent)))
    w_gd = float(sum(w * m['gd'] for w, m in zip(weights, recent)))
    d.update({
        f'form_points_{prefix}': int(pts),
        f'form_points_per_match_{prefix}': pts / available,
        f'goals_for_{prefix}': int(gf),
        f'goals_against_{prefix}': int(ga),
        f'avg_goals_for_{prefix}': gf / available,
        f'avg_goals_against_{prefix}': ga / available,
        f'goal_diff_sum_{prefix}': int(gd),
        f'avg_goal_diff_{prefix}': gd / available,
        f'wins_{prefix}': int(wins),
        f'draws_{prefix}': int(draws),
        f'losses_{prefix}': int(losses),
        f'win_rate_{prefix}': wins / available,
        f'draw_rate_{prefix}': draws / available,
        f'loss_rate_{prefix}': losses / available,
        f'days_since_last_match_{prefix}': days,
        f'weighted_form_points_{prefix}': w_pts,
        f'weighted_goal_diff_{prefix}': w_gd,
    })
    return d

matches = pd.read_csv(INPUT)
full = pd.read_csv(FULL)
# Keep completed matches only.
for c in ['home_score','away_score']:
    full[c] = pd.to_numeric(full[c], errors='coerce')
full = full.dropna(subset=['date','home_team','away_team','home_score','away_score']).copy()
full['home_score'] = full['home_score'].astype(int)
full['away_score'] = full['away_score'].astype(int)
full['date'] = pd.to_datetime(full['date']).dt.date
matches['date'] = pd.to_datetime(matches['date']).dt.date

# map exact source rows to target row indices, robust to duplicate keys by queuing indices.
target_by_key = defaultdict(deque)
for idx, row in matches.iterrows():
    target_by_key[make_key(row)].append(idx)

hist = defaultdict(lambda: deque(maxlen=1000))
features = {}
matched_targets = set()
full = full.sort_values(['date','home_team','away_team','tournament','city','country']).reset_index(drop=True)

for date, day_df in full.groupby('date', sort=True):
    # Calculate features before updating same-date matches to avoid leakage from matches on the same date.
    day_records = []
    for _, row in day_df.iterrows():
        key = make_key(row)
        idx = None
        if key in target_by_key and target_by_key[key]:
            idx = target_by_key[key].popleft()
            as_of_date = date
            hteam, ateam = row['home_team'], row['away_team']
            f = {}
            for side, team in [('home', hteam), ('away', ateam)]:
                for n in (5, 10):
                    s = stats_from_history(hist[team], as_of_date, n)
                    for k, v in s.items():
                        f[f'{side}_{k}'] = v
            # Difference features. Positive means home has stronger recent form.
            for n in (5,10):
                suffix = f'last{n}'
                f[f'form_points_diff_{suffix}'] = f[f'home_form_points_{suffix}'] - f[f'away_form_points_{suffix}']
                f[f'form_points_per_match_diff_{suffix}'] = np.nan_to_num(f[f'home_form_points_per_match_{suffix}'], nan=0.0) - np.nan_to_num(f[f'away_form_points_per_match_{suffix}'], nan=0.0)
                f[f'goals_for_diff_{suffix}'] = f[f'home_goals_for_{suffix}'] - f[f'away_goals_for_{suffix}']
                f[f'goals_against_diff_{suffix}'] = f[f'home_goals_against_{suffix}'] - f[f'away_goals_against_{suffix}']
                f[f'avg_goal_diff_delta_{suffix}'] = np.nan_to_num(f[f'home_avg_goal_diff_{suffix}'], nan=0.0) - np.nan_to_num(f[f'away_avg_goal_diff_{suffix}'], nan=0.0)
                f[f'win_rate_diff_{suffix}'] = np.nan_to_num(f[f'home_win_rate_{suffix}'], nan=0.0) - np.nan_to_num(f[f'away_win_rate_{suffix}'], nan=0.0)
                f[f'weighted_form_points_diff_{suffix}'] = f[f'home_weighted_form_points_{suffix}'] - f[f'away_weighted_form_points_{suffix}']
                f[f'weighted_goal_diff_delta_{suffix}'] = f[f'home_weighted_goal_diff_{suffix}'] - f[f'away_weighted_goal_diff_{suffix}']
                f[f'both_teams_have_{suffix}_history'] = bool((f[f'home_matches_available_{suffix}'] >= n) and (f[f'away_matches_available_{suffix}'] >= n))
            features[idx] = f
            matched_targets.add(idx)
        # collect updates after feature calculation
        day_records.append(row)
    # Now update histories using all matches on this date.
    for row in day_records:
        h, a = row['home_team'], row['away_team']
        hs, a_s = int(row['home_score']), int(row['away_score'])
        hp, hr = result_points(hs, a_s)
        ap, ar = result_points(a_s, hs)
        hist[h].append({'date': date, 'points': hp, 'gf': hs, 'ga': a_s, 'gd': hs-a_s, 'result': hr, 'opponent': a})
        hist[a].append({'date': date, 'points': ap, 'gf': a_s, 'ga': hs, 'gd': a_s-hs, 'result': ar, 'opponent': h})

# Assemble features in original target order.
if len(matched_targets) != len(matches):
    missing = sorted(set(matches.index)-matched_targets)[:10]
    raise RuntimeError(f'Matched {len(matched_targets)} of {len(matches)} target rows. First missing target indices: {missing}')
feat_df = pd.DataFrame.from_dict(features, orient='index').sort_index()
combined = pd.concat([matches.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
# Ensure dates are strings for CSV
combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')
# Rounding for readability
float_cols = combined.select_dtypes(include=['float64','float32']).columns
combined[float_cols] = combined[float_cols].round(6)
combined.to_csv(OUT, index=False)

# Coverage report by year and overall
cov_rows = []
for label, group in [('ALL', combined)] + [(str(y), g) for y, g in combined.groupby(pd.to_datetime(combined['date']).dt.year)]:
    cov_rows.append({
        'period': label,
        'matches': len(group),
        'home_has_at_least_5': int((group['home_matches_available_last5'] >= 5).sum()),
        'away_has_at_least_5': int((group['away_matches_available_last5'] >= 5).sum()),
        'both_have_at_least_5': int((group['both_teams_have_last5_history'] == True).sum()),
        'home_has_at_least_10': int((group['home_matches_available_last10'] >= 10).sum()),
        'away_has_at_least_10': int((group['away_matches_available_last10'] >= 10).sum()),
        'both_have_at_least_10': int((group['both_teams_have_last10_history'] == True).sum()),
        'median_home_days_since_last_match_last5': float(group['home_days_since_last_match_last5'].median()),
        'median_away_days_since_last_match_last5': float(group['away_days_since_last_match_last5'].median()),
    })
coverage = pd.DataFrame(cov_rows)
coverage.to_csv(COVERAGE, index=False)

metadata = {
    'created_for': 'Data Camp international football prediction pipeline - Milestone 4',
    'input_file': str(INPUT.relative_to(ROOT)),
    'history_file': str(FULL.relative_to(ROOT)),
    'output_file': str(OUT.relative_to(ROOT)),
    'rows': int(len(combined)),
    'columns': int(combined.shape[1]),
    'target_date_min': combined['date'].min(),
    'target_date_max': combined['date'].max(),
    'history_completed_matches_used': int(len(full)),
    'matched_target_rows': int(len(matched_targets)),
    'leakage_control': 'Features are calculated before updating team histories for the same match date; matches on the same date are not used as prior form.',
    'windows': [5,10],
    'key_model_features': [
        'form_points_diff_last5','form_points_per_match_diff_last5','avg_goal_diff_delta_last5',
        'weighted_form_points_diff_last5','weighted_goal_diff_delta_last5',
        'form_points_diff_last10','form_points_per_match_diff_last10','avg_goal_diff_delta_last10'
    ],
    'notes': [
        'Positive difference features mean the home team has stronger recent form.',
        'Rates and averages are NaN when a team has zero prior matches; difference features use 0 for missing side averages/rates.',
        'Histories are warmed up using completed matches before 2014 from results_full.csv.'
    ]
}
META.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

readme = f'''# Recent Form Feature Layer — Milestone 4\n\nThis layer adds pre-match recent-form features to `matches_with_elo_fifa.csv`.\n\n## Output\n\n- `data/processed/matches_with_elo_fifa_form.csv`\n- Rows: {len(combined):,}\n- Columns: {combined.shape[1]:,}\n- Date range: {combined['date'].min()} to {combined['date'].max()}\n\n## Leakage control\n\nFor each match, team form is calculated only from matches played before the current match date. Because the source has dates but not kickoff times, matches on the same date are not included as prior information. The script calculates all features for a date first, then updates team histories using that date's matches.\n\n## Main feature groups\n\nFor both home and away teams, the dataset includes last-5 and last-10 versions of:\n\n- matches available\n- form points\n- points per match\n- goals for and against\n- average goals for and against\n- goal-difference sum and average goal difference\n- wins, draws, losses\n- win/draw/loss rates\n- days since last match\n- weighted form points\n- weighted goal difference\n\nIt also includes home-minus-away difference features, such as:\n\n- `form_points_diff_last5`\n- `form_points_per_match_diff_last5`\n- `avg_goal_diff_delta_last5`\n- `weighted_form_points_diff_last5`\n- `form_points_diff_last10`\n- `avg_goal_diff_delta_last10`\n\nPositive difference values mean the home team had stronger recent form before the match.\n\n## Recommended modelling features\n\nStart with:\n\n```text\nelo_diff\nfifa_rank_diff_filled\nfifa_points_diff_filled\nform_points_per_match_diff_last5\navg_goal_diff_delta_last5\nweighted_form_points_diff_last5\nform_points_per_match_diff_last10\navg_goal_diff_delta_last10\nneutral\ntournament_importance\n```\n\n## Files\n\n- Script: `src/features/build_recent_form_features.py`\n- Metadata: `recent_form_metadata.json`\n- Coverage report: `data/processed/recent_form_coverage.csv`\n'''
README.write_text(readme, encoding='utf-8')
SCRIPT.write_text(Path('/tmp/build_form.py').read_text(encoding='utf-8'), encoding='utf-8')

# Zip selected project files.
if ZIP_OUT.exists():
    ZIP_OUT.unlink()
with zipfile.ZipFile(ZIP_OUT, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in ROOT.rglob('*'):
        if path.is_file():
            z.write(path, path.relative_to(ROOT.parent))

print('OUTPUT', OUT)
print('SHAPE', combined.shape)
print('COLUMNS_ADDED', feat_df.shape[1])
print('COVERAGE_ALL')
print(coverage.head(1).to_string(index=False))
print('NA_CHECK_DIFFS', combined[['form_points_diff_last5','form_points_per_match_diff_last5','avg_goal_diff_delta_last5','form_points_diff_last10']].isna().sum().to_dict())
print('ZIP', ZIP_OUT)
