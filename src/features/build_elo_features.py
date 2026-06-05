from pathlib import Path
import json, math
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
RAW_FULL = BASE/'data/raw/results_full.csv'
MATCHES_BASE = BASE/'data/processed/matches_base.csv'
OUT_RAW = BASE/'data/raw/elo_ratings_pre_match.csv'
OUT_PROC = BASE/'data/processed/matches_with_elo.csv'
OUT_CURRENT = BASE/'data/processed/current_elo_ratings.csv'
OUT_META = BASE/'elo_metadata.json'
OUT_README = BASE/'README_elo.md'
OUT_SCRIPT_DIR = BASE/'src/features'
OUT_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 100.0

def infer_k(tournament: str) -> int:
    t = str(tournament).lower()
    # approximate World Football Elo-style K classes
    if 'fifa world cup' in t and 'qualification' not in t:
        return 60
    if any(x in t for x in ['uefa euro', 'copa américa', 'copa america', 'african cup of nations', 'afc asian cup', 'concacaf gold cup', 'oceania nations cup']) and 'qualification' not in t:
        return 50
    if 'qualification' in t or 'qualifier' in t:
        return 40
    if any(x in t for x in ['nations league', 'confederations cup', 'finalissima']):
        return 40
    if 'friendly' in t:
        return 20
    return 30

def result_score(hg, ag):
    if hg > ag: return 1.0
    if hg < ag: return 0.0
    return 0.5

def expected_score(r_home, r_away, neutral):
    adj_home = r_home + (0.0 if bool(neutral) else HOME_ADVANTAGE)
    return 1.0 / (1.0 + 10 ** ((r_away - adj_home) / 400.0))

def goal_multiplier(gd):
    gd = abs(int(gd))
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) / 8.0

# Load all available completed results for historical warm-up
full = pd.read_csv(RAW_FULL)
full['date'] = pd.to_datetime(full['date'], errors='coerce')
full = full.dropna(subset=['date','home_team','away_team','home_score','away_score']).copy()
full['home_score'] = full['home_score'].astype(int)
full['away_score'] = full['away_score'].astype(int)
full = full.sort_values(['date','home_team','away_team','tournament']).reset_index(drop=True)

ratings = {}
rows = []
team_last_date = {}
for i, row in full.iterrows():
    home = row['home_team']; away = row['away_team']
    rh = ratings.get(home, INITIAL_ELO)
    ra = ratings.get(away, INITIAL_ELO)
    exp_h = expected_score(rh, ra, row['neutral'])
    actual_h = result_score(row['home_score'], row['away_score'])
    gd = int(row['home_score'] - row['away_score'])
    k = infer_k(row['tournament'])
    g = goal_multiplier(gd)
    change = k * g * (actual_h - exp_h)
    new_rh = rh + change
    new_ra = ra - change
    rows.append({
        'date': row['date'].date().isoformat(),
        'home_team': home,
        'away_team': away,
        'tournament': row['tournament'],
        'neutral': bool(row['neutral']),
        'home_score': int(row['home_score']),
        'away_score': int(row['away_score']),
        'home_elo_pre': round(rh, 2),
        'away_elo_pre': round(ra, 2),
        'elo_diff_pre': round(rh - ra, 2),
        'home_elo_expected_with_home_adv': round(exp_h, 6),
        'elo_k': k,
        'elo_goal_multiplier': round(g, 3),
        'home_elo_change': round(change, 2),
        'away_elo_change': round(-change, 2),
        'home_elo_post': round(new_rh, 2),
        'away_elo_post': round(new_ra, 2),
    })
    ratings[home] = new_rh
    ratings[away] = new_ra
    team_last_date[home] = row['date'].date().isoformat()
    team_last_date[away] = row['date'].date().isoformat()

elo_hist = pd.DataFrame(rows)
OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
elo_hist.to_csv(OUT_RAW, index=False)

# Merge into processed 2014-2026 table using exact match keys; add a duplicate sequence key to avoid duplicate collisions
base = pd.read_csv(MATCHES_BASE)
base['date'] = pd.to_datetime(base['date']).dt.date.astype(str)
key_cols = ['date','home_team','away_team','home_score','away_score','tournament','city','country','neutral']
# elo_hist lacks city/country, so use a count key on date/team/score/tournament/neutral
merge_cols = ['date','home_team','away_team','home_score','away_score','tournament','neutral']
base['neutral'] = base['neutral'].astype(bool)
for c in ['home_score','away_score']:
    base[c]=base[c].astype(int)
base['_dup'] = base.groupby(merge_cols).cumcount()
elo_hist['_dup'] = elo_hist.groupby(merge_cols).cumcount()
elo_cols = merge_cols + ['_dup','home_elo_pre','away_elo_pre','elo_diff_pre','home_elo_expected_with_home_adv','elo_k','elo_goal_multiplier','home_elo_change','away_elo_change','home_elo_post','away_elo_post']
merged = base.merge(elo_hist[elo_cols], on=merge_cols+['_dup'], how='left')
merged = merged.drop(columns=['_dup'])
# More model-friendly names
merged['home_elo'] = merged['home_elo_pre']
merged['away_elo'] = merged['away_elo_pre']
merged['elo_diff'] = merged['elo_diff_pre']
merged['elo_prob_home_win_proxy'] = merged['home_elo_expected_with_home_adv']
# Reorder
front = ['date','home_team','away_team','home_score','away_score','result','tournament','city','country','neutral',
         'home_elo','away_elo','elo_diff','elo_prob_home_win_proxy','elo_k','elo_goal_multiplier']
cols = front + [c for c in merged.columns if c not in front]
merged = merged[cols]
merged.to_csv(OUT_PROC, index=False)

current = pd.DataFrame([{'team': t, 'elo_rating': round(r,2), 'last_match_date': team_last_date.get(t)} for t,r in ratings.items()])
current = current.sort_values('elo_rating', ascending=False).reset_index(drop=True)
current.insert(0, 'rank', current.index+1)
current.to_csv(OUT_CURRENT, index=False)

# Save build script for reproducibility
script_dst = OUT_SCRIPT_DIR/'build_elo_features.py'
script_dst.write_text(Path('/tmp/build_elo.py').read_text(), encoding='utf-8')

meta = {
    'created_at': pd.Timestamp.utcnow().isoformat(),
    'input_results_full': str(RAW_FULL.relative_to(BASE)),
    'input_matches_base': str(MATCHES_BASE.relative_to(BASE)),
    'method': 'Local Elo-style ratings calculated from historical international results, using all completed matches in results_full.csv as warm-up and exporting pre-match ratings for the 2014-2026 modelling window.',
    'not_official_eloratings_net': True,
    'initial_elo': INITIAL_ELO,
    'home_advantage_points_when_not_neutral': HOME_ADVANTAGE,
    'k_rule_summary': {'World Cup finals':60,'Continental finals':50,'Qualifiers/Nations League':40,'Friendly':20,'Other':30},
    'goal_multiplier_rule': '1 goal=1.0, 2 goals=1.5, 3 goals=1.75, 4+ goals=1.75+(gd-3)/8',
    'outputs': {
        'elo_ratings_pre_match': str(OUT_RAW.relative_to(BASE)),
        'matches_with_elo': str(OUT_PROC.relative_to(BASE)),
        'current_elo_ratings': str(OUT_CURRENT.relative_to(BASE)),
        'build_script': str(script_dst.relative_to(BASE)),
    },
    'row_counts': {
        'elo_history_completed_matches': int(len(elo_hist)),
        'matches_with_elo': int(len(merged)),
        'missing_elo_rows_in_matches_with_elo': int(merged['home_elo'].isna().sum()),
        'teams_current_ratings': int(len(current)),
    },
    'top_10_current_ratings': current.head(10).to_dict(orient='records'),
}
OUT_META.write_text(json.dumps(meta, indent=2), encoding='utf-8')

readme = f"""# Elo Feature Layer for Data Camp World Cup Predictor

This layer adds pre-match Elo-style team-strength features to the international-results dataset.

## Important caveat

These are **locally calculated Elo-style ratings**, not a direct export from eloratings.net. They use the same general idea of Elo updating: expected result from rating difference, home advantage, tournament weight, and goal-difference multiplier. This is suitable for a reproducible university/data-science pipeline because every rating can be regenerated from the match-results file.

## Inputs

- `data/raw/results_full.csv`
- `data/processed/matches_base.csv`

## Outputs

- `data/raw/elo_ratings_pre_match.csv` — pre/post Elo for every completed historical match in the source file.
- `data/processed/matches_with_elo.csv` — modelling table for 2014-2026 with Elo features merged.
- `data/processed/current_elo_ratings.csv` — latest rating table after all completed matches available in the source.
- `src/features/build_elo_features.py` — reproducible build script.
- `elo_metadata.json` — build parameters and counts.

## Main modelling columns added

- `home_elo`
- `away_elo`
- `elo_diff`
- `elo_prob_home_win_proxy`
- `elo_k`
- `elo_goal_multiplier`
- `home_elo_change`
- `away_elo_change`
- `home_elo_post`
- `away_elo_post`

## Counts

- Historical completed matches used for Elo warm-up: {len(elo_hist):,}
- Rows in `matches_with_elo.csv`: {len(merged):,}
- Missing Elo rows after merge: {int(merged['home_elo'].isna().sum()):,}
- Rated teams: {len(current):,}

## Top 10 latest local Elo ratings

{current.head(10).to_markdown(index=False)}
"""
OUT_README.write_text(readme, encoding='utf-8')

print(json.dumps(meta['row_counts'], indent=2))
print(current.head(10).to_string(index=False))
print('WROTE', OUT_PROC)
