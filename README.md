# FIFA World Cup 2026 Prediction Simulator

## Project goal

This project builds a data-driven prediction system for the **FIFA World Cup 2026**.

The goal is to estimate:

- match-level win / draw / loss probabilities
- group-stage qualification probabilities
- knockout-stage advancement probabilities
- team-level champion probabilities

The final system uses a Monte Carlo tournament simulator powered by a calibrated probability model trained on historical international football data.

The current main output is a **10,000-run World Cup 2026 simulation**.

---

## Current headline result

Using the current Model v5.0 simulator with 10,000 tournament simulations, the top title probabilities are:

| Rank | Team | Champion probability | Final probability | Semi-final probability | Quarter-final probability | Round of 32 probability |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Spain | 29.09% | 40.61% | 50.12% | 61.13% | 95.88% |
| 2 | Argentina | 11.82% | 20.83% | 30.28% | 43.02% | 93.54% |
| 3 | England | 9.38% | 16.75% | 28.91% | 43.80% | 91.72% |
| 4 | France | 7.90% | 14.95% | 29.82% | 46.49% | 91.53% |
| 5 | Portugal | 6.48% | 12.46% | 20.65% | 37.71% | 88.82% |

Main result file:

```text
data/worldcup_2026/model_v5_team_probabilities.csv
```

---

## Project structure

```text
datacamp_worldcup_predictor/
  data/
    raw/
    processed/
    modeling/
    worldcup_2026/

  models/
    baseline_model_best.joblib
    v2_model_best.joblib
    v3_temperature_calibrator.joblib
    v4_probability_ensemble.joblib

  reports/
    baseline_model_report.md
    model_v2_report.md
    model_v3_calibration_report.md
    model_v4_ensemble_report.md

  src/
    features/
      build_elo_features.py
      build_fifa_features.py
      build_recent_form_features.py
      build_confederation_features.py
      build_tournament_h2h_features.py
      train_model_v2_fast.py
      train_model_v3_calibration.py
      train_model_v4_probability_ensemble.py
      simulate_worldcup_2026_v5.py
      simulate_worldcup_2026_v5_fast.py
      simulate_worldcup_2026_v5_10000.py

  predict_match.py
  requirements.txt
  README.md
```

---

## Data used

The project uses several football data layers.

### 1. Historical international results

The base dataset contains men’s international football matches from 2014 to 2026, with older historical matches used to warm up team-strength and experience features.

Main files:

```text
data/raw/international_results.csv
data/processed/matches_base.csv
```

Core columns include:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

These data are used to calculate match outcomes, goals, tournament categories, home/neutral effects, form, head-to-head history, and team experience.

---

### 2. Elo-style team strength

A local Elo-style rating system is calculated from historical international results.

Main files:

```text
data/raw/elo_ratings_pre_match.csv
data/processed/current_elo_ratings.csv
data/processed/matches_with_elo.csv
```

Important features:

```text
home_elo
away_elo
elo_diff
elo_prob_home_win_proxy
```

The Elo system uses match importance, home advantage, and goal-difference adjustment.

---

### 3. FIFA ranking features

Historical FIFA ranking data are merged into matches using date-safe pre-match lookup.

Main files:

```text
data/raw/fifa_rankings_raw.csv
data/raw/fifa_rankings_processed.csv
data/processed/matches_with_elo_fifa.csv
```

Important features:

```text
home_fifa_rank
away_fifa_rank
home_fifa_points
away_fifa_points
fifa_rank_diff
fifa_points_diff
home_fifa_days_since_update
away_fifa_days_since_update
```

Lower FIFA rank means stronger team. Therefore:

```text
fifa_rank_diff = away_fifa_rank - home_fifa_rank
```

Positive values favor the home/team A side.

---

### 4. Recent form

Recent form is calculated using only matches before the current match date to avoid data leakage.

Main file:

```text
data/processed/matches_with_elo_fifa_form.csv
```

Important features include last-5 and last-10:

```text
form points
points per match
goals for
goals against
goal difference
win rate
draw rate
loss rate
weighted form
days since last match
```

---

### 5. Confederation strength

Teams are mapped to FIFA confederations or `NON_FIFA` where appropriate.

Main files:

```text
data/processed/team_confederation_mapping.csv
data/processed/confederation_strength_snapshot.csv
data/processed/matches_with_elo_fifa_form_confed.csv
```

Important features:

```text
home_confederation
away_confederation
same_confederation
confederation_pair
confed_points_per_match_diff_prior
confed_goal_diff_per_match_diff_prior
confed_inter_points_per_match_diff_prior
confed_inter_goal_diff_per_match_diff_prior
```

---

### 6. Tournament experience and head-to-head history

The richest modelling dataset includes tournament experience and prior head-to-head information.

Main file:

```text
data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv
```

Important tournament experience features:

```text
total previous international matches
World Cup experience
continental tournament experience
major tournament points per match
same tournament experience
```

Important head-to-head features:

```text
h2h_matches_prior
h2h_home_team_points_per_match_prior
h2h_goal_diff_per_match_prior
h2h_matches_last5
h2h_days_since_last_meeting
```

---

### 7. FIFA World Cup 2026 fixtures

The simulator uses prepared World Cup 2026 group-stage fixtures and knockout slots.

Main files:

```text
data/worldcup_2026/worldcup_2026_group_fixtures.csv
data/worldcup_2026/worldcup_2026_knockout_slots.csv
data/worldcup_2026/worldcup_2026_all_fixtures.csv
data/worldcup_2026/worldcup_2026_groups.csv
```

The fixture layer contains:

```text
72 group-stage matches
32 knockout-stage slots
104 total matches
48 teams
12 groups
16 venues
```

Knockout slots are resolved dynamically during each simulation.

---

## Feature engineering summary

The final modelling dataset includes:

```text
match result labels
home/away goals
neutral venue indicator
tournament importance
Elo strength
FIFA ranking strength
recent form
confederation strength
inter-confederation strength
tournament experience
head-to-head history
```

The feature engineering process is designed to be time-safe. Historical features are calculated using only information available before each match.

---

## Models v1 to v5

### Model v1: First baseline

A multinomial logistic regression model trained on:

```text
Elo
FIFA ranking
recent form
confederation strength
tournament importance
neutral indicator
```

Purpose:

```text
Create a simple explainable baseline for W/D/L prediction.
```

---

### Model v2: Feature selection and stronger classifier

Model v2 introduced:

```text
curated feature selection
compact feature set
stronger classifier comparison
```

Candidate models included:

```text
logistic regression
random forest
dummy baseline
```

Best Model v2:

```text
random_forest_shallow
```

Model v2 improved hard W/D/L classification compared with the first baseline.

---

### Model v3: Probability calibration

Model v3 calibrated Model v2 using validation-set probability calibration.

Goal:

```text
preserve hard-class performance
improve log loss
improve Brier score
improve probability reliability
```

Calibration method:

```text
single-temperature scaling
```

---

### Model v4: Probability-focused ensemble

Model v4 combines:

```text
first baseline logistic model
+
calibrated Model v3 random forest
```

Formula:

```text
P_v4 = (1 - w) * P_baseline + w * P_model_v3
```

The selected validation-tuned weight was approximately:

```text
18.6% first baseline
81.4% calibrated Model v3
```

Model v4 is the best current model for probability prediction.

---

### Model v5: World Cup 2026 tournament simulator

Model v5 plugs Model v4 into the full World Cup 2026 fixture structure.

The simulator:

```text
1. Predicts all 72 group-stage matches
2. Samples W/D/L outcomes
3. Generates scorelines
4. Builds group tables
5. Selects top 2 from each group
6. Selects best 8 third-place teams
7. Resolves Round of 32 bracket slots
8. Simulates all knockout rounds
9. Resolves knockout draws with a strength-based tie-break approximation
10. Repeats the tournament many times
11. Aggregates team advancement and champion probabilities
```

Current final version:

```text
Model v5.0 = static pre-tournament strength simulator
```

That means simulated group-stage results update the tournament tables and bracket paths, but they are not fed back into the ML feature pipeline for later match probabilities.

---

## Model performance summary

The model comparison showed:

| Model | Best use |
|---|---|
| First baseline | Probability baseline |
| Model v2 | Better hard W/D/L prediction |
| Model v3 | Calibrated stronger classifier |
| Model v4 | Best probability model |
| Model v5 | Full tournament simulation |

Model v4 is used for the tournament simulator because World Cup prediction requires probability quality, not only hard class prediction.

---

## How to run on Windows

Recommended Python version:

```text
Python 3.11
```

### 1. Open PowerShell

Go to the project folder:

```powershell
cd "E:\WC _Prediction"
```

### 2. Create and activate virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the 10,000 simulation version

```powershell
python src\features\simulate_worldcup_2026_v5_10000.py
```

Expected success output:

```json
{
  "done": true,
  "n_simulations": 10000
}
```

### 5. Check the main result

```powershell
Get-Content data\worldcup_2026\model_v5_team_probabilities.csv -TotalCount 10
```

### 6. Check one group-stage match

```powershell
python predict_match.py --team-a Mexico --team-b "South Africa"
```

---

## How to run on Ubuntu / Linux

Recommended Python version:

```text
Python 3.11
```

### 1. Go to project folder

```bash
cd ~/Downloads/datacamp_worldcup_predictor
```

### 2. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If Python 3.11 is not installed:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run 10,000 simulations

```bash
python src/features/simulate_worldcup_2026_v5_10000.py
```

### 5. Check result

```bash
head data/worldcup_2026/model_v5_team_probabilities.csv
```

### 6. Check one group-stage match

```bash
python predict_match.py --team-a Mexico --team-b "South Africa"
```

---

## Match prediction CLI

The file:

```text
predict_match.py
```

allows quick lookup of group-stage match probabilities.

Example:

```bash
python predict_match.py --team-a Mexico --team-b "South Africa"
```

Example output:

```text
Mexico vs South Africa

Mexico win: 51.50%
Draw: 35.75%
South Africa win: 12.75%

Most likely result: Mexico win
```

This CLI currently uses:

```text
data/worldcup_2026/model_v5_group_match_probabilities.csv
```

So it is intended for group-stage matches whose teams are already known.

---

## Main output files

```text
data/worldcup_2026/model_v5_team_probabilities.csv
data/worldcup_2026/model_v5_champion_distribution.csv
data/worldcup_2026/model_v5_group_match_probabilities.csv
data/worldcup_2026/model_v5_group_finish_probabilities.csv
data/worldcup_2026/model_v5_knockout_matchup_probabilities.csv
data/worldcup_2026/model_v5_sample_simulated_matches_first_200_runs.csv
```

---

## Limitations

This project is a strong first complete World Cup simulator, but it has important limitations.

### 1. Static pre-tournament strength

Model v5.0 uses pre-tournament team strengths. Simulated group-stage performance does not dynamically update Elo, recent form, confidence, or momentum for knockout predictions.

### 2. Squad/player data not included yet

The current model does not include:

```text
player quality
club minutes
injuries
suspensions
squad age
squad market value
goalkeeper strength
attacking/defensive player depth
```

### 3. Betting-market probabilities not included yet

The current model does not use bookmaker odds or betting-market implied probabilities.

Betting markets can be useful because they aggregate public information, injuries, tactical expectations, and market sentiment.

### 4. Scoreline generation is approximate

Scorelines are sampled using historical scoreline patterns conditional on W/D/L outcomes. The scoreline model is not yet a full expected-goals or Poisson scoreline model.

### 5. Knockout draw resolution is approximate

If a knockout match is drawn after normal time, the simulator uses a strength-based extra-time/penalty approximation. It does not model extra time and penalties separately.

### 6. FIFA ranking freshness

The FIFA ranking layer currently depends on the available historical FIFA ranking source. Future versions should update FIFA rankings closer to the tournament.

### 7. Fixture assumptions

The simulator depends on the uploaded fixture files. If official fixtures, groups, venues, or qualified teams change, the fixture files should be updated.

---

## Future improvements

Recommended next improvements:

### Model v5.1: Dynamic tournament simulator

Update team momentum during the simulated tournament.

Possible dynamic updates:

```text
temporary Elo shift after group matches
recent tournament form
goals scored/conceded in current tournament
fatigue/rest days
confidence/momentum
injury/suspension placeholders
```

### Model v6: Player and squad-strength features

Add latest 2023–2026 player/squad data:

```text
squad market value
average player rating
club minutes
age profile
European top-league representation
attacking depth
defensive depth
goalkeeper quality
injury/suspension data
```

### Model v7: Betting-market ensemble

Add bookmaker odds and market-implied probabilities.

Potential features:

```text
home/team A implied probability
draw implied probability
away/team B implied probability
market overround-adjusted probabilities
market-vs-model disagreement
closing-line value analysis
```

### Model v8: Scoreline / expected-goals model

Build a dedicated scoreline model using:

```text
Poisson
bivariate Poisson
expected goals proxy
team attack strength
team defense strength
venue effect
tournament importance
```

### Interface layer

Add a user-friendly dashboard.

Recommended tools:

```text
Streamlit
FastAPI
Plotly
```

Potential dashboard tabs:

```text
Match predictor
Group fixtures
Group qualification probabilities
Champion probabilities
Knockout matchup explorer
Model explanation
```

---

## Reproducibility notes

For best reproducibility, use:

```text
Python 3.11
scikit-learn==1.8.0
```

The model artifacts were saved with scikit-learn 1.8.0. Newer versions may still work, but can produce compatibility warnings.

---

## Suggested project description

```text
A Monte Carlo FIFA World Cup 2026 prediction simulator using historical international results, Elo-style ratings, FIFA rankings, recent form, confederation strength, tournament experience, head-to-head history, calibrated probability modelling, and full-tournament bracket simulation.
```
