# Streamlit Interface Layer

This interface adds a local dashboard for the FIFA World Cup 2026 Prediction Simulator.

## Main feature improvement

The `Match predictor` page now uses dependent dropdowns:

- Select `Team A` first.
- `Team B` only shows valid group-stage opponents from the same group.

Example:

- If `Norway` is selected as Team A, Team B only shows `France`, `Senegal`, and `Iraq`.
- If `Mexico` is selected as Team A, Team B only shows `South Africa`, `South Korea`, and `Czech Republic`.

This prevents impossible fixture selections such as `Iraq vs South Africa`.

## How to install

```powershell
pip install streamlit plotly
```

Make sure `requirements.txt` includes:

```text
streamlit
plotly
```

## How to run

From the project root:

```powershell
streamlit run app.py
```

Open the displayed local URL, usually:

```text
http://localhost:8501
```

## Important

Run the dashboard from the project root folder so the app can find:

```text
data/worldcup_2026/model_v5_team_probabilities.csv
data/worldcup_2026/model_v5_group_match_probabilities.csv
data/worldcup_2026/model_v5_group_finish_probabilities.csv
```
## Dashboard pages

Overview: 

```text
Shows the main project summary:

48 teams
72 group-stage matches
10,000 simulations
Model v5.0
top title probabilities
Champion probabilities

Shows the tournament advancement probabilities for each team:

group winner probability
Round of 32 probability
Round of 16 probability
quarter-final probability
semi-final probability
final probability
champion probability
Match predictor

Allows the user to select a group-stage fixture and view win/draw/loss probabilities.

The Team B dropdown is dependent on Team A. When Team A is selected, Team B only shows valid group-stage opponents from the same group.

Example:

Team A: Norway
Team B options: France, Senegal, Iraq
Group explorer

Allows the user to select a group and view:

teams in the group
group-stage fixtures
group-related probabilities
Team profile

Allows the user to select one team and view:

champion probability
final probability
semi-final probability
quarter-final probability
Round of 16 probability
Round of 32 probability
group-stage fixtures
Compare teams

Allows the user to compare two teams by overall tournament chances, not only by direct match probability.

Group-stage matches

Shows all 72 group-stage match probabilities.

Group finish probabilities

Shows simulated group finish and qualification probabilities.

Knockout matchups

Shows simulated knockout matchup frequencies and probabilities.

About

Explains the data layers, model versions, limitations, and links to GitHub and Kaggle.
```