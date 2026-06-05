"""Build date-safe confederation features for the Data Camp football dataset.

Input:
  data/processed/matches_with_elo_fifa_form.csv

Output:
  data/processed/matches_with_elo_fifa_form_confed.csv
  data/processed/team_confederation_mapping.csv
  data/processed/confederation_strength_snapshot.csv
  confederation_metadata.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INFILE = ROOT / "data/processed/matches_with_elo_fifa_form.csv"
OUTFILE = ROOT / "data/processed/matches_with_elo_fifa_form_confed.csv"
MAPFILE = ROOT / "data/processed/team_confederation_mapping.csv"
SNAPSHOT_FILE = ROOT / "data/processed/confederation_strength_snapshot.csv"
META_FILE = ROOT / "confederation_metadata.json"
README_FILE = ROOT / "README_confederation.md"

CONFED_FULL_NAMES = {
    "AFC": "Asian Football Confederation",
    "CAF": "Confederation of African Football",
    "CONCACAF": "Confederation of North, Central American and Caribbean Association Football",
    "CONMEBOL": "Confederación Sudamericana de Fútbol",
    "OFC": "Oceania Football Confederation",
    "UEFA": "Union of European Football Associations",
    "NON_FIFA": "Non-FIFA / regional / unofficial team",
    "UNKNOWN": "Unknown or unmapped",
}

AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia",
    "China PR", "Guam", "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
    "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia",
    "Myanmar", "Nepal", "North Korea", "Northern Mariana Islands", "Oman", "Pakistan", "Palestine",
    "Philippines", "Qatar", "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria",
    "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates",
    "Uzbekistan", "Vietnam", "Yemen",
}
CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon",
    "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "DR Congo", "Djibouti",
    "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mayotte", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Réunion", "Rwanda", "São Tomé and Príncipe",
    "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan",
    "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zanzibar", "Zimbabwe", "Saint Helena",
}
CONCACAF = {
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda",
    "Bonaire", "British Virgin Islands", "Canada", "Cayman Islands", "Costa Rica", "Cuba", "Curaçao",
    "Dominica", "Dominican Republic", "El Salvador", "French Guiana", "Grenada", "Guadeloupe",
    "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Montserrat",
    "Nicaragua", "Panama", "Puerto Rico", "Saint Barthélemy", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Martin", "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Islands", "United States", "United States Virgin Islands",
}
CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela",
}
OFC = {
    "American Samoa", "Cook Islands", "Fiji", "Marshall Islands", "New Caledonia", "New Zealand",
    "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
}
UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark",
    "England", "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar",
    "Greece", "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro", "Netherlands",
    "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal", "Republic of Ireland",
    "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
}

# Known aliases from venue/country fields or legacy names.
ALIASES = {
    "United States of America": "United States",
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "Ireland": "Republic of Ireland",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "Türkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Vietnam Republic": "Vietnam",
}

SET_TO_CONFED = []
for name, s in [("AFC", AFC), ("CAF", CAF), ("CONCACAF", CONCACAF), ("CONMEBOL", CONMEBOL), ("OFC", OFC), ("UEFA", UEFA)]:
    SET_TO_CONFED.extend([(team, name) for team in s])
TEAM_TO_CONFED: Dict[str, str] = dict(SET_TO_CONFED)


def canonical_name(name: object) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return ALIASES.get(name, name)


def confed_of(name: object) -> str:
    name = canonical_name(name)
    if not name:
        return "UNKNOWN"
    return TEAM_TO_CONFED.get(name, "NON_FIFA")


class RunningStats:
    def __init__(self) -> None:
        self.appearances = 0
        self.points = 0.0
        self.goals_for = 0.0
        self.goals_against = 0.0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.elo_sum = 0.0
        self.elo_count = 0

    def update(self, points: float, gf: float, ga: float, elo_pre: float | None = None) -> None:
        self.appearances += 1
        self.points += points
        self.goals_for += gf
        self.goals_against += ga
        if points == 3:
            self.wins += 1
        elif points == 1:
            self.draws += 1
        else:
            self.losses += 1
        if elo_pre is not None and not pd.isna(elo_pre):
            self.elo_sum += float(elo_pre)
            self.elo_count += 1

    def features(self) -> dict:
        n = self.appearances
        if n == 0:
            return {
                "appearances": 0,
                "points_per_match": np.nan,
                "goal_diff_per_match": np.nan,
                "goals_for_per_match": np.nan,
                "goals_against_per_match": np.nan,
                "win_rate": np.nan,
                "draw_rate": np.nan,
                "loss_rate": np.nan,
                "avg_pre_match_elo": np.nan,
            }
        return {
            "appearances": n,
            "points_per_match": self.points / n,
            "goal_diff_per_match": (self.goals_for - self.goals_against) / n,
            "goals_for_per_match": self.goals_for / n,
            "goals_against_per_match": self.goals_against / n,
            "win_rate": self.wins / n,
            "draw_rate": self.draws / n,
            "loss_rate": self.losses / n,
            "avg_pre_match_elo": self.elo_sum / self.elo_count if self.elo_count else np.nan,
        }


def result_points(gf: int, ga: int) -> tuple[int, int]:
    if gf > ga:
        return 3, 0
    if gf < ga:
        return 0, 3
    return 1, 1


def update_stats_for_row(row: pd.Series, overall: dict, inter: dict) -> None:
    hc, ac = row["home_confederation"], row["away_confederation"]
    hg, ag = int(row["home_score"]), int(row["away_score"])
    hp, ap = result_points(hg, ag)
    if hc not in {"UNKNOWN", "NON_FIFA"}:
        overall[hc].update(hp, hg, ag, row.get("home_elo_pre", row.get("home_elo")))
    if ac not in {"UNKNOWN", "NON_FIFA"}:
        overall[ac].update(ap, ag, hg, row.get("away_elo_pre", row.get("away_elo")))
    if hc != ac and hc not in {"UNKNOWN", "NON_FIFA"} and ac not in {"UNKNOWN", "NON_FIFA"}:
        inter[hc].update(hp, hg, ag, row.get("home_elo_pre", row.get("home_elo")))
        inter[ac].update(ap, ag, hg, row.get("away_elo_pre", row.get("away_elo")))


def main() -> None:
    df = pd.read_csv(INFILE, parse_dates=["date"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    df["home_confederation"] = df["home_team"].map(confed_of)
    df["away_confederation"] = df["away_team"].map(confed_of)
    df["same_confederation"] = df["home_confederation"].eq(df["away_confederation"])
    df["confederation_pair"] = df["home_confederation"] + "_vs_" + df["away_confederation"]
    df["home_confederation_full"] = df["home_confederation"].map(CONFED_FULL_NAMES)
    df["away_confederation_full"] = df["away_confederation"].map(CONFED_FULL_NAMES)
    df["host_confederation"] = df["country"].map(confed_of)
    df["home_confed_matches_host_confed"] = df["home_confederation"].eq(df["host_confederation"])
    df["away_confed_matches_host_confed"] = df["away_confederation"].eq(df["host_confederation"])

    overall = defaultdict(RunningStats)
    inter = defaultdict(RunningStats)

    # Warm up confederation running statistics with all completed matches before the
    # modelling window. This gives January 2014 matches a realistic historical prior
    # while still avoiding any leakage from the target match itself or future results.
    full_results_path = ROOT / "data/raw/results_full.csv"
    warmup_rows_used = 0
    if full_results_path.exists():
        warm = pd.read_csv(full_results_path, parse_dates=["date"])
        warm = warm[warm["date"] < df["date"].min()].copy()
        warm["home_confederation"] = warm["home_team"].map(confed_of)
        warm["away_confederation"] = warm["away_team"].map(confed_of)
        warm = warm.dropna(subset=["home_score", "away_score"])
        warm = warm.sort_values(["date", "home_team", "away_team"])
        for _, row in warm.iterrows():
            update_stats_for_row(row, overall, inter)
        warmup_rows_used = int(len(warm))

    feature_rows = []
    for date, day in df.groupby("date", sort=True):
        # Feature creation first: no same-day matches can leak into another same-day match.
        for idx, row in day.iterrows():
            rec = {"_idx": idx}
            hc, ac = row["home_confederation"], row["away_confederation"]
            for side, conf in [("home", hc), ("away", ac)]:
                f = overall[conf].features()
                fi = inter[conf].features()
                for k, v in f.items():
                    rec[f"{side}_confed_{k}_prior"] = v
                for k, v in fi.items():
                    rec[f"{side}_confed_inter_{k}_prior"] = v
            # Pair-specific prior metrics are directional from home confed perspective.
            h = overall[hc].features(); a = overall[ac].features()
            hi = inter[hc].features(); ai = inter[ac].features()
            rec["confed_points_per_match_diff_prior"] = h["points_per_match"] - a["points_per_match"] if pd.notna(h["points_per_match"]) and pd.notna(a["points_per_match"]) else np.nan
            rec["confed_goal_diff_per_match_diff_prior"] = h["goal_diff_per_match"] - a["goal_diff_per_match"] if pd.notna(h["goal_diff_per_match"]) and pd.notna(a["goal_diff_per_match"]) else np.nan
            rec["confed_avg_elo_diff_prior"] = h["avg_pre_match_elo"] - a["avg_pre_match_elo"] if pd.notna(h["avg_pre_match_elo"]) and pd.notna(a["avg_pre_match_elo"]) else np.nan
            rec["confed_inter_points_per_match_diff_prior"] = hi["points_per_match"] - ai["points_per_match"] if pd.notna(hi["points_per_match"]) and pd.notna(ai["points_per_match"]) else np.nan
            rec["confed_inter_goal_diff_per_match_diff_prior"] = hi["goal_diff_per_match"] - ai["goal_diff_per_match"] if pd.notna(hi["goal_diff_per_match"]) and pd.notna(ai["goal_diff_per_match"]) else np.nan
            feature_rows.append(rec)
        # Update after all matches for this date.
        for _, row in day.iterrows():
            update_stats_for_row(row, overall, inter)

    feat = pd.DataFrame(feature_rows).set_index("_idx").sort_index()
    out = pd.concat([df, feat], axis=1)

    # Fill neutral priors for model readiness while retaining raw prior columns.
    prior_cols = [c for c in out.columns if c.endswith("_prior") or "_diff_prior" in c]
    for c in prior_cols:
        filled = c + "_filled"
        if c.endswith("appearances_prior"):
            out[filled] = out[c].fillna(0)
        elif "points_per_match" in c:
            out[filled] = out[c].fillna(1.0)  # draw-level neutral prior
        elif "win_rate" in c or "draw_rate" in c or "loss_rate" in c:
            out[filled] = out[c].fillna(1/3)
        elif "avg_pre_match_elo" in c:
            out[filled] = out[c].fillna(1500.0)
        else:
            out[filled] = out[c].fillna(0.0)

    # Coverage/mapping files.
    teams = sorted(set(out["home_team"]) | set(out["away_team"]))
    mapping = pd.DataFrame({"team": teams})
    mapping["confederation"] = mapping["team"].map(confed_of)
    mapping["confederation_full"] = mapping["confederation"].map(CONFED_FULL_NAMES)
    mapping["is_fifa_or_confed_association"] = ~mapping["confederation"].isin(["NON_FIFA", "UNKNOWN"])

    snapshot = []
    for name in ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA", "NON_FIFA", "UNKNOWN"]:
        f = overall[name].features(); fi = inter[name].features()
        snapshot.append({
            "confederation": name,
            "confederation_full": CONFED_FULL_NAMES[name],
            **{f"overall_{k}": v for k, v in f.items()},
            **{f"inter_confed_{k}": v for k, v in fi.items()},
        })
    snapshot_df = pd.DataFrame(snapshot)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTFILE, index=False)
    mapping.to_csv(MAPFILE, index=False)
    snapshot_df.to_csv(SNAPSHOT_FILE, index=False)

    missing_home = int(out["home_confederation"].isin(["UNKNOWN", "NON_FIFA"]).sum())
    missing_away = int(out["away_confederation"].isin(["UNKNOWN", "NON_FIFA"]).sum())
    both_known = int((~out["home_confederation"].isin(["UNKNOWN", "NON_FIFA"]) & ~out["away_confederation"].isin(["UNKNOWN", "NON_FIFA"])).sum())
    metadata = {
        "input_file": str(INFILE.relative_to(ROOT)),
        "output_file": str(OUTFILE.relative_to(ROOT)),
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "new_columns": int(out.shape[1] - df.shape[1]),
        "unique_teams": int(len(teams)),
        "mapped_fifa_or_association_teams": int(mapping["is_fifa_or_confed_association"].sum()),
        "non_fifa_or_unknown_teams": int((~mapping["is_fifa_or_confed_association"]).sum()),
        "matches_both_teams_known_confederation": both_known,
        "home_non_fifa_or_unknown_rows": missing_home,
        "away_non_fifa_or_unknown_rows": missing_away,
        "warmup_rows_before_model_window": warmup_rows_used,
        "method": "Date-safe running confederation form/strength with pre-2014 historical warm-up. Features are computed before updating same-date results.",
        "confederations": CONFED_FULL_NAMES,
    }
    META_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    README_FILE.write_text(f"""# Confederation feature layer\n\nCreated `data/processed/matches_with_elo_fifa_form_confed.csv` from `data/processed/matches_with_elo_fifa_form.csv`.\n\n## Method\n\nTeams were assigned to one of the six FIFA football confederations when they are FIFA members or widely used association/associate teams in the dataset. Regional and unofficial teams were kept as `NON_FIFA` rather than being forced into a FIFA confederation.\n\nThe strength features are date-safe. For each match date, all confederation prior features are calculated first. Only after that are the results from that date used to update the running confederation statistics.\n\n## Main columns\n\n- `home_confederation`, `away_confederation`\n- `same_confederation`, `confederation_pair`\n- `host_confederation`\n- `home_confed_matches_host_confed`, `away_confed_matches_host_confed`\n- `home_confed_points_per_match_prior`, `away_confed_points_per_match_prior`\n- `home_confed_goal_diff_per_match_prior`, `away_confed_goal_diff_per_match_prior`\n- `home_confed_avg_pre_match_elo_prior`, `away_confed_avg_pre_match_elo_prior`\n- `confed_points_per_match_diff_prior`\n- `confed_goal_diff_per_match_diff_prior`\n- `confed_avg_elo_diff_prior`\n- `confed_inter_points_per_match_diff_prior`\n- `confed_inter_goal_diff_per_match_diff_prior`\n\nFilled versions ending in `_filled` are included for direct modelling.\n\n## Coverage\n\n- Rows: {len(out):,}\n- Columns: {out.shape[1]:,}\n- Unique teams: {len(teams):,}\n- FIFA/confederation-associated teams mapped: {int(mapping['is_fifa_or_confed_association'].sum()):,}\n- Non-FIFA/unknown teams: {int((~mapping['is_fifa_or_confed_association']).sum()):,}\n- Matches where both teams have known confederation: {both_known:,}\n\n## Important note\n\nThe confederation strength values are empirical features from this modelling dataset, not official FIFA ratings. They are useful as model features because they summarize the historical strength of teams' regional football environments before each match.\n""", encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
