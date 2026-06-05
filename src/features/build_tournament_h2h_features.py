#!/usr/bin/env python3
"""Build tournament-experience and head-to-head features for Data Camp football dataset.

Input:  data/processed/matches_with_elo_fifa_form_confed.csv
        data/raw/results_full.csv for pre-2014 warm-up history
Output: data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv
        data/processed/tournament_h2h_feature_coverage.csv
        tournament_h2h_metadata.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "matches_with_elo_fifa_form_confed.csv"
FULL_RESULTS = ROOT / "data" / "raw" / "results_full.csv"
OUTPUT = ROOT / "data" / "processed" / "matches_with_elo_fifa_form_confed_exp_h2h.csv"
COVERAGE = ROOT / "data" / "processed" / "tournament_h2h_feature_coverage.csv"
META = ROOT / "tournament_h2h_metadata.json"
README = ROOT / "README_tournament_h2h.md"

MAJOR_CONTINENTAL = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "Oceania Nations Cup",
    "OFC Nations Cup",
}
NATIONS_LEAGUE = {
    "UEFA Nations League",
    "CONCACAF Nations League",
}


def tournament_family(t: str) -> str:
    t = str(t)
    if t == "FIFA World Cup":
        return "world_cup_finals"
    if "FIFA World Cup qualification" in t:
        return "world_cup_qualification"
    if t in MAJOR_CONTINENTAL:
        return "continental_finals"
    if "qualification" in t.lower():
        return "qualification"
    if t in NATIONS_LEAGUE or "Nations League" in t:
        return "nations_league"
    if t == "Friendly":
        return "friendly"
    return "other"


def tournament_importance(t: str) -> int:
    fam = tournament_family(t)
    if fam == "world_cup_finals":
        return 5
    if fam == "continental_finals":
        return 4
    if fam in {"world_cup_qualification", "qualification", "nations_league"}:
        return 3
    if fam == "friendly":
        return 1
    return 2


def points_for(gf: int, ga: int) -> int:
    return 3 if gf > ga else 1 if gf == ga else 0


@dataclass
class TeamStats:
    total_matches: int = 0
    first_match_date: pd.Timestamp | None = None
    last_match_date: pd.Timestamp | None = None
    tournament_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    family_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    high_importance_matches: int = 0
    importance_sum: float = 0.0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    major_matches: int = 0
    major_points: int = 0
    major_gd: int = 0
    world_cup_matches: int = 0
    continental_matches: int = 0
    qualifier_matches: int = 0
    nations_league_matches: int = 0
    last_major_date: pd.Timestamp | None = None

    def snapshot(self, date: pd.Timestamp, tournament: str) -> dict:
        fam = tournament_family(tournament)
        years_since_first = np.nan if self.first_match_date is None else (date - self.first_match_date).days / 365.25
        days_since_last_major = np.nan if self.last_major_date is None else (date - self.last_major_date).days
        return {
            "exp_total_matches_prior": self.total_matches,
            "exp_same_tournament_matches_prior": self.tournament_counts.get(tournament, 0),
            "exp_same_family_matches_prior": self.family_counts.get(fam, 0),
            "exp_world_cup_matches_prior": self.world_cup_matches,
            "exp_continental_matches_prior": self.continental_matches,
            "exp_qualifier_matches_prior": self.qualifier_matches,
            "exp_nations_league_matches_prior": self.nations_league_matches,
            "exp_major_matches_prior": self.major_matches,
            "exp_major_points_per_match_prior": self.major_points / self.major_matches if self.major_matches else np.nan,
            "exp_major_goal_diff_per_match_prior": self.major_gd / self.major_matches if self.major_matches else np.nan,
            "exp_points_per_match_prior": self.points / self.total_matches if self.total_matches else np.nan,
            "exp_goal_diff_per_match_prior": (self.goals_for - self.goals_against) / self.total_matches if self.total_matches else np.nan,
            "exp_avg_tournament_importance_prior": self.importance_sum / self.total_matches if self.total_matches else np.nan,
            "exp_high_importance_matches_prior": self.high_importance_matches,
            "exp_years_since_first_match": years_since_first,
            "exp_days_since_last_major_match": days_since_last_major,
        }

    def update(self, date: pd.Timestamp, tournament: str, gf: int, ga: int) -> None:
        fam = tournament_family(tournament)
        imp = tournament_importance(tournament)
        if self.first_match_date is None or date < self.first_match_date:
            self.first_match_date = date
        if self.last_match_date is None or date > self.last_match_date:
            self.last_match_date = date
        self.total_matches += 1
        self.tournament_counts[tournament] += 1
        self.family_counts[fam] += 1
        self.importance_sum += imp
        self.high_importance_matches += int(imp >= 4)
        self.goals_for += gf
        self.goals_against += ga
        self.points += points_for(gf, ga)
        if fam in {"world_cup_finals", "continental_finals"}:
            self.major_matches += 1
            self.major_points += points_for(gf, ga)
            self.major_gd += gf - ga
            self.last_major_date = date
        if fam == "world_cup_finals":
            self.world_cup_matches += 1
        elif fam == "continental_finals":
            self.continental_matches += 1
        elif fam in {"world_cup_qualification", "qualification"}:
            self.qualifier_matches += 1
        elif fam == "nations_league":
            self.nations_league_matches += 1


@dataclass
class H2HMatch:
    date: pd.Timestamp
    team1: str
    team2: str
    goals1: int
    goals2: int
    neutral: bool
    tournament: str


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([str(a), str(b)]))


def h2h_snapshot(history: list[H2HMatch], home: str, away: str, date: pd.Timestamp) -> dict:
    n = len(history)
    if n == 0:
        return {
            "h2h_matches_prior": 0,
            "h2h_home_team_wins_prior": 0,
            "h2h_draws_prior": 0,
            "h2h_home_team_losses_prior": 0,
            "h2h_home_team_points_per_match_prior": np.nan,
            "h2h_goal_diff_per_match_prior": np.nan,
            "h2h_goals_for_per_match_prior": np.nan,
            "h2h_goals_against_per_match_prior": np.nan,
            "h2h_matches_last5": 0,
            "h2h_home_team_points_per_match_last5": np.nan,
            "h2h_goal_diff_per_match_last5": np.nan,
            "h2h_home_team_win_rate_last5": np.nan,
            "h2h_days_since_last_meeting": np.nan,
            "h2h_same_tournament_matches_prior": 0,
        }
    wins = draws = losses = points = gf_sum = ga_sum = same_tournament = 0
    for m in history:
        if m.team1 == home:
            gf, ga = m.goals1, m.goals2
        else:
            gf, ga = m.goals2, m.goals1
        gf_sum += gf
        ga_sum += ga
        p = points_for(gf, ga)
        points += p
        wins += int(p == 3)
        draws += int(p == 1)
        losses += int(p == 0)
    last5 = history[-5:]
    l5_points = l5_gd = l5_wins = 0
    for m in last5:
        if m.team1 == home:
            gf, ga = m.goals1, m.goals2
        else:
            gf, ga = m.goals2, m.goals1
        p = points_for(gf, ga)
        l5_points += p
        l5_gd += gf - ga
        l5_wins += int(p == 3)
    days_since = (date - history[-1].date).days
    return {
        "h2h_matches_prior": n,
        "h2h_home_team_wins_prior": wins,
        "h2h_draws_prior": draws,
        "h2h_home_team_losses_prior": losses,
        "h2h_home_team_points_per_match_prior": points / n,
        "h2h_goal_diff_per_match_prior": (gf_sum - ga_sum) / n,
        "h2h_goals_for_per_match_prior": gf_sum / n,
        "h2h_goals_against_per_match_prior": ga_sum / n,
        "h2h_matches_last5": len(last5),
        "h2h_home_team_points_per_match_last5": l5_points / len(last5),
        "h2h_goal_diff_per_match_last5": l5_gd / len(last5),
        "h2h_home_team_win_rate_last5": l5_wins / len(last5),
        "h2h_days_since_last_meeting": days_since,
        # Exact tournament context is added in outer function because current tournament is needed.
        "h2h_same_tournament_matches_prior": 0,
    }


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["neutral"] = False
    return df.sort_values(["date", "home_team", "away_team", "tournament"]).reset_index(drop=True)


def add_update_match(team_stats: dict[str, TeamStats], h2h: dict[Tuple[str, str], list[H2HMatch]], row: pd.Series) -> None:
    date = row["date"]
    ht, at = str(row["home_team"]), str(row["away_team"])
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    tournament = str(row["tournament"])
    team_stats[ht].update(date, tournament, hs, as_)
    team_stats[at].update(date, tournament, as_, hs)
    key = pair_key(ht, at)
    h2h[key].append(H2HMatch(date=date, team1=ht, team2=at, goals1=hs, goals2=as_, neutral=bool(row.get("neutral", False)), tournament=tournament))


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "home_team", "away_team", "tournament"]).reset_index(drop=True)

    full = clean_results(pd.read_csv(FULL_RESULTS, low_memory=False))
    min_date = df["date"].min()
    warmup = full[full["date"] < min_date].copy()

    team_stats: dict[str, TeamStats] = defaultdict(TeamStats)
    h2h: dict[Tuple[str, str], list[H2HMatch]] = defaultdict(list)

    for _, row in warmup.iterrows():
        add_update_match(team_stats, h2h, row)

    feature_records = []
    # Date grouping prevents leakage among matches played on the same date.
    for date, group in df.groupby("date", sort=True):
        day_records = []
        for idx, row in group.iterrows():
            ht, at = str(row["home_team"]), str(row["away_team"])
            tournament = str(row["tournament"])
            rec = {"_idx": idx}
            home_snap = team_stats[ht].snapshot(date, tournament)
            away_snap = team_stats[at].snapshot(date, tournament)
            for k, v in home_snap.items():
                rec[f"home_{k}"] = v
            for k, v in away_snap.items():
                rec[f"away_{k}"] = v
            # Experience difference features; positive = home has more experience / stronger prior record.
            diff_pairs = [
                "exp_total_matches_prior", "exp_same_tournament_matches_prior", "exp_same_family_matches_prior",
                "exp_world_cup_matches_prior", "exp_continental_matches_prior", "exp_qualifier_matches_prior",
                "exp_nations_league_matches_prior", "exp_major_matches_prior", "exp_major_points_per_match_prior",
                "exp_major_goal_diff_per_match_prior", "exp_points_per_match_prior", "exp_goal_diff_per_match_prior",
                "exp_avg_tournament_importance_prior", "exp_high_importance_matches_prior", "exp_years_since_first_match",
            ]
            for k in diff_pairs:
                rec[f"{k}_diff"] = rec[f"home_{k}"] - rec[f"away_{k}"]
            # Lower days since major means more recent major experience, so away minus home gives positive = home more recent.
            rec["exp_major_recency_advantage"] = rec["away_exp_days_since_last_major_match"] - rec["home_exp_days_since_last_major_match"]

            hist = h2h[pair_key(ht, at)]
            hrec = h2h_snapshot(hist, ht, at, date)
            hrec["h2h_same_tournament_matches_prior"] = sum(1 for m in hist if m.tournament == tournament)
            hrec["h2h_same_family_matches_prior"] = sum(1 for m in hist if tournament_family(m.tournament) == tournament_family(tournament))
            rec.update(hrec)
            day_records.append(rec)
        feature_records.extend(day_records)
        # Update after all features for the date are captured.
        for _, row in group.iterrows():
            add_update_match(team_stats, h2h, row)

    features = pd.DataFrame(feature_records).set_index("_idx").sort_index()
    out = pd.concat([df, features], axis=1)

    # Fill model-ready versions. Missing rate-type features mean no history; use neutral prior values.
    fill_zero = [c for c in out.columns if (
        c.startswith("home_exp_") or c.startswith("away_exp_") or c.startswith("exp_") or c.startswith("h2h_")
    ) and not any(s in c for s in ["points_per_match", "goal_diff_per_match", "goals_for_per_match", "goals_against_per_match", "win_rate", "days_since", "years_since", "recency", "avg_tournament_importance"])]
    for c in fill_zero:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[f"{c}_filled"] = out[c].fillna(0)

    neutral_rate_cols = [c for c in out.columns if any(s in c for s in ["points_per_match", "goal_diff_per_match", "goals_for_per_match", "goals_against_per_match", "win_rate"])]
    for c in neutral_rate_cols:
        if (c.startswith("home_exp_") or c.startswith("away_exp_") or c.startswith("exp_") or c.startswith("h2h_")) and pd.api.types.is_numeric_dtype(out[c]):
            out[f"{c}_filled"] = out[c].fillna(0)

    recency_cols = [c for c in out.columns if ("days_since" in c or "years_since" in c or "recency" in c or "avg_tournament_importance" in c) and pd.api.types.is_numeric_dtype(out[c])]
    for c in recency_cols:
        if c.startswith("home_exp_") or c.startswith("away_exp_") or c.startswith("exp_") or c.startswith("h2h_"):
            median = float(out[c].median()) if not out[c].dropna().empty else 0.0
            out[f"{c}_filled"] = out[c].fillna(median)

    # Explicit booleans/coverage flags.
    out["both_teams_have_major_tournament_history"] = ((out["home_exp_major_matches_prior"] > 0) & (out["away_exp_major_matches_prior"] > 0)).astype(int)
    out["has_h2h_history"] = (out["h2h_matches_prior"] > 0).astype(int)
    out["has_h2h_last5"] = (out["h2h_matches_last5"] > 0).astype(int)

    out.to_csv(OUTPUT, index=False)

    coverage = pd.DataFrame([
        {"metric": "rows", "value": int(len(out))},
        {"metric": "columns", "value": int(out.shape[1])},
        {"metric": "new_columns_added", "value": int(out.shape[1] - df.shape[1])},
        {"metric": "matches_with_h2h_history", "value": int((out["h2h_matches_prior"] > 0).sum())},
        {"metric": "matches_without_h2h_history", "value": int((out["h2h_matches_prior"] == 0).sum())},
        {"metric": "both_teams_have_major_tournament_history", "value": int(out["both_teams_have_major_tournament_history"].sum())},
        {"metric": "warmup_matches_before_2014", "value": int(len(warmup))},
        {"metric": "teams_in_experience_history", "value": int(len(team_stats))},
        {"metric": "pair_histories_tracked", "value": int(len(h2h))},
    ])
    coverage.to_csv(COVERAGE, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT.relative_to(ROOT)),
        "output_file": str(OUTPUT.relative_to(ROOT)),
        "warmup_file": str(FULL_RESULTS.relative_to(ROOT)),
        "warmup_matches_before_model_window": int(len(warmup)),
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "new_columns_added": int(out.shape[1] - df.shape[1]),
        "leakage_protection": "Features are calculated before updating histories for the current match date; matches on the same date do not influence each other.",
        "feature_families": ["tournament_experience", "head_to_head"],
        "output_files": {
            "dataset": str(OUTPUT.relative_to(ROOT)),
            "coverage": str(COVERAGE.relative_to(ROOT)),
            "readme": str(README.relative_to(ROOT)),
        },
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    README.write_text(f"""# Tournament Experience and Head-to-Head Feature Layer

Input: `data/processed/matches_with_elo_fifa_form_confed.csv`

Output: `data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv`

This layer adds date-safe pre-match tournament-experience and head-to-head features. The script warms up histories using completed matches before 2014, then processes the 2014–2026 modelling window by date. It calculates features for all matches on a date before updating histories with that date's results, preventing same-day leakage.

## Main tournament-experience features

- prior total international matches
- prior same-tournament matches
- prior same-family tournament matches
- prior World Cup final-tournament matches
- prior continental final-tournament matches
- prior qualifier matches
- prior Nations League matches
- prior major-tournament points per match
- prior major-tournament goal difference per match
- average prior tournament importance
- experience difference features: home minus away

## Main head-to-head features

- prior meetings between the two teams
- home team's wins/draws/losses in the pairing, regardless of venue
- prior head-to-head points per match from the current home team's perspective
- prior head-to-head goal difference per match
- last-5 head-to-head form
- days since last meeting
- prior same-tournament and same-family meetings

## Coverage

- Rows: {len(out):,}
- Columns: {out.shape[1]:,}
- New columns added: {out.shape[1] - df.shape[1]:,}
- Matches with H2H history: {(out['h2h_matches_prior'] > 0).sum():,}
- Matches where both teams have major-tournament history: {out['both_teams_have_major_tournament_history'].sum():,}
""", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output": str(OUTPUT),
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "new_columns_added": int(out.shape[1] - df.shape[1]),
        "matches_with_h2h_history": int((out["h2h_matches_prior"] > 0).sum()),
        "both_teams_have_major_history": int(out["both_teams_have_major_tournament_history"].sum()),
    }, indent=2))

if __name__ == "__main__":
    main()
