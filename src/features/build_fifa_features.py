#!/usr/bin/env python3
"""Build date-aware FIFA ranking features for the Data Camp international football dataset.

The script expects:
- data/processed/matches_with_elo.csv
- data/raw/fifa_rankings_raw.csv from Dato-Futbol/fifa-ranking

It outputs:
- data/raw/fifa_rankings_processed.csv
- data/processed/matches_with_elo_fifa.csv
- data/processed/fifa_team_coverage.csv
- fifa_metadata.json
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MATCHES_PATH = ROOT / "data" / "processed" / "matches_with_elo.csv"
FIFA_RAW_PATH = ROOT / "data" / "raw" / "fifa_rankings_raw.csv"
FIFA_PROCESSED_PATH = ROOT / "data" / "raw" / "fifa_rankings_processed.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "matches_with_elo_fifa.csv"
COVERAGE_PATH = ROOT / "data" / "processed" / "fifa_team_coverage.csv"
METADATA_PATH = ROOT / "fifa_metadata.json"

ALIASES: Dict[str, str] = {
    # International results naming -> FIFA ranking naming/key harmonisation
    "usa": "united states",
    "united states of america": "united states",
    "united states virgin islands": "us virgin islands",
    "u s virgin islands": "us virgin islands",
    "ir iran": "iran",
    "iran islamic republic of": "iran",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "korea dpr": "north korea",
    "dpr korea": "north korea",
    "china pr": "china",
    "pr china": "china",
    "congo dr": "dr congo",
    "democratic republic of congo": "dr congo",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "cape verde islands": "cape verde",
    "eswatini": "swaziland",
    "fy r macedonia": "north macedonia",
    "fyr macedonia": "north macedonia",
    "macedonia": "north macedonia",
    "republic of ireland": "ireland",
    "st kitts and nevis": "saint kitts and nevis",
    "st vincent and the grenadines": "saint vincent and the grenadines",
    "st lucia": "saint lucia",
    "sao tome and principe": "sao tome and principe",
    "kyrgyz republic": "kyrgyzstan",
    "brunei darussalam": "brunei",
    "east timor": "timor leste",
    "chinese taipei": "taiwan",
    "hong kong china": "hong kong",
    "bosnia herzegovina": "bosnia and herzegovina",
    "the gambia": "gambia",
    "viet nam": "vietnam",
    "lao": "laos",
    "moldova republic": "moldova",
    "uae": "united arab emirates",
}


def normalise_team(value: str) -> str:
    s = str(value).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", "and")
    s = re.sub(r"\([^)]*\)", "", s)  # remove e.g. '(unranked)'
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return ALIASES.get(s, s)


def build_processed_fifa(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["ranking_date"] = pd.to_datetime(raw["date"])
    raw["team_clean"] = raw["team"].astype(str).str.replace(r"\s*\(unranked\)\s*$", "", regex=True).str.strip()
    raw["team_key"] = raw["team_clean"].map(normalise_team)
    raw["team_short"] = raw["team_short"].astype(str)
    raw["fifa_points"] = pd.to_numeric(raw["total_points"], errors="coerce")

    # Calculate rank from points for each publication date. This preserves ties with the same rank.
    raw["fifa_rank"] = raw.groupby("ranking_date")["fifa_points"].rank(method="min", ascending=False)
    raw.loc[raw["fifa_points"].isna(), "fifa_rank"] = np.nan
    raw["fifa_rank"] = raw["fifa_rank"].astype("Int64")

    raw = raw.sort_values(["team_key", "ranking_date"])
    raw["previous_fifa_rank"] = raw.groupby("team_key")["fifa_rank"].shift(1).astype("Int64")
    raw["previous_fifa_points"] = raw.groupby("team_key")["fifa_points"].shift(1)
    raw["fifa_rank_change"] = raw["previous_fifa_rank"] - raw["fifa_rank"]
    raw["fifa_points_change"] = raw["fifa_points"] - raw["previous_fifa_points"]

    cols = [
        "ranking_date",
        "team_clean",
        "team_key",
        "team_short",
        "fifa_rank",
        "fifa_points",
        "previous_fifa_rank",
        "previous_fifa_points",
        "fifa_rank_change",
        "fifa_points_change",
    ]
    return raw[cols].sort_values(["ranking_date", "fifa_rank", "team_clean"], na_position="last")


def lookup_team_rankings(matches: pd.DataFrame, fifa: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return latest ranking <= match date for home or away team."""
    team_col = f"{side}_team"
    work = matches[["match_id", "date", team_col]].copy()
    work["team_key"] = work[team_col].map(normalise_team)

    out_parts = []
    fifa_by_team = {team: group.sort_values("ranking_date") for team, group in fifa.groupby("team_key")}

    for team_key, group in work.groupby("team_key", sort=False):
        rank_hist = fifa_by_team.get(team_key)
        g = group.sort_values("date")
        if rank_hist is None or rank_hist.empty:
            missing = g[["match_id", "date", team_col, "team_key"]].copy()
            for col in ["ranking_date", "team_clean", "team_short", "fifa_rank", "fifa_points", "fifa_rank_change", "fifa_points_change"]:
                missing[col] = np.nan
            out_parts.append(missing)
            continue
        merged = pd.merge_asof(
            g.sort_values("date"),
            rank_hist[["ranking_date", "team_clean", "team_short", "fifa_rank", "fifa_points", "fifa_rank_change", "fifa_points_change"]].sort_values("ranking_date"),
            left_on="date",
            right_on="ranking_date",
            direction="backward",
        )
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True).sort_values("match_id")
    out = out.rename(columns={
        "team_key": f"{side}_fifa_team_key",
        "ranking_date": f"{side}_fifa_rank_date",
        "team_clean": f"{side}_fifa_team_name",
        "team_short": f"{side}_fifa_team_code",
        "fifa_rank": f"{side}_fifa_rank",
        "fifa_points": f"{side}_fifa_points",
        "fifa_rank_change": f"{side}_fifa_rank_change",
        "fifa_points_change": f"{side}_fifa_points_change",
    })
    out[f"{side}_fifa_rank_missing"] = out[f"{side}_fifa_rank"].isna()
    out[f"{side}_fifa_days_since_update"] = (out["date"] - out[f"{side}_fifa_rank_date"]).dt.days
    return out[[
        "match_id",
        f"{side}_fifa_team_key",
        f"{side}_fifa_rank_date",
        f"{side}_fifa_team_name",
        f"{side}_fifa_team_code",
        f"{side}_fifa_rank",
        f"{side}_fifa_points",
        f"{side}_fifa_rank_change",
        f"{side}_fifa_points_change",
        f"{side}_fifa_rank_missing",
        f"{side}_fifa_days_since_update",
    ]]


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    if "match_id" not in matches.columns:
        matches = matches.reset_index(drop=True)
        matches.insert(0, "match_id", np.arange(1, len(matches) + 1))

    raw_fifa = pd.read_csv(FIFA_RAW_PATH)
    fifa = build_processed_fifa(raw_fifa)
    FIFA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fifa.to_csv(FIFA_PROCESSED_PATH, index=False)

    home = lookup_team_rankings(matches, fifa, "home")
    away = lookup_team_rankings(matches, fifa, "away")
    combined = matches.merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")

    max_rank = int(fifa["fifa_rank"].max(skipna=True))
    rank_fill = max_rank + 100
    combined["home_fifa_rank_filled"] = combined["home_fifa_rank"].fillna(rank_fill).astype(int)
    combined["away_fifa_rank_filled"] = combined["away_fifa_rank"].fillna(rank_fill).astype(int)
    combined["home_fifa_points_filled"] = combined["home_fifa_points"].fillna(0.0)
    combined["away_fifa_points_filled"] = combined["away_fifa_points"].fillna(0.0)

    # Lower FIFA rank is better. Positive rank_diff means home team is ranked better.
    combined["fifa_rank_diff"] = combined["away_fifa_rank"] - combined["home_fifa_rank"]
    combined["fifa_rank_diff_filled"] = combined["away_fifa_rank_filled"] - combined["home_fifa_rank_filled"]
    combined["fifa_points_diff"] = combined["home_fifa_points"] - combined["away_fifa_points"]
    combined["fifa_points_diff_filled"] = combined["home_fifa_points_filled"] - combined["away_fifa_points_filled"]

    latest_rank_date = fifa["ranking_date"].max()
    combined["fifa_ranking_source_latest_date"] = latest_rank_date.date().isoformat()
    combined["home_fifa_stale_over_90_days"] = combined["home_fifa_days_since_update"].gt(90)
    combined["away_fifa_stale_over_90_days"] = combined["away_fifa_days_since_update"].gt(90)

    combined.to_csv(OUTPUT_PATH, index=False)

    # Coverage table by match team.
    teams = pd.DataFrame({"team": sorted(set(combined["home_team"]).union(set(combined["away_team"])))})
    teams["team_key"] = teams["team"].map(normalise_team)
    ranked_keys = set(fifa.loc[fifa["fifa_rank"].notna(), "team_key"])
    teams["has_fifa_history"] = teams["team_key"].isin(ranked_keys)
    match_counts = pd.concat([
        combined["home_team"].rename("team"),
        combined["away_team"].rename("team"),
    ]).value_counts().rename_axis("team").reset_index(name="match_count_2014_2026")
    teams = teams.merge(match_counts, on="team", how="left").sort_values(["has_fifa_history", "match_count_2014_2026", "team"], ascending=[True, False, True])
    teams.to_csv(COVERAGE_PATH, index=False)

    metadata = {
        "source": "Dato-Futbol/fifa-ranking ranking_fifa_historical.csv",
        "source_url": "https://github.com/Dato-Futbol/fifa-ranking",
        "raw_file": str(FIFA_RAW_PATH.relative_to(ROOT)),
        "processed_ranking_file": str(FIFA_PROCESSED_PATH.relative_to(ROOT)),
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "ranking_date_min": fifa["ranking_date"].min().date().isoformat(),
        "ranking_date_max": latest_rank_date.date().isoformat(),
        "ranking_publication_dates": int(fifa["ranking_date"].nunique()),
        "ranking_rows_processed": int(len(fifa)),
        "ranking_team_keys": int(fifa["team_key"].nunique()),
        "matches_rows": int(len(combined)),
        "matches_with_both_fifa_rank_available": int((~combined["home_fifa_rank_missing"] & ~combined["away_fifa_rank_missing"]).sum()),
        "matches_with_either_fifa_rank_missing": int((combined["home_fifa_rank_missing"] | combined["away_fifa_rank_missing"]).sum()),
        "home_fifa_missing_rows": int(combined["home_fifa_rank_missing"].sum()),
        "away_fifa_missing_rows": int(combined["away_fifa_rank_missing"].sum()),
        "matches_after_latest_ranking_date": int((combined["date"] > latest_rank_date).sum()),
        "rank_fill_value_for_modeling": rank_fill,
        "notes": [
            "The source ranking dataset ends on 2024-09-19, so later matches reuse each team's latest available ranking and are flagged via days_since_update/stale columns.",
            "Non-FIFA or unranked teams keep missing raw FIFA rank/points and receive filled modelling columns plus missingness flags.",
            "fifa_rank_diff = away_fifa_rank - home_fifa_rank, so larger positive values favour the home team because lower FIFA rank is better.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
