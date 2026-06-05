import argparse
from pathlib import Path
import pandas as pd


def find_match(df, team_a, team_b):
    team_a_lower = team_a.lower()
    team_b_lower = team_b.lower()

    match = df[
        (
            (df["team_a"].str.lower() == team_a_lower)
            & (df["team_b"].str.lower() == team_b_lower)
        )
        |
        (
            (df["team_a"].str.lower() == team_b_lower)
            & (df["team_b"].str.lower() == team_a_lower)
        )
    ]

    return match


def main():
    parser = argparse.ArgumentParser(
        description="Check World Cup 2026 group-stage match probabilities."
    )
    parser.add_argument("--team-a", required=True, help="First team name")
    parser.add_argument("--team-b", required=True, help="Second team name")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    prob_path = root / "data" / "worldcup_2026" / "model_v5_group_match_probabilities.csv"

    if not prob_path.exists():
        raise FileNotFoundError(f"Could not find probability file: {prob_path}")

    df = pd.read_csv(prob_path)

    match = find_match(df, args.team_a, args.team_b)

    if match.empty:
        print(f"No group-stage match found for: {args.team_a} vs {args.team_b}")
        print()
        print("Tip: available team names include:")
        teams = sorted(set(df["team_a"]).union(set(df["team_b"])))
        for team in teams:
            print(f"  - {team}")
        return

    row = match.iloc[0]

    team_a = row["team_a"]
    team_b = row["team_b"]

    print()
    print(f"{team_a} vs {team_b}")
    print("-" * (len(team_a) + len(team_b) + 4))

    if "match_date" in row:
        print(f"Date: {row['match_date']}")
    elif "date" in row:
        print(f"Date: {row['date']}")

    if "stadium" in row and "city" in row:
        print(f"Venue: {row['stadium']}, {row['city']}")
    elif "venue" in row:
        print(f"Venue: {row['venue']}")

    if "group" in row:
        print(f"Group: {row['group']}")

    print()
    print(f"{team_a} win: {row['p_team_a_win']:.2%}")
    print(f"Draw: {row['p_draw']:.2%}")
    print(f"{team_b} win: {row['p_team_b_win']:.2%}")

    print()
    most_likely = row["most_likely_result"]

    if most_likely == "team_a_win":
        readable = f"{team_a} win"
    elif most_likely == "team_b_win":
        readable = f"{team_b} win"
    elif most_likely == "draw":
        readable = "Draw"
    else:
        readable = str(most_likely)

    print(f"Most likely result: {readable}")
    print()


if __name__ == "__main__":
    main()