from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="FIFA World Cup 2026 Prediction Simulator",
    page_icon="⚽",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent
WC_DIR = ROOT / "data" / "worldcup_2026"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            st.error(f"Required file not found: `{path}`")
            st.stop()
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_data():
    team_probs = load_csv(WC_DIR / "model_v5_team_probabilities.csv")
    group_matches = load_csv(WC_DIR / "model_v5_group_match_probabilities.csv")
    group_finish = load_csv(WC_DIR / "model_v5_group_finish_probabilities.csv")
    champions = load_csv(WC_DIR / "model_v5_champion_distribution.csv", required=False)
    knockout_matchups = load_csv(WC_DIR / "model_v5_knockout_matchup_probabilities.csv", required=False)
    groups_file = load_csv(WC_DIR / "worldcup_2026_groups.csv", required=False)
    fixtures = load_csv(WC_DIR / "worldcup_2026_all_fixtures.csv", required=False)

    return team_probs, group_matches, group_finish, champions, knockout_matchups, groups_file, fixtures


team_probs, group_matches, group_finish, champions, knockout_matchups, groups_file, fixtures = load_data()


# ============================================================
# Helper functions
# ============================================================
def all_teams() -> list[str]:
    teams = set()
    if "team" in team_probs.columns:
        teams.update(team_probs["team"].dropna().astype(str).tolist())
    if {"team_a", "team_b"}.issubset(group_matches.columns):
        teams.update(group_matches["team_a"].dropna().astype(str).tolist())
        teams.update(group_matches["team_b"].dropna().astype(str).tolist())
    return sorted(teams)


TEAMS = all_teams()


def percent_value(x) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return ""


def format_percent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(percent_value)
    return out


def infer_group_map() -> dict[str, str]:
    group_map: dict[str, str] = {}

    if {"group", "team"}.issubset(groups_file.columns):
        for _, row in groups_file.iterrows():
            group_map[str(row["team"])] = str(row["group"])
        return group_map

    if {"group", "team_name"}.issubset(groups_file.columns):
        for _, row in groups_file.iterrows():
            group_map[str(row["team_name"])] = str(row["group"])
        return group_map

    if {"group", "team_a", "team_b"}.issubset(group_matches.columns):
        for _, row in group_matches.iterrows():
            group = str(row["group"])
            group_map[str(row["team_a"])] = group
            group_map[str(row["team_b"])] = group

    return group_map


GROUP_MAP = infer_group_map()


def get_group_for_team(team: str) -> str | None:
    return GROUP_MAP.get(team)


def teams_in_group(group: str) -> list[str]:
    if group is None:
        return []

    teams = set()
    if {"group", "team_a", "team_b"}.issubset(group_matches.columns):
        rows = group_matches[group_matches["group"].astype(str) == str(group)]
        teams.update(rows["team_a"].dropna().astype(str).tolist())
        teams.update(rows["team_b"].dropna().astype(str).tolist())
    return sorted(teams)


def get_valid_group_opponents(team: str) -> list[str]:
    group = get_group_for_team(team)
    return [t for t in teams_in_group(group) if t != team]


def find_group_match(team_a: str, team_b: str) -> pd.DataFrame:
    if not {"team_a", "team_b"}.issubset(group_matches.columns):
        return pd.DataFrame()

    return group_matches[
        ((group_matches["team_a"] == team_a) & (group_matches["team_b"] == team_b))
        | ((group_matches["team_a"] == team_b) & (group_matches["team_b"] == team_a))
    ]


def most_likely_readable(row: pd.Series) -> str:
    result = str(row.get("most_likely_result", ""))
    if result == "team_a_win":
        return f"{row.get('team_a', 'Team A')} win"
    if result == "team_b_win":
        return f"{row.get('team_b', 'Team B')} win"
    if result == "draw":
        return "Draw"
    return result


def team_probability_row(team: str) -> pd.Series | None:
    if "team" not in team_probs.columns:
        return None
    rows = team_probs[team_probs["team"] == team]
    if rows.empty:
        return None
    return rows.iloc[0]


def team_fixture_rows(team: str) -> pd.DataFrame:
    if not {"team_a", "team_b"}.issubset(group_matches.columns):
        return pd.DataFrame()
    return group_matches[(group_matches["team_a"] == team) | (group_matches["team_b"] == team)].copy()


def filtered_team_selectbox(label: str, teams: list[str], key_prefix: str, default: str | None = None) -> str:
    search = st.text_input(f"Search {label}", key=f"{key_prefix}_search")
    options = teams
    if search:
        options = [t for t in teams if search.lower() in t.lower()]
        if not options:
            st.warning("No teams match your search. Showing all teams.")
            options = teams

    default_index = 0
    if default in options:
        default_index = options.index(default)

    return st.selectbox(label, options, index=default_index, key=f"{key_prefix}_select")


def display_probability_metrics(row: pd.Series):
    metric_map = [
        ("Champion", "champion_probability"),
        ("Final", "final_probability"),
        ("Semi-final", "semi_final_probability"),
        ("Quarter-final", "quarter_final_probability"),
        ("Round of 16", "round_of_16_probability"),
        ("Round of 32", "round_of_32_probability"),
    ]

    cols = st.columns(3)
    for idx, (label, col) in enumerate(metric_map):
        value = percent_value(row[col]) if col in row else "N/A"
        cols[idx % 3].metric(label, value)


def make_advancement_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    available = ["team"] + [c for c in cols if c in df.columns]
    long_df = df[available].melt(id_vars="team", var_name="stage", value_name="probability")
    stage_labels = {
        "round_of_32_probability": "Round of 32",
        "round_of_16_probability": "Round of 16",
        "quarter_final_probability": "Quarter-final",
        "semi_final_probability": "Semi-final",
        "final_probability": "Final",
        "champion_probability": "Champion",
    }
    long_df["stage"] = long_df["stage"].map(stage_labels).fillna(long_df["stage"])
    return long_df


def normalize_probability_column(df: pd.DataFrame) -> pd.DataFrame:
    """Try to expose a readable matchup probability column if simulator file uses a different name."""
    out = df.copy()
    probability_like = [c for c in out.columns if "prob" in c.lower() or "rate" in c.lower()]
    if probability_like:
        return out
    return out


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("World Cup 2026")
page = st.sidebar.radio(
    "Choose page",
    [
        "Overview",
        "Champion probabilities",
        "Match predictor",
        "Group explorer",
        "Team profile",
        "Compare teams",
        "Group finish probabilities",
        "Knockout matchups",
        "About",
    ],
)


# ============================================================
# Header
# ============================================================
st.title("FIFA World Cup 2026 Prediction Simulator")
st.caption(
    "Monte Carlo tournament simulation using historical results, Elo-style ratings, "
    "FIFA rankings, recent form, confederation strength, tournament experience, "
    "head-to-head history, and a calibrated probability ensemble."
)


# ============================================================
# Overview
# ============================================================
if page == "Overview":
    st.header("Project overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Teams", f"{team_probs['team'].nunique() if 'team' in team_probs.columns else len(TEAMS)}")
    col2.metric("Group matches", f"{len(group_matches)}")
    col3.metric("Simulations", "10,000")
    col4.metric("Model", "v5.0")

    st.subheader("Top 10 title probabilities")

    top10 = team_probs.sort_values("champion_probability", ascending=False).head(10)
    percent_cols = [
        "champion_probability",
        "final_probability",
        "semi_final_probability",
        "quarter_final_probability",
        "round_of_16_probability",
        "round_of_32_probability",
    ]

    fig = px.bar(
        top10.sort_values("champion_probability"),
        x="champion_probability",
        y="team",
        orientation="h",
        title="Top 10 Champion Probabilities",
        labels={"champion_probability": "Champion probability", "team": "Team"},
        text=top10.sort_values("champion_probability")["champion_probability"].map(lambda x: f"{x:.2%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        format_percent(
            top10[
                [
                    "team",
                    "champion_probability",
                    "final_probability",
                    "semi_final_probability",
                    "quarter_final_probability",
                    "round_of_16_probability",
                    "round_of_32_probability",
                ]
            ],
            percent_cols,
        ),
        use_container_width=True,
    )

    st.info(
        "Model v5.0 simulates the full tournament bracket using static pre-tournament "
        "team-strength features. It updates group tables and knockout paths, but it does "
        "not dynamically update team strength during the tournament."
    )


# ============================================================
# Champion probabilities
# ============================================================
elif page == "Champion probabilities":
    st.header("Champion and advancement probabilities")

    top_n = st.slider("Number of teams to show", 5, 48, 10)
    sorted_probs = team_probs.sort_values("champion_probability", ascending=False).head(top_n)

    fig = px.bar(
        sorted_probs.sort_values("champion_probability"),
        x="champion_probability",
        y="team",
        orientation="h",
        title=f"Top {top_n} Champion Probabilities",
        labels={"champion_probability": "Champion probability", "team": "Team"},
        text=sorted_probs.sort_values("champion_probability")["champion_probability"].map(lambda x: f"{x:.2%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    cols = [
        "team",
        "group_winner_probability",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    cols = [c for c in cols if c in sorted_probs.columns]
    st.dataframe(format_percent(sorted_probs[cols], [c for c in cols if c != "team"]), use_container_width=True)

    st.download_button(
        "Download team probabilities CSV",
        data=team_probs.to_csv(index=False),
        file_name="model_v5_team_probabilities.csv",
        mime="text/csv",
    )


# ============================================================
# Match predictor
# ============================================================
elif page == "Match predictor":
    st.header("Group-stage match predictor")

    st.write(
        "Select a team first. The opponent dropdown will only show valid group-stage opponents "
        "from the same group."
    )

    col1, col2 = st.columns(2)

    with col1:
        default_team = "Mexico" if "Mexico" in TEAMS else TEAMS[0]
        team_a = filtered_team_selectbox("Team A", TEAMS, "match_team_a", default=default_team)

    group = get_group_for_team(team_a)
    opponents = get_valid_group_opponents(team_a)

    with col2:
        if not opponents:
            st.warning("No group-stage opponents found for this team.")
            st.stop()

        default_opponent = opponents[0]
        if team_a == "Mexico" and "South Africa" in opponents:
            default_opponent = "South Africa"
        elif team_a == "Norway" and "France" in opponents:
            default_opponent = "France"

        team_b = st.selectbox("Team B — valid group opponents only", opponents, index=opponents.index(default_opponent))

    st.caption(f"Selected group: **{group}** | Valid opponents for **{team_a}**: {', '.join(opponents)}")

    match = find_group_match(team_a, team_b)

    if match.empty:
        st.error("No fixture found. This should not happen if the dependent dropdown is working correctly.")
    else:
        row = match.iloc[0]

        st.subheader(f"{row['team_a']} vs {row['team_b']}")

        meta_cols = st.columns(4)
        meta_cols[0].metric("Group", row.get("group", "N/A"))
        meta_cols[1].metric("Match ID", row.get("match_id", "N/A"))
        meta_cols[2].metric("Date", row.get("match_date", row.get("date", "N/A")))
        meta_cols[3].metric("City", row.get("city", "N/A"))

        probs = pd.DataFrame(
            {
                "Outcome": [
                    f"{row['team_a']} win",
                    "Draw",
                    f"{row['team_b']} win",
                ],
                "Probability": [
                    row["p_team_a_win"],
                    row["p_draw"],
                    row["p_team_b_win"],
                ],
            }
        )

        fig = px.bar(
            probs,
            x="Outcome",
            y="Probability",
            title="Match outcome probabilities",
            text=probs["Probability"].map(lambda x: f"{x:.2%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(format_percent(probs, ["Probability"]), use_container_width=True)
        st.success(f"Most likely result: {most_likely_readable(row)}")

        st.subheader(f"All group-stage fixtures involving {team_a}")
        fixtures_for_team = team_fixture_rows(team_a)
        display_cols = [
            "match_id",
            "group",
            "match_date",
            "team_a",
            "team_b",
            "p_team_a_win",
            "p_draw",
            "p_team_b_win",
            "most_likely_result",
        ]
        display_cols = [c for c in display_cols if c in fixtures_for_team.columns]
        st.dataframe(
            format_percent(fixtures_for_team[display_cols], ["p_team_a_win", "p_draw", "p_team_b_win"]),
            use_container_width=True,
        )


# ============================================================
# Group explorer
# ============================================================
elif page == "Group explorer":
    st.header("Group explorer")

    groups = sorted(group_matches["group"].dropna().astype(str).unique().tolist()) if "group" in group_matches.columns else []
    selected_group = st.selectbox("Select group", groups)

    group_teams = teams_in_group(selected_group)
    st.subheader(f"Group {selected_group} teams")
    st.write(", ".join(group_teams))

    # Team strength snapshot for the group
    if group_teams:
        group_team_probs = team_probs[team_probs["team"].isin(group_teams)].copy()
        show_cols = [
            "team",
            "group_winner_probability",
            "round_of_32_probability",
            "round_of_16_probability",
            "champion_probability",
        ]
        show_cols = [c for c in show_cols if c in group_team_probs.columns]
        st.dataframe(
            format_percent(group_team_probs.sort_values("group_winner_probability", ascending=False)[show_cols],
                           [c for c in show_cols if c != "team"]),
            use_container_width=True,
        )

        fig = px.bar(
            group_team_probs.sort_values("group_winner_probability"),
            x="group_winner_probability",
            y="team",
            orientation="h",
            title=f"Group {selected_group}: Group Winner Probabilities",
            labels={"group_winner_probability": "Group winner probability", "team": "Team"},
            text=group_team_probs.sort_values("group_winner_probability")["group_winner_probability"].map(lambda x: f"{x:.2%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Group {selected_group} fixtures")
    view = group_matches[group_matches["group"].astype(str) == str(selected_group)].copy()
    display_cols = [
        "match_id",
        "match_date",
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "most_likely_result",
    ]
    display_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(format_percent(view[display_cols], ["p_team_a_win", "p_draw", "p_team_b_win"]), use_container_width=True)


# ============================================================
# Team profile
# ============================================================
elif page == "Team profile":
    st.header("Team profile")

    default = "Spain" if "Spain" in TEAMS else TEAMS[0]
    selected_team = filtered_team_selectbox("Team", TEAMS, "profile_team", default=default)

    row = team_probability_row(selected_team)
    if row is None:
        st.error("No probability row found for this team.")
        st.stop()

    group = get_group_for_team(selected_team)
    st.subheader(f"{selected_team}")
    st.caption(f"Group: **{group}**")

    display_probability_metrics(row)

    st.subheader("Advancement path")
    profile_df = pd.DataFrame([row])
    long_df = make_advancement_long(profile_df)

    fig = px.bar(
        long_df,
        x="stage",
        y="probability",
        title=f"{selected_team}: Tournament Advancement Probabilities",
        text=long_df["probability"].map(lambda x: f"{x:.2%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Group-stage fixtures")
    fixtures_for_team = team_fixture_rows(selected_team)
    display_cols = [
        "match_id",
        "group",
        "match_date",
        "team_a",
        "team_b",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "most_likely_result",
    ]
    display_cols = [c for c in display_cols if c in fixtures_for_team.columns]
    st.dataframe(
        format_percent(fixtures_for_team[display_cols], ["p_team_a_win", "p_draw", "p_team_b_win"]),
        use_container_width=True,
    )


# ============================================================
# Compare teams
# ============================================================
elif page == "Compare teams":
    st.header("Compare two teams")

    st.write(
        "This comparison shows each team's overall tournament probabilities, not only a direct match probability."
    )

    c1, c2 = st.columns(2)
    with c1:
        team_1 = filtered_team_selectbox("First team", TEAMS, "compare_team_1", default="Spain" if "Spain" in TEAMS else TEAMS[0])
    with c2:
        default_2 = "Argentina" if "Argentina" in TEAMS else TEAMS[min(1, len(TEAMS) - 1)]
        team_2 = filtered_team_selectbox("Second team", TEAMS, "compare_team_2", default=default_2)

    rows = team_probs[team_probs["team"].isin([team_1, team_2])].copy()
    if len(rows) < 2:
        st.warning("Please select two different teams.")
        st.stop()

    st.subheader("Overall tournament comparison")
    cols = [
        "team",
        "group_winner_probability",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    cols = [c for c in cols if c in rows.columns]
    st.dataframe(format_percent(rows[cols], [c for c in cols if c != "team"]), use_container_width=True)

    long_df = make_advancement_long(rows)
    fig = px.bar(
        long_df,
        x="stage",
        y="probability",
        color="team",
        barmode="group",
        title=f"{team_1} vs {team_2}: Advancement Probability Comparison",
        text=long_df["probability"].map(lambda x: f"{x:.1%}"),
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Direct group-stage fixture check")
    direct = find_group_match(team_1, team_2)
    if direct.empty:
        st.info("These teams do not have a scheduled group-stage match against each other.")
    else:
        row = direct.iloc[0]
        probs = pd.DataFrame(
            {
                "Outcome": [f"{row['team_a']} win", "Draw", f"{row['team_b']} win"],
                "Probability": [row["p_team_a_win"], row["p_draw"], row["p_team_b_win"]],
            }
        )
        st.write(f"Scheduled fixture: **{row['team_a']} vs {row['team_b']}**")
        st.dataframe(format_percent(probs, ["Probability"]), use_container_width=True)


# ============================================================
# Group finish probabilities
# ============================================================
elif page == "Group finish probabilities":
    st.header("Group finish probabilities")

    st.write(
        "This table shows simulated group finish probabilities. Exact columns depend on the generated simulator output."
    )

    st.dataframe(group_finish, use_container_width=True)

    st.download_button(
        "Download group finish probabilities CSV",
        data=group_finish.to_csv(index=False),
        file_name="model_v5_group_finish_probabilities.csv",
        mime="text/csv",
    )


# ============================================================
# Knockout matchups
# ============================================================
elif page == "Knockout matchups":
    st.header("Knockout matchup probabilities")

    if knockout_matchups.empty:
        st.warning("No knockout matchup probability file was found.")
    else:
        st.write(
            "This table summarizes simulated knockout matchups and how often they appeared."
        )

        view = normalize_probability_column(knockout_matchups)

        # Provide light filtering if stage column exists
        stage_cols = [c for c in view.columns if "stage" in c.lower() or "round" in c.lower()]
        if stage_cols:
            stage_col = stage_cols[0]
            stages = ["All"] + sorted(view[stage_col].dropna().astype(str).unique().tolist())
            selected_stage = st.selectbox("Filter by knockout stage", stages)
            if selected_stage != "All":
                view = view[view[stage_col].astype(str) == selected_stage]

        st.dataframe(view, use_container_width=True)

        st.download_button(
            "Download knockout matchup probabilities CSV",
            data=knockout_matchups.to_csv(index=False),
            file_name="model_v5_knockout_matchup_probabilities.csv",
            mime="text/csv",
        )


# ============================================================
# About
# ============================================================
elif page == "About":
    st.header("About this simulator")

    st.markdown(
        """
        ## Data layers

        The simulator uses:

        - historical international match results
        - local Elo-style team strength
        - FIFA ranking features
        - recent form
        - confederation strength
        - tournament experience
        - head-to-head history
        - World Cup 2026 fixtures and knockout slots

        ## Model versions

        - **Model v1:** logistic-regression baseline
        - **Model v2:** curated features and stronger classifier
        - **Model v3:** calibrated Model v2
        - **Model v4:** probability-focused ensemble
        - **Model v5:** full World Cup 2026 Monte Carlo simulator

        ## Current simulator

        This dashboard presents **Model v5.0**, based on **10,000 tournament simulations**.

        ## Limitations

        - Team strengths are static pre-tournament snapshots.
        - Player/squad/injury data are not included yet.
        - Betting-market probabilities are not included yet.
        - Knockout draws are resolved approximately.
        - Scorelines are generated approximately from historical patterns.
        """
    )

    st.subheader("Project links")

    github_url = "https://github.com/pranishkessi/worldcup-2026-predictor"
    kaggle_url = "https://www.kaggle.com/datasets/pranishkessi/fifa-world-cup-2026-prediction-simulator"

    st.markdown(
        f"""
        - [GitHub repository]({github_url})
        - [Kaggle dataset]({kaggle_url})
        """
    )

    st.subheader("Main local files")

    st.code(
        """
data/worldcup_2026/model_v5_team_probabilities.csv
data/worldcup_2026/model_v5_group_match_probabilities.csv
data/worldcup_2026/model_v5_group_finish_probabilities.csv
data/worldcup_2026/model_v5_knockout_matchup_probabilities.csv
models/v4_probability_ensemble.joblib
        """.strip()
    )
