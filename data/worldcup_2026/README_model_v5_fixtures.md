# Model v5 Fixture-Preparation Layer

This layer prepares the uploaded FIFA World Cup 2026 fixture spreadsheets for the future tournament simulator.

## Source files

- `datalab_export_2026-06-03 16_17_15.xlsx`: group-stage fixtures.
- `datalab_export_2026-06-03 16_17_37.xlsx`: knockout-stage bracket slots.

## Deliverables

- `data/worldcup_2026/worldcup_2026_group_fixtures.csv`
- `data/worldcup_2026/worldcup_2026_knockout_slots.csv`
- `data/worldcup_2026/worldcup_2026_all_fixtures.csv`
- `data/worldcup_2026/worldcup_2026_groups.csv`

## Summary

- Group-stage matches: 72
- Knockout-stage slot matches: 32
- Total tournament matches: 104
- Groups: 12
- Teams: 48
- Venues: 16

## File descriptions

### `worldcup_2026_group_fixtures.csv`

Known group-stage matches with real teams. It includes match date, venue, stadium, city, host country, team names, group, and a preliminary `neutral_for_model` flag.

### `worldcup_2026_knockout_slots.csv`

Knockout bracket structure with unresolved slots such as `Winner Group A`, `Runner-up Group B`, `Best 3rd (Groups C/E/F/H/I)`, and `Winner Match 101`. These slots must be resolved by the simulator after group standings and knockout results are known.

### `worldcup_2026_all_fixtures.csv`

Canonical combined fixture table for Model v5. Group-stage rows have known `home_team` and `away_team`. Knockout rows keep teams blank and use `home_slot` / `away_slot` plus parsed slot metadata.

### `worldcup_2026_groups.csv`

One row per team in each group. The `team_slot` values, such as `A1`, `A2`, etc., are assigned by first appearance order in the uploaded group-stage file because the source did not include official draw-position labels.

## Leakage / simulation note

This layer does not predict results. It only prepares fixtures. In Model v5, group-stage teams can be predicted directly with Model v4. Knockout-stage teams must be resolved dynamically in each Monte Carlo simulation.

## Next step

Build `predict_worldcup_match(team_a, team_b, match_date, venue, stage)` using Model v4, then use this fixture layer to run the group-stage and knockout Monte Carlo simulator.
