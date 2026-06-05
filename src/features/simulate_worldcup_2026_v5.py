#!/usr/bin/env python3
"""Model v5: World Cup 2026 Monte Carlo tournament simulator.

Uses Model v4 probability ensemble:
  P_v4 = 0.186 * baseline logistic probabilities + 0.814 * temperature-calibrated Model v2 probabilities.

Notes:
- Match strengths/features are pre-tournament snapshots from the historical feature dataset.
- Group standings are simulated dynamically using sampled scorelines.
- Knockout draws after 90 minutes are resolved by a strength-based extra-time/penalty draw resolver.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
WC = DATA / 'worldcup_2026'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
SRC = ROOT / 'src' / 'features'

RNG_SEED = 20260603
N_SIMULATIONS = 1000
LABELS = ['H', 'D', 'A']
HOSTS_CANON = {'Mexico', 'Canada', 'United States'}
DISPLAY_TO_CANON = {
    'USA': 'United States',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cabo Verde': 'Cape Verde',
}
CANON_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_CANON.items()}


def canon(team: str) -> str:
    if pd.isna(team):
        return team
    return DISPLAY_TO_CANON.get(str(team), str(team))


def display(team: str) -> str:
    return CANON_TO_DISPLAY.get(str(team), str(team))


def normalize_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {'true', '1', 'yes'}
    return bool(x)


class TemperatureCalibrator:
    def __init__(self, temperature: float):
        self.temperature = float(temperature)

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        eps = 1e-12
        p = np.clip(p, eps, 1.0)
        logits = np.log(p)
        logits = logits / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        out = np.exp(logits)
        return out / out.sum(axis=1, keepdims=True)


def points_for(result: str) -> Tuple[int, int]:
    if result == 'H':
        return 3, 0
    if result == 'A':
        return 0, 3
    return 1, 1


@dataclass
class TeamStats:
    team: str
    elo: float = 1500.0
    fifa_rank: float = np.nan
    fifa_points: float = np.nan
    confed: str = 'NON_FIFA'
    form_ppm5: float = 1.0
    form_ppm10: float = 1.0
    avg_gd5: float = 0.0
    avg_gd10: float = 0.0
    win_rate5: float = 0.333
    win_rate10: float = 0.333
    weighted_form5: float = 5.0
    weighted_form10: float = 10.0
    weighted_gd5: float = 0.0
    weighted_gd10: float = 0.0
    days_since: float = 30.0
    exp_total: float = 0.0
    exp_wc: float = 0.0
    exp_continental: float = 0.0
    exp_qualifier: float = 0.0
    exp_nations: float = 0.0
    exp_major: float = 0.0
    exp_major_ppm: float = 1.0
    exp_major_gdpm: float = 0.0
    exp_ppm: float = 1.0
    exp_gdpm: float = 0.0
    exp_avg_importance: float = 1.0
    exp_high_importance: float = 0.0
    years_since_first: float = 0.0
    exp_major_recency: float = 0.0


def build_team_stats(hist: pd.DataFrame, current_elo: pd.DataFrame, fifa: pd.DataFrame, confed_map: pd.DataFrame, teams: List[str]) -> Dict[str, TeamStats]:
    # Current Elo
    elo_map = {}
    for _, r in current_elo.iterrows():
        elo_map[canon(r['team'])] = float(r['elo_rating'])

    # Latest FIFA per team
    fifa_map = {}
    if 'ranking_date' in fifa.columns:
        fifa['ranking_date'] = pd.to_datetime(fifa['ranking_date'], errors='coerce')
        fifa = fifa.sort_values(['team_clean', 'ranking_date'])
        for t, g in fifa.groupby('team_clean'):
            last = g.iloc[-1]
            fifa_map[canon(t)] = (float(last.get('fifa_rank', np.nan)), float(last.get('fifa_points', np.nan)))

    # Confederation
    confed_lookup = {}
    for _, r in confed_map.iterrows():
        confed_lookup[canon(r['team'])] = r.get('confederation', 'NON_FIFA')

    # Build per-team match histories from all enriched data prior to WC.
    h = hist.copy()
    h['date'] = pd.to_datetime(h['date'])
    h = h.sort_values('date')
    team_matches = defaultdict(list)
    first_date = {}
    exp_counters = defaultdict(lambda: defaultdict(float))

    for _, r in h.iterrows():
        ht, at = canon(r['home_team']), canon(r['away_team'])
        hg, ag = int(r['home_score']), int(r['away_score'])
        date = r['date']
        res = 'H' if hg > ag else ('A' if ag > hg else 'D')
        hp, ap = points_for(res)
        for team, gf, ga, pts, opp in [(ht, hg, ag, hp, at), (at, ag, hg, ap, ht)]:
            first_date.setdefault(team, date)
            team_matches[team].append({'date': date, 'gf': gf, 'ga': ga, 'pts': pts, 'opp': opp})
            c = exp_counters[team]
            c['total'] += 1
            c['points'] += pts
            c['gd'] += gf - ga
            imp = float(r.get('tournament_importance', 1.0))
            c['importance_sum'] += imp
            if int(r.get('is_world_cup', 0)) == 1:
                c['wc'] += 1
            if int(r.get('is_continental', 0)) == 1:
                c['continental'] += 1
            if int(r.get('is_qualifier', 0)) == 1:
                c['qualifier'] += 1
            if 'Nations League' in str(r.get('tournament', '')):
                c['nations'] += 1
            if int(r.get('is_world_cup', 0)) == 1 or int(r.get('is_continental', 0)) == 1:
                c['major'] += 1
                c['major_points'] += pts
                c['major_gd'] += gf - ga
                c['last_major_date_ord'] = date.toordinal()
            if imp >= 3:
                c['high_importance'] += 1

    def summarize_form(matches: list, n: int) -> dict:
        m = matches[-n:]
        if not m:
            return dict(ppm=1.0, avg_gd=0.0, win_rate=0.333, wpts=float(n), wgd=0.0, days=60.0)
        pts = np.array([x['pts'] for x in m], dtype=float)
        gd = np.array([x['gf'] - x['ga'] for x in m], dtype=float)
        win = np.array([1.0 if x['pts'] == 3 else 0.0 for x in m])
        weights = np.arange(1, len(m)+1, dtype=float)  # most recent gets biggest weight
        today = pd.Timestamp('2026-06-10')
        days = max(0.0, float((today - m[-1]['date']).days))
        return dict(
            ppm=float(pts.mean()), avg_gd=float(gd.mean()), win_rate=float(win.mean()),
            wpts=float(np.average(pts, weights=weights) * len(m)),
            wgd=float(np.average(gd, weights=weights)), days=days
        )

    out: Dict[str, TeamStats] = {}
    for t in teams:
        t = canon(t)
        matches = team_matches[t]
        f5, f10 = summarize_form(matches, 5), summarize_form(matches, 10)
        c = exp_counters[t]
        total = max(c['total'], 1.0)
        major = max(c['major'], 1.0)
        first = first_date.get(t, pd.Timestamp('2026-06-10'))
        years_since_first = max(0.0, (pd.Timestamp('2026-06-10') - first).days / 365.25)
        last_major_ord = c.get('last_major_date_ord', np.nan)
        if not np.isnan(last_major_ord):
            last_major = pd.Timestamp.fromordinal(int(last_major_ord))
            major_recency = max(0.0, (pd.Timestamp('2026-06-10') - last_major).days / 365.25)
        else:
            major_recency = 99.0
        rank, points = fifa_map.get(t, (np.nan, np.nan))
        out[t] = TeamStats(
            team=t,
            elo=elo_map.get(t, 1500.0),
            fifa_rank=rank,
            fifa_points=points,
            confed=confed_lookup.get(t, 'NON_FIFA'),
            form_ppm5=f5['ppm'], form_ppm10=f10['ppm'], avg_gd5=f5['avg_gd'], avg_gd10=f10['avg_gd'],
            win_rate5=f5['win_rate'], win_rate10=f10['win_rate'],
            weighted_form5=f5['wpts'], weighted_form10=f10['wpts'], weighted_gd5=f5['wgd'], weighted_gd10=f10['wgd'],
            days_since=f5['days'],
            exp_total=c['total'], exp_wc=c['wc'], exp_continental=c['continental'], exp_qualifier=c['qualifier'],
            exp_nations=c['nations'], exp_major=c['major'],
            exp_major_ppm=c['major_points']/major, exp_major_gdpm=c['major_gd']/major,
            exp_ppm=c['points']/total, exp_gdpm=c['gd']/total, exp_avg_importance=c['importance_sum']/total,
            exp_high_importance=c['high_importance'], years_since_first=years_since_first, exp_major_recency=major_recency
        )
    return out


def build_pair_history(hist: pd.DataFrame) -> Dict[Tuple[str, str], List[dict]]:
    pair_hist = defaultdict(list)
    h = hist.copy()
    h['date'] = pd.to_datetime(h['date'])
    h = h.sort_values('date')
    for _, r in h.iterrows():
        ht, at = canon(r['home_team']), canon(r['away_team'])
        hg, ag = int(r['home_score']), int(r['away_score'])
        for a, b, gf, ga in [(ht, at, hg, ag), (at, ht, ag, hg)]:
            pts = 3 if gf > ga else (1 if gf == ga else 0)
            pair_hist[(a, b)].append({
                'date': r['date'], 'gf': gf, 'ga': ga, 'pts': pts,
                'tournament': str(r.get('tournament', '')), 'family': tournament_family(r)
            })
    return pair_hist


def tournament_family(row_or_stage) -> str:
    t = str(row_or_stage.get('tournament', row_or_stage.get('stage', '')) if hasattr(row_or_stage, 'get') else row_or_stage)
    if 'World Cup' in t or 'Final' in t or 'Round' in t or 'Quarter' in t or 'Semi' in t:
        return 'World Cup'
    if 'qualification' in t.lower():
        return 'Qualifier'
    if 'Nations League' in t:
        return 'Nations League'
    if 'Friendly' in t:
        return 'Friendly'
    return 'Other'


def h2h_features(pair_hist, team_a: str, team_b: str) -> dict:
    m = pair_hist.get((team_a, team_b), [])
    if not m:
        return dict(h2h_matches_prior=0, h2h_home_team_points_per_match_prior_filled=1.0,
                    h2h_goal_diff_per_match_prior_filled=0.0, h2h_goals_for_per_match_prior_filled=1.0,
                    h2h_goals_against_per_match_prior_filled=1.0, h2h_matches_last5=0,
                    h2h_home_team_points_per_match_last5_filled=1.0, h2h_goal_diff_per_match_last5_filled=0.0,
                    h2h_home_team_win_rate_last5_filled=0.333, h2h_days_since_last_meeting_filled=9999,
                    h2h_same_tournament_matches_prior=0, h2h_same_family_matches_prior=0,
                    has_h2h_history=0, has_h2h_last5=0)
    pts = np.array([x['pts'] for x in m], float)
    gd = np.array([x['gf']-x['ga'] for x in m], float)
    gf = np.array([x['gf'] for x in m], float)
    ga = np.array([x['ga'] for x in m], float)
    last5 = m[-5:]
    pts5 = np.array([x['pts'] for x in last5], float)
    gd5 = np.array([x['gf']-x['ga'] for x in last5], float)
    win5 = np.array([1.0 if x['pts']==3 else 0.0 for x in last5], float)
    days = (pd.Timestamp('2026-06-10') - m[-1]['date']).days
    return dict(h2h_matches_prior=len(m), h2h_home_team_points_per_match_prior_filled=float(pts.mean()),
                h2h_goal_diff_per_match_prior_filled=float(gd.mean()), h2h_goals_for_per_match_prior_filled=float(gf.mean()),
                h2h_goals_against_per_match_prior_filled=float(ga.mean()), h2h_matches_last5=len(last5),
                h2h_home_team_points_per_match_last5_filled=float(pts5.mean()), h2h_goal_diff_per_match_last5_filled=float(gd5.mean()),
                h2h_home_team_win_rate_last5_filled=float(win5.mean()), h2h_days_since_last_meeting_filled=float(days),
                h2h_same_tournament_matches_prior=sum(1 for x in m if 'World Cup' in x['tournament']),
                h2h_same_family_matches_prior=sum(1 for x in m if x['family']=='World Cup'),
                has_h2h_history=1, has_h2h_last5=1)


def load_confed_strength() -> dict:
    snap = pd.read_csv(DATA/'processed'/'confederation_strength_snapshot.csv')
    d = {}
    for _, r in snap.iterrows():
        c = r['confederation']
        d[c] = r.to_dict()
    return d


def base_feature_row(team_a: str, team_b: str, host_country: str, stage: str, team_stats: Dict[str, TeamStats], pair_hist: dict, conf_strength: dict, medians: dict) -> dict:
    a, b = canon(team_a), canon(team_b)
    sa, sb = team_stats[a], team_stats[b]
    host = canon(host_country) if host_country else ''
    is_world_cup = 1
    is_friendly = 0
    is_qualifier = 0
    is_continental = 0
    tournament_importance = 4.0 if stage == 'Group stage' else 5.0
    neutral = not (a in HOSTS_CANON and a == host)
    same_confed = int(sa.confed == sb.confed)
    host_confed = team_stats.get(host, TeamStats(host)).confed if host in team_stats else 'HOST_UNKNOWN'

    fifa_rank_diff = (sb.fifa_rank - sa.fifa_rank) if not (np.isnan(sa.fifa_rank) or np.isnan(sb.fifa_rank)) else medians.get('fifa_rank_diff_filled', 0.0)
    fifa_points_diff = (sa.fifa_points - sb.fifa_points) if not (np.isnan(sa.fifa_points) or np.isnan(sb.fifa_points)) else medians.get('fifa_points_diff_filled', 0.0)
    elo_diff = sa.elo - sb.elo
    p_elo = 1 / (1 + 10 ** (-(elo_diff + (0 if neutral else 100)) / 400))

    ca, cb = conf_strength.get(sa.confed, {}), conf_strength.get(sb.confed, {})
    def cs(col, default=0.0):
        return float(ca.get(col, default) or default) - float(cb.get(col, default) or default)

    row = {
        'elo_diff': elo_diff, 'elo_diff_pre': elo_diff, 'elo_prob_home_win_proxy': p_elo,
        'fifa_rank_diff_filled': fifa_rank_diff, 'fifa_points_diff_filled': fifa_points_diff,
        'neutral': int(neutral), 'tournament_importance': tournament_importance,
        'is_friendly': is_friendly, 'is_qualifier': is_qualifier, 'is_world_cup': is_world_cup, 'is_continental': is_continental,
        'form_points_per_match_diff_last5': sa.form_ppm5 - sb.form_ppm5,
        'avg_goal_diff_delta_last5': sa.avg_gd5 - sb.avg_gd5,
        'win_rate_diff_last5': sa.win_rate5 - sb.win_rate5,
        'weighted_form_points_diff_last5': sa.weighted_form5 - sb.weighted_form5,
        'weighted_goal_diff_delta_last5': sa.weighted_gd5 - sb.weighted_gd5,
        'form_points_per_match_diff_last10': sa.form_ppm10 - sb.form_ppm10,
        'avg_goal_diff_delta_last10': sa.avg_gd10 - sb.avg_gd10,
        'win_rate_diff_last10': sa.win_rate10 - sb.win_rate10,
        'weighted_form_points_diff_last10': sa.weighted_form10 - sb.weighted_form10,
        'weighted_goal_diff_delta_last10': sa.weighted_gd10 - sb.weighted_gd10,
        'home_days_since_last_match_last5': sa.days_since, 'away_days_since_last_match_last5': sb.days_since,
        'same_confederation': same_confed,
        'home_confed_matches_host_confed': int(sa.confed == host_confed),
        'away_confed_matches_host_confed': int(sb.confed == host_confed),
        'confed_points_per_match_diff_prior_filled': cs('overall_points_per_match', 0.0),
        'confed_goal_diff_per_match_diff_prior_filled': cs('overall_goal_diff_per_match', 0.0),
        'confed_avg_elo_diff_prior_filled': cs('overall_avg_pre_match_elo', 0.0),
        'confed_inter_points_per_match_diff_prior_filled': cs('inter_confed_points_per_match', 0.0),
        'confed_inter_goal_diff_per_match_diff_prior_filled': cs('inter_confed_goal_diff_per_match', 0.0),
        'home_confederation': sa.confed, 'away_confederation': sb.confed, 'confederation_pair': f'{sa.confed}-{sb.confed}', 'host_confederation': host_confed,
        # tournament experience
        'exp_total_matches_prior_diff': sa.exp_total - sb.exp_total,
        'exp_same_tournament_matches_prior_diff': sa.exp_wc - sb.exp_wc,
        'exp_same_family_matches_prior_diff': sa.exp_wc - sb.exp_wc,
        'exp_world_cup_matches_prior_diff': sa.exp_wc - sb.exp_wc,
        'exp_continental_matches_prior_diff': sa.exp_continental - sb.exp_continental,
        'exp_qualifier_matches_prior_diff': sa.exp_qualifier - sb.exp_qualifier,
        'exp_nations_league_matches_prior_diff': sa.exp_nations - sb.exp_nations,
        'exp_major_matches_prior_diff': sa.exp_major - sb.exp_major,
        'exp_major_points_per_match_prior_diff_filled': sa.exp_major_ppm - sb.exp_major_ppm,
        'exp_major_goal_diff_per_match_prior_diff_filled': sa.exp_major_gdpm - sb.exp_major_gdpm,
        'exp_points_per_match_prior_diff_filled': sa.exp_ppm - sb.exp_ppm,
        'exp_goal_diff_per_match_prior_diff_filled': sa.exp_gdpm - sb.exp_gdpm,
        'exp_avg_tournament_importance_prior_diff_filled': sa.exp_avg_importance - sb.exp_avg_importance,
        'exp_high_importance_matches_prior_diff': sa.exp_high_importance - sb.exp_high_importance,
        'exp_years_since_first_match_diff_filled': sa.years_since_first - sb.years_since_first,
        'exp_major_recency_advantage_filled': sb.exp_major_recency - sa.exp_major_recency,
        'both_teams_have_major_tournament_history': int(sa.exp_major > 0 and sb.exp_major > 0),
    }
    row.update(h2h_features(pair_hist, a, b))
    # Fill any model feature not set with median/default
    for k, v in medians.items():
        row.setdefault(k, v)
    return row


class ProbabilityEngine:
    def __init__(self, team_stats, pair_hist, conf_strength, medians, v2_features, baseline_features_num, baseline_features_cat):
        self.baseline = joblib.load(MODELS/'baseline_model_best.joblib')
        self.v2 = joblib.load(MODELS/'v2_model_best_train_plus_validation.joblib')
        self.cal = TemperatureCalibrator(json.load(open(ROOT/'model_v3_calibration_metadata.json'))['temperature'])
        self.team_stats = team_stats
        self.pair_hist = pair_hist
        self.conf_strength = conf_strength
        self.medians = medians
        self.v2_features = v2_features
        self.cache = {}

    def predict(self, team_a_display: str, team_b_display: str, host_country: str, stage: str) -> dict:
        # If team B is the host in its own host country and team A is not, orient model with B as home, then invert.
        a, b = canon(team_a_display), canon(team_b_display)
        host = canon(host_country)
        invert = False
        if b in HOSTS_CANON and b == host and not (a in HOSTS_CANON and a == host):
            a, b = b, a
            invert = True
        key = (a, b, host, stage, invert)
        if key in self.cache:
            return self.cache[key]
        row = base_feature_row(a, b, host, stage, self.team_stats, self.pair_hist, self.conf_strength, self.medians)
        df = pd.DataFrame([row])
        # baseline can select its own columns by name
        p_base = self.baseline.predict_proba(df)[0][[2,1,0]]
        # v2 expects curated feature columns
        Xv2 = df.reindex(columns=self.v2_features)
        p_v2 = self.v2.predict_proba(Xv2)[0][[2,1,0]]
        p_v3 = self.cal.transform(p_v2.reshape(1, -1))[0]
        p = 0.186 * p_base + 0.814 * p_v3
        p = p / p.sum()
        if invert:
            # Model was B-vs-A; return original A-vs-B probabilities.
            p = np.array([p[2], p[1], p[0]])
        out = {'p_team_a_win': float(p[0]), 'p_draw': float(p[1]), 'p_team_b_win': float(p[2])}
        self.cache[key] = out
        return out


def make_scoreline_distributions(hist: pd.DataFrame):
    d = {'H': [], 'D': [], 'A': []}
    for _, r in hist.iterrows():
        hg, ag = int(r['home_score']), int(r['away_score'])
        if hg > ag:
            d['H'].append((hg, ag))
        elif hg < ag:
            d['A'].append((hg, ag))
        else:
            d['D'].append((hg, ag))
    # Use recency/tournament-ish common distributions, but raw is enough.
    return {k: np.array(v, dtype=int) for k, v in d.items()}


def sample_outcome_and_score(probs: dict, rng: np.random.Generator, score_dist: dict) -> Tuple[str, int, int]:
    labels = ['H', 'D', 'A']
    p = np.array([probs['p_team_a_win'], probs['p_draw'], probs['p_team_b_win']], dtype=float)
    p = p / p.sum()
    outcome = rng.choice(labels, p=p)
    arr = score_dist[outcome]
    hg, ag = arr[rng.integers(0, len(arr))]
    return outcome, int(hg), int(ag)


def init_table(teams: List[str]) -> Dict[str, dict]:
    return {t: {'team': t, 'played': 0, 'points': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'wins':0, 'draws':0, 'losses':0, 'fair': random.random()} for t in teams}


def update_table(table: dict, team_a: str, team_b: str, ga: int, gb: int):
    pa, pb = (3,0) if ga>gb else ((0,3) if ga<gb else (1,1))
    for t, gf, gc, pts in [(team_a, ga, gb, pa), (team_b, gb, ga, pb)]:
        r=table[t]; r['played']+=1; r['points']+=pts; r['gf']+=gf; r['ga']+=gc; r['gd']=r['gf']-r['ga']
        if pts==3: r['wins']+=1
        elif pts==1: r['draws']+=1
        else: r['losses']+=1


def rank_group(table: dict, rng: np.random.Generator) -> List[dict]:
    rows = list(table.values())
    # Approximate FIFA ranking rules: points, goal difference, goals scored, wins, random draw.
    for r in rows:
        r['_lot'] = rng.random()
    rows.sort(key=lambda r: (r['points'], r['gd'], r['gf'], r['wins'], r['_lot']), reverse=True)
    return rows


def rank_thirds(third_rows: List[dict], rng: np.random.Generator) -> List[dict]:
    rows=[dict(r) for r in third_rows]
    for r in rows: r['_lot'] = rng.random()
    rows.sort(key=lambda r: (r['points'], r['gd'], r['gf'], r['wins'], r['_lot']), reverse=True)
    return rows


def assign_best_thirds(slots: List[Tuple[int, str, str]], qualified: List[dict]) -> Dict[Tuple[int,str], str]:
    # slots: (match_id, side, allowed_csv). qualified rows are sorted best-to-worst thirds.
    qgroups = [r['group'] for r in qualified]
    qteam_by_group = {r['group']: r['team'] for r in qualified}
    slot_allowed = []
    for mid, side, allowed in slots:
        allowed_set = set(str(allowed).split(',')) if pd.notna(allowed) else set(qgroups)
        slot_allowed.append((mid, side, [g for g in qgroups if g in allowed_set]))
    # Backtracking: assign better third-place groups to earliest possible slots while satisfying all constraints.
    assignment = {}
    used = set()
    slot_allowed.sort(key=lambda x: len(x[2]))
    def bt(i):
        if i == len(slot_allowed):
            return True
        mid, side, allowed = slot_allowed[i]
        for g in allowed:
            if g not in used:
                used.add(g); assignment[(mid, side)] = qteam_by_group[g]
                if bt(i+1): return True
                used.remove(g); assignment.pop((mid, side), None)
        return False
    if not bt(0):
        # Greedy fallback ignoring constraints only if fixture constraints are impossible.
        for (mid, side, allowed), r in zip(slot_allowed, qualified):
            assignment[(mid, side)] = r['team']
    return assignment


def resolve_slot(row: pd.Series, side: str, group_results: dict, match_results: dict, third_assignment: dict) -> str:
    typ = row[f'{side}_slot_type']
    if typ == 'winner_group':
        return group_results[row[f'{side}_group_ref']]['winner']
    if typ == 'runner_up_group':
        return group_results[row[f'{side}_group_ref']]['runner_up']
    if typ == 'best_third':
        return third_assignment[(int(row['match_id']), side)]
    if typ == 'winner_match':
        return match_results[int(row[f'{side}_match_ref'])]['winner']
    if typ == 'loser_match':
        return match_results[int(row[f'{side}_match_ref'])]['loser']
    raise ValueError(f'Unknown slot type {typ}')


def simulate_tournament(prob_engine: ProbabilityEngine, group_fixtures: pd.DataFrame, groups: pd.DataFrame, ko: pd.DataFrame, score_dist: dict, rng: np.random.Generator):
    group_results = {}
    team_stage = defaultdict(lambda: {'group_winner':0, 'group_runner_up':0, 'best_third':0, 'round_of_32':0, 'round_of_16':0, 'quarter_final':0, 'semi_final':0, 'final':0, 'third_place_match':0, 'champion':0, 'runner_up':0, 'third_place':0, 'fourth_place':0})
    group_pos_records = []
    match_results = {}

    # Group stage.
    group_tables = {}
    for g, gg in groups.groupby('group'):
        ts = [display(canon(t)) for t in gg['team'].tolist()]
        group_tables[g] = init_table(ts)

    for _, r in group_fixtures.sort_values(['match_date','match_id']).iterrows():
        a, b = display(canon(r['home_team'])), display(canon(r['away_team']))
        probs = prob_engine.predict(a, b, r['host_country'], 'Group stage')
        outcome, ga, gb = sample_outcome_and_score(probs, rng, score_dist)
        update_table(group_tables[r['group']], a, b, ga, gb)
        winner = a if ga>gb else (b if gb>ga else None)
        match_results[int(r['match_id'])] = {'team_a': a, 'team_b': b, 'ga':ga, 'gb':gb, 'outcome':outcome, 'winner':winner, 'loser': None}

    third_rows = []
    for g, table in group_tables.items():
        ranked = rank_group(table, rng)
        group_results[g] = {'winner': ranked[0]['team'], 'runner_up': ranked[1]['team'], 'third': ranked[2]['team'], 'ranked': ranked}
        team_stage[ranked[0]['team']]['group_winner'] += 1
        team_stage[ranked[0]['team']]['round_of_32'] += 1
        team_stage[ranked[1]['team']]['group_runner_up'] += 1
        team_stage[ranked[1]['team']]['round_of_32'] += 1
        third = dict(ranked[2]); third['group']=g
        third_rows.append(third)
        for pos, rr in enumerate(ranked, start=1):
            group_pos_records.append({'group':g, 'team':rr['team'], 'position':pos})

    ranked_thirds = rank_thirds(third_rows, rng)
    best_thirds = ranked_thirds[:8]
    for r in best_thirds:
        team_stage[r['team']]['best_third'] += 1
        team_stage[r['team']]['round_of_32'] += 1

    # Assign best thirds to their bracket slots.
    r32 = ko[ko['stage']=='Round of 32']
    slots=[]
    for _, r in r32.iterrows():
        if r['home_slot_type']=='best_third': slots.append((int(r['match_id']),'home',r['home_allowed_third_groups']))
        if r['away_slot_type']=='best_third': slots.append((int(r['match_id']),'away',r['away_allowed_third_groups']))
    third_assignment = assign_best_thirds(slots, best_thirds)

    # Knockout stage.
    stage_to_adv = {
        'Round of 32':'round_of_16',
        'Round of 16':'quarter_final',
        'Quarter-finals':'semi_final',
        'Quarter-final':'semi_final',
        'Semi-finals':'final',
        'Semi-final':'final',
        'Final':'champion'
    }
    for _, r in ko.sort_values('match_id').iterrows():
        mid = int(r['match_id'])
        a = resolve_slot(r, 'home', group_results, match_results, third_assignment)
        b = resolve_slot(r, 'away', group_results, match_results, third_assignment)
        probs = prob_engine.predict(a, b, r['host_country'], r['stage'])
        outcome, ga, gb = sample_outcome_and_score(probs, rng, score_dist)
        # Knockout winner: if 90-minute draw, resolve by non-draw strength share.
        if ga > gb:
            winner, loser = a, b
        elif gb > ga:
            winner, loser = b, a
        else:
            pa, pb = probs['p_team_a_win'], probs['p_team_b_win']
            pwina = pa / (pa + pb) if (pa + pb) > 0 else 0.5
            if rng.random() < pwina:
                winner, loser = a, b
            else:
                winner, loser = b, a
        match_results[mid] = {'team_a': a, 'team_b': b, 'ga':ga, 'gb':gb, 'outcome':outcome, 'winner':winner, 'loser':loser}
        if r['stage'] == 'Semi-finals':
            team_stage[winner]['final'] += 1
        elif r['stage'] == 'Third-place playoff':
            team_stage[a]['third_place_match'] += 1
            team_stage[b]['third_place_match'] += 1
            team_stage[winner]['third_place'] += 1
            team_stage[loser]['fourth_place'] += 1
        elif r['stage'] == 'Final':
            team_stage[winner]['champion'] += 1
            team_stage[loser]['runner_up'] += 1
        else:
            adv = stage_to_adv.get(r['stage'])
            if adv:
                team_stage[winner][adv] += 1
    return team_stage, group_pos_records, match_results


def main():
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)
    WC.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    hist = pd.read_csv(DATA/'processed'/'matches_with_elo_fifa_form_confed_exp_h2h.csv')
    hist['date'] = pd.to_datetime(hist['date'])
    current_elo = pd.read_csv(DATA/'processed'/'current_elo_ratings.csv')
    fifa = pd.read_csv(DATA/'raw'/'fifa_rankings_processed.csv')
    confed_map = pd.read_csv(DATA/'processed'/'team_confederation_mapping.csv')
    groups = pd.read_csv(WC/'worldcup_2026_groups.csv')
    group_fixtures = pd.read_csv(WC/'worldcup_2026_group_fixtures.csv')
    ko = pd.read_csv(WC/'worldcup_2026_knockout_slots.csv')

    # Canonicalize display files in-memory.
    wc_teams = sorted({canon(t) for t in groups['team']})
    missing = sorted(set(wc_teams) - (set(hist['home_team'].map(canon)) | set(hist['away_team'].map(canon))))
    if missing:
        raise RuntimeError(f'Missing historical teams after canonicalization: {missing}')

    team_stats = build_team_stats(hist, current_elo, fifa, confed_map, wc_teams + list(HOSTS_CANON))
    pair_hist = build_pair_history(hist)
    conf_strength = load_confed_strength()

    v2_meta = json.load(open(ROOT/'model_v2_metadata.json'))
    v2_features = v2_meta['features_used']
    baseline_meta = json.load(open(ROOT/'baseline_model_metadata.json'))
    model_cols = set(v2_features + baseline_meta['numeric_features'] + baseline_meta['categorical_features'])
    model_ds = pd.read_csv(DATA/'modeling'/'model_dataset_v2_curated.csv')
    medians = {c: float(model_ds[c].median()) for c in model_ds.columns if c in model_cols and pd.api.types.is_numeric_dtype(model_ds[c])}
    for c in model_cols:
        medians.setdefault(c, 0.0)

    engine = ProbabilityEngine(team_stats, pair_hist, conf_strength, medians, v2_features, baseline_meta['numeric_features'], baseline_meta['categorical_features'])
    score_dist = make_scoreline_distributions(hist[hist['date'] >= '2014-01-01'])

    # Group fixture probability file.
    prob_rows = []
    for _, r in group_fixtures.sort_values('match_id').iterrows():
        a, b = display(canon(r['home_team'])), display(canon(r['away_team']))
        p = engine.predict(a, b, r['host_country'], 'Group stage')
        prob_rows.append({
            'match_id': int(r['match_id']), 'stage': r['stage'], 'group': r['group'], 'match_date': r['match_date'],
            'stadium': r['stadium'], 'city': r['city'], 'host_country': r['host_country'],
            'team_a': a, 'team_b': b, **p,
            'most_likely_result': ['team_a_win','draw','team_b_win'][int(np.argmax([p['p_team_a_win'],p['p_draw'],p['p_team_b_win']]))]
        })
    group_probs = pd.DataFrame(prob_rows)
    group_probs.to_csv(WC/'model_v5_group_match_probabilities.csv', index=False)

    teams_display = [display(canon(t)) for t in groups['team']]
    aggregate = {t: defaultdict(int) for t in teams_display}
    group_pos_counts = defaultdict(lambda: defaultdict(int))
    champion_counts = defaultdict(int)
    matchup_counts = defaultdict(lambda: defaultdict(int))
    sample_match_rows = []

    for sim in range(1, N_SIMULATIONS+1):
        ts, pos_records, match_results = simulate_tournament(engine, group_fixtures, groups, ko, score_dist, rng)
        for t, d in ts.items():
            for k, v in d.items():
                aggregate[t][k] += v
        for rec in pos_records:
            group_pos_counts[(rec['group'], rec['team'])][rec['position']] += 1
        champ = match_results[104]['winner']
        champion_counts[champ] += 1
        for mid, mr in match_results.items():
            if mid >= 73:
                key = tuple(sorted([mr['team_a'], mr['team_b']]))
                matchup_counts[(mid, key[0], key[1])]['meetings'] += 1
                matchup_counts[(mid, key[0], key[1])][f"wins_{mr['winner']}"] += 1
        if sim <= 200:
            for mid, mr in match_results.items():
                sample_match_rows.append({'simulation': sim, 'match_id': mid, **mr})

    # Team probabilities.
    stage_cols = ['group_winner','group_runner_up','best_third','round_of_32','round_of_16','quarter_final','semi_final','final','third_place_match','champion','runner_up','third_place','fourth_place']
    team_rows=[]
    for t in sorted(teams_display):
        row={'team':t, 'canonical_team': canon(t)}
        for c in stage_cols:
            row[f'{c}_probability'] = aggregate[t][c] / N_SIMULATIONS
        row['group_stage_exit_probability'] = 1 - row['round_of_32_probability']
        row['top_4_probability'] = row['final_probability'] + row['third_place_match_probability']
        team_rows.append(row)
    team_probs=pd.DataFrame(team_rows).sort_values('champion_probability', ascending=False)
    team_probs.to_csv(WC/'model_v5_team_probabilities.csv', index=False)
    team_probs[['team','champion_probability']].to_csv(WC/'model_v5_champion_distribution.csv', index=False)

    # Group finish probabilities.
    gp_rows=[]
    for (g,t), counts in group_pos_counts.items():
        row={'group':g, 'team':t}
        for pos in [1,2,3,4]: row[f'finish_{pos}_probability']=counts[pos]/N_SIMULATIONS
        row['top2_probability'] = row['finish_1_probability'] + row['finish_2_probability']
        gp_rows.append(row)
    group_finish = pd.DataFrame(gp_rows).sort_values(['group','finish_1_probability'], ascending=[True,False])
    group_finish.to_csv(WC/'model_v5_group_finish_probabilities.csv', index=False)

    # Knockout matchup aggregate.
    mu_rows=[]
    for (mid, a, b), counts in matchup_counts.items():
        row={'match_id':mid,'team_a_alpha':a,'team_b_alpha':b,'meeting_probability':counts['meetings']/N_SIMULATIONS}
        row[f'{a}_win_given_meeting'] = counts.get(f'wins_{a}',0) / max(counts['meetings'],1)
        row[f'{b}_win_given_meeting'] = counts.get(f'wins_{b}',0) / max(counts['meetings'],1)
        mu_rows.append(row)
    pd.DataFrame(mu_rows).sort_values(['match_id','meeting_probability'], ascending=[True,False]).to_csv(WC/'model_v5_knockout_matchup_probabilities.csv', index=False)

    pd.DataFrame(sample_match_rows).to_csv(WC/'model_v5_sample_simulated_matches_first_200_runs.csv', index=False)

    # Summary stats and README.
    top = team_probs.head(12)[['team','champion_probability','final_probability','semi_final_probability','quarter_final_probability','round_of_16_probability','round_of_32_probability']]
    meta = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': 'Model v5 World Cup 2026 Monte Carlo simulator',
        'n_simulations': N_SIMULATIONS,
        'rng_seed': RNG_SEED,
        'probability_engine': 'Model v4 ensemble = 0.186 baseline logistic + 0.814 calibrated Model v3 random forest',
        'scoreline_method': 'sampled historical scoreline conditional on simulated W/D/L outcome',
        'group_tiebreakers_approximation': ['points','goal_difference','goals_for','wins','random_draw'],
        'best_third_slot_resolution': 'constraint-satisfying assignment using uploaded allowed best-third group sets',
        'knockout_draw_resolution': 'draw after 90 minutes resolved by p_team_a_win/(p_team_a_win+p_team_b_win)',
        'static_feature_note': 'Team-strength features are pre-tournament snapshots; simulated group results update standings but are not fed back into model features for later matches.',
        'outputs': {
            'team_probabilities': str(WC/'model_v5_team_probabilities.csv'),
            'champion_distribution': str(WC/'model_v5_champion_distribution.csv'),
            'group_match_probabilities': str(WC/'model_v5_group_match_probabilities.csv'),
            'group_finish_probabilities': str(WC/'model_v5_group_finish_probabilities.csv'),
            'knockout_matchup_probabilities': str(WC/'model_v5_knockout_matchup_probabilities.csv'),
            'sample_simulated_matches': str(WC/'model_v5_sample_simulated_matches_first_200_runs.csv'),
        },
        'top_12_champion_probabilities': top.to_dict(orient='records'),
    }
    json.dump(meta, open(ROOT/'model_v5_simulator_metadata.json','w'), indent=2)

    readme = f"""# Model v5: World Cup 2026 Monte Carlo Simulator

This layer plugs the Model v4 probability ensemble into the prepared World Cup 2026 fixture structure.

## Simulation design

- Simulations: **{N_SIMULATIONS:,}**
- Match probability engine: **Model v4** = 18.6% first baseline logistic model + 81.4% calibrated Model v3 random forest.
- Group stage: all 72 matches are predicted, sampled, scored, and ranked into group tables.
- Qualification: top 2 from each of 12 groups plus the 8 best third-place teams advance to the Round of 32.
- Knockout stage: slots are resolved dynamically from group outcomes and previous knockout winners/losers.
- Draws in knockout matches are resolved by a strength-based extra-time/penalty approximation.

## Important assumptions

1. Team-strength features are pre-tournament snapshots from the data available before the World Cup fixtures begin.
2. Simulated group results update tables and bracket paths, but they are not fed back into the ML feature pipeline for later match probabilities.
3. Scorelines are generated by sampling historical international scorelines conditional on the sampled W/D/L result.
4. Tie-breakers approximate FIFA ordering with points, goal difference, goals scored, wins, then random draw.
5. Best-third slot allocation is solved against the allowed group sets in the uploaded knockout fixture file.

## Main outputs

- `data/worldcup_2026/model_v5_team_probabilities.csv`
- `data/worldcup_2026/model_v5_champion_distribution.csv`
- `data/worldcup_2026/model_v5_group_match_probabilities.csv`
- `data/worldcup_2026/model_v5_group_finish_probabilities.csv`
- `data/worldcup_2026/model_v5_knockout_matchup_probabilities.csv`
- `data/worldcup_2026/model_v5_sample_simulated_matches_first_200_runs.csv`

## Top 12 title probabilities

{top.to_markdown(index=False)}
"""
    (ROOT/'README_model_v5_simulator.md').write_text(readme)

    # zip package
    zip_path = ROOT.parent / 'datacamp_worldcup_predictor_model_v5_simulator.zip'
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(zip_path).replace('.zip',''), 'zip', ROOT)
    print(json.dumps({'done': True, 'zip': str(zip_path), 'top': meta['top_12_champion_probabilities'][:5], 'cache_size': len(engine.cache)}, indent=2))

if __name__ == '__main__':
    main()
