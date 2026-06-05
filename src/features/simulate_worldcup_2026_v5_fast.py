#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, random, sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'; WC=DATA/'worldcup_2026'; MODELS=ROOT/'models'
sys.path.insert(0, str(ROOT/'src'/'features'))
import simulate_worldcup_2026_v5 as sim

N_SIMULATIONS=1000
RNG_SEED=20260603

class BatchProbabilityEngine:
    def __init__(self, team_stats, pair_hist, conf_strength, medians, v2_features, teams_display, hosts, stages):
        self.team_stats=team_stats; self.pair_hist=pair_hist; self.conf_strength=conf_strength; self.medians=medians; self.v2_features=v2_features
        self.baseline=joblib.load(MODELS/'baseline_model_best.joblib')
        self.v2=joblib.load(MODELS/'v2_model_best_train_plus_validation.joblib')
        temp=json.load(open(ROOT/'model_v3_calibration_metadata.json'))['temperature']
        self.cal=sim.TemperatureCalibrator(temp)
        self.cache={}
        self._precompute(teams_display, hosts, stages)
    def _precompute(self, teams_display, hosts, stages):
        rows=[]; keys=[]; inv_flags=[]
        teams=list(dict.fromkeys([sim.display(sim.canon(t)) for t in teams_display]))
        for host in hosts:
            for stage in stages:
                for a in teams:
                    for b in teams:
                        if a==b: continue
                        ca, cb, ch = sim.canon(a), sim.canon(b), sim.canon(host)
                        inv=False; aa, bb = ca, cb
                        if cb in sim.HOSTS_CANON and cb == ch and not (ca in sim.HOSTS_CANON and ca == ch):
                            aa, bb = cb, ca; inv=True
                        row=sim.base_feature_row(aa, bb, ch, stage, self.team_stats, self.pair_hist, self.conf_strength, self.medians)
                        rows.append(row); keys.append((ca,cb,ch,stage)); inv_flags.append(inv)
        df=pd.DataFrame(rows)
        p_base=self.baseline.predict_proba(df)[:,[2,1,0]]
        p_v2=self.v2.predict_proba(df.reindex(columns=self.v2_features))[:,[2,1,0]]
        p_v3=self.cal.transform(p_v2)
        p=0.186*p_base+0.814*p_v3
        p=p/p.sum(axis=1, keepdims=True)
        for key, inv, probs in zip(keys, inv_flags, p):
            if inv:
                probs=np.array([probs[2], probs[1], probs[0]])
            self.cache[key]={'p_team_a_win':float(probs[0]), 'p_draw':float(probs[1]), 'p_team_b_win':float(probs[2])}
    def predict(self, team_a_display, team_b_display, host_country, stage):
        key=(sim.canon(team_a_display), sim.canon(team_b_display), sim.canon(host_country), stage)
        if key not in self.cache:
            # Rare fallback for any unexpected team/stage.
            ca, cb, ch=key
            row=sim.base_feature_row(ca, cb, ch, stage, self.team_stats, self.pair_hist, self.conf_strength, self.medians)
            df=pd.DataFrame([row]); p_base=self.baseline.predict_proba(df)[0][[2,1,0]]; p_v2=self.v2.predict_proba(df.reindex(columns=self.v2_features))[0][[2,1,0]]
            p_v3=self.cal.transform(p_v2.reshape(1,-1))[0]; p=0.186*p_base+0.814*p_v3; p=p/p.sum()
            self.cache[key]={'p_team_a_win':float(p[0]), 'p_draw':float(p[1]), 'p_team_b_win':float(p[2])}
        return self.cache[key]

def main():
    np.random.seed(RNG_SEED); random.seed(RNG_SEED); rng=np.random.default_rng(RNG_SEED)
    hist=pd.read_csv(DATA/'processed'/'matches_with_elo_fifa_form_confed_exp_h2h.csv'); hist['date']=pd.to_datetime(hist['date'])
    current_elo=pd.read_csv(DATA/'processed'/'current_elo_ratings.csv')
    fifa=pd.read_csv(DATA/'raw'/'fifa_rankings_processed.csv')
    confed_map=pd.read_csv(DATA/'processed'/'team_confederation_mapping.csv')
    groups=pd.read_csv(WC/'worldcup_2026_groups.csv')
    group_fixtures=pd.read_csv(WC/'worldcup_2026_group_fixtures.csv')
    ko=pd.read_csv(WC/'worldcup_2026_knockout_slots.csv')
    wc_teams=sorted({sim.canon(t) for t in groups['team']})
    team_stats=sim.build_team_stats(hist, current_elo, fifa, confed_map, wc_teams + list(sim.HOSTS_CANON))
    pair_hist=sim.build_pair_history(hist); conf_strength=sim.load_confed_strength()
    v2_features=json.load(open(ROOT/'model_v2_metadata.json'))['features_used']
    base_meta=json.load(open(ROOT/'baseline_model_metadata.json'))
    model_cols=set(v2_features + base_meta['numeric_features'] + base_meta['categorical_features'])
    model_ds=pd.read_csv(DATA/'modeling'/'model_dataset_v2_curated.csv')
    medians={c:float(model_ds[c].median()) for c in model_ds.columns if c in model_cols and pd.api.types.is_numeric_dtype(model_ds[c])}
    for c in model_cols: medians.setdefault(c,0.0)
    teams_display=[sim.display(sim.canon(t)) for t in groups['team']]
    hosts=sorted(set(group_fixtures['host_country']).union(set(ko['host_country'])))
    stages=['Group stage']+ko['stage'].drop_duplicates().tolist()
    engine=BatchProbabilityEngine(team_stats,pair_hist,conf_strength,medians,v2_features,teams_display,hosts,stages)
    score_dist=sim.make_scoreline_distributions(hist[hist['date']>='2014-01-01'])
    # save group probabilities
    prob_rows=[]
    for _,r in group_fixtures.sort_values('match_id').iterrows():
        a,b=sim.display(sim.canon(r['home_team'])), sim.display(sim.canon(r['away_team']))
        p=engine.predict(a,b,r['host_country'],'Group stage')
        prob_rows.append({'match_id':int(r['match_id']),'stage':r['stage'],'group':r['group'],'match_date':r['match_date'],'stadium':r['stadium'],'city':r['city'],'host_country':r['host_country'],'team_a':a,'team_b':b,**p,'most_likely_result':['team_a_win','draw','team_b_win'][int(np.argmax([p['p_team_a_win'],p['p_draw'],p['p_team_b_win']]))]})
    pd.DataFrame(prob_rows).to_csv(WC/'model_v5_group_match_probabilities.csv', index=False)
    aggregate={t:defaultdict(int) for t in teams_display}; group_pos_counts=defaultdict(lambda:defaultdict(int)); matchup_counts=defaultdict(lambda:defaultdict(int)); sample=[]
    for run in range(1,N_SIMULATIONS+1):
        ts,pos_records,match_results=sim.simulate_tournament(engine, group_fixtures, groups, ko, score_dist, rng)
        for t,d in ts.items():
            for k,v in d.items(): aggregate[t][k]+=v
        for rec in pos_records: group_pos_counts[(rec['group'],rec['team'])][rec['position']]+=1
        for mid,mr in match_results.items():
            if mid>=73:
                a,b=tuple(sorted([mr['team_a'],mr['team_b']]))
                matchup_counts[(mid,a,b)]['meetings']+=1; matchup_counts[(mid,a,b)][f"wins_{mr['winner']}"]+=1
            if run<=200: sample.append({'simulation':run,'match_id':mid,**mr})
    stage_cols=['group_winner','group_runner_up','best_third','round_of_32','round_of_16','quarter_final','semi_final','final','third_place_match','champion','runner_up','third_place','fourth_place']
    team_rows=[]
    for t in sorted(teams_display):
        row={'team':t,'canonical_team':sim.canon(t)}
        for c in stage_cols: row[f'{c}_probability']=aggregate[t][c]/N_SIMULATIONS
        row['group_stage_exit_probability']=1-row['round_of_32_probability']; row['top_4_probability']=row['final_probability']+row['third_place_match_probability']
        team_rows.append(row)
    team_probs=pd.DataFrame(team_rows).sort_values('champion_probability', ascending=False)
    team_probs.to_csv(WC/'model_v5_team_probabilities.csv', index=False)
    team_probs[['team','champion_probability']].to_csv(WC/'model_v5_champion_distribution.csv', index=False)
    gp=[]
    for (g,t),counts in group_pos_counts.items():
        row={'group':g,'team':t}
        for pos in [1,2,3,4]: row[f'finish_{pos}_probability']=counts[pos]/N_SIMULATIONS
        row['top2_probability']=row['finish_1_probability']+row['finish_2_probability']; gp.append(row)
    pd.DataFrame(gp).sort_values(['group','finish_1_probability'], ascending=[True,False]).to_csv(WC/'model_v5_group_finish_probabilities.csv', index=False)
    mu=[]
    for (mid,a,b),counts in matchup_counts.items():
        meetings=counts['meetings']; row={'match_id':mid,'team_a_alpha':a,'team_b_alpha':b,'meeting_probability':meetings/N_SIMULATIONS, f'{a}_win_given_meeting':counts.get(f'wins_{a}',0)/max(meetings,1), f'{b}_win_given_meeting':counts.get(f'wins_{b}',0)/max(meetings,1)}; mu.append(row)
    pd.DataFrame(mu).sort_values(['match_id','meeting_probability'], ascending=[True,False]).to_csv(WC/'model_v5_knockout_matchup_probabilities.csv', index=False)
    pd.DataFrame(sample).to_csv(WC/'model_v5_sample_simulated_matches_first_200_runs.csv', index=False)
    top=team_probs.head(12)[['team','champion_probability','final_probability','semi_final_probability','quarter_final_probability','round_of_16_probability','round_of_32_probability']]
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'model_name':'Model v5 World Cup 2026 Monte Carlo simulator','n_simulations':N_SIMULATIONS,'rng_seed':RNG_SEED,'probability_engine':'Model v4 ensemble = 0.186 baseline logistic + 0.814 calibrated Model v3 random forest','scoreline_method':'sampled historical scoreline conditional on simulated W/D/L outcome','group_tiebreakers_approximation':['points','goal_difference','goals_for','wins','random_draw'],'best_third_slot_resolution':'constraint-satisfying assignment using uploaded allowed best-third group sets','knockout_draw_resolution':'draw after 90 minutes resolved by p_team_a_win/(p_team_a_win+p_team_b_win)','static_feature_note':'Team-strength features are pre-tournament snapshots; simulated group results update standings but are not fed back into model features for later matches.','top_12_champion_probabilities':top.to_dict(orient='records')}
    json.dump(meta, open(ROOT/'model_v5_simulator_metadata.json','w'), indent=2)
    readme=f"""# Model v5: World Cup 2026 Monte Carlo Simulator\n\nThis layer plugs the Model v4 probability ensemble into the prepared World Cup 2026 fixture structure.\n\n## Simulation design\n\n- Simulations: **{N_SIMULATIONS:,}**\n- Match probability engine: **Model v4** = 18.6% first baseline logistic model + 81.4% calibrated Model v3 random forest.\n- Group stage: all 72 matches are predicted, sampled, scored, and ranked into group tables.\n- Qualification: top 2 from each of 12 groups plus the 8 best third-place teams advance to the Round of 32.\n- Knockout stage: slots are resolved dynamically from group outcomes and previous knockout winners/losers.\n- Draws in knockout matches are resolved by a strength-based extra-time/penalty approximation.\n\n## Important assumptions\n\n1. Team-strength features are pre-tournament snapshots from data available before the World Cup fixtures begin.\n2. Simulated group results update tables and bracket paths, but they are not fed back into the ML feature pipeline for later match probabilities.\n3. Scorelines are generated by sampling historical international scorelines conditional on the sampled W/D/L result.\n4. Tie-breakers approximate FIFA ordering with points, goal difference, goals scored, wins, then random draw.\n5. Best-third slot allocation is solved against the allowed group sets in the uploaded knockout fixture file.\n\n## Main outputs\n\n- `data/worldcup_2026/model_v5_team_probabilities.csv`\n- `data/worldcup_2026/model_v5_champion_distribution.csv`\n- `data/worldcup_2026/model_v5_group_match_probabilities.csv`\n- `data/worldcup_2026/model_v5_group_finish_probabilities.csv`\n- `data/worldcup_2026/model_v5_knockout_matchup_probabilities.csv`\n- `data/worldcup_2026/model_v5_sample_simulated_matches_first_200_runs.csv`\n\n## Top 12 title probabilities\n\n{top.to_markdown(index=False)}\n"""
    (ROOT/'README_model_v5_simulator.md').write_text(readme)
    zip_path=ROOT.parent / 'datacamp_worldcup_predictor_model_v5_simulator.zip'
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(zip_path).replace('.zip',''),'zip',ROOT)
    print(json.dumps({'done':True,'n_simulations':N_SIMULATIONS,'zip':str(zip_path),'top5':top.head(5).to_dict(orient='records'),'cache_size':len(engine.cache)}, indent=2))
if __name__=='__main__': main()
