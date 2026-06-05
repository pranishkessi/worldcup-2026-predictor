#!/usr/bin/env python3
from __future__ import annotations
import json, time, zipfile, shutil
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'data/processed/matches_with_elo_fifa_form_confed_exp_h2h.csv'
MODELS=ROOT/'models'; REPORTS=ROOT/'reports'; MODELING=ROOT/'data/modeling'
for d in [MODELS,REPORTS,MODELING]: d.mkdir(parents=True, exist_ok=True)
CLASSES=['H','D','A']; TARGET='result'
FEATURES=[
 'elo_diff','elo_diff_pre','elo_prob_home_win_proxy','fifa_rank_diff_filled','fifa_points_diff_filled',
 'neutral','tournament_importance','is_friendly','is_qualifier','is_world_cup','is_continental',
 'form_points_per_match_diff_last5','avg_goal_diff_delta_last5','win_rate_diff_last5','weighted_form_points_diff_last5','weighted_goal_diff_delta_last5',
 'form_points_per_match_diff_last10','avg_goal_diff_delta_last10','win_rate_diff_last10','weighted_form_points_diff_last10','weighted_goal_diff_delta_last10',
 'home_days_since_last_match_last5','away_days_since_last_match_last5',
 'same_confederation','home_confed_matches_host_confed','away_confed_matches_host_confed',
 'confed_points_per_match_diff_prior_filled','confed_goal_diff_per_match_diff_prior_filled','confed_avg_elo_diff_prior_filled',
 'confed_inter_points_per_match_diff_prior_filled','confed_inter_goal_diff_per_match_diff_prior_filled',
 'exp_total_matches_prior_diff','exp_same_tournament_matches_prior_diff','exp_same_family_matches_prior_diff','exp_world_cup_matches_prior_diff',
 'exp_continental_matches_prior_diff','exp_qualifier_matches_prior_diff','exp_nations_league_matches_prior_diff','exp_major_matches_prior_diff',
 'exp_major_points_per_match_prior_diff_filled','exp_major_goal_diff_per_match_prior_diff_filled','exp_points_per_match_prior_diff_filled',
 'exp_goal_diff_per_match_prior_diff_filled','exp_avg_tournament_importance_prior_diff_filled','exp_high_importance_matches_prior_diff',
 'exp_years_since_first_match_diff_filled','exp_major_recency_advantage_filled','both_teams_have_major_tournament_history',
 'h2h_matches_prior','h2h_home_team_points_per_match_prior_filled','h2h_goal_diff_per_match_prior_filled','h2h_goals_for_per_match_prior_filled',
 'h2h_goals_against_per_match_prior_filled','h2h_matches_last5','h2h_home_team_points_per_match_last5_filled','h2h_goal_diff_per_match_last5_filled',
 'h2h_home_team_win_rate_last5_filled','h2h_days_since_last_meeting_filled','h2h_same_tournament_matches_prior','h2h_same_family_matches_prior','has_h2h_history','has_h2h_last5'
]
BOOLS={'neutral','is_friendly','is_qualifier','is_world_cup','is_continental','same_confederation','home_confed_matches_host_confed','away_confed_matches_host_confed','both_teams_have_major_tournament_history','has_h2h_history','has_h2h_last5'}

def brier(y, proba):
    idx={c:i for i,c in enumerate(CLASSES)}; oh=np.zeros_like(proba)
    for i,v in enumerate(y): oh[i,idx[v]]=1
    return float(np.mean(np.sum((proba-oh)**2,axis=1)))
def rps(y, proba):
    idx={c:i for i,c in enumerate(CLASSES)}; vals=[]
    for yy,p in zip(y,proba):
        obs=np.zeros(len(CLASSES)); obs[idx[yy]]=1; vals.append(np.mean((np.cumsum(p)-np.cumsum(obs))**2))
    return float(np.mean(vals))
def align(model,X):
    p=model.predict_proba(X); out=np.zeros((len(X),len(CLASSES))); mc=list(model.classes_)
    for j,c in enumerate(CLASSES):
        if c in mc: out[:,j]=p[:,mc.index(c)]
    s=out.sum(1); out[s==0]=1/len(CLASSES); return out/out.sum(1,keepdims=True)
def evaluate(name,model,X,y,split):
    pred=model.predict(X); proba=align(model,X)
    return {'model':name,'split':split,'rows':len(y),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
            'macro_f1':f1_score(y,pred,average='macro'),'weighted_f1':f1_score(y,pred,average='weighted'),
            'log_loss':float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), [CLASSES.index(v) for v in y]],1e-15,1)))),'multiclass_brier':brier(np.array(y),proba),'ranked_probability_score':rps(np.array(y),proba)}

def main():
    print('loading')
    df=pd.read_csv(INPUT, low_memory=False); df['date']=pd.to_datetime(df['date']); df=df[df.result.isin(CLASSES)].copy()
    feats=[c for c in FEATURES if c in df.columns]
    for c in feats:
        if c in BOOLS: df[c]=df[c].astype(int)
    cols=list(dict.fromkeys(['match_id','date','home_team','away_team','home_score','away_score','result','tournament','neutral']+feats))
    m=df[cols].copy()
    m['split']=np.select([m.date<pd.Timestamp('2022-01-01'),(m.date>=pd.Timestamp('2022-01-01'))&(m.date<pd.Timestamp('2023-01-01')),m.date>=pd.Timestamp('2023-01-01')],['train','validation','test'], default='unused')
    m.to_csv(MODELING/'model_dataset_v2_curated.csv', index=False)
    tr=m[m.split=='train']; va=m[m.split=='validation']; te=m[m.split=='test']
    Xtr,ytr=tr[feats],tr.result; Xva,yva=va[feats],va.result; Xte,yte=te[feats],te.result
    pre=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    pre_tree=Pipeline([('impute',SimpleImputer(strategy='median'))])
    models={
      'dummy_most_frequent': DummyClassifier(strategy='most_frequent'),
      'logistic_l2_curated': Pipeline([('preprocess',pre),('model',LogisticRegression(max_iter=2000,C=0.4,class_weight='balanced',solver='lbfgs'))]),
      'logistic_kbest_35': Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('select',SelectKBest(f_classif,k=min(35,len(feats)))),('model',LogisticRegression(max_iter=2000,C=0.7,class_weight='balanced',solver='lbfgs'))]),
      'random_forest_shallow': Pipeline([('preprocess',pre_tree),('model',RandomForestClassifier(n_estimators=160,max_depth=7,min_samples_leaf=30,class_weight='balanced_subsample',random_state=42,n_jobs=-1))]),
    }
    metrics=[]
    for name,model in models.items():
        print('fit',name, flush=True); t=time.time(); model.fit(Xtr,ytr); print('done',name,time.time()-t, flush=True)
        joblib.dump(model, MODELS/f'v2_fast_model_{name}.joblib')
        for split,X,y in [('train',Xtr,ytr),('validation',Xva,yva),('test',Xte,yte)]: metrics.append(evaluate(name,model,X,y,split))
    met=pd.DataFrame(metrics); met.to_csv(MODELS/'v2_model_metrics.csv',index=False)
    best=met[(met.split=='validation')&(met.model!='dummy_most_frequent')].sort_values(['log_loss','multiclass_brier','macro_f1'],ascending=[True,True,False]).iloc[0].model
    best_model=models[best]; joblib.dump(best_model, MODELS/'v2_model_best.joblib')
    dep=clone(best_model); dep.fit(pd.concat([Xtr,Xva]), pd.concat([ytr,yva])); joblib.dump(dep, MODELS/'v2_model_best_train_plus_validation.joblib')
    # predictions and diagnostics
    parts=[]
    for split,dfx,X,y in [('validation',va,Xva,yva),('test',te,Xte,yte)]:
        pred=best_model.predict(X); proba=align(best_model,X)
        tmp=dfx[['match_id','date','home_team','away_team','home_score','away_score','result','tournament','neutral','split']].copy()
        tmp['model']=best; tmp['predicted_result']=pred; tmp['prob_home_win']=proba[:,0]; tmp['prob_draw']=proba[:,1]; tmp['prob_away_win']=proba[:,2]; tmp['confidence']=proba.max(1); tmp['correct']=(pred==np.array(y)).astype(int); parts.append(tmp)
        pd.DataFrame(confusion_matrix(y,pred,labels=CLASSES),index=[f'actual_{c}' for c in CLASSES],columns=[f'pred_{c}' for c in CLASSES]).to_csv(MODELS/f'v2_confusion_matrix_{split}.csv')
    preds=pd.concat(parts); preds.to_csv(MODELS/'v2_validation_test_predictions.csv',index=False); preds[preds.split=='test'].to_csv(MODELS/'v2_test_predictions.csv',index=False)
    reports={}
    for split,X,y in [('validation',Xva,yva),('test',Xte,yte)]: reports[split]=classification_report(y,best_model.predict(X),labels=CLASSES,output_dict=True,zero_division=0)
    (MODELS/'v2_classification_reports.json').write_text(json.dumps(reports,indent=2))
    # feature screening and importance
    Xnum=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(Xtr),columns=feats); f,p=f_classif(Xnum,ytr)
    screen=pd.DataFrame({'feature':feats,'f_score':f,'p_value':p}).sort_values('f_score',ascending=False); screen.to_csv(MODELS/'v2_numeric_feature_screening.csv',index=False)
    imp=[]
    try:
      if best.startswith('logistic'):
        if best=='logistic_kbest_35':
          names=np.array(feats)[best_model.named_steps['select'].get_support()]
          est=best_model.named_steps['model']
        else:
          names=np.array(feats); est=best_model.named_steps['model']
        for i,cls in enumerate(est.classes_):
          for feat,val in zip(names,est.coef_[i]): imp.append({'model':best,'class':cls,'feature':feat,'importance':float(val),'abs_importance':abs(float(val))})
      elif 'random_forest' in best:
        est=best_model.named_steps['model']
        for feat,val in zip(feats,est.feature_importances_): imp.append({'model':best,'feature':feat,'importance':float(val),'abs_importance':abs(float(val))})
    except Exception as e: imp=[{'error':repr(e)}]
    pd.DataFrame(imp).sort_values('abs_importance',ascending=False if imp and 'abs_importance' in imp[0] else True).to_csv(MODELS/'v2_feature_importance.csv',index=False)
    # comparisons
    fam=[]
    if (MODELS/'baseline_metrics.csv').exists():
      b=pd.read_csv(MODELS/'baseline_metrics.csv'); b=b[b.model=='logistic_regression'].copy(); b['model_family']='previous_baseline'; fam.append(b)
    if (MODELS/'exp_h2h_metrics.csv').exists():
      e=pd.read_csv(MODELS/'exp_h2h_metrics.csv'); e=e[e.model=='logistic_regression_exp_h2h'].copy(); e['model_family']='exp_h2h_logistic_all_features'; fam.append(e)
    v=met[met.model==best].copy(); v['model_family']='model_v2_best'; fam.append(v)
    family=pd.concat(fam); family.to_csv(MODELS/'v2_model_family_metrics_comparison.csv',index=False)
    rows=[]
    if len(fam)>=2:
      base=family[family.model_family=='previous_baseline']; v2=family[family.model_family=='model_v2_best']
      for split in ['validation','test']:
        br=base[base.split==split].iloc[0]; nr=v2[v2.split==split].iloc[0]
        for metric in ['accuracy','balanced_accuracy','macro_f1','weighted_f1','log_loss','multiclass_brier','ranked_probability_score']:
          higher=metric in ['accuracy','balanced_accuracy','macro_f1','weighted_f1']; change=float(nr[metric]-br[metric]); rows.append({'split':split,'metric':metric,'baseline':float(br[metric]),'model_v2':float(nr[metric]),'change':change,'direction':'higher_better' if higher else 'lower_better','improved':bool(change>0 if higher else change<0)})
    comp=pd.DataFrame(rows); comp.to_csv(MODELS/'baseline_vs_v2_comparison.csv',index=False)
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_file':str(INPUT.relative_to(ROOT)),'modeling_dataset':'data/modeling/model_dataset_v2_curated.csv','rows':{'train':len(tr),'validation':len(va),'test':len(te),'total':len(m)},'candidate_models':list(models.keys()),'selection_metric':'validation log_loss','best_model':best,'features_used':feats,'best_validation':met[(met.model==best)&(met.split=='validation')].iloc[0].to_dict(),'best_test':met[(met.model==best)&(met.split=='test')].iloc[0].to_dict()}
    (ROOT/'model_v2_metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    # report
    cand_val=met[met.split=='validation'][['model','accuracy','macro_f1','log_loss','multiclass_brier','ranked_probability_score']].sort_values('log_loss')
    cand_test=met[met.split=='test'][['model','accuracy','macro_f1','log_loss','multiclass_brier','ranked_probability_score']].sort_values('log_loss')
    def family_table(split): return family[family.split==split][['model_family','model','accuracy','macro_f1','log_loss','multiclass_brier','ranked_probability_score']].sort_values('log_loss').to_markdown(index=False,floatfmt='.4f')
    best_val=met[(met.model==best)&(met.split=='validation')].iloc[0]; best_test=met[(met.model==best)&(met.split=='test')].iloc[0]
    if not comp.empty:
      tl=comp[(comp.split=='test')&(comp.metric=='log_loss')].iloc[0]; ta=comp[(comp.split=='test')&(comp.metric=='accuracy')].iloc[0]
      verdict=f"Compared with the first baseline, Model v2 test log loss changed from **{tl.baseline:.4f}** to **{tl.model_v2:.4f}** ({tl.change:+.4f}), and test accuracy changed from **{ta.baseline:.4f}** to **{ta.model_v2:.4f}** ({ta.change:+.4f})."
    else: verdict='Baseline comparison was not available.'
    top_imp=pd.read_csv(MODELS/'v2_feature_importance.csv') if (MODELS/'v2_feature_importance.csv').exists() else pd.DataFrame()
    top_imp_md=top_imp.head(25).to_markdown(index=False,floatfmt='.4f') if len(top_imp) else 'No model-specific importance available.'
    report=f"""# Model v2: Feature Selection + Stronger Classifiers\n\n## Goal\nTrain a stronger second milestone using a compact, selected feature set and several model families. Selection is based on **validation log loss**.\n\n## Time-safe split\n\n| Split | Date range | Rows |\n|---|---:|---:|\n| Train | 2014-01-01 to 2021-12-31 | {len(tr):,} |\n| Validation | 2022-01-01 to 2022-12-31 | {len(va):,} |\n| Test | 2023-01-01 to 2026-06-02 | {len(te):,} |\n\n## Candidate ranking on validation\n\n{cand_val.to_markdown(index=False,floatfmt='.4f')}\n\n## Candidate ranking on test\n\n{cand_test.to_markdown(index=False,floatfmt='.4f')}\n\n## Best selected model\n\n`{best}`\n\n| Metric | Validation | Test |\n|---|---:|---:|\n| Accuracy | {best_val.accuracy:.4f} | {best_test.accuracy:.4f} |\n| Macro F1 | {best_val.macro_f1:.4f} | {best_test.macro_f1:.4f} |\n| Log loss | {best_val.log_loss:.4f} | {best_test.log_loss:.4f} |\n| Multiclass Brier | {best_val.multiclass_brier:.4f} | {best_test.multiclass_brier:.4f} |\n| Ranked probability score | {best_val.ranked_probability_score:.4f} | {best_test.ranked_probability_score:.4f} |\n\n## Family comparison: validation\n\n{family_table('validation')}\n\n## Family comparison: test\n\n{family_table('test')}\n\n## Verdict\n\n{verdict}\n\nA lower log loss is better for probability prediction. A higher accuracy is better for hard W/D/L classification.\n\n## Top model feature signals\n\n{top_imp_md}\n\n## Top numeric screening signals\n\n{screen.head(25).to_markdown(index=False,floatfmt='.4f')}\n\n## Output files\n\n- `data/modeling/model_dataset_v2_curated.csv`\n- `models/v2_model_metrics.csv`\n- `models/v2_model_family_metrics_comparison.csv`\n- `models/baseline_vs_v2_comparison.csv`\n- `models/v2_validation_test_predictions.csv`\n- `models/v2_test_predictions.csv`\n- `models/v2_feature_importance.csv`\n- `models/v2_numeric_feature_screening.csv`\n- `models/v2_confusion_matrix_validation.csv`\n- `models/v2_confusion_matrix_test.csv`\n- `models/v2_classification_reports.json`\n- `models/v2_model_best.joblib`\n- `models/v2_model_best_train_plus_validation.joblib`\n- `model_v2_metadata.json`\n"""
    (REPORTS/'model_v2_report.md').write_text(report)
    print(json.dumps({'status':'ok','best_model':best,'validation':meta['best_validation'],'test':meta['best_test'],'baseline_vs_v2_test':comp[comp.split=='test'].to_dict(orient='records')},indent=2,default=str))
if __name__=='__main__': main()
