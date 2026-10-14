#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20260721
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'outputs/oai_knee_analysis_ready.csv.gz'
PRIMARY_PRED_PATH = ROOT / 'outputs/primary_model/heldout_predictions.csv.gz'
OUTER_ASSIGN_PATH = ROOT / 'outputs/primary_model/frozen_outer_fold_assignments.csv'
OUTDIR = ROOT / 'outputs/sensitivity'
OUTDIR.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    'womac_pain_v00','medial_jsn_v00','lateral_jsn_v00','age_years','bmi',
    'prior_knee_injury','prior_knee_surgery'
]
CATEGORICAL = ['kl_grade_v00','sex','race','hispanic']
FEATURES = NUMERIC + CATEGORICAL
GRID = {
    'model__alpha':[0.001,0.003,0.01,0.03,0.1,0.3,1.0],
    'model__l1_ratio':[0.05,0.25,0.5,0.75,0.95],
}


def elastic_pipeline() -> Pipeline:
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    pre = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('categorical', categorical, CATEGORICAL),
    ])
    return Pipeline([
        ('preprocess', pre),
        ('model', ElasticNet(max_iter=50_000, selection='cyclic')),
    ])


def metrics(obs, pred):
    slope, intercept = np.polyfit(pred, obs, 1)
    return {
        'mae': float(mean_absolute_error(obs, pred)),
        'rmse': float(mean_squared_error(obs, pred) ** 0.5),
        'mean_bias_predicted_minus_observed': float(np.mean(pred - obs)),
        'calibration_intercept': float(intercept),
        'calibration_slope': float(slope),
    }


def clustered_ols_multi(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    if x.ndim == 1:
        x = x[:,None]
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    resid = y - design @ beta
    ug = np.unique(groups)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for g in ug:
        mask = groups == g
        score = design[mask].T @ resid[mask]
        meat += np.outer(score, score)
    n,k,G = len(y), design.shape[1], len(ug)
    correction = (G/(G-1))*((n-1)/(n-k))
    cov = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    z = beta/se
    p = 2*norm.sf(np.abs(z))
    lo = beta - norm.ppf(.975)*se
    hi = beta + norm.ppf(.975)*se
    return beta,se,p,lo,hi,cov


def nested_elastic_predictions(data: pd.DataFrame, outcome: str, label: str, fixed_assignments: pd.DataFrame|None=None):
    data = data.copy().reset_index(drop=True)
    data['kl_grade_v00'] = data['kl_grade_v00'].astype('Int64').astype('string')
    if fixed_assignments is None:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        data['outer_fold'] = -1
        for fold,(_,te) in enumerate(splitter.split(np.zeros((len(data),1)),groups=data.participant_key)):
            data.loc[te,'outer_fold'] = fold
    else:
        data = data.merge(fixed_assignments[['participant_key','outer_fold']], on='participant_key', how='left', validate='many_to_one')
        if data.outer_fold.isna().any():
            raise AssertionError('Missing fixed outer fold assignment')
        data['outer_fold'] = data.outer_fold.astype(int)

    frames=[]; tunes=[]
    for outer in range(5):
        train=data[data.outer_fold!=outer].copy(); test=data[data.outer_fold==outer].copy()
        if set(train.participant_key) & set(test.participant_key):
            raise AssertionError('Participant leakage')
        expected_by_kl=train.groupby('kl_grade_v00',observed=True).womac_pain_v00.mean()
        expected=test.kl_grade_v00.map(expected_by_kl).astype(float)
        if expected.isna().any():
            raise AssertionError('Test KL absent from training')
        signed=test.womac_pain_v00.to_numpy() - expected.to_numpy()
        inner=GroupKFold(n_splits=5,shuffle=True,random_state=SEED+outer+1)
        search=GridSearchCV(elastic_pipeline(),GRID,scoring='neg_mean_absolute_error',cv=inner,n_jobs=-1,refit=True)
        search.fit(train[FEATURES],train[outcome],groups=train.participant_key)
        pred=search.predict(test[FEATURES])
        frames.append(pd.DataFrame({
            'participant_key':test.participant_key.astype(int).to_numpy(),
            'knee_key':test.knee_key.to_numpy(),
            'outer_fold':outer,
            'observed':test[outcome].astype(float).to_numpy(),
            'predicted':pred,
            'signed_discordance':signed,
            'absolute_discordance':np.abs(signed),
            'absolute_error':np.abs(test[outcome].to_numpy()-pred),
        }))
        tunes.append({'outer_fold':outer,'alpha':search.best_params_['model__alpha'],'l1_ratio':search.best_params_['model__l1_ratio'],'inner_cv_mae':-search.best_score_})
    pred=pd.concat(frames,ignore_index=True).sort_values('knee_key').reset_index(drop=True)
    if len(pred)!=len(data) or not pred.knee_key.is_unique:
        raise AssertionError('Prediction count/uniqueness failed')
    fit=clustered_ols_multi(pred.absolute_error.to_numpy(),pred.absolute_discordance.to_numpy(),pred.participant_key.to_numpy())
    beta,se,p,lo,hi,_=fit
    result={
        'analysis':label,
        'knees':int(len(pred)),
        'participants':int(pred.participant_key.nunique()),
        'discordance_error_slope':float(beta[1]),
        'slope_se_clustered':float(se[1]),
        'ci_95_low':float(lo[1]),
        'ci_95_high':float(hi[1]),
        'p_value':float(p[1]),
        **metrics(pred.observed.to_numpy(),pred.predicted.to_numpy()),
    }
    return pred,pd.DataFrame(tunes),result


def s1(full):
    d=full[(full.womac_pain_v06.notna()) & (full.kl_grade_v00>=2)].copy()
    pred,tune,res=nested_elastic_predictions(d,'womac_pain_v06','S1 KL>=2 restriction')
    pred.to_csv(OUTDIR/'s1_kl2_predictions.csv.gz',index=False,compression='gzip')
    tune.to_csv(OUTDIR/'s1_kl2_tuning.csv',index=False)
    return res


def s2(primary):
    pos=np.clip(primary.signed_discordance.to_numpy(),0,None)
    neg=np.clip(-primary.signed_discordance.to_numpy(),0,None)
    beta,se,p,lo,hi,cov=clustered_ols_multi(primary.absolute_error.to_numpy(),np.column_stack([pos,neg]),primary.participant_key.to_numpy())
    # Wald test H0: beta_pos = beta_neg
    diff=beta[1]-beta[2]
    var=cov[1,1]+cov[2,2]-2*cov[1,2]
    se_diff=np.sqrt(var)
    z=diff/se_diff
    p_diff=float(2*norm.sf(abs(z)))
    return {
        'analysis':'S2 signed discordance asymmetry',
        'knees':int(len(primary)),'participants':int(primary.participant_key.nunique()),
        'positive_discordance_slope':float(beta[1]),'positive_ci_low':float(lo[1]),'positive_ci_high':float(hi[1]),'positive_p':float(p[1]),
        'negative_discordance_magnitude_slope':float(beta[2]),'negative_ci_low':float(lo[2]),'negative_ci_high':float(hi[2]),'negative_p':float(p[2]),
        'slope_difference_positive_minus_negative':float(diff),'difference_se':float(se_diff),'wald_p_equal_slopes':p_diff,
    }


def s3(full):
    d=full[full.womac_pain_v06.notna()].copy()
    d['womac_change_48m']=d.womac_pain_v06-d.womac_pain_v00
    pred,tune,res=nested_elastic_predictions(d,'womac_change_48m','S3 48-month WOMAC change outcome')
    pred.to_csv(OUTDIR/'s3_change_predictions.csv.gz',index=False,compression='gzip')
    tune.to_csv(OUTDIR/'s3_change_tuning.csv',index=False)
    return res


def s4(full, primary, assignments):
    d=full[full.womac_pain_v06.notna()].copy().merge(assignments,on='participant_key',how='left',validate='many_to_one')
    d['kl_grade_v00']=d.kl_grade_v00.astype('Int64').astype('string')
    adjusted=[]
    adj_num=['age_years','bmi','prior_knee_injury','prior_knee_surgery']
    adj_cat=['kl_grade_v00','sex']
    for outer in range(5):
        tr=d[d.outer_fold!=outer].copy(); te=d[d.outer_fold==outer].copy()
        pre=ColumnTransformer([
            ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),adj_num),
            ('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='Missing')),('oh',OneHotEncoder(handle_unknown='ignore'))]),adj_cat),
        ])
        pipe=Pipeline([('pre',pre),('model',LinearRegression())])
        pipe.fit(tr[adj_num+adj_cat],tr.womac_pain_v00)
        expected=pipe.predict(te[adj_num+adj_cat])
        adjusted.append(pd.DataFrame({'knee_key':te.knee_key.to_numpy(),'adjusted_signed_discordance':te.womac_pain_v00.to_numpy()-expected}))
    adj=pd.concat(adjusted,ignore_index=True)
    m=primary.merge(adj,on='knee_key',how='left',validate='one_to_one')
    m['adjusted_absolute_discordance']=m.adjusted_signed_discordance.abs()
    beta,se,p,lo,hi,_=clustered_ols_multi(m.absolute_error.to_numpy(),m.adjusted_absolute_discordance.to_numpy(),m.participant_key.to_numpy())
    m[['participant_key','knee_key','adjusted_signed_discordance','adjusted_absolute_discordance','absolute_error']].to_csv(OUTDIR/'s4_adjusted_discordance.csv.gz',index=False,compression='gzip')
    return {
        'analysis':'S4 covariate-adjusted discordance',
        'knees':int(len(m)),'participants':int(m.participant_key.nunique()),
        'discordance_error_slope':float(beta[1]),'slope_se_clustered':float(se[1]),'ci_95_low':float(lo[1]),'ci_95_high':float(hi[1]),'p_value':float(p[1]),
    }


def s5(full):
    cols=FEATURES+['womac_pain_v06']
    d=full.dropna(subset=cols).copy()
    pred,tune,res=nested_elastic_predictions(d,'womac_pain_v06','S5 complete-case analysis')
    pred.to_csv(OUTDIR/'s5_complete_case_predictions.csv.gz',index=False,compression='gzip')
    tune.to_csv(OUTDIR/'s5_complete_case_tuning.csv',index=False)
    return res


def s6(full):
    d=full[full.womac_pain_v06.notna()].copy().sort_values(['participant_key','knee_key'])
    rng=np.random.default_rng(SEED)
    chosen=[]
    for _,g in d.groupby('participant_key',sort=True):
        chosen.append(g.index[int(rng.integers(0,len(g)))])
    d=d.loc[chosen].copy().sort_values('participant_key').reset_index(drop=True)
    pred,tune,res=nested_elastic_predictions(d,'womac_pain_v06','S6 one knee per participant')
    pred.to_csv(OUTDIR/'s6_one_knee_predictions.csv.gz',index=False,compression='gzip')
    tune.to_csv(OUTDIR/'s6_one_knee_tuning.csv',index=False)
    pd.DataFrame({'knee_key':d.knee_key}).to_csv(OUTDIR/'s6_selected_knees.csv',index=False)
    return res


def gbt_pipeline():
    numeric=Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True))])
    categorical=Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='Missing')),('oh',OneHotEncoder(handle_unknown='ignore'))])
    pre=ColumnTransformer([('num',numeric,NUMERIC),('cat',categorical,CATEGORICAL)])
    model=XGBRegressor(
        objective='reg:squarederror', tree_method='hist', random_state=SEED,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
    )
    return Pipeline([('pre',pre),('model',model)])

def s7(full, primary, assignments):
    d=full[full.womac_pain_v06.notna()].copy().merge(assignments,on='participant_key',how='left',validate='many_to_one')
    d['kl_grade_v00']=d.kl_grade_v00.astype('Int64').astype('string')
    grid={'model__n_estimators':[200,500],'model__max_depth':[2,3],'model__learning_rate':[0.03,0.10]}
    frames=[]; tunes=[]
    for outer in range(5):
        tr=d[d.outer_fold!=outer].copy(); te=d[d.outer_fold==outer].copy()
        inner=GroupKFold(n_splits=5,shuffle=True,random_state=SEED+outer+1)
        search=GridSearchCV(gbt_pipeline(),grid,scoring='neg_mean_absolute_error',cv=inner,n_jobs=1,refit=True)
        search.fit(tr[FEATURES],tr.womac_pain_v06,groups=tr.participant_key)
        pr=search.predict(te[FEATURES])
        frames.append(pd.DataFrame({'participant_key':te.participant_key.astype(int).to_numpy(),'knee_key':te.knee_key.to_numpy(),'outer_fold':outer,'observed':te.womac_pain_v06.to_numpy(),'predicted':pr,'absolute_error_gbt':np.abs(te.womac_pain_v06.to_numpy()-pr)}))
        tunes.append({'outer_fold':outer,**search.best_params_,'inner_cv_mae':-search.best_score_})
    g=pd.concat(frames,ignore_index=True).sort_values('knee_key').reset_index(drop=True)
    m=g.merge(primary[['knee_key','absolute_error','absolute_discordance']],on='knee_key',how='left',validate='one_to_one')
    beta,se,p,lo,hi,_=clustered_ols_multi(m.absolute_error_gbt.to_numpy(),m.absolute_discordance.to_numpy(),m.participant_key.to_numpy())
    mm=metrics(m.observed.to_numpy(),m.predicted.to_numpy())
    # paired participant bootstrap: GBT MAE minus elastic-net MAE
    rng=np.random.default_rng(SEED+77)
    group_idx={k:g.index.to_numpy() for k,g in m.groupby('participant_key')}
    keys=np.array(list(group_idx))
    diffs=[]
    for _ in range(1000):
        sampled=rng.choice(keys,size=len(keys),replace=True)
        idx=np.concatenate([group_idx[k] for k in sampled])
        s=m.loc[idx]
        diffs.append(s.absolute_error_gbt.mean()-s.absolute_error.mean())
    diffs=np.asarray(diffs)
    point=float(m.absolute_error_gbt.mean()-m.absolute_error.mean())
    out={
        'analysis':'S7 gradient-boosted-tree comparator','knees':int(len(m)),'participants':int(m.participant_key.nunique()),
        **mm,
        'mae_difference_gbt_minus_elastic':point,
        'mae_difference_ci_95_low':float(np.quantile(diffs,.025)),'mae_difference_ci_95_high':float(np.quantile(diffs,.975)),
        'gbt_discordance_error_slope':float(beta[1]),'gbt_slope_ci_95_low':float(lo[1]),'gbt_slope_ci_95_high':float(hi[1]),'gbt_slope_p_value':float(p[1]),
    }
    m.to_csv(OUTDIR/'s7_gbt_predictions_comparison.csv.gz',index=False,compression='gzip')
    pd.DataFrame(tunes).to_csv(OUTDIR/'s7_gbt_tuning.csv',index=False)
    pd.DataFrame({'gbt_minus_elastic_mae':diffs}).to_csv(OUTDIR/'s7_paired_participant_bootstrap_mae_difference.csv.gz',index=False,compression='gzip')
    return out


def main():
    full=pd.read_csv(DATA_PATH)
    primary=pd.read_csv(PRIMARY_PRED_PATH)
    assignments=pd.read_csv(OUTER_ASSIGN_PATH)
    results=[]
    for name,func in [
        ('S1',lambda:s1(full)),
        ('S2',lambda:s2(primary)),
        ('S3',lambda:s3(full)),
        ('S4',lambda:s4(full,primary,assignments)),
        ('S5',lambda:s5(full)),
        ('S6',lambda:s6(full)),
        ('S7',lambda:s7(full,primary,assignments)),
    ]:
        print(f'Running {name}...',flush=True)
        res=func(); results.append(res)
        (OUTDIR/f'{name.lower()}_result.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
        print(json.dumps(res,indent=2),flush=True)
    # normalize union of keys for csv
    pd.DataFrame(results).to_csv(OUTDIR/'sensitivity_summary.csv',index=False)
    (OUTDIR/'sensitivity_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
