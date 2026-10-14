#!/usr/bin/env python3
"""Synthetic-only checks of grouped folds, preprocessing and clustered regression."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from primary_model_pipeline import grouped_folds,model_pipeline,NUMERIC,CATEGORICAL
from primary_hypothesis_analysis import clustered_ols

def main():
    rng=np.random.default_rng(42);n=120
    d=pd.DataFrame({c:rng.normal(size=n) for c in NUMERIC})
    for c in CATEGORICAL:d[c]=pd.Series(np.tile(['0','1','2'],40),dtype='string')
    d['participant_key']=np.repeat(np.arange(n//2),2)
    folds,assignments=grouped_folds(d)
    assert len(assignments)==60 and d.assign(fold=folds).groupby('participant_key').fold.nunique().eq(1).all()
    again,_=grouped_folds(d);assert folds.equals(again)
    model=model_pipeline();train=folds!=0;model.fit(d.loc[train],rng.normal(size=int(train.sum())))
    assert np.isfinite(model.predict(d.loc[~train])).all()
    x=rng.normal(size=n);result=clustered_ols(1+0.3*x+rng.normal(scale=.01,size=n),x,d.participant_key.to_numpy())
    assert abs(result['beta'][1]-.3)<.01
    print('PASS: synthetic grouped splitting, preprocessing/prediction, and clustered regression')
if __name__=='__main__':main()
