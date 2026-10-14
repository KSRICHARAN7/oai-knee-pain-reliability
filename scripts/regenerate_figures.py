#!/usr/bin/env python3
"""Public-release figures from aggregate tables only; no fitting or row data."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results/tables'
OUT = ROOT / 'local_runs/figures'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'oai-public-v1','axes.spines.top':False,'axes.spines.right':False})
    metadata=[]
    def save(fig,name,sources):
        fig.tight_layout()
        fig.savefig(OUT/f'{name}.svg',metadata={'Date':None,'Creator':'OAI public aggregate figure generator'})
        plt.close(fig)
        metadata.append({'figure':f'{name}.svg','sources':[{'path':'results/tables/'+s,'sha256':hashlib.sha256((TABLES/s).read_bytes()).hexdigest()} for s in sources],'status':'Public-release presentation; original aggregate values unchanged'})
    d=pd.read_csv(TABLES/'risk_coverage.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(d.coverage_percent,d.targeted_mae,'o-',color='#087e8b',label='Targeted deferral')
    ax.plot(d.coverage_percent,d.random_abstention_mean_mae,'o--',color='#76818e',label='Random deferral')
    ax.fill_between(d.coverage_percent,d.random_abstention_ci_2_5,d.random_abstention_ci_97_5,color='#76818e',alpha=.15,label='Random deferral 95% reference interval')
    ax.set(xlabel='Retained coverage (%)',ylabel='Retained-case MAE (WOMAC points)',title='Prediction error and retained coverage')
    ax.legend(fontsize=9); save(fig,'risk_coverage',['risk_coverage.csv'])
    d=pd.read_csv(TABLES/'error_by_discordance_quartile_with_ci.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.errorbar(d.discordance_quartile,d.absolute_error_mean,yerr=[d.absolute_error_mean-d.ci_2_5,d.ci_97_5-d.absolute_error_mean],fmt='o',capsize=5,color='#087e8b')
    ax.set(xlabel='Absolute baseline discordance quartile',ylabel='Mean absolute prediction error',title='Held-out error by discordance quartile')
    save(fig,'discordance_quartiles',['error_by_discordance_quartile_with_ci.csv'])
    for filename,name,label,title in [('figureS01_sensitivity_forest_data.csv','sensitivity_forest','Analysis','Primary and sensitivity associations'),('figureS02_longitudinal_slope_data.csv','longitudinal_slopes','Visit month','Discordance–error association across visits')]:
        d=pd.read_csv(TABLES/filename); center=d.iloc[:,1]; lo=d.iloc[:,2]; hi=d.iloc[:,3]
        fig,ax=plt.subplots(figsize=(10,5))
        ax.errorbar(center,range(len(d)),xerr=[center-lo,hi-center],fmt='o',capsize=4,color='#087e8b')
        ax.set_yticks(range(len(d)),d[label].astype(str)); ax.axvline(0,color='#bcc5ce',lw=1)
        ax.set(xlabel='Coefficient (95% CI)',title=title); ax.invert_yaxis()
        save(fig,name,[filename])
    d=pd.read_csv(TABLES/'figure01_cohort_flow_data.csv')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(d.Stage,d['Knees remaining'],color='#087e8b');ax.invert_yaxis()
    ax.set(xlabel='Knees',title='Cohort construction and outcome availability')
    save(fig,'cohort_flow',['figure01_cohort_flow_data.csv'])
    (OUT/'figure_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Regenerated {len(metadata)} figures from aggregate tables in {OUT.relative_to(ROOT)}')
if __name__=='__main__':main()
