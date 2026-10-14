#!/usr/bin/env python3
"""Cross-check frozen headline values across independent aggregate tables."""
from pathlib import Path
import csv,math
ROOT=Path(__file__).resolve().parents[1]/'results/tables'
def read(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def main():
    key={r['Item']:float(r['Frozen value']) for r in read('frozen_key_results.csv')}
    metrics={r['metric']:float(r['estimate']) for r in read('primary_model_metrics.csv')}
    risk={int(r['abstain_percent']):r for r in read('risk_coverage.csv')}
    assert math.isclose(key['Primary MAE'],metrics['mae'],abs_tol=1e-6)
    assert math.isclose(key['Primary RMSE'],metrics['rmse'],abs_tol=1e-6)
    assert int(risk[0]['retained_n'])==key['Primary eligible knees']
    assert math.isclose(float(risk[10]['targeted_mae']),key['10% targeted abstention retained MAE'],abs_tol=1e-6)
    forest=read('figureS01_sensitivity_forest_data.csv')[0]
    assert math.isclose(float(forest['Coefficient']),key['Primary discordance-error coefficient'],abs_tol=1e-12)
    quartiles=read('error_by_discordance_quartile_with_ci.csv')
    n=sum(int(r['knees_n']) for r in quartiles)
    mean=sum(int(r['knees_n'])*float(r['absolute_error_mean']) for r in quartiles)/n
    assert n==key['Primary eligible knees'] and math.isclose(mean,key['Primary MAE'],abs_tol=1e-6)
    print('PASS: frozen metrics, deferral results, primary coefficient, and quartile totals agree')
if __name__=='__main__':main()
