#!/usr/bin/env python3
"""Public verification or authorized private analysis, without changing frozen code."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'analysis'
def run(script,*args,cwd=ROOT):
    subprocess.run([sys.executable,str(script),*map(str,args)],cwd=cwd,check=True)
def compare_primary(config):
    import numpy as np
    import pandas as pd
    one=A/'outputs/primary_model';two=A/'outputs_repeat/primary_model'
    a=pd.read_csv(one/'heldout_predictions.csv.gz');b=pd.read_csv(two/'heldout_predictions.csv.gz')
    same_keys=a[['knee_key','participant_key','outer_fold']].equals(b[['knee_key','participant_key','outer_fold']])
    same_folds=(one/'frozen_outer_fold_assignments.csv').read_bytes()==(two/'frozen_outer_fold_assignments.csv').read_bytes()
    cols=['predicted_womac_48m','expected_baseline_pain_from_kl','signed_discordance','absolute_discordance','absolute_error']
    differences={c:float(np.max(np.abs(a[c].to_numpy()-b[c].to_numpy()))) for c in cols}
    integrity=pd.read_csv(one/'leakage_and_integrity_checks.csv').passed.eq(True).all()
    ok=bool(same_keys and same_folds and integrity and all(np.isfinite(v) and v<=config['primary_repeat_atol'] for v in differences.values()))
    report={'fold_files_identical':same_folds,'knee_order_identical':same_keys,'max_absolute_numeric_difference':differences,'reproducible':ok,'scope':'Independent repeat in public-release environment; not byte-for-byte historical model verification'}
    (one/'reproducibility_check.json').write_text(json.dumps(report,indent=2)+'\n')
    if not ok:raise RuntimeError('Primary repeatability check failed; downstream analysis blocked')
    frozen=pd.read_csv(ROOT/'results/tables/primary_model_metrics.csv').set_index('metric')
    current=pd.read_csv(one/'primary_model_metrics.csv').set_index('metric')
    if not np.allclose(current.loc[frozen.index,'estimate'],frozen.estimate,rtol=0,atol=config['frozen_metric_atol']):raise RuntimeError('Refit metrics differ from frozen reference; investigate data/environment before proceeding')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['public','full'],default='public');p.add_argument('--zip-dir',type=Path);p.add_argument('--secondary',action='store_true');args=p.parse_args()
    run(ROOT/'scripts/verify_public.py')
    config=json.loads((ROOT/'config/reproducibility.json').read_text())
    if args.mode=='public':
        run(ROOT/'scripts/check_aggregate_consistency.py');run(ROOT/'scripts/regenerate_figures.py');return
    if not args.zip_dir:p.error('--zip-dir is required for full mode')
    z=args.zip_dir.resolve()
    missing=[n for n in config['source_archives'] if not (z/n).is_file()]
    if missing:p.error(f'Missing authorized OAI archives: {missing}')
    if args.secondary:
        import importlib.util
        for module in ['statsmodels','xgboost']:
            if importlib.util.find_spec(module) is None:p.error('Install requirements-secondary.txt before requesting secondary analyses')
    run(A/'build_knee_cohort.py','--zip-dir',z,cwd=A)
    run(A/'descriptive_qc.py',cwd=A)
    gate=A/'outputs/primary_model/reproducibility_check.json'
    gate.unlink(missing_ok=True)
    run(A/'primary_model_pipeline.py',cwd=A)
    run(A/'primary_model_pipeline.py','--output-dir','outputs_repeat/primary_model',cwd=A)
    compare_primary(config)
    run(A/'primary_hypothesis_analysis.py',cwd=A)
    run(A/'uncertainty_abstention_analysis.py',cwd=A)
    if args.secondary:
        for script in ['sensitivity_analyses.py','followup_attrition_analysis.py','longitudinal_analysis.py','subgroup_abstention_audit.py','knee_replacement_outcome_audit.py']:run(A/script,cwd=A)
    print('Full run completed locally. All generated participant-level artifacts must remain private.')
if __name__=='__main__':main()
