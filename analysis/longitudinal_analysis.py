#!/usr/bin/env python3
"""Secondary longitudinal predictability analysis for the OAI structure-pain study.

Implements LONGITUDINAL_ANALYSIS_SPECIFICATION_2026-09-13.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

SEED = 20260721
BOOT_SEED = 20260913
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "oai_knee_analysis_ready.csv.gz"
FOLDS = ROOT / "outputs" / "primary_model" / "frozen_outer_fold_assignments.csv"
OUT = ROOT / "outputs" / "longitudinal"
OUT.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "womac_pain_v00", "medial_jsn_v00", "lateral_jsn_v00",
    "age_years", "bmi", "prior_knee_injury", "prior_knee_surgery",
]
CATEGORICAL = ["kl_grade_v00", "sex", "race", "hispanic"]
FEATURES = NUMERIC + CATEGORICAL
VISITS = {
    "24": "womac_pain_v03",
    "36": "womac_pain_v05",
    "48": "womac_pain_v06",
}
GRID = {
    "model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "model__l1_ratio": [0.05, 0.25, 0.5, 0.75, 0.95],
}


def model_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", ElasticNet(max_iter=50_000, selection="cyclic")),
    ])


def metric_values(y, pred):
    slope, intercept = np.polyfit(pred, y, 1)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "mean_bias_predicted_minus_observed": float(np.mean(pred-y)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def main():
    full = pd.read_csv(DATA)
    folds = pd.read_csv(FOLDS)

    complete = full[["womac_pain_v00", *VISITS.values()]].notna().all(axis=1)
    data = full.loc[complete].copy().reset_index(drop=True)
    data = data.merge(folds, on="participant_key", how="left", validate="many_to_one")
    if data.outer_fold.isna().any():
        raise AssertionError("Complete longitudinal participant missing frozen outer-fold assignment")
    data["outer_fold"] = data.outer_fold.astype(int)
    data["kl_grade_v00"] = data["kl_grade_v00"].astype("Int64").astype("string")

    # Generate cross-fitted discordance once per knee from training data only.
    discordance_frames = []
    for outer_fold in range(5):
        train = data.loc[data.outer_fold.ne(outer_fold)]
        test = data.loc[data.outer_fold.eq(outer_fold)]
        if set(train.participant_key) & set(test.participant_key):
            raise AssertionError("Participant leakage in frozen outer fold")
        expected_by_kl = train.groupby("kl_grade_v00", observed=True).womac_pain_v00.mean()
        expected = test.kl_grade_v00.map(expected_by_kl).astype(float)
        if expected.isna().any():
            raise AssertionError("Outer-test KL grade absent from longitudinal outer training set")
        signed = test.womac_pain_v00.astype(float) - expected
        discordance_frames.append(pd.DataFrame({
            "participant_key": test.participant_key.astype(int),
            "knee_key": test.knee_key.astype(str),
            "outer_fold": outer_fold,
            "expected_baseline_pain_from_kl": expected.to_numpy(),
            "signed_discordance": signed.to_numpy(),
            "absolute_discordance": signed.abs().to_numpy(),
        }))
    disc = pd.concat(discordance_frames, ignore_index=True)
    if len(disc) != len(data) or not disc.knee_key.is_unique:
        raise AssertionError("Discordance cross-fitting did not yield exactly one row per knee")

    all_predictions = []
    tuning_rows = []
    metrics_rows = []

    for visit_idx, (visit, outcome) in enumerate(VISITS.items()):
        visit_frames = []
        for outer_fold in range(5):
            train = data.loc[data.outer_fold.ne(outer_fold)].copy()
            test = data.loc[data.outer_fold.eq(outer_fold)].copy()
            inner = GroupKFold(n_splits=5, shuffle=True, random_state=SEED + 100*visit_idx + outer_fold + 1)
            search = GridSearchCV(
                model_pipeline(), GRID, scoring="neg_mean_absolute_error", cv=inner,
                n_jobs=-1, refit=True, return_train_score=False,
            )
            search.fit(train[FEATURES], train[outcome], groups=train.participant_key)
            pred = search.predict(test[FEATURES])
            fold = pd.DataFrame({
                "participant_key": test.participant_key.astype(int),
                "knee_key": test.knee_key.astype(str),
                "outer_fold": outer_fold,
                "visit_month": int(visit),
                "observed_womac": test[outcome].astype(float).to_numpy(),
                "predicted_womac": pred,
                "absolute_error": np.abs(test[outcome].astype(float).to_numpy() - pred),
            }).merge(disc[["knee_key", "expected_baseline_pain_from_kl", "signed_discordance", "absolute_discordance"]],
                     on="knee_key", how="left", validate="one_to_one")
            visit_frames.append(fold)
            tuning_rows.append({
                "visit_month": int(visit),
                "outer_fold": outer_fold,
                "train_knees": len(train),
                "test_knees": len(test),
                "train_participants": train.participant_key.nunique(),
                "test_participants": test.participant_key.nunique(),
                "best_alpha": search.best_params_["model__alpha"],
                "best_l1_ratio": search.best_params_["model__l1_ratio"],
                "inner_cv_mae": -search.best_score_,
            })
        visit_pred = pd.concat(visit_frames, ignore_index=True).sort_values("knee_key").reset_index(drop=True)
        if len(visit_pred) != len(data) or not visit_pred.knee_key.is_unique:
            raise AssertionError(f"Visit {visit}: not exactly one held-out prediction per knee")
        vals = metric_values(visit_pred.observed_womac.to_numpy(), visit_pred.predicted_womac.to_numpy())
        metrics_rows.append({"visit_month": int(visit), **vals})
        all_predictions.append(visit_pred)

    long = pd.concat(all_predictions, ignore_index=True)
    long["visit36"] = (long.visit_month == 36).astype(int)
    long["visit48"] = (long.visit_month == 48).astype(int)
    long["disc_x_36"] = long.absolute_discordance * long.visit36
    long["disc_x_48"] = long.absolute_discordance * long.visit48

    # Mixed model: participant random intercept + nested knee variance component.
    mixed = smf.mixedlm(
        "absolute_error ~ absolute_discordance + visit36 + visit48 + disc_x_36 + disc_x_48",
        data=long,
        groups=long["participant_key"],
        re_formula="1",
        vc_formula={"knee": "0 + C(knee_key)"},
    )
    fit = mixed.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)

    params = fit.params
    cov = fit.cov_params()
    fixed_names = ["Intercept", "absolute_discordance", "visit36", "visit48", "disc_x_36", "disc_x_48"]
    fixed = pd.DataFrame({
        "term": fixed_names,
        "estimate": [params[n] for n in fixed_names],
        "se": [fit.bse[n] for n in fixed_names],
        "z": [fit.tvalues[n] for n in fixed_names],
        "p_value": [fit.pvalues[n] for n in fixed_names],
        "ci_2_5": [fit.conf_int().loc[n, 0] for n in fixed_names],
        "ci_97_5": [fit.conf_int().loc[n, 1] for n in fixed_names],
    })

    def linear_combo(weights: dict[str, float]):
        est = sum(weights.get(n, 0.0)*params[n] for n in weights)
        names = list(weights)
        w = np.array([weights[n] for n in names])
        cv = cov.loc[names, names].to_numpy()
        var = float(w @ cv @ w)
        se = math.sqrt(max(var, 0.0))
        z = est / se if se > 0 else np.nan
        p = 2*stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
        return est, se, est-1.96*se, est+1.96*se, z, p

    slopes = []
    for visit, weights in [
        (24, {"absolute_discordance": 1}),
        (36, {"absolute_discordance": 1, "disc_x_36": 1}),
        (48, {"absolute_discordance": 1, "disc_x_48": 1}),
    ]:
        est, se, lo, hi, z, p = linear_combo(weights)
        slopes.append({
            "visit_month": visit,
            "discordance_error_slope": est,
            "se": se,
            "ci_2_5": lo,
            "ci_97_5": hi,
            "z": z,
            "p_value": p,
        })
    slopes = pd.DataFrame(slopes)

    # Omnibus test that discordance slopes are all zero: beta_disc = beta_disc+int36 = beta_disc+int48 = 0.
    # Equivalent full-rank contrasts: beta_disc=0, int36=0, int48=0.
    R = np.zeros((3, len(params)))
    pnames = list(params.index)
    for row, name in enumerate(["absolute_discordance", "disc_x_36", "disc_x_48"]):
        R[row, pnames.index(name)] = 1
    wald = fit.wald_test(R, scalar=True)

    # Descriptive quartiles from cross-fitted discordance; same quartile for each knee at all visits.
    disc_for_q = disc[["knee_key", "absolute_discordance"]].copy()
    disc_for_q["discordance_quartile"] = pd.qcut(
        disc_for_q.absolute_discordance.rank(method="first"), 4,
        labels=["Q1: lowest", "Q2", "Q3", "Q4: highest"]
    )
    long = long.merge(disc_for_q[["knee_key", "discordance_quartile"]], on="knee_key", how="left", validate="many_to_one")
    quart = long.groupby(["visit_month", "discordance_quartile"], observed=True).absolute_error.mean().reset_index(name="mean_absolute_error")

    # Participant bootstrap CIs for descriptive quartile means.
    rng = np.random.default_rng(BOOT_SEED)
    part_frames = {k: g.index.to_numpy() for k, g in long.groupby("participant_key")}
    keys = np.array(list(part_frames))
    boot_rows = []
    for b in range(2000):
        samp = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([part_frames[k] for k in samp])
        s = long.loc[idx]
        g = s.groupby(["visit_month", "discordance_quartile"], observed=True).absolute_error.mean()
        for (v, q), val in g.items():
            boot_rows.append((b, int(v), str(q), float(val)))
    boot = pd.DataFrame(boot_rows, columns=["draw", "visit_month", "discordance_quartile", "mean_absolute_error"])
    cis = boot.groupby(["visit_month", "discordance_quartile"], observed=True).mean_absolute_error.quantile([.025,.975]).unstack().reset_index()
    cis.columns = ["visit_month", "discordance_quartile", "ci_2_5", "ci_97_5"]
    quart = quart.merge(cis, on=["visit_month", "discordance_quartile"], how="left")

    # Basic checks.
    checks = pd.DataFrame({
        "check": [
            "Complete longitudinal cohort has 7807 knees",
            "Complete longitudinal cohort has 3937 participants",
            "Exactly three held-out predictions per knee",
            "Every participant occurs in one frozen outer fold",
            "All predictions finite",
            "All absolute errors nonnegative",
            "All discordance values finite",
            "Mixed model converged",
        ],
        "passed": [
            len(data) == 7807,
            data.participant_key.nunique() == 3937,
            long.groupby("knee_key").size().eq(3).all(),
            long.groupby("participant_key").outer_fold.nunique().eq(1).all(),
            np.isfinite(long.predicted_womac).all(),
            long.absolute_error.ge(0).all(),
            np.isfinite(long.absolute_discordance).all(),
            bool(fit.converged),
        ],
    })
    if not checks.passed.all():
        raise AssertionError(checks.loc[~checks.passed, "check"].tolist())

    # Save outputs.
    long.to_csv(OUT / "longitudinal_heldout_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(tuning_rows).to_csv(OUT / "longitudinal_tuning.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(OUT / "timepoint_model_metrics.csv", index=False)
    fixed.to_csv(OUT / "mixed_model_fixed_effects.csv", index=False)
    slopes.to_csv(OUT / "discordance_error_slopes_by_visit.csv", index=False)
    quart.to_csv(OUT / "error_by_discordance_quartile_and_visit.csv", index=False)
    checks.to_csv(OUT / "longitudinal_checks.csv", index=False)
    (OUT / "mixed_model_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")

    summary = {
        "analysis_cohort_knees": int(len(data)),
        "analysis_cohort_participants": int(data.participant_key.nunique()),
        "long_rows": int(len(long)),
        "mixed_model_converged": bool(fit.converged),
        "participant_random_intercept_variance": float(fit.cov_re.iloc[0,0]),
        "knee_variance_component": float(fit.vcomp[0]) if len(fit.vcomp) else None,
        "residual_variance": float(fit.scale),
        "omnibus_wald_statistic": float(wald.statistic),
        "omnibus_wald_df": int(wald.df_denom),
        "omnibus_p_value": float(wald.pvalue),
        "slopes_by_visit": slopes.to_dict(orient="records"),
        "timepoint_metrics": pd.DataFrame(metrics_rows).to_dict(orient="records"),
    }
    (OUT / "longitudinal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Descriptive longitudinal figure (archived now; publication inclusion decided later).
    qorder = ["Q1: lowest", "Q2", "Q3", "Q4: highest"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for q in qorder:
        d = quart.loc[quart.discordance_quartile.astype(str).eq(q)].sort_values("visit_month")
        ax.errorbar(d.visit_month, d.mean_absolute_error,
                    yerr=[d.mean_absolute_error-d.ci_2_5, d.ci_97_5-d.mean_absolute_error],
                    marker="o", capsize=3, label=q)
    ax.set_xlabel("Follow-up month")
    ax.set_ylabel("Mean absolute held-out WOMAC prediction error")
    ax.set_title("Prediction error across follow-up by baseline discordance quartile")
    ax.set_xticks([24,36,48])
    ax.legend(title="Baseline discordance")
    fig.tight_layout()
    fig.savefig(OUT / "longitudinal_error_by_discordance.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nVisit-specific discordance-error slopes\n", slopes.to_string(index=False))
    print("\nTimepoint metrics\n", pd.DataFrame(metrics_rows).to_string(index=False))
    print("\nQuartile means\n", quart.to_string(index=False))

if __name__ == "__main__":
    main()
