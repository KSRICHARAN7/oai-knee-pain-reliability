#!/usr/bin/env python3
"""Nested uncertainty estimation and prespecified abstention evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from primary_hypothesis_analysis import clustered_ols
from primary_model_pipeline import CATEGORICAL, FEATURES, NUMERIC, OUTCOME, SEED, model_pipeline


UNCERTAINTY_NUMERIC = NUMERIC + [
    "primary_prediction", "expected_baseline_pain_from_kl",
    "signed_discordance", "absolute_discordance",
]
UNCERTAINTY_FEATURES = UNCERTAINTY_NUMERIC + CATEGORICAL


def uncertainty_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, UNCERTAINTY_NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", ElasticNet(max_iter=50_000, selection="cyclic")),
    ])


def add_uncertainty_features(frame: pd.DataFrame, primary_prediction: np.ndarray,
                             expected: pd.Series, signed: pd.Series) -> pd.DataFrame:
    out = frame[["participant_key", "knee_key", "outer_fold", *FEATURES]].copy()
    out["primary_prediction"] = primary_prediction
    out["expected_baseline_pain_from_kl"] = expected.to_numpy()
    out["signed_discordance"] = signed.to_numpy()
    out["absolute_discordance"] = signed.abs().to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/oai_knee_analysis_ready.csv.gz"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/primary_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/uncertainty"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checks = pd.read_csv(args.model_dir / "leakage_and_integrity_checks.csv")
    repro = json.loads((args.model_dir / "reproducibility_check.json").read_text())
    if not checks.passed.all() or not repro["reproducible"]:
        raise RuntimeError("Uncertainty analysis blocked by primary pipeline checks")

    data = pd.read_csv(args.data)
    data = data.loc[data[OUTCOME].notna()].copy()
    data["kl_grade_v00"] = data["kl_grade_v00"].astype("Int64").astype("string")
    assignments = pd.read_csv(args.model_dir / "frozen_outer_fold_assignments.csv")
    data = data.drop(columns=["outer_fold"], errors="ignore").merge(
        assignments, on="participant_key", how="left", validate="many_to_one"
    )
    heldout = pd.read_csv(args.model_dir / "heldout_predictions.csv.gz")
    tuning = pd.read_csv(args.model_dir / "fold_tuning_summary.csv").set_index("outer_fold")

    score_outputs = []
    uncertainty_tuning = []
    leakage_rows = []
    error_grid = {
        "model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "model__l1_ratio": [0.05, 0.25, 0.5, 0.75, 0.95],
    }

    for outer_fold in range(5):
        outer_train = data.loc[data.outer_fold.ne(outer_fold)].copy().reset_index(drop=True)
        outer_test = data.loc[data.outer_fold.eq(outer_fold)].copy().reset_index(drop=True)
        overlap = set(outer_train.participant_key) & set(outer_test.participant_key)
        leakage_rows.append({"outer_fold": outer_fold, "participant_overlap": len(overlap)})
        if overlap:
            raise AssertionError("Outer participant leakage in uncertainty analysis")

        best_alpha = float(tuning.loc[outer_fold, "best_alpha"])
        best_l1 = float(tuning.loc[outer_fold, "best_l1_ratio"])
        inner = GroupKFold(n_splits=5, shuffle=True, random_state=SEED + 100 + outer_fold)
        inner_outputs = []
        dummy = np.zeros((len(outer_train), 1))
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
            inner.split(dummy, groups=outer_train.participant_key)
        ):
            inner_train = outer_train.iloc[inner_train_idx].copy()
            inner_valid = outer_train.iloc[inner_valid_idx].copy()
            if set(inner_train.participant_key) & set(inner_valid.participant_key):
                raise AssertionError("Inner participant leakage")
            primary = model_pipeline().set_params(
                model__alpha=best_alpha, model__l1_ratio=best_l1
            )
            primary.fit(inner_train[FEATURES], inner_train[OUTCOME])
            inner_prediction = primary.predict(inner_valid[FEATURES])
            means = inner_train.groupby("kl_grade_v00", observed=True).womac_pain_v00.mean()
            expected = inner_valid.kl_grade_v00.map(means).astype(float)
            signed = inner_valid.womac_pain_v00 - expected
            features = add_uncertainty_features(inner_valid, inner_prediction, expected, signed)
            features["inner_absolute_error_target"] = np.abs(
                inner_valid[OUTCOME].to_numpy() - inner_prediction
            )
            features["inner_fold"] = inner_fold
            inner_outputs.append(features)

        error_training = pd.concat(inner_outputs, ignore_index=True)
        if not error_training.knee_key.is_unique or len(error_training) != len(outer_train):
            raise AssertionError("Inner OOF error targets are incomplete or duplicated")

        outer_primary = heldout.loc[heldout.outer_fold.eq(outer_fold)].copy()
        outer_primary = outer_test[["knee_key"]].merge(
            outer_primary, on="knee_key", how="left", validate="one_to_one"
        )
        test_features = add_uncertainty_features(
            outer_test,
            outer_primary.predicted_womac_48m.to_numpy(),
            outer_primary.expected_baseline_pain_from_kl,
            outer_primary.signed_discordance,
        )

        error_cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED + 200 + outer_fold)
        search = GridSearchCV(
            uncertainty_pipeline(), error_grid, scoring="neg_mean_absolute_error",
            cv=error_cv, n_jobs=-1, refit=True,
        )
        search.fit(
            error_training[UNCERTAINTY_FEATURES],
            error_training.inner_absolute_error_target,
            groups=error_training.participant_key,
        )
        uncertainty = np.clip(search.predict(test_features[UNCERTAINTY_FEATURES]), 0, None)
        output = outer_primary[[
            "participant_key", "knee_key", "side", "outer_fold",
            "observed_womac_48m", "predicted_womac_48m", "absolute_discordance", "absolute_error",
        ]].copy()
        output["estimated_uncertainty"] = uncertainty
        score_outputs.append(output)
        uncertainty_tuning.append({
            "outer_fold": outer_fold,
            "best_alpha": search.best_params_["model__alpha"],
            "best_l1_ratio": search.best_params_["model__l1_ratio"],
            "inner_error_model_mae": -search.best_score_,
        })

    scores = pd.concat(score_outputs, ignore_index=True).sort_values("knee_key").reset_index(drop=True)
    leakage = pd.DataFrame(leakage_rows)
    assertions = {
        "Exactly one uncertainty score per held-out knee": len(scores) == len(heldout) and scores.knee_key.is_unique,
        "No outer participant overlap": leakage.participant_overlap.eq(0).all(),
        "All uncertainty scores are finite": np.isfinite(scores.estimated_uncertainty).all(),
        "No outcome or actual error is an uncertainty feature": OUTCOME not in UNCERTAINTY_FEATURES and "absolute_error" not in UNCERTAINTY_FEATURES,
        "Outer folds agree with primary predictions": scores.set_index("knee_key").outer_fold.equals(heldout.set_index("knee_key").sort_index().outer_fold),
    }
    assertion_frame = pd.DataFrame({"check": assertions.keys(), "passed": assertions.values()})
    if not assertion_frame.passed.all():
        raise AssertionError(assertion_frame.loc[~assertion_frame.passed, "check"].tolist())
    assertion_frame.to_csv(args.output_dir / "uncertainty_leakage_checks.csv", index=False)
    scores.to_csv(args.output_dir / "heldout_uncertainty_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(uncertainty_tuning).to_csv(args.output_dir / "uncertainty_tuning.csv", index=False)

    rho, rho_p = spearmanr(scores.estimated_uncertainty, scores.absolute_error)
    cluster_fit = clustered_ols(
        scores.absolute_error.to_numpy(), scores.estimated_uncertainty.to_numpy(),
        scores.participant_key.to_numpy(),
    )

    rng = np.random.default_rng(SEED + 300)
    risk_rows = []
    n = len(scores)
    for abstain_percent in [0, 5, 10, 15, 20]:
        n_abstain = int(np.floor(n * abstain_percent / 100))
        order = np.argsort(-scores.estimated_uncertainty.to_numpy())
        retained = np.ones(n, dtype=bool)
        retained[order[:n_abstain]] = False
        targeted_mae = mean_absolute_error(
            scores.loc[retained, "observed_womac_48m"], scores.loc[retained, "predicted_womac_48m"]
        )
        random_maes = []
        for _ in range(1000):
            random_retained = np.ones(n, dtype=bool)
            if n_abstain:
                random_retained[rng.choice(n, size=n_abstain, replace=False)] = False
            random_maes.append(scores.loc[random_retained, "absolute_error"].mean())
        risk_rows.append({
            "abstain_percent": abstain_percent,
            "coverage_percent": 100 - abstain_percent,
            "retained_n": int(retained.sum()),
            "targeted_mae": targeted_mae,
            "random_abstention_mean_mae": float(np.mean(random_maes)),
            "random_abstention_ci_2_5": float(np.quantile(random_maes, .025)),
            "random_abstention_ci_97_5": float(np.quantile(random_maes, .975)),
        })
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(args.output_dir / "risk_coverage.csv", index=False, float_format="%.6f")

    primary_row = risk.loc[risk.abstain_percent.eq(10)].iloc[0]
    baseline_mae = risk.loc[risk.abstain_percent.eq(0), "targeted_mae"].iloc[0]
    summary = {
        "eligible_knees": len(scores),
        "participants": int(scores.participant_key.nunique()),
        "spearman_uncertainty_error": float(rho),
        "spearman_p_value": float(rho_p),
        "clustered_error_per_uncertainty_point": float(cluster_fit["beta"][1]),
        "clustered_ci_95_lower": float(cluster_fit["ci_low"][1]),
        "clustered_ci_95_upper": float(cluster_fit["ci_high"][1]),
        "clustered_p_value": float(cluster_fit["p"][1]),
        "mae_no_abstention": float(baseline_mae),
        "mae_after_targeted_10_percent_abstention": float(primary_row.targeted_mae),
        "absolute_mae_reduction": float(baseline_mae - primary_row.targeted_mae),
        "relative_mae_reduction_percent": float(100 * (baseline_mae - primary_row.targeted_mae) / baseline_mae),
        "random_10_percent_abstention_mean_mae": float(primary_row.random_abstention_mean_mae),
        "targeted_beats_random_at_10_percent": bool(primary_row.targeted_mae < primary_row.random_abstention_ci_2_5),
        "all_uncertainty_leakage_checks_passed": bool(assertion_frame.passed.all()),
    }
    (args.output_dir / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(risk.coverage_percent, risk.targeted_mae, marker="o", linewidth=2, label="Targeted abstention")
    ax.plot(risk.coverage_percent, risk.random_abstention_mean_mae, marker="o", linestyle="--", label="Random abstention")
    ax.invert_xaxis()
    ax.set(title="Prediction error as uncertain cases are deferred",
           xlabel="Coverage: cases retaining a prediction (%)", ylabel="Mean absolute WOMAC error")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_5_risk_coverage.png", dpi=220)
    plt.close(fig)

    report = f"""# Uncertainty and abstention result

All uncertainty leakage checks passed. Estimated uncertainty was associated with observed held-out absolute error (Spearman rho={rho:.3f}).

Without abstention, held-out MAE was {baseline_mae:.3f} WOMAC points. After deferring the 10% of knees with the highest nested uncertainty score, MAE among the retained 90% was {primary_row.targeted_mae:.3f}, a {summary['relative_mae_reduction_percent']:.1f}% relative reduction. Random 10% abstention produced a mean MAE of {primary_row.random_abstention_mean_mae:.3f}.

This is an internally validated research result, not evidence that the system is ready for clinical use.
"""
    (args.output_dir / "UNCERTAINTY_RESULT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n", risk.to_string(index=False))


if __name__ == "__main__":
    main()
