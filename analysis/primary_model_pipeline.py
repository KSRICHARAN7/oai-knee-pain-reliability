#!/usr/bin/env python3
"""Generate leakage-protected held-out predictions for the frozen primary model.

This script does not test the discordance-error hypothesis. It freezes grouped
folds, generates cross-fitted discordance, tunes elastic net inside each outer
training set, and saves exactly one outer held-out prediction per eligible knee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260721
OUTCOME = "womac_pain_v06"
NUMERIC = [
    "womac_pain_v00", "medial_jsn_v00", "lateral_jsn_v00",
    "age_years", "bmi", "prior_knee_injury", "prior_knee_surgery",
]
CATEGORICAL = ["kl_grade_v00", "sex", "race", "hispanic"]
FEATURES = NUMERIC + CATEGORICAL


def grouped_folds(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = pd.Series(index=data.index, dtype="int8")
    assignment_rows = []
    x_dummy = np.zeros((len(data), 1))
    for fold, (_, test_index) in enumerate(splitter.split(x_dummy, groups=data.participant_key)):
        folds.iloc[test_index] = fold
        assignment_rows.extend((int(key), fold) for key in data.iloc[test_index].participant_key.unique())
    assignments = pd.DataFrame(assignment_rows, columns=["participant_key", "outer_fold"])
    if assignments.participant_key.duplicated().any():
        raise AssertionError("A participant was assigned to more than one outer fold")
    return folds.astype(int), assignments.sort_values("participant_key").reset_index(drop=True)


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


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(predicted, observed, 1)
    return {
        "mae": mean_absolute_error(observed, predicted),
        "rmse": mean_squared_error(observed, predicted) ** 0.5,
        "mean_bias_predicted_minus_observed": float(np.mean(predicted - observed)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
    }


def participant_bootstrap(predictions: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    groups = {key: frame.index.to_numpy() for key, frame in predictions.groupby("participant_key")}
    keys = np.array(list(groups))
    rows = []
    for draw in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        sample = predictions.loc[indices]
        values = metric_values(sample.observed_womac_48m.to_numpy(), sample.predicted_womac_48m.to_numpy())
        rows.append({"bootstrap_draw": draw, **values})
    return pd.DataFrame(rows)


def validate(data: pd.DataFrame, predictions: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    fold_counts = predictions.groupby("participant_key").outer_fold.nunique()
    checks = {
        "Primary outcome cohort has 8232 knees": len(data) == 8232,
        "Exactly one prediction per eligible knee": len(predictions) == len(data) and predictions.knee_key.is_unique,
        "Every participant occurs in one outer fold": fold_counts.eq(1).all(),
        "Fold assignment is complete": len(assignments) == data.participant_key.nunique(),
        "Predictions are finite": np.isfinite(predictions.predicted_womac_48m).all(),
        "Discordance scores are finite": np.isfinite(predictions.absolute_discordance).all(),
        "Errors are nonnegative": predictions.absolute_error.ge(0).all(),
        "Observed outcome remains in valid range": predictions.observed_womac_48m.between(0, 20).all(),
        "No follow-up variable is a predictor": not any("v03" in x or "v05" in x or "v06" in x for x in FEATURES),
        "Original OAI ID is absent": "ID" not in data.columns and "id" not in data.columns,
    }
    result = pd.DataFrame({"check": checks.keys(), "passed": checks.values()})
    if not result.passed.all():
        raise AssertionError(result.loc[~result.passed, "check"].tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/oai_knee_analysis_ready.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/primary_model"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.data)
    data = full.loc[full[OUTCOME].notna()].copy().reset_index(drop=True)
    data["kl_grade_v00"] = data["kl_grade_v00"].astype("Int64").astype("string")
    folds, assignments = grouped_folds(data)
    data["outer_fold"] = folds
    assignments.to_csv(args.output_dir / "frozen_outer_fold_assignments.csv", index=False)
    fold_hash = hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()

    grid = {
        "model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "model__l1_ratio": [0.05, 0.25, 0.5, 0.75, 0.95],
    }
    output_frames = []
    tuning_rows = []
    for outer_fold in range(5):
        train = data.loc[data.outer_fold.ne(outer_fold)].copy()
        test = data.loc[data.outer_fold.eq(outer_fold)].copy()
        train_groups = set(train.participant_key)
        test_groups = set(test.participant_key)
        if train_groups & test_groups:
            raise AssertionError("Participant leakage between outer train and test")

        expected_by_kl = train.groupby("kl_grade_v00", observed=True).womac_pain_v00.mean()
        expected = test.kl_grade_v00.map(expected_by_kl).astype(float)
        if expected.isna().any():
            raise AssertionError("An outer-test KL grade was absent from its training set")
        signed_discordance = test.womac_pain_v00 - expected

        inner = GroupKFold(n_splits=5, shuffle=True, random_state=SEED + outer_fold + 1)
        search = GridSearchCV(
            model_pipeline(), grid, scoring="neg_mean_absolute_error", cv=inner,
            n_jobs=-1, refit=True, return_train_score=False,
        )
        search.fit(train[FEATURES], train[OUTCOME], groups=train.participant_key)
        predicted = search.predict(test[FEATURES])

        fold_output = pd.DataFrame({
            "participant_key": test.participant_key.astype(int),
            "knee_key": test.knee_key,
            "side": test.side,
            "outer_fold": outer_fold,
            "observed_womac_48m": test[OUTCOME].astype(float),
            "predicted_womac_48m": predicted,
            "expected_baseline_pain_from_kl": expected,
            "signed_discordance": signed_discordance,
            "absolute_discordance": signed_discordance.abs(),
            "absolute_error": np.abs(test[OUTCOME].to_numpy() - predicted),
        })
        output_frames.append(fold_output)
        tuning_rows.append({
            "outer_fold": outer_fold,
            "train_knees": len(train),
            "test_knees": len(test),
            "train_participants": train.participant_key.nunique(),
            "test_participants": test.participant_key.nunique(),
            "best_alpha": search.best_params_["model__alpha"],
            "best_l1_ratio": search.best_params_["model__l1_ratio"],
            "inner_cv_mae": -search.best_score_,
        })

    predictions = pd.concat(output_frames, ignore_index=True).sort_values("knee_key").reset_index(drop=True)
    checks = validate(data, predictions, assignments)
    checks.to_csv(args.output_dir / "leakage_and_integrity_checks.csv", index=False)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv.gz", index=False, compression="gzip")
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(args.output_dir / "fold_tuning_summary.csv", index=False, float_format="%.6f")

    point = metric_values(predictions.observed_womac_48m.to_numpy(), predictions.predicted_womac_48m.to_numpy())
    boot = participant_bootstrap(predictions)
    boot.to_csv(args.output_dir / "participant_bootstrap_metrics.csv.gz", index=False, compression="gzip")
    metric_rows = []
    for name, value in point.items():
        metric_rows.append({
            "metric": name,
            "estimate": value,
            "ci_2_5": boot[name].quantile(.025),
            "ci_97_5": boot[name].quantile(.975),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "primary_model_metrics.csv", index=False, float_format="%.6f")

    summary = {
        "seed": SEED,
        "eligible_knees": len(data),
        "eligible_participants": int(data.participant_key.nunique()),
        "outer_folds": 5,
        "inner_folds": 5,
        "fold_assignment_sha256": fold_hash,
        "all_checks_passed": bool(checks.passed.all()),
        "features": FEATURES,
        "outcome": OUTCOME,
    }
    (args.output_dir / "pipeline_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nFold tuning\n", tuning.to_string(index=False))
    print("\nHeld-out metrics\n", metrics.to_string(index=False))


if __name__ == "__main__":
    main()
