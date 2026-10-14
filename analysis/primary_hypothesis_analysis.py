#!/usr/bin/env python3
"""Run the single frozen primary discordance-error hypothesis test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def clustered_ols(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """OLS with CR1 participant-clustered sandwich covariance."""
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    unique_groups = np.unique(groups)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for group in unique_groups:
        mask = groups == group
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(y), design.shape[1], len(unique_groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(covariance))
    z = beta / se
    p = 2 * norm.sf(np.abs(z))
    ci_low = beta - norm.ppf(.975) * se
    ci_high = beta + norm.ppf(.975) * se
    return {"beta": beta, "se": se, "p": p, "ci_low": ci_low, "ci_high": ci_high}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("outputs/primary_model/heldout_predictions.csv.gz"))
    parser.add_argument("--checks", type=Path, default=Path("outputs/primary_model/leakage_and_integrity_checks.csv"))
    parser.add_argument("--repro", type=Path, default=Path("outputs/primary_model/reproducibility_check.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/primary_analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checks = pd.read_csv(args.checks)
    reproducibility = json.loads(args.repro.read_text())
    if not checks.passed.all() or not reproducibility["reproducible"]:
        raise RuntimeError("Primary test blocked: leakage/integrity/reproducibility checks did not pass")

    data = pd.read_csv(args.predictions)
    fit = clustered_ols(
        data.absolute_error.to_numpy(), data.absolute_discordance.to_numpy(),
        data.participant_key.to_numpy(),
    )
    coefficient = float(fit["beta"][1])
    se = float(fit["se"][1])
    ci_low = float(fit["ci_low"][1])
    ci_high = float(fit["ci_high"][1])
    p_value = float(fit["p"][1])

    # Independent participant-bootstrap verification of the primary coefficient.
    rng_coef = np.random.default_rng(20260722)
    grouped_rows = {k: g.index.to_numpy() for k, g in data.groupby("participant_key")}
    participant_keys = np.array(list(grouped_rows))
    bootstrap_coefficients = []
    for _ in range(1000):
        sampled_keys = rng_coef.choice(participant_keys, size=len(participant_keys), replace=True)
        sampled_rows = np.concatenate([grouped_rows[k] for k in sampled_keys])
        sampled = data.loc[sampled_rows]
        slope = np.polyfit(sampled.absolute_discordance, sampled.absolute_error, 1)[0]
        bootstrap_coefficients.append(slope)
    bootstrap_coefficients = np.asarray(bootstrap_coefficients)
    bootstrap_ci_low, bootstrap_ci_high = np.quantile(bootstrap_coefficients, [.025, .975])
    pd.DataFrame({"coefficient": bootstrap_coefficients}).to_csv(
        args.output_dir / "participant_bootstrap_primary_coefficient.csv.gz",
        index=False, compression="gzip",
    )

    result = {
        "analysis": "Primary participant-clustered OLS",
        "outcome": "Absolute held-out 48-month WOMAC prediction error",
        "exposure": "Cross-fitted absolute baseline KL-WOMAC discordance",
        "eligible_knees": len(data),
        "participants": int(data.participant_key.nunique()),
        "coefficient_per_womac_point": coefficient,
        "cluster_robust_standard_error": se,
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "two_sided_p_value": p_value,
        "participant_bootstrap_ci_95_lower": float(bootstrap_ci_low),
        "participant_bootstrap_ci_95_upper": float(bootstrap_ci_high),
        "alpha": 0.05,
        "reject_null": p_value < 0.05,
    }
    (args.output_dir / "primary_hypothesis_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result]).to_csv(args.output_dir / "primary_hypothesis_result.csv", index=False)

    # Prespecified descriptive error levels. Quantiles are for visualization only.
    data["discordance_quartile"] = pd.qcut(
        data.absolute_discordance.rank(method="first"), 4,
        labels=["Q1: lowest", "Q2", "Q3", "Q4: highest"]
    )
    quartiles = data.groupby("discordance_quartile", observed=True).agg(
        knees_n=("knee_key", "size"),
        discordance_mean=("absolute_discordance", "mean"),
        discordance_min=("absolute_discordance", "min"),
        discordance_max=("absolute_discordance", "max"),
        absolute_error_mean=("absolute_error", "mean"),
        absolute_error_median=("absolute_error", "median"),
    ).reset_index()
    quartiles.to_csv(args.output_dir / "error_by_discordance_quartile.csv", index=False, float_format="%.6f")

    # Participant bootstrap CIs for quartile mean errors.
    rng = np.random.default_rng(20260723)
    group_indices = {k: g.index.to_numpy() for k, g in data.groupby("participant_key")}
    keys = np.array(list(group_indices))
    boot_rows = []
    for draw in range(1000):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([group_indices[k] for k in sampled])
        sample = data.loc[idx]
        means = sample.groupby("discordance_quartile", observed=True).absolute_error.mean()
        for quartile, value in means.items():
            boot_rows.append((draw, str(quartile), value))
    boot = pd.DataFrame(boot_rows, columns=["draw", "discordance_quartile", "mean_error"])
    cis = boot.groupby("discordance_quartile").mean_error.quantile([.025, .975]).unstack()
    cis.columns = ["ci_2_5", "ci_97_5"]
    quartiles = quartiles.merge(cis, on="discordance_quartile", how="left")
    quartiles.to_csv(args.output_dir / "error_by_discordance_quartile_with_ci.csv", index=False, float_format="%.6f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = np.arange(len(quartiles))
    y = quartiles.absolute_error_mean.to_numpy()
    low = y - quartiles.ci_2_5.to_numpy()
    high = quartiles.ci_97_5.to_numpy() - y
    ax.errorbar(x, y, yerr=np.vstack([low, high]), marker="o", linewidth=2,
                capsize=5, color="#315b7d")
    ax.set_xticks(x, quartiles.discordance_quartile)
    ax.set(title="Held-out prediction error by baseline discordance",
           xlabel="Absolute structure-pain discordance quartile",
           ylabel="Mean absolute 48-month WOMAC error")
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_4_error_by_discordance.png", dpi=220)
    plt.close(fig)

    report = f"""# Primary hypothesis result

The frozen primary test was run only after all leakage, integrity, and reproducibility checks passed.

Across {len(data):,} knees from {data.participant_key.nunique():,} participants, each additional WOMAC point of absolute baseline structure-pain discordance was associated with a {coefficient:.3f}-point increase in absolute held-out 48-month prediction error (participant-clustered 95% CI {ci_low:.3f} to {ci_high:.3f}; two-sided p={p_value:.3g}).

This result evaluates predictability, not disease causation. It does not show that discordance causes later pain or justify clinical use of the model.
"""
    (args.output_dir / "PRIMARY_RESULT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\n", quartiles.to_string(index=False))


if __name__ == "__main__":
    main()
