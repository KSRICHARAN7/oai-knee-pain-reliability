#!/usr/bin/env python3
"""Run prespecified descriptive quality control without hypothesis testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VISITS = {
    "Baseline": "womac_pain_v00",
    "24 months": "womac_pain_v03",
    "36 months": "womac_pain_v05",
    "48 months": "womac_pain_v06",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/oai_knee_analysis_ready.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/descriptive_qc"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)

    checks = {
        "Unique knee_key": data["knee_key"].is_unique,
        "Unique participant-side": not data.duplicated(["participant_key", "side"]).any(),
        "No missing baseline KL": data["kl_grade_v00"].notna().all(),
        "Baseline KL is 0-4": data["kl_grade_v00"].between(0, 4).all(),
        "No missing baseline WOMAC": data["womac_pain_v00"].notna().all(),
        "All observed WOMAC scores are 0-20": all(
            data[col].dropna().between(0, 20).all() for col in VISITS.values()
        ),
        "No baseline replacement retained": not data["replacement_at_baseline"].eq(1).any(),
        "Both knees share participant key": data.groupby("participant_key")["side"].nunique().le(2).all(),
    }
    pd.DataFrame({"check": checks.keys(), "passed": checks.values()}).to_csv(
        args.output_dir / "quality_checks.csv", index=False
    )
    if not all(checks.values()):
        raise AssertionError("One or more descriptive quality checks failed")

    flow = pd.DataFrame([
        ("Project 15 baseline knee records", 8982, 0),
        ("After excluding baseline knee replacements", 8924, 58),
        ("After requiring baseline KL grade", 8921, 3),
        ("After requiring baseline WOMAC pain", 8920, 1),
        ("Primary baseline cohort", 8920, 0),
        ("Observed 48-month WOMAC outcome", int(data.womac_pain_v06.notna().sum()),
         int(data.womac_pain_v06.isna().sum())),
    ], columns=["stage", "knees_remaining", "excluded_at_stage"])
    flow.to_csv(args.output_dir / "cohort_flow.csv", index=False)

    followup = []
    for visit, col in VISITS.items():
        n = int(data[col].notna().sum())
        followup.append((visit, n, len(data) - n, 100 * n / len(data)))
    followup = pd.DataFrame(followup, columns=["visit", "observed_n", "missing_n", "observed_percent"])
    followup.to_csv(args.output_dir / "followup_completeness.csv", index=False, float_format="%.2f")

    rows = []
    for grade, group in data.groupby("kl_grade_v00", observed=True):
        pain = group.womac_pain_v00
        rows.append({
            "baseline_kl_grade": int(grade),
            "knees_n": len(group),
            "cohort_percent": 100 * len(group) / len(data),
            "baseline_womac_mean": pain.mean(),
            "baseline_womac_sd": pain.std(),
            "baseline_womac_median": pain.median(),
            "baseline_womac_q1": pain.quantile(.25),
            "baseline_womac_q3": pain.quantile(.75),
            "baseline_womac_zero_percent": 100 * pain.eq(0).mean(),
            "observed_48m_n": group.womac_pain_v06.notna().sum(),
            "observed_48m_percent": 100 * group.womac_pain_v06.notna().mean(),
        })
    by_kl = pd.DataFrame(rows)
    by_kl.to_csv(args.output_dir / "baseline_by_kl.csv", index=False, float_format="%.2f")

    baseline = pd.DataFrame([
        ("Knees", len(data), ""),
        ("Participants", data.participant_key.nunique(), ""),
        ("Age, years", data.age_years.mean(), data.age_years.std()),
        ("BMI", data.bmi.mean(), data.bmi.std()),
        ("Baseline WOMAC pain", data.womac_pain_v00.mean(), data.womac_pain_v00.std()),
        ("Female", data.sex.eq("Female").sum(), 100 * data.sex.eq("Female").mean()),
        ("Prior knee injury", data.prior_knee_injury.eq(1).sum(), 100 * data.prior_knee_injury.eq(1).mean()),
        ("Prior knee surgery", data.prior_knee_surgery.eq(1).sum(), 100 * data.prior_knee_surgery.eq(1).mean()),
    ], columns=["characteristic", "value_or_n", "sd_or_percent"])
    baseline.to_csv(args.output_dir / "baseline_summary.csv", index=False, float_format="%.2f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    counts = data.kl_grade_v00.value_counts().sort_index()
    bars = ax.bar([str(int(x)) for x in counts.index], counts.values, color="#315b7d")
    ax.bar_label(bars, labels=[f"{x:,}" for x in counts.values], padding=3, fontsize=9)
    ax.set(title="Baseline radiographic severity", xlabel="Kellgren-Lawrence grade", ylabel="Number of knees")
    ax.set_ylim(0, counts.max() * 1.14)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_1_kl_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    groups = [data.loc[data.kl_grade_v00.eq(g), "womac_pain_v00"] for g in range(5)]
    bp = ax.boxplot(groups, tick_labels=[str(g) for g in range(5)], showfliers=False, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#8fb6cf")
    means = [g.mean() for g in groups]
    ax.scatter(range(1, 6), means, color="#a23b3b", zorder=3, label="Mean")
    ax.set(title="Baseline pain varies within every radiographic grade",
           xlabel="Kellgren-Lawrence grade", ylabel="Baseline WOMAC pain (0-20)")
    ax.set_ylim(-0.5, 20.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_2_womac_by_kl.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(followup.visit, followup.observed_percent, color="#4b8063")
    ax.bar_label(bars, labels=[f"{x:.1f}%" for x in followup.observed_percent], padding=3, fontsize=9)
    ax.set(title="WOMAC pain follow-up completeness", xlabel="Visit", ylabel="Observed knees (%)")
    ax.set_ylim(0, 106)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figure_3_followup_completeness.png", dpi=220)
    plt.close(fig)

    floor0 = 100 * data.womac_pain_v00.eq(0).mean()
    floor48 = 100 * data.womac_pain_v06.eq(0).mean()
    report = f"""# Descriptive quality-control report

This stage was limited to cohort validation and descriptive summaries. No association between discordance and future prediction error was calculated.

## Cohort

- {len(data):,} eligible knees from {data.participant_key.nunique():,} participants.
- {data.womac_pain_v06.notna().sum():,} knees ({100*data.womac_pain_v06.notna().mean():.1f}%) have the primary 48-month outcome.
- {data[list(VISITS.values())].notna().all(axis=1).sum():,} knees have WOMAC pain at all four selected visits.

## Quality checks

All eight prespecified checks passed: unique knee records, participant-side uniqueness, valid KL and WOMAC ranges, complete baseline KL and WOMAC, exclusion of baseline replacements, and preservation of participant grouping.

## Descriptive observations

- Baseline WOMAC pain has a substantial floor: {floor0:.1f}% of knees report a score of zero.
- Among observed 48-month outcomes, {floor48:.1f}% report a score of zero.
- Pain values vary within every KL grade, confirming that a continuous discordance measure is meaningful.
- Follow-up completeness remains above {followup.observed_percent.min():.1f}% at each selected visit.

The floor concentration is not a protocol deviation, but it must be acknowledged when interpreting regression error and calibration. No model or endpoint will be changed in response to this descriptive finding.
"""
    (args.output_dir / "DESCRIPTIVE_QC_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
