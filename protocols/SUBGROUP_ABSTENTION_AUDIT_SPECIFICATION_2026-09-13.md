# Subgroup and abstention audit specification

**Date:** 2026-09-13  
**Status:** Exploratory secondary audit required by the frozen July 21 preregistration. This audit does not alter the frozen primary analysis, model, uncertainty scores, or 10% abstention rule.

## Analysis population
Use the 8,232 knees with observed 48-month WOMAC pain that received exactly one frozen held-out primary prediction and one nested held-out uncertainty score.

## Prespecified subgroup variables
- Sex: Female, Male
- Race: White or Caucasian, Black or African American, Asian, Other Non-white
- Hispanic ethnicity: Yes, No
- Baseline Kellgren-Lawrence grade: 0, 1, 2, 3, 4
- Prior knee injury: Yes, No
- Prior knee surgery: Yes, No

Missing subgroup values will be counted separately in metadata but will not be interpreted as clinical subgroups.

## Frozen abstention definition
The subgroup audit will reuse the primary abstention rule exactly: defer the `floor(8232 * 0.10) = 823` knees with the highest already-computed nested uncertainty scores across the full eligible cohort. No subgroup-specific thresholds will be tuned. The cutoff is applied globally, not separately within each subgroup.

## Metrics
For every subgroup level with at least 50 knees:
- Number of knees and unique participants
- Mean absolute error (MAE) before abstention
- Participant-bootstrap 95% CI for MAE
- Mean prediction bias
- 10% global-rule subgroup abstention frequency and coverage
- Participant-bootstrap 95% CI for subgroup abstention frequency
- MAE among retained subgroup cases after the global 10% abstention rule
- Absolute change in subgroup MAE after abstention

Calibration intercept and calibration slope will be reported only for subgroup levels with at least 100 knees because smaller samples can yield unstable calibration estimates. Calibration is defined by an ordinary linear calibration regression of observed 48-month WOMAC pain on held-out predicted WOMAC pain within the subgroup. Confidence intervals for calibration intercept and slope will be generated with participant-level bootstrap resampling.

## Bootstrap
Use 5,000 bootstrap replicates with fixed seed `20260913`. Resample participants, not individual knees, within each subgroup so both knees from a sampled participant remain together. Percentile 95% confidence intervals will be reported.

## Interpretation
The audit is descriptive/exploratory. No subgroup difference will be treated as a confirmatory statistical claim. Small subgroups will be labeled as imprecise where relevant. The analysis will specifically assess whether the globally defined 10% abstention rule disproportionately defers predictions from any demographic or clinical subgroup and whether retained-case error improves across groups.
