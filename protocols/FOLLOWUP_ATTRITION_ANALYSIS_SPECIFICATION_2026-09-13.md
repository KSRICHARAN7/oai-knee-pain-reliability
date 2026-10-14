# Follow-up / attrition analysis specification

**Date:** 2026-09-13  
**Status:** Secondary descriptive analysis already required by the July 21 preregistration.

The frozen preregistration specified that baseline characteristics of knees with and without observed
48-month WOMAC pain would be compared descriptively to assess possible follow-up bias.

## Cohort
- Start from all 8,920 knees in the frozen baseline cohort.
- Observed-outcome group: knees with nonmissing 48-month WOMAC pain.
- Missing-outcome group: knees with missing 48-month WOMAC pain.
- No 48-month outcome values will be imputed for this analysis.

## Prespecified baseline characteristics examined
- Baseline WOMAC pain
- Baseline KL grade / radiographic OA status
- Baseline medial and lateral joint-space narrowing
- Age
- BMI
- Sex
- Race
- Hispanic ethnicity
- Prior knee injury
- Prior knee surgery

## Reporting
- Continuous variables: mean (SD), missing-minus-observed mean difference, standardized mean difference (SMD).
- Categorical variables: proportions, missing-minus-observed percentage-point difference, SMD.
- 95% confidence intervals are obtained by resampling participants, not individual knees, with 5,000 bootstrap replicates.
- Both knees from a sampled participant remain together in each bootstrap replicate.
- Fixed bootstrap seed: 20260913.
- P values are not used because this analysis is descriptive rather than a new confirmatory hypothesis test.
- Absolute SMDs around 0.10 or greater are flagged as potentially meaningful imbalance, but interpretation considers magnitude and clinical context rather than a rigid cutoff.
- Missing baseline values are not imputed for this descriptive comparison; available-case denominators are reported.

This analysis does not alter the frozen primary model, primary hypothesis, or primary result.
