# Secondary longitudinal analysis specification

**Date frozen:** 2026-09-13  
**Status:** Secondary analysis implementing the longitudinal hypothesis already listed in the July 21 preregistration.  
**Important:** The 48-month primary result is already known. This specification was written before examining 24- or 36-month longitudinal prediction results.

## Scientific question
Does greater baseline absolute structure-pain discordance identify knees whose WOMAC pain is less predictable across 24, 36, and 48 months?

## Analysis cohort
- Use the 7,807 knees with baseline WOMAC pain and observed WOMAC pain at all three selected follow-up visits (24, 36, and 48 months).
- These knees belong to 3,937 participants.
- Restricting to complete longitudinal follow-up ensures that every knee contributes the same three repeated measurements.
- This is a secondary complete-trajectory analysis and does not replace the frozen 48-month primary analysis.

## Outer folds and leakage control
- Reuse the participant-level outer-fold assignments frozen for the primary 48-month model.
- Both knees from one participant therefore remain in the same outer fold at all three time points.
- Within each outer training fold, expected baseline pain is estimated from categorical baseline KL grade using training data only.
- Held-out baseline discordance for each knee is observed baseline WOMAC pain minus the training-fold KL-specific expected pain; the absolute value is the longitudinal exposure.

## Time-specific prediction models
For each follow-up time (24, 36, and 48 months):
- Fit an elastic-net regression model using the same baseline predictor set as the frozen primary model:
  - baseline WOMAC pain;
  - baseline KL grade, categorical;
  - medial and lateral joint-space narrowing;
  - age;
  - BMI;
  - sex;
  - race;
  - Hispanic ethnicity;
  - prior knee injury;
  - prior knee surgery.
- Use participant-grouped nested five-fold cross-validation.
- Reuse the frozen outer participant folds; tune alpha and l1_ratio only within each outer training set using grouped inner five-fold CV.
- Preprocessing and imputation are learned from training data only.
- Generate one held-out prediction and one absolute held-out prediction error for each knee at each visit.

## Longitudinal mixed-effects model
Stack the 24-, 36-, and 48-month held-out absolute errors in long format (three rows per knee).

Fit a linear mixed-effects model with:
- outcome: absolute held-out WOMAC prediction error;
- fixed effects: absolute baseline discordance, visit as a categorical factor (24 months as reference), and discordance-by-visit interaction;
- participant random intercept;
- knee-within-participant random intercept as a variance component.

The discordance coefficient estimates the association at 24 months. Interaction terms estimate how that association differs at 36 and 48 months. Linear combinations will report the discordance-error slope separately at 24, 36, and 48 months with 95% confidence intervals.

## Interpretation
- This is a secondary/exploratory longitudinal analysis; the frozen primary 48-month hypothesis remains primary.
- The analysis addresses predictability, not causation or disease progression.
- A positive discordance-error slope at multiple visits would support the idea that structure-pain discordance marks persistently harder-to-predict trajectories.
- Null or inconsistent results will be reported.
