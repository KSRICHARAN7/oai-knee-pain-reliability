# Preregistration lock: OAI structure–pain discordance study

**Protocol status:** Frozen before examining any association between discordance and future prediction error.  
**Lock date:** 2026-07-21  
**Study type:** Secondary longitudinal analysis with internal validation.

## 1. Primary question

Among adults at risk for or living with knee osteoarthritis, does the magnitude of baseline discordance between radiographic severity and patient-reported pain identify knees with less accurate predictions of 48-month WOMAC pain?

## 2. Primary cohort and unit of analysis

- Unit: individual knee.
- Source: Osteoarthritis Initiative public-use data.
- X-ray readings: Project 15 only.
- Include: baseline Kellgren–Lawrence (KL) grade and baseline knee-specific WOMAC pain available.
- Exclude: knee replacement visible on the baseline OAI radiograph.
- Primary prediction analysis additionally requires observed 48-month WOMAC pain; the outcome will not be imputed.
- Primary baseline cohort: 8,920 knees from 4,490 participants.
- Anticipated primary outcome cohort: 8,232 knees with 48-month WOMAC pain.
- Two knees from one participant may be retained, but participant grouping must be preserved in every split and uncertainty calculation.

## 3. Primary outcome

Observed knee-specific WOMAC pain at 48 months (`womac_pain_v06`, range 0–20). The prediction target is the absolute 48-month score, with baseline WOMAC included as a predictor.

Secondary outcomes are WOMAC pain at 24 and 36 months, the 48-month change from baseline, and adjudicated knee replacement. These cannot replace the primary outcome after results are examined.

## 4. Primary discordance exposure

Within each training fold, expected baseline pain will be estimated from categorical KL grade:

`expected baseline pain = mean baseline WOMAC pain within the knee's KL grade in the training data`

Signed discordance is:

`observed baseline WOMAC pain - expected baseline WOMAC pain`

The **primary exposure** is the absolute value of signed discordance, measured in WOMAC points. Positive and negative signed discordance will be examined separately only as secondary analyses. Expected pain for held-out knees must be calculated from their training fold, never from the full dataset.

## 5. Primary prediction model

The primary outcome model will be elastic-net regression. Prespecified baseline predictors are:

- Baseline WOMAC pain
- KL grade entered categorically
- Medial and lateral joint-space narrowing
- Age
- Body mass index
- Sex
- Race
- Hispanic ethnicity
- Prior injury to the corresponding knee
- Prior surgery on the corresponding knee

Continuous variables will not be selected or removed on the basis of univariate significance. Categorical predictors will be one-hot encoded. Missing numeric predictors will receive training-fold median imputation; missing categorical predictors will receive an explicit `Missing` category. Preprocessing must be learned in training data only.

## 6. Validation design

- Primary validation: nested five-fold cross-validation grouped by participant.
- Outer folds generate exactly one held-out prediction for every eligible knee.
- Inner grouped folds select the elastic-net penalty and mixing parameter.
- Fixed random seed: `20260721`.
- The right and left knees of one participant must never appear on opposite sides of a split.
- A gradient-boosted tree model may be evaluated as a secondary nonlinear comparator, but it cannot replace elastic net as the primary model.

## 7. Primary error and hypothesis test

Absolute prediction error is:

`abs(observed 48-month WOMAC - predicted 48-month WOMAC)`

The primary hypothesis will be tested with a regression of absolute held-out prediction error on absolute baseline discordance. Inference will account for clustering of knees within participants. The primary coefficient is expected to be greater than zero. A two-sided alpha of 0.05 will be used for this single primary hypothesis.

Primary model performance measures are mean absolute error (MAE), root mean squared error, mean prediction bias, calibration intercept, and calibration slope. Ninety-five percent confidence intervals will be calculated by resampling participants rather than individual knees.

## 8. Uncertainty and abstention

An uncertainty model will estimate expected absolute error without accessing outer-test outcomes. Within each outer training set, inner out-of-fold prediction errors will be generated and used as targets for a regularized secondary error model. The trained uncertainty model will then assign an uncertainty score to the untouched outer-test knees.

The primary abstention threshold is the 10% of held-out knees with the highest estimated uncertainty. Secondary thresholds are 5%, 15%, and 20%.

Primary uncertainty evaluations are:

1. Association between estimated uncertainty and observed absolute error.
2. MAE among the 90% of knees retained after 10% abstention.
3. Change in MAE compared with no abstention.
4. Risk–coverage curve across all prespecified thresholds.
5. Comparison with random abstention at equal coverage.

The uncertainty method will be considered informative if higher uncertainty corresponds to higher held-out error and targeted 10% abstention lowers retained-case MAE more than random 10% abstention. Abstention is a research flag for review, not a treatment recommendation.

## 9. Missingness and subgroup audits

The primary outcome will not be imputed. Baseline characteristics of knees with and without observed 48-month WOMAC pain will be compared descriptively. Complete-case analysis will be a sensitivity analysis.

Error, calibration, and abstention frequency will be audited across sex, race, Hispanic ethnicity, KL grade, injury history, and surgery history when sample sizes permit. These subgroup analyses are exploratory and will be reported with confidence intervals rather than used for confirmatory claims.

## 10. Prespecified sensitivity analyses

1. Restrict the cohort to radiographic OA at baseline (KL grade at least 2).
2. Use signed discordance to distinguish more pain than expected from less pain than expected.
3. Use 48-month change in WOMAC rather than the absolute 48-month value.
4. Recalculate discordance after adjustment for age, BMI, sex, injury, and surgery.
5. Use complete cases only.
6. Select one knee per participant to remove within-person dependence.
7. Compare elastic net with a gradient-boosted tree model.

## 11. Reporting and stopping rules

- No test-set-driven feature selection, cutoff selection, or model replacement.
- All deviations from this document must be dated, justified, and labeled as amendments made before or after outcome analysis.
- Null, mixed, or unfavorable findings will be reported.
- Results will not be described as causal or externally validated.
- Participant-level OAI data will remain private; only code and aggregate results may be shared under the OAI data-use terms.

## Frozen primary analysis in one sentence

Using participant-grouped nested five-fold cross-validation, test whether cross-fitted absolute baseline KL–WOMAC discordance predicts absolute held-out error from an elastic-net model of 48-month WOMAC pain, then test whether abstaining on the 10% of knees with the highest nested uncertainty estimate reduces retained-case MAE more than random abstention.
