# Sensitivity analysis specification lock

Lock date: 2026-09-13
Status: specified before running the seven prespecified sensitivity analyses below.

The frozen primary analysis remains unchanged. These analyses are sensitivity/secondary analyses and cannot replace the primary result.

## S1. Radiographic OA restriction
Restrict to knees with baseline KL grade >=2 and observed 48-month WOMAC pain. Refit the entire participant-grouped nested five-fold elastic-net pipeline using the same primary predictors, preprocessing, hyperparameter grid, outcome, and seed. Re-estimate expected baseline pain by KL inside each outer training fold. Test absolute discordance against absolute held-out 48-month prediction error with participant-clustered inference.

## S2. Signed discordance / directional asymmetry
Use the original frozen primary held-out predictions. Define positive discordance=max(signed discordance,0) and negative discordance=max(-signed discordance,0). Regress absolute held-out error simultaneously on positive and negative discordance with participant-clustered covariance. Report both slopes and a two-sided Wald test of equality of the two slopes. This asks whether unexpectedly high pain and unexpectedly low pain differ in their association with prediction error.

## S3. 48-month WOMAC change outcome
Define change = 48-month WOMAC pain - baseline WOMAC pain. Refit the entire participant-grouped nested five-fold elastic-net pipeline with the same baseline predictors and hyperparameter grid, but predict change instead of the absolute 48-month score. Re-estimate discordance within each outer training fold. Test absolute discordance against absolute held-out change-prediction error with participant-clustered inference.

## S4. Covariate-adjusted discordance
Keep the frozen primary held-out 48-month predictions and errors unchanged. Within each outer training fold, fit an expected-baseline-pain linear model using categorical KL grade plus age, BMI, sex, prior knee injury, and prior knee surgery. Numeric missing values use training-fold median imputation; categorical missing values use an explicit Missing category; categorical variables are one-hot encoded. Apply that model to the untouched outer-test fold. Adjusted discordance is observed baseline WOMAC pain minus this expected value. Regress frozen absolute held-out prediction error on absolute adjusted discordance with participant-clustered inference.

## S5. Complete-case analysis
Restrict to knees with observed 48-month WOMAC pain and no missing values in any primary predictor. Refit the entire participant-grouped nested five-fold elastic-net pipeline using the same model, grid, seed, and cross-fitted KL-based discordance. Test absolute discordance against absolute held-out error with participant-clustered inference.

## S6. One knee per participant
From participants with an observed 48-month outcome, select exactly one knee using a reproducible random draw with seed 20260721. Refit the entire nested five-fold elastic-net pipeline using the same predictors, preprocessing, grid, and cross-fitted KL-based discordance. Test absolute discordance against absolute held-out error. Participant clustering is retained in the inferential code although every participant contributes one knee.

## S7. Gradient-boosted-tree comparator
On the full primary cohort, keep the same frozen outer participant folds. Fit a scikit-learn HistGradientBoostingRegressor after training-fold median/missing-category preprocessing and dense one-hot encoding. Inner participant-grouped five-fold CV selects learning_rate in {0.03,0.10}, max_leaf_nodes in {15,31}, and l2_regularization in {0,1}; max_iter=200 and random_state=20260721 are fixed. Scoring is negative mean absolute error. Generate exactly one held-out prediction per eligible knee. Report MAE, RMSE, bias, calibration intercept/slope, the paired participant-bootstrap difference in MAE versus elastic net, and the participant-clustered association between the original cross-fitted absolute discordance and GBT absolute error.

## Reporting rule
All seven results will be reported regardless of direction or statistical significance. No sensitivity analysis can replace the frozen primary model or hypothesis result.

## Computational amendment before S7 results
The initial S7 implementation using scikit-learn HistGradientBoostingRegressor did not complete within the available runtime and produced no model result. Before observing any S7 performance result, the comparator implementation was changed to XGBoost (`XGBRegressor`) with histogram tree building. The scientific analysis remains the same: full primary cohort, frozen participant outer folds, participant-grouped inner five-fold tuning, MAE-based model selection, exactly one held-out prediction per knee, and paired comparison against elastic net. The tuning grid is n_estimators in {200,500}, max_depth in {2,3}, and learning_rate in {0.03,0.10}; subsample=0.8, colsample_bytree=0.8, reg_lambda=1, objective=`reg:squarederror`, tree_method=`hist`, random_state=20260721. This amendment was made for computational feasibility before any S7 result was available.
