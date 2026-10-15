# Research overview

The question is whether baseline structure–pain mismatch identifies knees with less accurate predictions of future pain. The unit is a knee; both knees can contribute, with participant grouping preserved in validation and inference.

Baseline discordance is absolute observed WOMAC pain minus expected baseline pain for the same Kellgren–Lawrence grade, where the expectation is estimated in the training fold. Elastic-net models predict the absolute 48-month WOMAC score using baseline pain, radiographic severity, demographics, injury, and surgery history. All preprocessing and hyperparameter tuning take place inside training data.

The primary held-out MAE was 1.787 points. The fitted association between each additional point of absolute discordance and absolute prediction error was 0.301 points (participant-clustered 95% CI 0.275–0.326). This is the frozen unadjusted association, not a causal effect. The supplied manuscript reports that adjustment for baseline pain in a post-freeze sensitivity analysis reduced the discordance coefficient to -0.011 (95% CI -0.043 to 0.022; p = 0.52). Thus, an independent discordance association was not detected. These manuscript-reported values were not recomputed from the original technical ZIP. Quartile means do not form a simple monotonic progression.

An error model trained on inner out-of-fold errors assigned uncertainty scores to untouched outer-test knees. At the prespecified 10% deferral threshold, retained-case MAE was 1.582, an 11.47% relative reduction. The score estimates expected error; it is not a calibrated prediction interval or clinical confidence guarantee. Deferral performance must be considered alongside coverage and subgroup differences.

Limitations include one cohort, internal validation, complete observed follow-up for the primary outcome, potential attrition and demographic differences, and limited external clinical applicability. The replacement audit did not support a defensible survival model. MOST replication remains future work.
