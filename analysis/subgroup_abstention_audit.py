#!/usr/bin/env python3
"""Exploratory subgroup and abstention audit for frozen OAI held-out predictions.

Uses only the already-frozen primary held-out predictions and nested uncertainty scores.
The 10% abstention set is global and unchanged from the primary uncertainty analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20260913
BOOTSTRAPS = 5000
MIN_REPORT_N = 50
MIN_CALIBRATION_N = 100

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "oai_knee_analysis_ready.csv.gz"
SCORES = ROOT / "outputs" / "uncertainty" / "heldout_uncertainty_scores.csv.gz"
OUT = ROOT / "outputs" / "subgroup_audit"
OUT.mkdir(parents=True, exist_ok=True)

SUBGROUPS = {
    "Sex": ("sex", ["Female", "Male"]),
    "Race": ("race", ["White or Caucasian", "Black or African American", "Asian", "Other Non-white"]),
    "Hispanic ethnicity": ("hispanic", ["No", "Yes"]),
    "Baseline KL grade": ("kl_grade_v00", [0, 1, 2, 3, 4]),
    "Prior knee injury": ("prior_knee_injury", [0.0, 1.0]),
    "Prior knee surgery": ("prior_knee_surgery", [0.0, 1.0]),
}


def label_level(dimension: str, level) -> str:
    if dimension in {"Prior knee injury", "Prior knee surgery"}:
        return "Yes" if float(level) == 1.0 else "No"
    return str(level)


def metrics_from_totals(t):
    n, sy, sx, sx2, sxy, sae, sbias, sabstain, sret_err, nret = t
    if n <= 0:
        return (np.nan,) * 7
    mae = sae / n
    bias = sbias / n
    abstain_rate = sabstain / n
    retained_mae = sret_err / nret if nret > 0 else np.nan
    denom = sx2 - sx * sx / n
    if denom > 0:
        slope = (sxy - sx * sy / n) / denom
        intercept = sy / n - slope * sx / n
    else:
        slope = np.nan
        intercept = np.nan
    return mae, bias, abstain_rate, retained_mae, intercept, slope, nret


def audit_group(frame: pd.DataFrame, rng: np.random.Generator):
    # Aggregate sufficient statistics by participant for cluster bootstrap.
    tmp = frame.copy()
    tmp["xy"] = tmp["predicted_womac_48m"] * tmp["observed_womac_48m"]
    tmp["x2"] = tmp["predicted_womac_48m"] ** 2
    tmp["bias"] = tmp["predicted_womac_48m"] - tmp["observed_womac_48m"]
    tmp["retained_error"] = np.where(tmp["abstained_10pct"], 0.0, tmp["absolute_error"])
    tmp["retained_n"] = (~tmp["abstained_10pct"]).astype(int)
    tmp["one"] = 1
    agg = tmp.groupby("participant_key", sort=False).agg(
        n=("one", "sum"),
        sy=("observed_womac_48m", "sum"),
        sx=("predicted_womac_48m", "sum"),
        sx2=("x2", "sum"),
        sxy=("xy", "sum"),
        sae=("absolute_error", "sum"),
        sbias=("bias", "sum"),
        sabstain=("abstained_10pct", "sum"),
        sret_err=("retained_error", "sum"),
        nret=("retained_n", "sum"),
    )
    A = agg.to_numpy(dtype=float)
    totals = A.sum(axis=0)
    point = metrics_from_totals(totals)

    b_mae = np.empty(BOOTSTRAPS)
    b_bias = np.empty(BOOTSTRAPS)
    b_abst = np.empty(BOOTSTRAPS)
    b_retmae = np.empty(BOOTSTRAPS)
    b_intercept = np.empty(BOOTSTRAPS)
    b_slope = np.empty(BOOTSTRAPS)
    P = len(A)
    chunk = 100
    for start in range(0, BOOTSTRAPS, chunk):
        r = min(chunk, BOOTSTRAPS-start)
        idx = rng.integers(0, P, size=(r, P))
        T = A[idx].sum(axis=1)
        n = T[:,0]; sy=T[:,1]; sx=T[:,2]; sx2=T[:,3]; sxy=T[:,4]
        sae=T[:,5]; sbias=T[:,6]; sab=T[:,7]; sre=T[:,8]; nr=T[:,9]
        mae=sae/n; bias=sbias/n; abst=sab/n; ret=np.divide(sre,nr,out=np.full(r,np.nan),where=nr>0)
        den=sx2-sx*sx/n
        slope=np.divide(sxy-sx*sy/n,den,out=np.full(r,np.nan),where=den>0)
        intercept=sy/n-slope*sx/n
        sl=slice(start,start+r)
        b_mae[sl]=mae; b_bias[sl]=bias; b_abst[sl]=abst; b_retmae[sl]=ret
        b_intercept[sl]=intercept; b_slope[sl]=slope

    def q(a):
        return np.nanpercentile(a,[2.5,97.5]).tolist()
    return {
        "n_knees": int(len(frame)),
        "n_participants": int(frame.participant_key.nunique()),
        "mae": float(point[0]), "mae_ci": q(b_mae),
        "bias": float(point[1]), "bias_ci": q(b_bias),
        "abstention_rate": float(point[2]), "abstention_rate_ci": q(b_abst),
        "retained_n": int(point[6]),
        "retained_mae": float(point[3]), "retained_mae_ci": q(b_retmae),
        "mae_change_after_abstention": float(point[3]-point[0]),
        "calibration_intercept": float(point[4]), "calibration_intercept_ci": q(b_intercept),
        "calibration_slope": float(point[5]), "calibration_slope_ci": q(b_slope),
    }


def main():
    data = pd.read_csv(DATA)
    scores = pd.read_csv(SCORES)
    if len(scores) != 8232 or not scores.knee_key.is_unique:
        raise RuntimeError("Frozen uncertainty score file does not match the primary cohort")

    merge_cols = ["knee_key", "sex", "race", "hispanic", "kl_grade_v00", "prior_knee_injury", "prior_knee_surgery"]
    scores = scores.merge(data[merge_cols], on="knee_key", how="left", validate="one_to_one")

    n_abstain = int(np.floor(len(scores)*0.10))
    order = np.argsort(-scores.estimated_uncertainty.to_numpy())
    scores["abstained_10pct"] = False
    scores.loc[order[:n_abstain], "abstained_10pct"] = True
    global_rate = float(scores.abstained_10pct.mean())
    cutoff = float(scores.loc[order[n_abstain-1], "estimated_uncertainty"])
    next_value = float(scores.loc[order[n_abstain], "estimated_uncertainty"])

    rows = []
    rng = np.random.default_rng(SEED)

    # Overall context row.
    overall = audit_group(scores, rng)
    rows.append({"dimension":"Overall", "level":"All", **overall})

    missing_rows=[]
    for dimension,(col,levels) in SUBGROUPS.items():
        missing_rows.append({"dimension":dimension,"missing_n":int(scores[col].isna().sum())})
        for level in levels:
            g = scores.loc[scores[col].eq(level)].copy()
            if len(g) < MIN_REPORT_N:
                status = "Below reporting threshold"
                rows.append({
                    "dimension":dimension,"level":label_level(dimension,level),
                    "n_knees":int(len(g)),"n_participants":int(g.participant_key.nunique()),
                    "reporting_status":status,
                })
                continue
            m = audit_group(g, rng)
            status = "Reported; calibration suppressed" if len(g) < MIN_CALIBRATION_N else "Reported"
            if len(g) < MIN_CALIBRATION_N:
                m["calibration_intercept"] = np.nan
                m["calibration_intercept_ci"] = [np.nan,np.nan]
                m["calibration_slope"] = np.nan
                m["calibration_slope_ci"] = [np.nan,np.nan]
            rows.append({"dimension":dimension,"level":label_level(dimension,level),"reporting_status":status,**m})

    out = pd.DataFrame(rows)
    # Expand CI list fields for flat CSV.
    for stem in ["mae","bias","abstention_rate","retained_mae","calibration_intercept","calibration_slope"]:
        ci_col = stem+"_ci"
        out[stem+"_ci_low"] = out[ci_col].apply(lambda x: x[0] if isinstance(x,list) else np.nan)
        out[stem+"_ci_high"] = out[ci_col].apply(lambda x: x[1] if isinstance(x,list) else np.nan)
        out.drop(columns=[ci_col], inplace=True)
    out["abstention_percent"] = 100*out["abstention_rate"]
    out["abstention_ci_low_percent"] = 100*out["abstention_rate_ci_low"]
    out["abstention_ci_high_percent"] = 100*out["abstention_rate_ci_high"]
    out["coverage_percent"] = 100-out["abstention_percent"]
    out["mae_improvement_after_abstention"] = -out["mae_change_after_abstention"]

    cols = [
        "dimension","level","reporting_status","n_knees","n_participants",
        "mae","mae_ci_low","mae_ci_high","bias","bias_ci_low","bias_ci_high",
        "calibration_intercept","calibration_intercept_ci_low","calibration_intercept_ci_high",
        "calibration_slope","calibration_slope_ci_low","calibration_slope_ci_high",
        "abstention_percent","abstention_ci_low_percent","abstention_ci_high_percent","coverage_percent",
        "retained_n","retained_mae","retained_mae_ci_low","retained_mae_ci_high",
        "mae_improvement_after_abstention"
    ]
    out = out.reindex(columns=cols)
    out.to_csv(OUT/"subgroup_audit_summary.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(OUT/"subgroup_missingness_counts.csv", index=False)

    # Audit details and technical summary.
    reported = out[(out.dimension != "Overall") & out.reporting_status.str.startswith("Reported", na=False)].copy()
    summary = {
        "eligible_knees": int(len(scores)),
        "participants": int(scores.participant_key.nunique()),
        "global_abstained_n": n_abstain,
        "global_abstention_percent": 100*global_rate,
        "uncertainty_cutoff_for_global_10pct": cutoff,
        "next_uncertainty_value_below_cutoff": next_value,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "min_n_for_error_and_abstention_reporting": MIN_REPORT_N,
        "min_n_for_calibration_reporting": MIN_CALIBRATION_N,
        "reported_subgroup_rows": int(len(reported)),
        "abstention_percent_min": float(reported.abstention_percent.min()),
        "abstention_percent_max": float(reported.abstention_percent.max()),
        "mae_min": float(reported.mae.min()),
        "mae_max": float(reported.mae.max()),
        "all_groups_retained_mae_lower_than_pre_abstention": bool((reported.mae_improvement_after_abstention > 0).all()),
    }
    (OUT/"subgroup_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    # Technical plot: subgroup abstention frequency with participant-bootstrap CIs.
    plot = reported.copy()
    plot["label"] = plot["dimension"] + ": " + plot["level"]
    # Preserve audit specification order.
    y = np.arange(len(plot))
    x = plot.abstention_percent.to_numpy()
    lo = x - plot.abstention_ci_low_percent.to_numpy()
    hi = plot.abstention_ci_high_percent.to_numpy() - x
    fig, ax = plt.subplots(figsize=(9, max(6, 0.38*len(plot)+1.6)))
    ax.errorbar(x, y, xerr=np.vstack([lo,hi]), fmt="o", capsize=3)
    ax.axvline(100*global_rate, linestyle="--", linewidth=1.5, label="Overall global-rule abstention rate")
    ax.set_yticks(y)
    ax.set_yticklabels(plot.label)
    ax.invert_yaxis()
    ax.set_xlabel("Abstained at global 10% uncertainty rule (%)")
    ax.set_title("Subgroup abstention frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/"subgroup_abstention_frequency.png", dpi=220)
    plt.close(fig)

    print(out.to_string(index=False))
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
