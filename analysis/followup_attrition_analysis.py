#!/usr/bin/env python3
"""Prespecified follow-up/attrition analysis for the OAI structure-pain project.

Reads:
    outputs/oai_knee_analysis_ready.csv.gz

Writes:
    outputs/attrition/attrition_summary.csv
    outputs/attrition/attrition_detailed_categories.csv
    outputs/attrition/attrition_summary.json

Analysis:
- compares baseline characteristics for knees with vs without observed 48-month WOMAC pain;
- reports missing-minus-observed differences and standardized mean differences;
- uses 5,000 participant-level bootstrap resamples for 95% CIs;
- keeps both knees from each participant together in every resample;
- fixed seed 20260913;
- descriptive only; no new confirmatory p-values.
"""

from pathlib import Path
import json, math
import numpy as np
import pandas as pd

SEED = 20260913
BOOTSTRAPS = 5000

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "oai_knee_analysis_ready.csv.gz"
OUT = ROOT / "outputs" / "attrition"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["outcome_observed"] = df["womac_pain_v06"].notna()
obs = df["outcome_observed"].to_numpy()
miss = ~obs

participants = pd.Index(df["participant_key"].unique())
pmap = {p: i for i, p in enumerate(participants)}
pidx = df["participant_key"].map(pmap).to_numpy()
P = len(participants)

def smd_binary(p_miss, p_obs):
    den = math.sqrt((p_miss*(1-p_miss) + p_obs*(1-p_obs))/2)
    return (p_miss-p_obs)/den if den > 0 else np.nan

continuous_specs = {
    "Baseline WOMAC pain": "womac_pain_v00",
    "Age, years": "age_years",
    "BMI, kg/m^2": "bmi",
}
binary_defs = {
    "Radiographic OA (KL >=2)": ((df["kl_grade_v00"] >= 2).to_numpy(float), df["kl_grade_v00"].notna().to_numpy()),
    "Any medial JSN (grade >=1)": ((df["medial_jsn_v00"] >= 1).to_numpy(float), df["medial_jsn_v00"].notna().to_numpy()),
    "Any lateral JSN (grade >=1)": ((df["lateral_jsn_v00"] >= 1).to_numpy(float), df["lateral_jsn_v00"].notna().to_numpy()),
    "Female sex": ((df["sex"] == "Female").to_numpy(float), df["sex"].notna().to_numpy()),
    "White or Caucasian race": ((df["race"] == "White or Caucasian").to_numpy(float), df["race"].notna().to_numpy()),
    "Black or African American race": ((df["race"] == "Black or African American").to_numpy(float), df["race"].notna().to_numpy()),
    "Asian race": ((df["race"] == "Asian").to_numpy(float), df["race"].notna().to_numpy()),
    "Other non-white race": ((df["race"] == "Other Non-white").to_numpy(float), df["race"].notna().to_numpy()),
    "Hispanic ethnicity": ((df["hispanic"] == "Yes").to_numpy(float), df["hispanic"].notna().to_numpy()),
    "Prior knee injury": ((df["prior_knee_injury"] == 1).to_numpy(float), df["prior_knee_injury"].notna().to_numpy()),
    "Prior knee surgery": ((df["prior_knee_surgery"] == 1).to_numpy(float), df["prior_knee_surgery"].notna().to_numpy()),
}

cols, desc = [], {}
def add_col(a):
    cols.append(np.asarray(a, dtype=float))
    return len(cols)-1

for label, col in continuous_specs.items():
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    valid = ~np.isnan(x)
    for gname, gmask in [("obs", obs), ("miss", miss)]:
        m = valid & gmask
        s = np.bincount(pidx[m], weights=x[m], minlength=P)
        n = np.bincount(pidx[m], minlength=P)
        ss = np.bincount(pidx[m], weights=x[m]**2, minlength=P)
        desc[(label, gname)] = ("continuous", (add_col(s), add_col(n), add_col(ss)))

for label, (x, valid) in binary_defs.items():
    for gname, gmask in [("obs", obs), ("miss", miss)]:
        m = valid & gmask
        y = np.bincount(pidx[m], weights=x[m], minlength=P)
        n = np.bincount(pidx[m], minlength=P)
        desc[(label, gname)] = ("binary", (add_col(y), add_col(n)))

contrib = np.column_stack(cols)
labels = list(continuous_specs) + list(binary_defs)

def calc_stat(totals, label):
    typ = desc[(label, "obs")][0]
    if typ == "continuous":
        osum, on, oss = [totals[i] for i in desc[(label, "obs")][1]]
        msum, mn, mss = [totals[i] for i in desc[(label, "miss")][1]]
        om, mm = osum/on, msum/mn
        ov = (oss-osum*osum/on)/(on-1)
        mv = (mss-msum*msum/mn)/(mn-1)
        diff = mm-om
        return {
            "observed": om, "missing": mm, "difference": diff,
            "smd": diff/math.sqrt((ov+mv)/2),
            "n_observed": int(on), "n_missing": int(mn),
            "sd_observed": math.sqrt(ov), "sd_missing": math.sqrt(mv),
        }
    oy, on = [totals[i] for i in desc[(label, "obs")][1]]
    my, mn = [totals[i] for i in desc[(label, "miss")][1]]
    op, mp = oy/on, my/mn
    return {
        "observed": op, "missing": mp, "difference": mp-op,
        "smd": smd_binary(mp, op),
        "n_observed": int(on), "n_missing": int(mn),
        "count_observed": int(round(oy)), "count_missing": int(round(my)),
    }

full_totals = contrib.sum(axis=0)
point = {lab: calc_stat(full_totals, lab) for lab in labels}

rng = np.random.default_rng(SEED)
boot_diff = {lab: np.empty(BOOTSTRAPS) for lab in labels}
boot_smd = {lab: np.empty(BOOTSTRAPS) for lab in labels}

for start in range(0, BOOTSTRAPS, 100):
    r = min(100, BOOTSTRAPS-start)
    sample_idx = rng.integers(0, P, size=(r, P))
    totals = contrib[sample_idx].sum(axis=1)
    for lab in labels:
        typ = desc[(lab, "obs")][0]
        if typ == "continuous":
            osum, on, oss = [totals[:, i] for i in desc[(lab, "obs")][1]]
            msum, mn, mss = [totals[:, i] for i in desc[(lab, "miss")][1]]
            om, mm = osum/on, msum/mn
            ov = (oss-osum*osum/on)/(on-1)
            mv = (mss-msum*msum/mn)/(mn-1)
            d = mm-om
            s = d/np.sqrt((ov+mv)/2)
        else:
            oy, on = [totals[:, i] for i in desc[(lab, "obs")][1]]
            my, mn = [totals[:, i] for i in desc[(lab, "miss")][1]]
            op, mp = oy/on, my/mn
            d = mp-op
            den = np.sqrt((mp*(1-mp)+op*(1-op))/2)
            s = np.where(den > 0, d/den, np.nan)
        boot_diff[lab][start:start+r] = d
        boot_smd[lab][start:start+r] = s

rows = []
for lab in labels:
    p = point[lab]
    dci = np.nanpercentile(boot_diff[lab], [2.5, 97.5])
    sci = np.nanpercentile(boot_smd[lab], [2.5, 97.5])
    row = {
        "Characteristic": lab,
        "Type": desc[(lab, "obs")][0],
        "Observed": p["observed"],
        "Missing": p["missing"],
        "Missing_minus_observed": p["difference"],
        "Difference_CI_low": dci[0],
        "Difference_CI_high": dci[1],
        "SMD_missing_minus_observed": p["smd"],
        "SMD_CI_low": sci[0],
        "SMD_CI_high": sci[1],
        "N_observed": p["n_observed"],
        "N_missing": p["n_missing"],
    }
    if row["Type"] == "continuous":
        row["SD_observed"] = p["sd_observed"]
        row["SD_missing"] = p["sd_missing"]
    else:
        row["Count_observed"] = p["count_observed"]
        row["Count_missing"] = p["count_missing"]
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "attrition_summary.csv", index=False)

detailed_specs = {
    "KL grade": ("kl_grade_v00", [0,1,2,3,4]),
    "Medial JSN grade": ("medial_jsn_v00", [0,1,2,3]),
    "Lateral JSN grade": ("lateral_jsn_v00", [0,1,2,3]),
    "Sex": ("sex", ["Female","Male"]),
    "Race": ("race", ["White or Caucasian","Black or African American","Asian","Other Non-white"]),
    "Hispanic ethnicity": ("hispanic", ["Yes","No"]),
    "Prior knee injury": ("prior_knee_injury", [1.0,0.0]),
    "Prior knee surgery": ("prior_knee_surgery", [1.0,0.0]),
}
detail_rows = []
for varlabel, (col, levels) in detailed_specs.items():
    valid = df[col].notna()
    for lev in levels:
        ind = df[col] == lev
        op = ind[obs & valid].mean()
        mp = ind[miss & valid].mean()
        detail_rows.append({
            "Variable": varlabel,
            "Level": str(lev),
            "Observed_count": int(ind[obs & valid].sum()),
            "Observed_denominator": int((obs & valid).sum()),
            "Observed_percent": 100*op,
            "Missing_count": int(ind[miss & valid].sum()),
            "Missing_denominator": int((miss & valid).sum()),
            "Missing_percent": 100*mp,
            "Difference_pp_missing_minus_observed": 100*(mp-op),
            "SMD_missing_minus_observed": smd_binary(mp, op),
        })
    missind = df[col].isna()
    op, mp = missind[obs].mean(), missind[miss].mean()
    detail_rows.append({
        "Variable": varlabel,
        "Level": "Missing baseline value",
        "Observed_count": int(missind[obs].sum()),
        "Observed_denominator": int(obs.sum()),
        "Observed_percent": 100*op,
        "Missing_count": int(missind[miss].sum()),
        "Missing_denominator": int(miss.sum()),
        "Missing_percent": 100*mp,
        "Difference_pp_missing_minus_observed": 100*(mp-op),
        "SMD_missing_minus_observed": smd_binary(mp, op) if (op > 0 or mp > 0) else 0.0,
    })

pd.DataFrame(detail_rows).to_csv(OUT / "attrition_detailed_categories.csv", index=False)

out_json = {
    "baseline_knees": int(len(df)),
    "baseline_participants": int(df["participant_key"].nunique()),
    "observed_48m_knees": int(obs.sum()),
    "observed_48m_participants": int(df.loc[obs, "participant_key"].nunique()),
    "missing_48m_knees": int(miss.sum()),
    "missing_48m_participants": int(df.loc[miss, "participant_key"].nunique()),
    "participants_represented_in_both_groups": int(len(set(df.loc[obs,"participant_key"]) & set(df.loc[miss,"participant_key"]))),
    "bootstrap_replicates": BOOTSTRAPS,
    "bootstrap_seed": SEED,
}
with open(OUT / "attrition_summary.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=2)

print(summary.to_string(index=False))
