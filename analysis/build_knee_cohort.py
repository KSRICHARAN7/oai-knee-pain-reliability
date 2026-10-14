#!/usr/bin/env python3
"""Build a de-identified, knee-level OAI analysis table from four original ZIPs."""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


ZIP_NAMES = {
    "clinical": "AllClinical_ASCII.zip",
    "general": "General_ASCII.zip",
    "xray": "X-Ray Image Assessments_ASCII.zip",
}


def extract_named(zip_path: Path, basenames: list[str], destination: Path) -> dict[str, Path]:
    wanted = {name.lower(): name for name in basenames}
    found: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            base = Path(member).name
            if base.lower() in wanted:
                target = destination / wanted[base.lower()]
                target.write_bytes(archive.read(member))
                found[base.lower()] = target
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing from {zip_path.name}: {missing}")
    return found


def read_pipe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", dtype=str, low_memory=False)


def number(series: pd.Series) -> pd.Series:
    """Read the leading number from OAI values such as '2: Female'."""
    extracted = series.astype("string").str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def label(series: pd.Series) -> pd.Series:
    """Keep the human-readable portion of an OAI categorical value."""
    out = series.astype("string").str.replace(r"^\s*-?\d+(?:\.\d+)?\s*:\s*", "", regex=True)
    return out.mask(out.str.startswith(".:", na=False))


def clinical_knees(frame: pd.DataFrame, visit: str) -> pd.DataFrame:
    cols = [f"{visit}WOMKPR", f"{visit}WOMKPL", f"{visit}KOOSKPR", f"{visit}KOOSKPL"]
    for col in cols:
        if col not in frame:
            frame[col] = pd.NA
    right = frame[["ID", cols[0], cols[2]]].rename(
        columns={cols[0]: f"womac_pain_{visit.lower()}", cols[2]: f"koos_pain_{visit.lower()}"}
    ).assign(side="right")
    left = frame[["ID", cols[1], cols[3]]].rename(
        columns={cols[1]: f"womac_pain_{visit.lower()}", cols[3]: f"koos_pain_{visit.lower()}"}
    ).assign(side="left")
    out = pd.concat([right, left], ignore_index=True)
    for col in out.columns.difference(["ID", "side"]):
        out[col] = number(out[col])
    return out


def xray_knees(frame: pd.DataFrame, visit: str) -> pd.DataFrame:
    frame = frame[number(frame["READPRJ"]).eq(15)].copy()
    out = frame[["ID", "SIDE", f"{visit}XRKL", f"{visit}XRJSM", f"{visit}XRJSL"]].copy()
    out["side"] = number(out.pop("SIDE")).map({1: "right", 2: "left"})
    out = out.rename(columns={
        f"{visit}XRKL": f"kl_grade_{visit.lower()}",
        f"{visit}XRJSM": f"medial_jsn_{visit.lower()}",
        f"{visit}XRJSL": f"lateral_jsn_{visit.lower()}",
    })
    for col in out.columns.difference(["ID", "side"]):
        out[col] = number(out[col])
    if out.duplicated(["ID", "side"]).any():
        raise ValueError(f"Duplicate Project 15 records at {visit}")
    return out


def baseline_covariates(clinical: pd.DataFrame, enrollees: pd.DataFrame) -> pd.DataFrame:
    person = clinical[["ID", "V00AGE", "P01BMI"]].copy()
    person["age_years"] = number(person.pop("V00AGE"))
    person["bmi"] = number(person.pop("P01BMI"))
    enrol = enrollees[["ID", "P02SEX", "P02RACE", "P02HISP"]].copy()
    enrol["sex"] = label(enrol.pop("P02SEX"))
    enrol["race"] = label(enrol.pop("P02RACE"))
    enrol["hispanic"] = number(enrol.pop("P02HISP")).map({0: "No", 1: "Yes"})
    person = person.merge(enrol, on="ID", how="left", validate="one_to_one")

    right = clinical[["ID", "P01INJR", "P01KSURGR"]].rename(
        columns={"P01INJR": "prior_knee_injury", "P01KSURGR": "prior_knee_surgery"}
    ).assign(side="right")
    left = clinical[["ID", "P01INJL", "P01KSURGL"]].rename(
        columns={"P01INJL": "prior_knee_injury", "P01KSURGL": "prior_knee_surgery"}
    ).assign(side="left")
    joint = pd.concat([right, left], ignore_index=True)
    for col in ["prior_knee_injury", "prior_knee_surgery"]:
        joint[col] = number(joint[col])
    return joint.merge(person, on="ID", how="left", validate="many_to_one")


def knee_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"id": "ID"})
    right = frame[["ID", "V99ERKBLRP", "V99ERKRPCF", "V99ERKDAYS"]].rename(columns={
        "V99ERKBLRP": "replacement_at_baseline",
        "V99ERKRPCF": "replacement_status_code",
        "V99ERKDAYS": "replacement_days",
    }).assign(side="right")
    left = frame[["ID", "V99ELKBLRP", "V99ELKRPCF", "V99ELKDAYS"]].rename(columns={
        "V99ELKBLRP": "replacement_at_baseline",
        "V99ELKRPCF": "replacement_status_code",
        "V99ELKDAYS": "replacement_days",
    }).assign(side="left")
    out = pd.concat([right, left], ignore_index=True)
    for col in ["replacement_at_baseline", "replacement_status_code", "replacement_days"]:
        out[col] = number(out[col])
    out["confirmed_followup_replacement"] = out["replacement_status_code"].eq(3).fillna(False).astype("int8")
    return out


def build(zip_directory: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        clinical = extract_named(zip_directory / ZIP_NAMES["clinical"], [
            "AllClinical00.txt", "AllClinical03.txt", "AllClinical05.txt", "AllClinical06.txt"
        ], temp)
        general = extract_named(zip_directory / ZIP_NAMES["general"], ["Enrollees.txt", "OUTCOMES99.txt"], temp)
        xray = extract_named(zip_directory / ZIP_NAMES["xray"], ["KXR_SQ_BU00.txt", "KXR_SQ_BU06.txt"], temp)

        c0 = read_pipe(clinical["allclinical00.txt"])
        cohort = xray_knees(read_pipe(xray["kxr_sq_bu00.txt"]), "V00")
        for visit, key in [("V00", "allclinical00.txt"), ("V03", "allclinical03.txt"),
                           ("V05", "allclinical05.txt"), ("V06", "allclinical06.txt")]:
            cohort = cohort.merge(clinical_knees(read_pipe(clinical[key]), visit),
                                  on=["ID", "side"], how="left", validate="one_to_one")
        cohort = cohort.merge(xray_knees(read_pipe(xray["kxr_sq_bu06.txt"]), "V06"),
                              on=["ID", "side"], how="left", validate="one_to_one")
        cohort = cohort.merge(baseline_covariates(c0, read_pipe(general["enrollees.txt"])),
                              on=["ID", "side"], how="left", validate="one_to_one")
        cohort = cohort.merge(knee_outcomes(read_pipe(general["outcomes99.txt"])),
                              on=["ID", "side"], how="left", validate="one_to_one")

    initial_n = len(cohort)
    cohort = cohort[cohort["replacement_at_baseline"].fillna(0).eq(0)].copy()
    cohort = cohort[cohort["kl_grade_v00"].notna() & cohort["womac_pain_v00"].notna()].copy()
    id_map = {value: i + 1 for i, value in enumerate(sorted(cohort["ID"].unique()))}
    cohort.insert(0, "participant_key", cohort["ID"].map(id_map).astype("int32"))
    cohort.insert(1, "knee_key", cohort["participant_key"].astype(str) + "_" + cohort["side"].str[0].str.upper())
    cohort = cohort.drop(columns=["ID"]).sort_values(["participant_key", "side"]).reset_index(drop=True)

    analysis_path = output_directory / "oai_knee_analysis_ready.csv.gz"
    cohort.to_csv(analysis_path, index=False, compression="gzip")
    missing = pd.DataFrame({
        "variable": cohort.columns,
        "nonmissing_n": [cohort[c].notna().sum() for c in cohort],
        "missing_n": [cohort[c].isna().sum() for c in cohort],
        "missing_percent": [100 * cohort[c].isna().mean() for c in cohort],
    })
    missing.to_csv(output_directory / "missingness_report.csv", index=False, float_format="%.2f")
    summary = [
        "OAI knee cohort audit",
        "=====================",
        f"Project 15 baseline knee records: {initial_n:,}",
        f"Final eligible baseline cohort: {len(cohort):,} knees from {cohort.participant_key.nunique():,} participants",
        f"Complete WOMAC at baseline/24/36/48 months: {cohort[['womac_pain_v00','womac_pain_v03','womac_pain_v05','womac_pain_v06']].notna().all(axis=1).sum():,}",
        f"Complete KL and WOMAC at baseline and 48 months: {cohort[['kl_grade_v00','womac_pain_v00','kl_grade_v06','womac_pain_v06']].notna().all(axis=1).sum():,}",
        "",
        "Rules: READPRJ=15; no baseline knee replacement; baseline KL and WOMAC pain required.",
        "Participant IDs were replaced with project-specific sequential keys.",
    ]
    (output_directory / "cohort_audit.txt").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-dir", type=Path, required=True, help="Directory containing the original OAI ZIPs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    build(args.zip_dir, args.output_dir)


if __name__ == "__main__":
    main()
