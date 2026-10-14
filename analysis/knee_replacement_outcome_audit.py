#!/usr/bin/env python3
"""Audit feasibility of the prespecified knee-replacement secondary outcome.

This script intentionally does NOT fit a survival model. It verifies whether the
archived analysis-ready table contains the event and censoring information needed
for a defensible time-to-event analysis.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "oai_knee_analysis_ready.csv.gz"
OUT = ROOT / "outputs" / "replacement_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

field_rows = []
for c in [
    "replacement_at_baseline",
    "replacement_status_code",
    "replacement_days",
    "confirmed_followup_replacement",
]:
    s = df[c]
    field_rows.append({
        "field": c,
        "dtype": str(s.dtype),
        "nonmissing_n": int(s.notna().sum()),
        "missing_n": int(s.isna().sum()),
        "unique_nonmissing_n": int(s.nunique(dropna=True)),
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
    })
pd.DataFrame(field_rows).to_csv(OUT / "replacement_field_audit.csv", index=False)

status = (
    df.assign(status_display=df["replacement_status_code"].astype("Int64").astype(str).replace("<NA>", "missing"))
      .groupby("status_display", dropna=False)
      .agg(
          knees=("knee_key", "size"),
          replacement_days_nonmissing=("replacement_days", lambda s: int(s.notna().sum())),
          confirmed_flag_n=("confirmed_followup_replacement", lambda s: int((s == 1).sum())),
      )
      .reset_index()
)
status["percent_of_baseline_cohort"] = 100 * status["knees"] / len(df)
status.to_csv(OUT / "replacement_status_counts.csv", index=False)

confirmed = df[df["confirmed_followup_replacement"].eq(1)]
horizons = [("48_months", 1461), ("5_years", 1826), ("6_years", 2192),
            ("8_years", 2922), ("9_years", 3287), ("10_years", 3653)]
pd.DataFrame([
    {
        "horizon": label,
        "days_threshold": days,
        "confirmed_total_replacement_events_by_horizon_n": int((confirmed["replacement_days"] <= days).sum()),
        "note": "Event count only; not an incidence/risk estimate because non-event censoring times are unavailable in the current package.",
    }
    for label, days in horizons
]).to_csv(OUT / "confirmed_total_replacement_event_counts_by_horizon.csv", index=False)

requirements = pd.DataFrame([
    ["Side-specific total knee replacement event time", "V99ERKDAYS / V99ELKDAYS", "present", "replacement_days", "Available for 914 coded records; 885 records meet the project's current confirmed-event mapping."],
    ["Confirmed/adjudicated total replacement indicator", "V99ERKRPCF / V99ELKRPCF", "present", "replacement_status_code + confirmed_followup_replacement", "Current build maps status code 3 to confirmed follow-up replacement."],
    ["Partial knee replacement event/status", "separate OAI partial-replacement variables", "absent", "", "Published OAI KR analyses treat partial replacement separately; current derived table does not extract it."],
    ["Non-event censoring time", "earliest of death, withdrawal of consent, or last contact", "absent", "", "Required for defensible survival/time-to-event analysis; current derived table has no censoring date/time for non-events."],
    ["Baseline replacement exclusion", "baseline replacement indicator", "present", "replacement_at_baseline", "All 8,920 knees equal 0 after cohort exclusion."],
], columns=["requirement", "oai_source_or_definition", "availability_in_current_package", "current_field", "audit_note"])
requirements.to_csv(OUT / "replacement_time_to_event_requirements.csv", index=False)

decision = {
    "audit_date": "2026-09-13",
    "baseline_knees": int(len(df)),
    "baseline_participants": int(df["participant_key"].nunique()),
    "baseline_replacement_values_unique": [int(x) for x in sorted(df["replacement_at_baseline"].dropna().unique())],
    "replacement_status_counts": {
        "missing": int(df["replacement_status_code"].isna().sum()),
        "0": int((df["replacement_status_code"] == 0).sum()),
        "1": int((df["replacement_status_code"] == 1).sum()),
        "3": int((df["replacement_status_code"] == 3).sum()),
    },
    "replacement_days_nonmissing_n": int(df["replacement_days"].notna().sum()),
    "confirmed_total_replacement_n": int(df["confirmed_followup_replacement"].sum()),
    "confirmed_total_replacement_days_all_nonmissing": bool(confirmed["replacement_days"].notna().all()),
    "status_code_1_with_days_n": int(((df["replacement_status_code"] == 1) & df["replacement_days"].notna()).sum()),
    "partial_replacement_fields_present": False,
    "non_event_censoring_fields_present": False,
    "survival_analysis_run": False,
    "decision": "Do not run a time-to-event knee-replacement model from the current package.",
    "reason_codes": [
        "non_event_censoring_time_absent",
        "partial_replacement_fields_absent",
        "status_code_semantics_not_fully_embedded_in_package_documentation",
    ],
    "required_to_enable_survival_analysis": [
        "extract partial knee replacement variables if the intended endpoint is any knee replacement",
        "extract participant censoring information sufficient to compute earliest death, consent withdrawal, or last contact",
        "retain/verify the OAI codebook definitions for replacement confirmation/status variables",
        "then freeze the survival-model specification before examining the discordance-replacement association",
    ],
    "public_source_metadata": [
        {
            "source": "D'Assignies et al., Cartilage, 2025",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12681341/",
            "technical_point": "Uses V99ELKDAYS/V99ERKDAYS for total knee replacement and separate partial-replacement variables; censors non-events at earliest death, withdrawal, or last contact.",
        },
        {
            "source": "Saxer et al., Osteoarthritis and Cartilage Open, 2024",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10944111/",
            "technical_point": "Defines total and partial knee replacement separately and uses death/withdrawal/last-contact censoring for non-events.",
        },
        {
            "source": "NDA OAI featured dataset",
            "url": "https://nda.nih.gov/general-query.html?q=query%3Dfeatured-datasets%3AOsteoarthritis+Initiative+%28OAI%29",
            "technical_point": "Lists OAI Outcomes as containing death and knee/hip replacement information.",
        },
    ],
}
with open(OUT / "replacement_outcome_audit.json", "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)

print(json.dumps(decision, indent=2))
