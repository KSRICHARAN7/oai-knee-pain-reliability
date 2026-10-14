# Data access and publication boundary

Obtain OAI public-use files directly from the OAI data custodian and comply with the terms accompanying your access. This repository does not supply or authorize redistribution of those files.

Required archive names are `AllClinical_ASCII.zip`, `General_ASCII.zip`, and `X-Ray Image Assessments_ASCII.zip`. The hip archive is not required for this knee analysis. Exact member filenames and variable mapping are in `analysis/build_knee_cohort.py` and `variable_dictionary.csv`.

Public files were selected using an explicit allowlist, not by copying the technical package wholesale. Shared materials are analysis source code, internal frozen specifications, variable definitions, cohort/model-level aggregate tables, and figures regenerated from those tables. Subgroup and detailed attrition/replacement outputs are conservatively excluded from this release; the audit code remains available to authorized analysts.

Never commit raw records, participant or knee keys in record tables, derived cohort rows, predictions, individual uncertainty scores, fold assignments, bootstrap draws, trained models, private ZIPs, or credentials. Project-specific keys remain record-level data even when original OAI IDs are absent. Local full runs write to ignored `analysis/outputs/` and `analysis/outputs_repeat/`.

A `.gitignore` is a guardrail, not a privacy guarantee. `scripts/verify_public.py` requires every distributable file to be named in `PUBLIC_MANIFEST.json`, validates its hash, and scans public CSV headers and content. Check any proposed additions manually and update the allowlist only after review. Never attach restricted analysis output to issues or pull requests.
