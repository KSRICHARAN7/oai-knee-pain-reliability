# Reproducibility

## What this release verifies

The public runner validates the allowlist/hash manifest and regenerates figures from aggregate sources without access to participant data. It also checks that frozen headline metrics agree across distributed tables. These checks do not verify the original model fitting.

The core dependency versions in `requirements.txt` are the versions available and tested during public packaging under Python 3.12. The original analysis environment was not included in the uploaded technical package. A byte-identical recreation of historical fitting cannot be guaranteed from this environment specification.

## Authorized full run

From the repository root, install `requirements.txt` and run:

```bash
python scripts/reproduce.py --mode full --zip-dir /absolute/path/to/oai-zips
```

The runner works inside `analysis/`, keeping original relative paths valid. It verifies required ZIPs before analysis, builds the cohort, runs descriptive QC and the primary pipeline twice, and compares fold files, knee ordering, predictions, discordance, and errors. Only if repeatability and integrity checks pass does it create the reproducibility gate consumed by the original hypothesis and uncertainty scripts. Primary metrics are then compared with frozen aggregate values at the configured rounding tolerance.

The command intentionally fails on a discrepancy. It does not silently change a frozen result, fabricate a successful gate, or carry on with an unverified downstream analysis. Differences can arise from source-data releases, row ordering, or software versions; investigate and document them.

Use `python -m pip install -r requirements-secondary.txt` and add `--secondary` to run sensitivity, attrition, longitudinal, subgroup, and replacement audits. These optional dependencies and the full refitting path were not executed during public packaging. Full nested CV and bootstrap calculations can be computationally expensive.

## Provenance and figures

`docs/source_provenance.json` maps preserved files to their paths and SHA-256 hashes in the supplied private source. `PUBLIC_MANIFEST.json` hashes all release files except itself. Git commits provide the repository version identity; the release manifest is an integrity aid, not a signed scientific attestation.

The supplied `publication_assets.py` was only a docstring placeholder. It was not published as a working generator. `scripts/regenerate_figures.py` is a new, aggregate-only generator for the public release. It recreates informative figures with new styling, not pixel-identical versions of historical figures. Original numerical CSV sources are preserved.

The uploaded manuscript and supplement are maintained separately; they are not republished here, and no journal publication or DOI is claimed.
