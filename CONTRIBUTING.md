# Contributing

Report reproducibility problems with software versions, the command, and a sanitized error message. Do not upload records, keys, predictions, or restricted archives. Do not include patient details in issues.

The original analysis is frozen. Proposed numerical corrections require a documented reason, affected files, old and new values, and date. New analyses belong in a clearly labeled exploratory or separate replication project. Presentation changes can be made without changing frozen tables.

Before proposing changes, run `python scripts/reproduce.py --mode public` and review every added file against `docs/data_policy.md`. Any changed release file requires a deliberate manifest update and privacy review.
