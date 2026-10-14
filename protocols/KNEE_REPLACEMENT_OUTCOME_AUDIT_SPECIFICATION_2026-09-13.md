# Knee-replacement secondary-outcome technical audit specification

Date: 2026-09-13
Status: Prespecified secondary-outcome feasibility audit.

Purpose
- Determine whether the currently archived OAI-derived technical dataset supports a defensible knee-replacement time-to-event analysis.
- Do not test any discordance-to-knee-replacement association unless the event and censoring definitions are complete.

Audit requirements
1. Side-specific adjudicated/confirmed knee-replacement event indicator.
2. Side-specific time from enrollment to replacement for events.
3. Complete definition of whether the endpoint includes total replacement only or total plus partial replacement.
4. A valid censoring time for every knee/participant without the event, based on OAI follow-up information (for example death, withdrawal of consent, or last contact as used in published OAI survival analyses).
5. Baseline replacement exclusion must remain intact.
6. No event-free participant may be assigned an arbitrary common censoring time unless supported by the source data and analysis definition.

Decision rule
- Run survival/time-to-event modeling only if event classification and non-event censoring are both reconstructable from archived source variables.
- If either is incomplete, archive the audit and report that the prespecified secondary outcome could not be analyzed defensibly from the current package.
- This decision does not alter the frozen primary analysis or any completed secondary analysis.

Current-package endpoint scope
- The archived table contains total-knee-replacement day/status fields derived from OAI OUTCOMES99.
- The current build does not include the separate partial-knee-replacement variables used in published OAI knee-replacement analyses.
- The archived table does not include non-event censoring dates/times.
