# Mapping audit policy (for CTDR / KG)

## Two separate judgments

1. **Material (`arch_vo`)** — Does the VO class correctly represent the vaccine *product / antigen / disease-use class* under study?
2. **Platform (`vo_platform`)** — Does the VO class correctly represent the *technology platform* (DNA, RNA, viral vector, protein subunit / VLP, inactivated, live attenuated, etc.)?

Different VO ids are **not** automatically an error (VO organizes disease, antigen, and preparation axes). Identical ids are **not** automatically correct.

`vo_platform=unknown` means **unlabeled**, not incorrect, until reviewed.

## Allowed labels

| Label | Meaning |
|-------|---------|
| `correct` | VO accurately represents the corresponding vaccine or platform |
| `incorrect` | Wrong vaccine, wrong valency/product family, or wrong technology platform |
| `ambiguous` | Multi-vaccine trial where the current single label covers only part of the interventions, or evidence insufficient |

(Former draft used `too_coarse`; prefer `incorrect` when the class is the wrong branch, and `ambiguous` when coverage is partial.)

## Locked rule (2026-10-30)

**Primary-investigational mapping** — `arch_vo` = primary investigational vaccine only.  
Comparators and co-administered vaccines are not written into `arch_vo`.  
`vo_platform` describes the technology of that primary vaccine. Multi-platform comparison arms → `vo_platform=unknown` (unlabeled / ambiguous), not a forced single platform.

(Former working name “Policy A” meant this same rule; there is no alternate lettered mapping policy.)

**Upstream (root) apply:** `python -m ctdr.apply_vo_corrections_upstream` writes the same overlay into:
- `ckpts/sea_ml_v14/disease_vo_map.csv` (`arch_vo`)
- `SEA-CDM …/v1.4/study_context_profile.csv` (`vaccine_type_vo_id`)

Backups: `*.pre_policy_a.bak` (historical filename). CTDR `build_table` overlay remains as an idempotent safety net. Correction targets: `vo_corrections_primary.csv` (legacy alias `vo_corrections_policy_a.csv`).

**Hard validation (must pass):** `python -m ctdr.validate_vo_mapping`  
Checks advisor-incorrect regressions, residual error patterns, upstream↔CTDR agreement, overlay idempotence.

## Registry fields vs ontology

Keep `primary_purpose` (and similar AACT fields) as **registry values**. Do not merge “looks odd for a preventive vaccine” into VO incorrectness without checking ClinicalTrials.gov.

## HPV VLP candidates

For Gardasil-9 / V503 / RG1-VLP style products, consider platform **`VO_0004914` (virus-like particle vaccine)** after checking current VO hierarchy—not `VO_0000484` (recombinant vector vaccine).
