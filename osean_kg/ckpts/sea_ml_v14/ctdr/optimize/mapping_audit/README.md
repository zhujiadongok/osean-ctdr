# VO mapping audit + primary-investigational mapping remediation

## Policy

See `POLICY.md` — **primary-investigational mapping**: `arch_vo` = primary investigational vaccine only.

## Current lock

| Item | Value |
|------|------:|
| CTDR cohort | 1418 |
| Overlay / upstream NCTs | **620** (~44% of CTDR) |
| Audit sample | **80/80 reviewed** |
| Hard validation | **PASSED** (`validation_report.json`) |

### Canonical rebuild (root → CTDR)

```text
python -m ctdr.build_vo_corrections_bulk
python -m ctdr.apply_vo_corrections_upstream
python -m ctdr.validate_vo_mapping          # must exit 0
python -m ctdr.build_table
python -m ctdr.eval_after_vo_fix
```

Upstream writes `disease_vo_map.csv` + `study_context_profile.csv` (backups `*.pre_policy_a.bak`). CTDR overlay stays as safety net.

## Files

| File | Role |
|------|------|
| `mapping_audit_sample.csv` | 80-row stratified sample worksheet |
| `mapping_audit_sample_reviewed.csv` | Advisor judgments (19 reviewed) |
| `vo_corrections_primary.csv` | Correction targets (298) |
| `bulk_corrections_log.csv` | Overlay + `source` |
| `validation_report.json` | Hard checks (advisor regressions + residuals) |
| `upstream_apply_report.json` | Upstream write summary |
| `after_vo_fix_eval.json` | exact / OntSim / gated |

## Overlay composition (298)

| Source | n |
|--------|--:|
| rule_hepb | 59 |
| rule_bcg | 50 |
| rule_mrna | 41 |
| rule_false_dna | 21 |
| rule_hpv_vlp | 17 |
| advisor | 16 |
| rule_inact_covid | 15 |
| rule_hepa | 12 |
| rule_ipv | 11 |
| rule_mrna_dc / gvax / v212 | 7 each |
| rule_rota / manual_ctgov | 6 each |
| rule_tumor | 5 |
| rule_hbv_core_vlp | 4 |
| other (HepE, JE, AVA, MVA, peptide, Shingrix, …) | ≤3 each |

## Post-fix metrics (neighbor)

- Chrono: exact = gated **0.659**; OntSim+role **0.663**
- Pooled (14 classes, n=583): exact **0.653**; role **0.649**; gated **0.659**
- LOMO: gated − exact class ≈ **+0.003**

## Still not “fully done”

High-confidence name/pattern + advisor + CT.gov co-admin. Remaining risk: multi-arm trials with incomplete `specific_vaccine_name`, cancer vaccines under coarse vector labels, and unaudited sample rows.
