# Mapping remediation status (locked)

## Done

| Item | Status |
|------|--------|
| Automated + CT.gov remediations | **620 / 1418** CTDR NCTs (~44%) |
| Upstream sidecars | Applied (`disease_vo_map` + `study_context_profile`) |
| Hard validation | **PASSED** (`validation_report.json`) |
| Audit sample (80) | **80/80 reviewed** (advisor + CT.gov final22 + auto-remediated verified) |
| Advisor-incorrect regressions | 20 locked to correction targets |

## Audit sample is closed

No blank `reviewer` rows remain in `mapping_audit_sample_reviewed.csv`.

Final-22 CT.gov fixes included (examples):
- Typhoid **Vi** ≠ live attenuated
- MenACWY / Menactra / PCV13 product ids
- Sabin **IPV** primary (not DTaP name extract)
- CYD/Dengvaxia (not JE DNA)
- Tdap pregnancy; PfSPZ product refine; peptide Material class

## What remains outside the sample

~800 CTDR trials not in the 620 overlay. Many are consistent licensed products.
~57 `arch=plat=vector` with clear viral-vector evidence — left on purpose.

Further gains need new high-confidence rules or full-cohort expert review, not re-opening the closed 80-row sample without cause.
