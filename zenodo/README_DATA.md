# OSEAN-CTDR supporting data (Zenodo draft package)

Companion to code repository: https://github.com/zhujiadongok/osean-ctdr

## Contents

| Path | Description |
|------|-------------|
| `vo/vo.owl` | Vaccine Ontology release **2026-06-12** (local copy used in analyses) |
| `sidecars/disease_vo_map.csv` | NCT → Material VO mapping used by CTDR |
| `sidecars/study_context_profile.csv` | Design / phase / enrollment fields |
| `sidecars/study_intervention_profile.csv` | Route / dose |
| `sidecars/subject_demographics_summary.csv` | Eligibility sex / age flags |
| `artifacts/` | Frozen paper result JSONs (`graph_value`, `onto_graph_knn`, `seacdm_linkage`, `push08`, AE `metrics.json`) |
| `labels/vo_eligible_labels.py` | Figure 6B / Table S7 class labels |

## Not included

- Full AACT pipe dump (large; exact NLM snapshot date was not retained). Identifier used locally: folder `6wvqgoof4dut7eaxn610uzfajvnt`; `studies.txt` SHA-256 `82870fc4a17cc1eb75529c603866db82207ff7366ab49569f4779f89ffaba7f8` (390,802,838 bytes). Readers should obtain AACT/ClinicalTrials.gov data under applicable terms.
- Full SEA-CDM official core table dump (only CTDR sidecars above).

## Related

- Code: https://github.com/zhujiadongok/osean-ctdr  
- VO official release IRI: http://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl  

## License

Code follows the GitHub repository. Ontology follows VO / OBO terms. Registry-derived tables are redistributed for research reproducibility; redistributors should respect ClinicalTrials.gov / AACT terms.
