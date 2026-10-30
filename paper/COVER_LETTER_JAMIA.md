# Cover letter draft — JAMIA Research and Applications

**To:** Editors, *Journal of the American Medical Informatics Association*

**Re:** OSEAN-CTDR: Ontology-based SEA Network Clinical Trial Design Recommendation for SEA-CDM–based vaccine trial evidence integration

Dear Editors,

Please consider our Research and Applications manuscript on organizing vaccine clinical-trial registry evidence in SEA-CDM and evaluating when Vaccine Ontology (VO) relations improve design-field suggestion versus exact identifier match under matched neighbor rules.

**Why this fits JAMIA.** The work is an informatics methods contribution: a reusable NCT-centered join path (SEA-CDM v1.4 + VO), a fair exact-versus-OntSim ablation, and an unseen-Material stress test (pooled holdout and leave-one-Material-out), including an exploratory IC-weighted subclass-and-sibling retrieval sensitivity with exact fallback for seen Material ids. We deliberately do not claim clinical decision change or AE-ranking indispensability of VO edges.

**Mapping quality.** Under locked primary-investigational mapping (`arch_vo` = primary investigational vaccine only), we remediated 620/1,418 CTDR trials upstream, recorded dispositions for an 80-trial audit sample (mixed advisor / CT.gov / auto-verified), locked regression checks (`validate_vo_mapping`), and report chronological, Setting B, pooled, and LOMO results—including primary OntSim, HistGBDT, and frequency on the remediated 23-class split.

**Data/code.** Code: https://github.com/zhujiadongok/osean-ctdr. Supporting data Zenodo DOI (reserved; publish with submission): https://doi.org/10.5281/zenodo.23773934. AACT dump is not redistributed; snapshot identifiers and checksums are in Table S8.

All authors have approved the submission. We confirm the work is original and not under consideration elsewhere.

Sincerely,  
Yongqun (Oliver) He (corresponding)  
on behalf of Jiadong Zhu, Feng-Yu (Leo) Yeh, Jie Zheng, and Yongqun He  
University of Michigan Medical School  
yongqunh@med.umich.edu
