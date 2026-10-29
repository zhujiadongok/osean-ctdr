# OSEAN Manuscript Draft (JAMIA Research and Applications)

**Working title:** OSEAN-CTDR: SEA-CDM–based vaccine trial evidence integration and ontology-guided design-field prediction

**Authors:** Jiadong Zhu, Feng-Yu (Leo) Yeh, Jie Zheng, Yongqun (Oliver) He  
**Affiliation:** University of Michigan Medical School  
**Corresponding author:** Yongqun (Oliver) He (yongqunh@med.umich.edu)

**Target:** *JAMIA* Research and Applications — structured abstract ≤250 words; main text ≤4,000 words; ≤4 tables; ≤6 figures; ≤5 keywords.

---

## Abstract

**Objective:** To develop a SEA-CDM–based representation of vaccine trial evidence and evaluate whether Vaccine Ontology relations support design-field prediction for temporally held-out trials and ontology-connected unseen Material classes.

**Materials and Methods:** AACT vaccine trials were organized in SEA-CDM v1.4 (`SEA-CDM AACT vaccine remapped/v1.4`) with VO grounding. OSEAN-CTDR predicts allocation, masking, intervention model, and enrollment from typed features. We used a temporal split (train start &lt;2022; test ≥2022) and an exploratory ontology-connected unseen-Material evaluation (nine mid-frequency classes): pooled simultaneous holdout and leave-one-Material-out (LOMO). Exact-match and OntSim kNN used identical weights and a shared missing-ID rule (unknown≠match); OntSim expands subclass and optional role neighborhoods.

**Results:** On the temporal test (n=283), OntSim and exact-match were similar (0.656 vs 0.658). In the pooled holdout (n=413), OntSim+role achieved the highest observed trial-mean accuracy among evaluated methods (0.635) versus 0.620 for exact-match (Δ +1.47 percentage points, 95% CI 0.44–2.52) and 0.628 for HistGBDT. Under LOMO, class-equal summaries favored tabular models, while trial-equal OntSim−exact was near zero (+0.0012).

**Discussion:** Ontology expansion benefited neighbor-based transfer under simultaneous material-class holdout; gains varied across protocols and weighting schemes and did not establish superiority over HistGBDT.

**Conclusion:** SEA-CDM provides a structured foundation for integrating vaccine-trial evidence, while VO relation expansion supports cross-Material retrieval in the exploratory pooled-holdout setting.

**Keywords:** clinical trial design; SEA-CDM; Vaccine Ontology; knowledge graph; ClinicalTrials.gov

---

## 1. Background and Significance

Designing a vaccine trial requires choosing allocation, masking, intervention model, and enrollment under product, indication, and phase constraints. Historical ClinicalTrials.gov registrations provide potentially reusable design information, but registry records combine structured fields with free-text descriptions that require harmonization for ontology-guided reuse [1,2]. Prior work has also represented trial entities as knowledge graphs for retrieval and related applications [3].

The Study–Experiment–Assay Common Data Model (SEA-CDM) and Ontology-based SEA Network (OSEAN) provide a published framework for immune-study structure [7]. Building on SEA-CDM and OSEAN, we extend their application to vaccine-registry evidence integration and ontology-guided design-field prediction. This paper’s additions are a vaccine-registry mapping into SEA-CDM v1.4, ClinicalTrials.gov design recommendation (OSEAN-CTDR), and evaluations of temporal prediction versus ontology-connected unseen Material transfer.

---

## 2. Objective

To organize vaccine-trial registry evidence in SEA-CDM and evaluate whether VO relation expansion improves neighbor retrieval and design-field prediction for temporally held-out trials and for ontology-connected Materials absent from training.

---

## 3. Materials and Methods

The pipeline is shown in **Figure 1**.

### 3.1 SEA-CDM resource (paper package)

All CTDR/AE analyses use **`SEA-CDM AACT vaccine remapped/v1.4`** (9,147 studies)—not the separate `SEA-CDM VaxCT VO enriched` export (9,149 studies). Core tables follow official v1.4 column layouts (`V14_COMPLIANCE.json`). Project sidecars used for features are listed in Table S0 and are not official core tables.

**Source versions.** AACT: pipe-delimited ClinicalTrials.gov extract used for remapping; the exact NLM/AACT snapshot date was not retained. Vaccine Ontology release **2026-06-12** [4,5]. File identifiers and checksums for reproduction are in Table S8.

Product/platform strings are linked via VO and VIOLIN [6] (**Figure 2**).

**Structural checks (S5).** We report primary-key uniqueness, row-level foreign-key resolve rates, and same-study path consistency for the Study→Experiment→Intervention/Subject/Results spine. Field completeness, FK validity, and semantic mapping correctness are distinct: `interventions.material_id` stores ontology CURIEs (here universally the placeholder `VO:0000001`), which are not `material.material_id` row keys; `subject.group_id=GROUP_1` follows singleton/population guidance in the local hardening audit and need not resolve to design-arm rows in `groups.csv`. Fill rate ≠ mapping accuracy.

**Feature sources (not core-table placeholders).** Prediction Material (`arch_vo`) comes from `disease_vo_map.csv` joined by NCT to `study_context_profile.csv`, with platform roll-up from `vaccine_type_vo_id` when needed. Demographics and route/dose come from `subject_demographics_summary.csv` and `study_intervention_profile.csv` (NCT join). RNA/mRNA platforms are reported as `VO_0000186`. Of 9,147 profile studies, 2,034 have a `disease_vo_map` row (platform 1,181 / product 479 / both 374); 7,113 lack a map row and are not CTDR Material-eligible.

### 3.2 OSEAN graphs (two layers)

**SEA-CDM instance resource graph** links study entities for organization and apps. **Figure 3** summarizes the entity types and relations in the OSEAN resource graph (SEA-CDM spine, VO grounding, and AE analytics overlays). **VO class relation graph** (explicit `subClassOf` / `has_role` in `vo.owl`, not full OWL reasoning) is used only to compute OntSim for neighbor ranking. Design targets are excluded from CTDR feature/retrieval graphs; they may exist elsewhere in the resource as registered fields.

### 3.3 OSEAN-CTDR cohort and labels

**Funnel (profile → CTDR).** Profile studies 9,147 → interventional 7,924 → non-missing phase 6,956 → labeled allocation 5,582 → Material VO (`arch_vo` starts with `VO_`) 1,428 → non-missing `start_date` **1,418**. Among 1,418: mapping how = platform 801 / product 320 / both 297; `vo_platform=unknown` in 242 trials.

**Splits.** Temporal: train start &lt;2022 (n=1,135); test ≥2022 (n=283). Inner lock: fit &lt;2019 (n=891), validation 2019–2021 (n=244). This is **retrospective prediction of registered design characteristics** from a fixed registry snapshot: `start_date` splits do not replay historical visibility of enrollment or other fields. Enrollment buckets use the profile `enrollment` number (AACT registry field; may be anticipated or actual depending on study status) with cutpoints 50/150/500.

**Targets (Setting A).** Allocation; masking (5-way); intervention_model; enrollment_bucket. Unknown gold counts in the 1,418 cohort: masking 16, intervention_model 29, enrollment_bucket 15 (excluded from per-field evaluation; kNN skips unknown neighbor votes; classifiers drop unknown labels when training).

On the temporal test, every test Material identifier also appears in training (283/283)—so temporal evaluation does not stress unseen Materials.

### 3.4 Neighbor scoring (exact vs OntSim)

Weights (matched across modes; fixed for the fair ablation, not tuned on the pooled holdout):

$$
s(q,t) = 1\cdot I(\mathrm{disease}) + 1\cdot I(\mathrm{phase}) + 0.5\cdot I(\mathrm{purpose}) + 1\cdot J(\mathrm{platform}) + 0.5\cdot J(\mathrm{Material}).
$$

Missing values (`unknown`/empty) never contribute a positive match on either side. For platform, OntSim applies only when both sides are usable `VO_` ids after canonicalization; otherwise both modes use the same categorical exact rule. For Material, exact uses canonical VO equality; OntSim uses Jaccard over relation-expanded neighborhoods: subclass = self + traversable parents; role mode also adds inherited `has_role` nodes and those roles’ parents. Top-$k$ training neighbors by score with **$k=20$** (from the exploratory grid $k\in\{5,10,15,20\}$); $k=20$ was fixed after informal chronological exploration and then used for all reported exact/OntSim contrasts—it was not validation-locked and was not re-tuned on the pooled or LOMO holdouts (ties broken by NCT id). Each target is a weighted vote with weight $\max(s,0)+10^{-6}$, skipping missing neighbor labels; if no valid votes → `unknown`. Paper scripts use the four Setting A targets (not config defaults that also list `primary_outcome_family`).

### 3.5 Unseen-Material protocols (exploratory)

**Unseen** = the final Material identifier does not appear among training trials; the class and public VO relations remain available in `vo.owl`. **Ontology-connected** means

$$
\mathrm{subclass\_closure}(a) \cap \mathrm{subclass\_closure}(b) \neq \emptyset
$$

for some other cohort Material (closure includes self). Eligible: count in [12, 80] (enough test mass, not cohort-dominant); all nine such classes (Table S7). Rule fixed for this analysis run (not a pre-registered protocol). (i) Pooled: all nine test-only. (ii) LOMO: each held out once. Both report trial-equal and class-equal summaries. Bootstrap: 1,500 trial resamples (not class-level inference).

### 3.6 Other models

HistGBDT/logistic: SEA-CDM/VO columns including route, dose, sex, age flags, sponsor_class (`regress_predict`; unknown train labels dropped). Frequency-by-phase baseline. BM25/TF-IDF k=20 are **exploratory / test-grid selected**. Setting B and AE details: supplement.

---

## 4. Results

### 4.1 Resource and structural validation

On the paper v1.4 package: no duplicate primary keys for core tables examined; experiment→study, subject→experiment, interventions→experiment/subject, occurrence→organism, and results→experiment resolve at **100%** of rows; all 20,001 interventions with both experiment and subject resolving land on the **same study**. `interventions.material_id` does not resolve to `material` row keys (placeholder CURIEs). Occurrence rows: 617,510 (601,707 assigned project-defined event-category codes for AE tooling—not observed CTCAE grades; AACT lacks CTCAE). Feature Material for CTDR is supplied by the mapping sidecars above, not by those placeholders.

### 4.2 Temporal evaluation (Setting A)

**Table 1.** Field-mean accuracy on 2022+ trials (n=283). Exact and OntSim use identical weights and missing-ID rules.

| Method | Field-mean acc | Macro-F1 |
|--------|---------------:|---------:|
| Frequency by phase | 0.633 | 0.257 |
| Exact-match kNN@20 | 0.658 | 0.321 |
| OntSim kNN@20 (subclass / +role) | 0.656 / 0.656 | 0.326 |
| Logistic regression | 0.661 | 0.338 |
| HistGBDT | 0.658 | 0.364 |
| BM25 / TF-IDF k=20† | 0.648 / 0.652 | 0.304 / 0.336 |

†Exploratory; k chosen on a test-period grid (not validation-locked).

OntSim − exact trial-mean Δ −0.0018 (95% CI −0.0177 to +0.0133).

**Figure 4** shows the 2019–2021 validation comparison.

**Figure 5** shows per-target HistGBDT versus phase-frequency on the temporal test.

### 4.3 Ontology-connected unseen Materials (exploratory)

**Table 2.** Pooled holdout of all nine eligible classes (train 1,005 / test 413). Class-mean = unweighted mean of per-class trial-means.

| Method | Trial-mean | Class-mean |
|--------|----------:|----------:|
| OntSim kNN@20 (subclass+role) | **0.6350** | 0.6283 |
| Frequency by phase | 0.6299 | 0.6255 |
| HistGBDT (no VO) | 0.6289 | 0.6192 |
| OntSim kNN@20 (subclass) | 0.6281 | 0.6233 |
| HistGBDT (full) | 0.6277 | 0.6211 |
| HistGBDT (no Material) | 0.6259 | 0.6200 |
| Exact-match kNN@20 | 0.6203 | 0.6138 |

In this exploratory pooled holdout, OntSim+role achieved the highest observed trial-mean accuracy among evaluated methods (0.6350), versus 0.6203 for exact-match (Δ **+0.0147**, 95% CI **+0.0044 to +0.0252**) and 0.6277 for HistGBDT (Δ +0.0073, 95% CI −0.0103 to +0.0242). Exact-match still uses VO identifiers; the ablation removes relation expansion only.

**Table 3.** LOMO summaries for the same nine classes (other eligible Materials remain in train). Class-equal = unweighted mean of per-class trial-means; trial-equal = size-weighted mean across held-out trials.

| Method | Class-equal | Trial-equal |
|--------|------------:|------------:|
| HistGBDT (no VO) | 0.6378 | 0.6435 |
| HistGBDT (full) | 0.6305 | 0.6386 |
| Frequency by phase | 0.6310 | 0.6332 |
| HistGBDT (no Material) | 0.6262 | 0.6308 |
| Exact-match kNN@20 | 0.6112 | 0.6174 |
| OntSim subclass / +role | 0.6061 / 0.6041 | 0.6186 / 0.6186 |
| Δ(role − exact) | −0.0071 | +0.0012 |

Averaging equally across classes, OntSim+role is slightly below exact-match; averaging equally across trials, it is slightly above. That pattern indicates heterogeneous effects by class size, so LOMO should not be summarized as uniformly worse or better for the graph. Per-class deltas: S4; full trial-equal method table: S4b.

**Figure 6** shows pooled method ranks and LOMO per-class OntSim−exact deltas.

### 4.4 Matched ablation and secondary analyses

Temporal matched ablation after the shared missing-ID rule: exact 0.658; OntSim 0.656 (CI for OntSim−exact includes 0). Setting B / AE (standard recall@10; Model B 0.383; no-VO ≈0.384): supplement—not used to claim graph indispensability.

---

## 5. Discussion

SEA-CDM organizes registry-derived trial evidence into typed entities and traceable links, with project-specific mapping tables supplying prediction features. Within this representation, ontology-expanded neighborhoods enable evidence retrieval across related Material identifiers.

**Why protocols disagree.** The temporal split asks whether neighbors help when Materials are already seen in earlier years—here exact and OntSim are essentially tied, and tabular models remain competitive. The pooled holdout removes nine mid-frequency Materials at once and asks whether VO subclass/role neighborhoods recover design signal from related classes still in train; that is where OntSim+role leads exact-match after the shared missing-ID rule. LOMO holds out one class at a time with other eligible Materials still training: class-equal and trial-equal summaries can disagree when larger classes move differently from smaller ones (Table 3). None of these protocols alone answers “is the graph always better?”; together they bound when relation expansion helps neighbor transfer.

In the exploratory pooled holdout, the observed benefit was relative to exact-match neighbor retrieval after aligning missing-ID rules. Results do not prove that CDM adoption itself raises accuracy, do not establish superiority over HistGBDT, and do not show uniform LOMO gains.

Limits: retrospective snapshot; exploratory eligibility; trial-level bootstrap ≠ class-level inference; placeholder Material CURIEs in core interventions; no full OWL reasoner; no expert mapping panel; Setting B / R-GCN artifacts predate the shared missing-ID alignment used for Setting A and `graph_value`.

---

## 6. Conclusion

OSEAN-CTDR combines SEA-CDM organization of vaccine registry evidence with VO-guided neighbor retrieval. Ontology relation expansion improved exact-match neighbors under simultaneous multi-class Material holdout in this exploratory evaluation. Temporal and LOMO protocols show different rankings. Expert mapping audit, public code/data archives, and independent evaluation remain next steps.

---

## Supplementary Appendix

### S0. Feature source table (CTDR)

| Feature | Source file | Join | Notes |
|---------|-------------|------|-------|
| disease, phase, purpose, enrollment, start_date, design labels | `study_context_profile.csv` | NCT | Registry snapshot fields |
| arch_vo (Material) | `disease_vo_map.csv` | NCT | product/platform/both |
| vo_platform | profile platform VO + roll-up map | — | unknown if non-VO |
| sex, older, child, adult | `subject_demographics_summary.csv` | NCT | eligibility cohort flags |
| route, dose | `study_intervention_profile.csv` | NCT | |
| sponsor_class | profile | — | |

Entrypoints and full paths: Table S8.

### S1. Setting B (coarse labels) and R-GCN

OPEN/BLINDED; PARALLEL/OTHER; SMALL&lt;150 / LARGE≥150; allocation unchanged. Unknown gold excluded. Reported numbers are the frozen exploratory artifact `ckpts/sea_ml_v14/ctdr/optimize/push08/` produced by `ctdr/push08_lock.py` (val-locked logistic best among classical models): HistGBDT test mean 0.821; logistic 0.815; frequency 0.802; ontology-kNN@20 0.800; R-GCN enriched 0.787. That run predates the shared missing-ID alignment later applied in `recommend.py` for Setting A / `graph_value`; Setting B was not re-scored after the fix and is not a primary claim.

**R-GCN (brief).** Supervised relational GCN on the enriched CTDR input graph (SEA-CDM typed nodes + VO subclass/`has_role` + same-context links; design targets excluded as edges). Architecture: 3 layers, embedding dim 128, Adam lr $10^{-2}$, weight decay $10^{-4}$, 450 epochs (`RGCNClf` in `ctdr/push08_lock.py`). Hybrid onto+R-GCN blends were evaluated but did not beat HistGBDT on the locked Setting B test mean.

### S2. Resource counts (v1.4 remapped)

Studies 9,147; experiments 18,294; subjects 9,147; interventions 20,001; occurrence 617,510; results 1,525,757. CTDR n=1,418.

### S3. Statistics and disclosure

Bootstrap: OntSim contrasts 1,500 (graph_value) or 2,000 (onto_graph_knn) trial resamples; percentile 95% CI; seed 7. Macro-F1: sklearn `f1_score(..., average="macro", zero_division=0)` after excluding rows with unknown gold; label set is sklearn’s default (union of remaining true and predicted labels), not gold-only. Prior 2022+ exploration disclosed; not an untouched confirmatory cohort.

### S4. LOMO per-class (after missing-ID alignment)

Exact / OntSim+role trial-mean and Δ:  
0561 0.672/0.662 (−0.009); 4673 0.652/0.663 (+0.011); 1178 0.575/0.599 (+0.024); 0642 0.624/0.624 (0); 5198 0.545/0.585 (+0.040); 0761 0.586/0.564 (−0.021); 0047 0.607/0.589 (−0.018); 7062 0.676/0.603 (−0.074); 7063 0.562/0.547 (−0.016).

Deltas are computed from unrounded floating-point trial-means; subtracting the three-decimal displays can differ from the reported Δ by about 0.001.

### S4b. LOMO trial-equal (all methods)

Frequency 0.6332; HistGBDT full 0.6386; HistGBDT no Material 0.6308; HistGBDT no VO 0.6435; exact 0.6174; OntSim subclass 0.6186; OntSim+role 0.6186.

### S5. SEA-CDM structural audit summary

See `optimize/seacdm_linkage/` and `v1.4/OFFICIAL_V14_*`. Spine FKs 100% row-resolved as in §4.1; material CURIE placeholders and GROUP_1 documented; results.group_id: 601 empty, 0 non-empty unresolvable among nonempty.

### S6. AE ranking

Unit: NCT × arm-role context (vaccine/control), not necessarily a design arm row. NCT 80/20 split by study (train 1,624 / test 407 studies); train 2,065 / test 506 contexts; **n_eval = 483** contexts with ≥1 positive label after train-only vocab (≥25 contexts, top 80 labels). Age impute from train median. Model A: Material VO; B: +platform + clinical C; no-VO context ablation ≈0.384 vs B 0.383 (`ckpts/sea_ml_v14/metrics.json`). Project event-category codes on occurrence are not used as CTCAE grades in the ranking claim.

### S7. Nine ontology-connected eligible Materials

| VO id | Label (vo.owl) | n trials |
|-------|----------------|---------:|
| VO_0000561 | toxoid vaccine | 79 |
| VO_0004673 | quadrivalent influenza vaccine | 69 |
| VO_0001178 | live attenuated influenza vaccine | 63 |
| VO_0000642 | influenza virus vaccine | 62 |
| VO_0005198 | COVID-19 RNA vaccine | 44 |
| VO_0000761 | malaria vaccine | 35 |
| VO_0000047 | Fluzone | 28 |
| VO_0007062 | Quadrivalent Human Papillomavirus (types 6, 11, 16, 18) Recombinant Vaccine | 17 |
| VO_0007063 | Recombinant Human Papillomavirus Nonavalent Vaccine | 16 |

Eligibility: count in [12, 80] and ontology-connected within the CTDR cohort; all nine reported.

### S8. Reproducibility file inventory

Public code package: https://github.com/zhujiadongok/osean-ctdr. Deposit should include, at minimum: SEA-CDM v1.4 vaccine remapped tables used here; `disease_vo_map.csv` and platform roll-up (`PROFILE_TO_OFFICIAL`); VO release 2026-06-12 OWL; CTDR cohort/split NCT lists; analysis configs and scripts under `osean_kg/ctdr/`; frozen result JSONs (`graph_value.json`, `onto_graph_knn.json`, `seacdm_linkage/`, `push08/`, AE `metrics.json`). Checksums identify files but do not replace a downloadable archive.

Local identifiers (not a substitute for public deposit):

- AACT dump: `AACT Database/` + folder id `6wvqgoof4dut7eaxn610uzfajvnt`
- `studies.txt` size 390,802,838 bytes; SHA-256  
  `82870fc4a17cc1eb` `75529c603866db82` `207ff7366ab49569` `f4779f89ffaba7f8`
- VO OWL: under `SEA-CDM AACT vaccine remapped/research_factors/vo_mapping/FOR_ADVISOR/` (`vo.owl`)
- Chart labels: `osean_neo4j/vo_eligible_labels.py`
- Primary contrasts (shared missing-ID): `ckpts/sea_ml_v14/ctdr/optimize/graph_value/graph_value.json`
- Entrypoints: `python -u -m ctdr.eval_onto_graph_knn`; `eval_graph_value`; `eval_seacdm_linkage`

---

## Funding / Data & Code Availability

**Funding.** *[Corresponding author to insert funding statement before submission.]*  
**Author contributions.** J. Zhu: analysis design, implementation, evaluation, and drafting. F.-Y. Yeh: VO/VaxCT mapping resources. J. Zheng: *[role to confirm]*. Y. He: supervision and critical revision. All authors approved the submitted version.  
**Conflict of interest.** None declared.  
**Data and code.** Analysis code, CTDR feature sidecars, and frozen result artifacts: https://github.com/zhujiadongok/osean-ctdr (see Table S8). A Zenodo DOI for any remaining large assets can be added at submission.  
**AI assistance.** Language and code editing assistance was used; authors verified all scientific content and results.

---

## Figure legends

**Figure 1.** Pipeline: AACT → SEA-CDM v1.4 (vaccine remapped) → VO grounding → CTDR / AE analytics. Instance graphs organize entities; OntSim uses the VO class graph.

**Figure 2.** VO grounding of platform/product strings; RNA/mRNA platform identifiers are normalized to `VO_0000186` before analysis.

**Figure 3.** Entity types and relations in the OSEAN resource graph. SEA-CDM entities organize trials; VO grounding and AE analytics are overlays. CTDR uses typed inputs plus VO class relations for OntSim; design targets are not retrieval edges.

**Figure 4.** Validation field-mean accuracy (2019–2021, n=244); zero-baseline bars.

**Figure 5.** Per-target test accuracy: HistGBDT vs phase-frequency (n=283).

**Figure 6.** Unseen Materials: (A) pooled trial-mean ranks (zero baseline); (B) LOMO per-class Δ(OntSim+role − exact).

---

## References

1. Zarin DA, Tse T, Williams RJ, Califf RM, Ide NC. The ClinicalTrials.gov results database—update and key issues. *N Engl J Med*. 2011;364:852–860.  
2. Tasneem A, Aberle L, Ananthakrishnan S, et al. The database for Aggregate Analysis of ClinicalTrials.gov (AACT) and subsequent regrouping by clinical specialty. *PLoS One*. 2012;7:e33677.  
3. Chen Z, Peng B, Ioannidis VN, et al. A knowledge graph of clinical trials (CTKG). *Sci Rep*. 2022;12:4724.  
4. He Y, Cowell L, Diehl AD, et al. VO: Vaccine Ontology. *Nature Precedings*. 2009.  
5. Lin Y, He Y. The ontology of vaccine adverse events (OVAE) and its usage in representing and analyzing adverse events associated with US-licensed vaccine products. *J Biomed Semantics*. 2012;3:17.  
6. Xiang Z, Todd T, Ku KP, et al. VIOLIN: vaccine investigation and online information network. *Nucleic Acids Res*. 2008;36:D923–D928.  
7. Huffman A, et al. The Study–Experiment–Assay Common Data Model (SEA-CDM) and Ontology-based SEA Network (OSEAN). *Sci Data*. 2026;13:238. doi:10.1038/s41597-026-06558-z  
8. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *JMLR*. 2011;12:2825–2830.

---

## Notes for coauthors

- Confirm J. Zheng CRediT role; insert funding statement and public archive URL before submission.  
- Figure/table labels: `vo_eligible_labels.py` ↔ Table S7.
