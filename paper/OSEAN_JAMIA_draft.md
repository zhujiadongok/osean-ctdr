# OSEAN Manuscript Draft (JAMIA Research and Applications)

**Working title:** OSEAN-CTDR: Ontology-based SEA Network Clinical Trial Design Recommendation for SEA-CDM–based vaccine trial evidence integration

**Authors:** Jiadong Zhu, Feng-Yu (Leo) Yeh, Jie Zheng, Yongqun (Oliver) He  
**Affiliation:** University of Michigan Medical School  
**Corresponding author:** Yongqun (Oliver) He (yongqunh@med.umich.edu)

**Target:** *JAMIA* Research and Applications — structured abstract ≤250 words; main text ≤4,000 words; ≤4 tables; ≤6 figures; ≤5 keywords.

---

## Abstract

**Objective:** Organize vaccine-trial registry evidence in SEA-CDM and evaluate when Vaccine Ontology (VO) relations improve design-field suggestion versus exact identifier match under matched neighbor rules.

**Materials and Methods:** AACT vaccine trials were mapped to SEA-CDM v1.4 with VO grounding under primary-investigational mapping. OSEAN-CTDR predicts allocation, masking, intervention model, and enrollment. We used a temporal split (train &lt;2022; test ≥2022) and an exploratory unseen-Material evaluation (ontology-connected classes): pooled holdout and leave-one-Material-out (LOMO). Exact-match and OntSim kNN shared weights and a missing-ID rule. Primary OntSim expands subclass neighborhoods (optional role). An exploratory IC-weighted subclass-and-sibling retrieval variant uses that similarity for Material ids absent from training and falls back to exact match when ids are seen.

**Results:** After primary-investigational mapping remediation (620 NCT overlays), temporal Setting A (n=283) showed HistGBDT = OntSim+role 0.664 vs exact = IC-sib 0.661. On Setting B, HistGBDT reached 0.821 (OntSim+role numerically above exact; Δ +1.24 pp, 95% CI 0.000 to +0.0256). In pooled unseen holdout (23 classes, n=888), exact≈OntSim≈IC-sib (~0.645). LOMO class-equal: exact 0.633; OntSim+role 0.646; IC-sib 0.654; HistGBDT 0.651; frequency 0.645.

**Discussion:** Ontology-related gains depended on the retrieval formulation and evaluation protocol. The exploratory IC-weighted subclass-and-sibling variant improved mean LOMO accuracy over exact matching, whereas pooled holdout showed no clear benefit. Gains do not beat HistGBDT overall. Output is decision support only.

**Conclusion:** SEA-CDM supports reusable vaccine-trial evidence integration; VO relation expansion improved observed LOMO accuracy over exact matching, with benefits depending on the retrieval formulation and evaluation protocol.

**Keywords:** clinical trial design; SEA-CDM; Vaccine Ontology; knowledge graph; ClinicalTrials.gov

---

## 1. Background and Significance

Designing a vaccine trial requires choosing allocation, masking, intervention model, and enrollment under product, indication, phase, and operational constraints. These choices are often informed by precedent: teams look for historically similar registrations and adapt what prior sponsors recorded. ClinicalTrials.gov and its Aggregate Analysis of ClinicalTrials.gov (AACT) extract make that precedent searchable at scale [1,2], but raw registry records mix structured fields with free-text product and condition strings. Without shared identifiers, “similar vaccine” retrieval collapses to string match or opaque embedding similarity, and it is hard to tell whether a recommendation reused ontology structure or merely phase-frequency defaults.

Knowledge-graph representations of trials support retrieval and related analytics [3], yet vaccine-specific design suggestion also needs a typed study model that separates interventions, eligibility cohorts, and registered design answers, plus a vaccine vocabulary that links products and platforms. The Vaccine Ontology (VO) and VIOLIN resources provide community identifiers and relations for vaccines and related entities [4–6]. Separately, the Study–Experiment–Assay Common Data Model (SEA-CDM) and Ontology-based SEA Network (OSEAN) provide a published framework for organizing immune-study evidence as typed entities and links [7]. SEA-CDM/OSEAN have not previously been evaluated as the substrate for ClinicalTrials.gov–based vaccine design-field prediction with an explicit exact-match versus ontology-similarity neighbor ablation.

This paper’s contributions are intentionally scoped for biomedical informatics reuse rather than end-to-end clinical decision claims: (i) a vaccine-registry mapping into SEA-CDM v1.4 with VO grounding that yields one NCT-centered join path for design suggestion and inspectable evidence lists; (ii) OSEAN-CTDR, which predicts registered design fields from typed features with optional ontology-expanded neighbors; and (iii) evaluations that separate temporal prediction (Materials already seen in earlier years) from ontology-connected unseen-Material transfer, under matched neighbor weights and a shared missing-identifier rule, including an exploratory IC-weighted subclass-and-sibling retrieval sensitivity with exact fallback for seen Material ids. A secondary adverse-event ranking analysis and a prototype interface are reported for completeness; they are not used to claim that VO edges are indispensable for AE ranking or that the prototype changes sponsor decisions.

---

## 2. Objective

To organize vaccine-trial registry evidence in SEA-CDM and to evaluate whether—and under which evaluation protocols—VO relation expansion improves neighbor retrieval and design-field prediction relative to exact identifier match.

---

## 3. Materials and Methods

The pipeline is shown in **Figure 1**.

### 3.1 SEA-CDM resource (paper package)

All CTDR/AE analyses use `**SEA-CDM AACT vaccine remapped/v1.4**` (9,147 studies)—not the separate `SEA-CDM VaxCT VO enriched` export (9,149 studies). Core tables follow official v1.4 column layouts (`V14_COMPLIANCE.json`). Project sidecars used for features are listed in Table S0 and are not official core tables.

**Source versions.** AACT: pipe-delimited ClinicalTrials.gov extract used for remapping; snapshot date **2026-04-12** (Table S8; `studies.txt` row timestamps and download-folder mtime). Vaccine Ontology release **2026-06-12** [4,5].

Product/platform strings are linked via VO and VIOLIN [6] (**Figure 2**).

**What the structure enables.** From a study NCT, SEA-CDM links the registered experiment to interventions (what was given), subject/eligibility summary (who was enrolled), and occurrence/results rows (what was reported), while VO sidecars attach vaccine Material/platform classes. The same typed join path supports design recommendation, AE context ranking, and NCT-preserving evidence lists without rebuilding a separate extract for each task. Structural audit details are in S5; fill rate is not mapping accuracy. **Primary-investigational mapping (locked).** `arch_vo` denotes the **primary investigational vaccine** only; comparators/coadministered vaccines are excluded. `vo_platform` is that vaccine’s technology class. Advisor audit plus high-confidence name/pattern rules and CT.gov checks (S10) remediates **620** NCTs upstream (`disease_vo_map` / profile) with CTDR overlay safety net. The 80-row audit sample has disposition records for all rows (not all expert-reviewed; S10). `validate_vo_mapping` must pass. Remaining CTDR rows outside the overlay are not claimed perfect.

**Feature sources (not core-table placeholders).** Prediction Material (`arch_vo`) comes from `disease_vo_map.csv` joined by NCT to `study_context_profile.csv`, with platform roll-up from `vaccine_type_vo_id` when needed. Demographics and route/dose come from `subject_demographics_summary.csv` and `study_intervention_profile.csv` (NCT join). RNA/mRNA platforms are reported as `VO_0000186`. Of 9,147 profile studies, 2,034 have a `disease_vo_map` row (how = platform 919 / product 260 / both 855); 7,113 lack a map row and are not CTDR Material-eligible.

### 3.2 OSEAN graphs (two layers)

**SEA-CDM instance resource graph** links study entities for organization and apps (**Figure 3**). **VO class relation graph** (explicit `subClassOf` / `has_role` in `vo.owl`, not full OWL reasoning) is used only to compute OntSim for neighbor ranking. Design targets are excluded from CTDR feature/retrieval graphs.

### 3.3 OSEAN-CTDR cohort and labels

**Funnel (profile → CTDR).** Profile studies 9,147 → interventional 7,924 → non-missing phase 6,956 → labeled allocation 5,582 → Material VO (`arch_vo` starts with `VO_`) 1,428 → non-missing `start_date` **1,418**. Among the **post-remediation** 1,418 CTDR rows: mapping how = platform 539 / product 101 / both 778; `vo_platform=unknown` in **24** trials. The same cohort before primary-investigational overlay (archived `optimize/ctdr_studies.csv` snapshot) had how = platform 801 / product 320 / both 297 and `vo_platform=unknown` in **241** trials. The large funnel drop reflects the requirement for an actionable VO Material identifier, not a claim that unmapped trials lack clinical value.

**Splits.** Temporal: train start <2022 (n=1,135); test ≥2022 (n=283). Inner lock: fit <2019 (n=891), validation 2019–2021 (n=244). This is **retrospective prediction of registered design characteristics** from a fixed registry snapshot: `start_date` splits do not replay historical visibility of enrollment or other fields as they would have appeared on the registration date. Enrollment buckets use the profile `enrollment` number (AACT registry field; may be anticipated or actual depending on study status) with cutpoints 50/150/500.

**Targets (Setting A).** Allocation; masking (5-way); intervention_model; enrollment_bucket. Field-mean accuracy averages the four per-field accuracies after excluding unknown gold on each field; macro-F1 is the corresponding mean of per-field macro-F1 values [8]. Trial-mean accuracy first averages correctness across evaluable targets within each trial and then across trials. For LOMO, class-equal accuracy averages the within-class trial means equally across held-out Materials, whereas trial-equal accuracy weights classes by their numbers of evaluated trials. Unknown gold counts in the 1,418 cohort: masking 16, intervention_model 29, enrollment_bucket 15.

On the temporal test, every test Material identifier also appears in training (283/283)—so temporal evaluation measures whether ontology expansion helps when identifiers are already known from earlier years, not whether it can invent neighbors for a brand-new Material id.

### 3.4 Neighbor scoring (exact vs OntSim)

**Primary fair ablation (main tables).** Weights matched across exact and OntSim; fixed for the fair contrast (not tuned on pooled/LOMO):

$$
s(q,t) = 1\cdot I(\mathrm{disease}) + 1\cdot I(\mathrm{phase}) + 0.5\cdot I(\mathrm{purpose}) + 1\cdot J(\mathrm{platform}) + 0.5\cdot J(\mathrm{Material}).
$$

Missing values (`unknown`/empty) never contribute a positive match on either side. For platform, OntSim applies only when both sides are usable `VO_` ids after canonicalization; otherwise both modes use the same categorical exact rule. For Material, exact uses canonical VO equality; primary OntSim uses Jaccard over relation-expanded neighborhoods: subclass = self + traversable parents; role mode also adds inherited `has_role` nodes and those roles’ parents. Top-$k$ training neighbors with **$k=20$** (exploratory grid $k\in5,10,15,20$; fixed after informal chronological exploration; not re-tuned on pooled/LOMO; ties by NCT id). Each target is a weighted vote with weight $\max(s,0)+10^{-6}$.

**Exploratory IC-weighted subclass-and-sibling variant.** Separately from the primary ablation, we evaluate an exploratory retrieval mode that replaces Material/platform Jaccard with an IC-weighted Jaccard over subclass neighborhoods expanded by sibling classes (same direct parent; no role union). Node information content is $\mathrm{IC}(c)=-\log(|\{d:\ c\in\mathrm{subclassClosure}(d)\}|/|C|)$ over non-deprecated VO classes $C$. A **gate** (exact fallback) applies this similarity only when the query Material id is absent from training; when the id is seen, scoring uses exact VO equality. Field weights and $k=20$ match the primary ablation. Because LOMO holds out every query Material by construction, the gate is open for all LOMO queries—so LOMO gains for this mode reflect the IC/sibling similarity package, not an isolated test of gating. Val-tuned weights: S11 (not a fair ablation).

**Mechanism example (post-remediation LOMO).** Hold out autologous tumor cell vaccine trials (`VO_0007319`) and query `NCT00085397` (phase 2). Under exact-match, the Material term is 0, so top neighbors are disease/phase/platform-tied mixes of GVAX and autologous dendritic-cell cancer vaccines (score 3.5). Under primary OntSim+role, shared ancestry concentrates GVAX pancreatic neighbors at the top (score ≈3.72) and changes the enrollment-bucket vote from `50_149` to `lt_50`. The ontology-based prediction matched the recorded enrollment bucket (&lt;50), whereas exact matching predicted 50–149 (artifact: `mechanism_example_NCT_neighbors.json`; class LOMO in Table 4 / S4).

### 3.5 Unseen-Material protocols (exploratory)

**Unseen** = the final Material identifier does not appear among training trials; public VO relations remain available. **Ontology-connected** means subclass closures intersect for some other cohort Material. Eligible: count in [12, 80]; **all 23 such classes** after primary-investigational mapping remediation (Table S7). Rule fixed for this analysis run (not pre-registered). (i) Pooled: all 23 as test-only (train 530 / test 888). (ii) LOMO: each held out once. Both report trial-equal and class-equal summaries. Bootstrap: 1,500 trial resamples (not class-level inference).

### 3.6 Other models and Setting B

**Tabular models.** HistGBDT and logistic regression use SEA-CDM/VO categorical columns including route, dose, sex, age flags, and sponsor_class [8]. They do not retrieve neighbors.

**Frequency by phase.** Majority label among same-phase training trials (global majority fallback).

**Text kNN (exploratory).** BM25/TF-IDF k=20; $k$ chosen on a test-period grid—not validation-locked.

**Setting B (post-hoc coarse Y).** Same models/split; labels coarsen (masking → OPEN/BLINDED; intervention → PARALLEL/OTHER; enrollment → SMALL&lt;150 / LARGE≥150). AE / R-GCN: supplement only.

**Prototype.** Streamlit Design view returns predicted fields and OntSim-ranked supporting NCTs (**Figure S9**); decision support only—not a clinical usability trial.

---

## 4. Results

### 4.1 Resource and structural validation

On the paper v1.4 package: no duplicate primary keys for core tables examined; experiment→study, subject→experiment, interventions→experiment/subject, occurrence→organism, and results→experiment resolve at **100%** of rows; all 20,001 interventions with both experiment and subject resolving land on the **same study**. `interventions.material_id` does not resolve to `material` row keys (placeholder CURIEs). Occurrence rows: 617,510. Feature Material for CTDR is supplied by mapping sidecars, not placeholders.

### 4.2 Temporal evaluation (Setting A)

**Table 1.** Field-mean accuracy on 2022+ trials (n=283). Exact and primary OntSim use identical weights and missing-ID rules.

| Method | Field-mean acc | Macro-F1 |
|--------|---------------:|---------:|
| HistGBDT | 0.664 | 0.361 |
| OntSim kNN@20 (+role) | 0.664 | 0.332 |
| Logistic regression | 0.663 | 0.355 |
| Exact-match kNN@20 | 0.661 | 0.324 |
| OntSim kNN@20 (subclass) | 0.661 | 0.325 |
| IC-sib kNN@20 (exploratory) | 0.661 | 0.324 |
| Frequency by phase | 0.633 | 0.257 |
| BM25 / TF-IDF k=20† | 0.648 / 0.652 | 0.304 / 0.336 |

†Exploratory text baselines; k chosen on a test-period grid (not validation-locked). All other rows locked after primary-investigational mapping remediation (`table1_lock.json`, S10).

On this split, HistGBDT ties OntSim+role at 0.664; exact = IC-sib = subclass OntSim at 0.661. Ontology expansion yields little chronological gain when most test Material ids already appear in training. Macro-F1 remains below accuracy, reflecting minority masking/enrollment categories.

**Figure 4** shows 2019–2021 validation context. **Figure 5** shows per-target HistGBDT versus phase-frequency.

### 4.3 Setting B (coarse labels; same models)

**Table 2.** Field-mean accuracy on the same 2022+ trials (n=283) after coarsening Y.

| Method | Field-mean acc | Macro-F1 |
|--------|---------------:|---------:|
| HistGBDT | 0.821 | 0.627 |
| Logistic regression | 0.818 | 0.610 |
| OntSim kNN@20 (subclass) | 0.806 | 0.571 |
| OntSim kNN@20 (+role) | 0.804 | 0.576 |
| Frequency by phase | 0.802 | 0.562 |
| Exact-match kNN@20 | 0.792 | 0.559 |

OntSim+role is numerically above exact (Δ **+0.0124**; 95% CI **0.000 to +0.0256**). Frequency already exceeds 0.80; HistGBDT is highest. Artifact: `setting_b/setting_b.json`.

### 4.4 Ontology-connected unseen Materials (exploratory)

**Table 3.** Pooled holdout of ontology-connected eligible classes after primary-investigational mapping remediation (23 classes; train 530 / test 888).

| Method | Trial-mean | Field-mean |
|--------|----------:|----------:|
| Exact-match kNN@20 | **0.6461** | 0.6437 |
| OntSim kNN@20 (subclass+role) | 0.6456 | 0.6434 |
| IC-sib kNN@20 (exploratory) | 0.6444 | 0.6420 |
| Frequency by phase | 0.6322 | 0.6303 |
| HistGBDT | 0.6149 | 0.6128 |

IC-sib vs exact: Δ **−0.0017** (95% CI **−0.0090 to +0.0053**). Primary OntSim+role vs exact: Δ **−0.0005** (95% CI **−0.0069 to +0.0063**). Exact-match still uses VO identifiers; the ablation removes relation expansion only. With cleaned labels and a broader eligible set, pooled neighbor methods converge; LOMO remains the clearer unseen-Material contrast.

**Table 4.** LOMO summaries for the same 23 eligible classes (neighbor methods and tabular baselines on the post-remediation split).

| Method | Class-equal | Trial-equal |
|--------|------------:|------------:|
| IC-sib (exploratory) | **0.6541** | 0.6555 |
| HistGBDT | 0.6512 | **0.6592** |
| OntSim kNN@20 (+role) | 0.6457 | 0.6483 |
| Frequency by phase | 0.6450 | 0.6445 |
| Exact-match kNN@20 | 0.6331 | 0.6398 |
| Δ(IC-sib − exact) | +0.0210 | +0.0157 |
| Δ(OntSim+role − exact) | +0.0125 | +0.0084 |

Primary OntSim+role improves both LOMO summaries over exact under matched weights. The IC-sib exploratory variant improves further on class-equal mean; because every LOMO query Material is unseen, that gain is attributable to the IC/sibling similarity package rather than to gating alone. IC-sib achieved the highest observed class-equal LOMO accuracy (0.6541), whereas HistGBDT achieved the highest trial-equal accuracy (0.6592). Per-class values: S4.

**Figure 6** shows pooled ranks (including primary OntSim and IC-sib) and LOMO per-class **primary OntSim+role − exact** deltas.

### 4.5 Matched ablation and secondary analyses

Temporal matched ablation (post-remediation): exact = IC-sib 0.661 / OntSim+role 0.664. AE recall@10 (Model B 0.383; no-VO ≈0.384) and legacy R-GCN: supplement only.

---

## 5. Discussion

SEA-CDM’s primary value here is operational reuse: one NCT-centered join path feeds design suggestion and inspectable evidence lists, with VO sidecars supplying Material/platform ids that core placeholder CURIEs do not. Relative to prior trial knowledge graphs [3] and SEA-CDM/OSEAN [7], OSEAN-CTDR adds a fair exact-versus-OntSim ablation, an unseen-Material stress test, and an exploratory IC-weighted subclass-and-sibling retrieval mode with exact fallback when Material ids are already seen in training.

**When ontology helps.** After primary-investigational mapping remediation, chronological Setting A keeps OntSim ≈ exact (0.664 vs 0.661); HistGBDT is tied with OntSim+role at the top of Table 1. Pooled holdout methods converge once labels are cleaned; **LOMO** is where both primary OntSim+role and the IC-sib exploratory variant lead exact (class Δ +0.013 and +0.021). Setting B favors HistGBDT (0.821), with OntSim+role numerically above exact under coarsened labels. The prototype (Figure S9) is decision support only.

**Implications for informatics practice.** Teams building registry-based design assistants should (i) ground products in a shared ontology when possible, (ii) evaluate exact vs relation-aware retrieval under *matched* rules, (iii) remediate mapping quality before interpreting ontology ablations, and (iv) treat ontology value as protocol- and formulation-dependent—clearest here under leave-one-Material-out transfer, not as a universal chronological upgrade.

**Limits.** Retrospective AACT snapshot (**2026-04-12**); fields not reconstructed as historically visible on each start date; unseen eligibility fixed for this run (not pre-registered); trial-level bootstrap ≠ class-level inference; placeholder core Material CURIEs; explicit VO traversal without full OWL reasoning; mapping substantially remediated (S10: **620** NCT remediations; audit sample dispositions are mixed human/auto—not 80/80 expert correctness); Setting B post-hoc; no prospective or user-study evidence that suggestions change design decisions; 2022+ also used in earlier exploration; val-tuned weights (S11) break the fair weight match and are not primary; IC-sib changes neighborhood, IC weighting, siblings, and gating together—components are not separately ablated.

---

## 6. Conclusion

OSEAN-CTDR organizes vaccine-registry evidence in SEA-CDM and uses VO relations for neighbor-based design suggestion. Chronological OntSim ≈ exact (HistGBDT competitive). Ontology-related gains depended on the retrieval formulation and evaluation protocol: both primary OntSim+role and the exploratory IC-weighted subclass-and-sibling variant improved observed LOMO accuracy over exact matching, whereas pooled holdout showed no clear benefit. Next steps: publish archives, optional broader expert mapping review, and independent or prospective evaluation of decision-support usefulness.

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

### S1. Setting B classical re-run and R-GCN

Main-text Table 2: `ckpts/sea_ml_v14/ctdr/optimize/setting_b/setting_b.json` from `ctdr/eval_setting_b.py`.

**R-GCN (brief; legacy).** Frozen Setting B test mean **0.787** (below frequency 0.802 and HistGBDT 0.821). Artifact predates missing-ID alignment; not re-trained for this revision.

### S2. Resource counts (v1.4 remapped)

Studies 9,147; experiments 18,294; subjects 9,147; interventions 20,001; occurrence 617,510; results 1,525,757. CTDR n=1,418.

### S3. Statistics and disclosure

Bootstrap: OntSim contrasts 1,500 trial resamples; percentile 95% CI; seed 7. Macro-F1: sklearn `f1_score(..., average="macro", zero_division=0)` after excluding unknown gold. Prior 2022+ exploration disclosed.

### S4. LOMO per-class (post primary-investigational mapping remediation)

Class-equal / trial-equal summaries: Table 4. Per-class trial-mean accuracy on the 23 eligible Materials (`after_vo_fix_eval.json`):

| VO id | Label | n | Exact | OntSim+role | IC-sib | Frequency | HistGBDT |
|-------|-------|--:|------:|------------:|-------:|----------:|---------:|
| VO_0000534 | subunit vaccine | 80 | 0.647 | 0.641 | 0.619 | 0.641 | 0.644 |
| VO_0000561 | toxoid vaccine | 77 | 0.689 | 0.686 | 0.686 | 0.676 | 0.696 |
| VO_0000032 | DNA vaccine | 75 | 0.653 | 0.677 | 0.680 | 0.630 | 0.663 |
| VO_0004673 | quadrivalent influenza vaccine | 67 | 0.649 | 0.646 | 0.646 | 0.657 | 0.731 |
| VO_0000367 | live attenuated vaccine | 66 | 0.620 | 0.639 | 0.663 | 0.631 | 0.645 |
| VO_0001178 | live attenuated influenza vaccine | 60 | 0.582 | 0.590 | 0.604 | 0.637 | 0.604 |
| VO_0003150 | Hepatitis B surface antigen vaccine | 58 | 0.621 | 0.638 | 0.655 | 0.621 | 0.677 |
| VO_0000484 | recombinant vector vaccine | 57 | 0.697 | 0.697 | 0.719 | 0.678 | 0.711 |
| VO_0000642 | influenza virus vaccine | 56 | 0.641 | 0.637 | 0.637 | 0.624 | 0.624 |
| VO_0000771 | BCG vaccine | 51 | 0.675 | 0.675 | 0.685 | 0.673 | 0.649 |
| VO_0000761 | malaria vaccine | 31 | 0.548 | 0.589 | 0.637 | 0.589 | 0.653 |
| VO_0006034 | meningococcal vaccine (unknown formulation/serogroups) | 31 | 0.629 | 0.621 | 0.621 | 0.645 | 0.653 |
| VO_0000047 | Fluzone | 28 | 0.580 | 0.598 | 0.607 | 0.580 | 0.625 |
| VO_0000577 | peptide vaccine | 18 | 0.764 | 0.736 | 0.764 | 0.750 | 0.778 |
| VO_0003075 | Gardasil 9 vaccine | 18 | 0.625 | 0.639 | 0.653 | 0.667 | 0.611 |
| VO_0007062 | Quadrivalent HPV (6/11/16/18) recombinant vaccine | 17 | 0.647 | 0.647 | 0.632 | 0.750 | 0.603 |
| VO_0004246 | Anthrax vaccine adsorbed with squalene adjuvant | 16 | 0.594 | 0.609 | 0.625 | 0.578 | 0.609 |
| VO_0007319 | autologous tumor cell vaccine | 16 | 0.651 | 0.776 | 0.760 | 0.667 | 0.682 |
| VO_0005158 | Oxford-AstraZeneca COVID-19 vaccine | 15 | 0.583 | 0.600 | 0.617 | 0.567 | 0.567 |
| VO_0007063 | Recombinant HPV nonavalent vaccine | 14 | 0.625 | 0.625 | 0.607 | 0.625 | 0.661 |
| VO_0000186 | RNA vaccine | 13 | 0.673 | 0.635 | 0.615 | 0.615 | 0.558 |
| VO_0003464 | Hepatitis A vaccine, inactivated | 12 | 0.625 | 0.708 | 0.667 | 0.646 | 0.688 |
| VO_0003834 | poliovirus vaccine inactivated | 12 | 0.542 | 0.542 | 0.646 | 0.688 | 0.646 |

### S5. SEA-CDM structural audit summary

See `optimize/seacdm_linkage/` and `v1.4/OFFICIAL_V14_*`. Spine FKs 100% row-resolved as in §4.1.

### S6. AE ranking

Unit: NCT × arm-role context. NCT 80/20 split; **n_eval = 483** contexts. Model B recall@10 0.383; no-VO ≈0.384. Not used to claim graph indispensability for AE ranking.

### S7. Ontology-connected eligible Materials (post-remediation)

Eligible rule: Material count ∈ [12, 80] and ontology-connected to another cohort Material. After primary-investigational mapping remediation this yields **23** classes (Table S4 lists n and LOMO scores; labels from `vo.owl` via `vo_eligible_labels.py`).

| VO id | Label (vo.owl) | n |
|-------|----------------|--:|
| VO_0000534 | subunit vaccine | 80 |
| VO_0000561 | toxoid vaccine | 77 |
| VO_0000032 | DNA vaccine | 75 |
| VO_0004673 | quadrivalent influenza vaccine | 67 |
| VO_0000367 | live attenuated vaccine | 66 |
| VO_0001178 | live attenuated influenza vaccine | 60 |
| VO_0003150 | Hepatitis B surface antigen vaccine | 58 |
| VO_0000484 | recombinant vector vaccine | 57 |
| VO_0000642 | influenza virus vaccine | 56 |
| VO_0000771 | BCG vaccine | 51 |
| VO_0000761 | malaria vaccine | 31 |
| VO_0006034 | meningococcal vaccine (unknown formulation/serogroups) | 31 |
| VO_0000047 | Fluzone | 28 |
| VO_0000577 | peptide vaccine | 18 |
| VO_0003075 | Gardasil 9 vaccine | 18 |
| VO_0007062 | Quadrivalent HPV (6/11/16/18) recombinant vaccine | 17 |
| VO_0004246 | Anthrax vaccine adsorbed with squalene adjuvant | 16 |
| VO_0007319 | autologous tumor cell vaccine | 16 |
| VO_0005158 | Oxford-AstraZeneca COVID-19 vaccine | 15 |
| VO_0007063 | Recombinant HPV nonavalent vaccine | 14 |
| VO_0000186 | RNA vaccine | 13 |
| VO_0003464 | Hepatitis A vaccine, inactivated | 12 |
| VO_0003834 | poliovirus vaccine inactivated | 12 |

### S8. Reproducibility file inventory

Public code: https://github.com/zhujiadongok/osean-ctdr. Supporting data (Zenodo; DOI reserved): https://doi.org/10.5281/zenodo.23773934.

**AACT snapshot.** Pipe-delimited ClinicalTrials.gov / AACT extract dated **2026-04-12**. Evidence: (i) `AACT Database/6wvqgoof4dut7eaxn610uzfajvnt.zip` filesystem mtime 2026-04-12; (ii) `studies.txt` row `version` / update timestamps `2026-04-12 04:05:04.982706`. Local folder id `6wvqgoof4dut7eaxn610uzfajvnt`; `studies.txt` size 390,802,838 bytes; SHA-256 `82870fc4a17cc1eb` `75529c603866db82` `207ff7366ab49569` `f4779f89ffaba7f8`.

**VO OWL.** Release **2026-06-12** under `SEA-CDM AACT vaccine remapped/research_factors/vo_mapping/FOR_ADVISOR/vo.owl`.

- Table 1 lock: `optimize/table1_lock.json` (`python -m ctdr.eval_table1_lock`)
- Post-remediation neighbors / LOMO: `optimize/mapping_audit/after_vo_fix_eval.json`
- Mechanism neighbors: `optimize/mapping_audit/mechanism_example_NCT_neighbors.json`
- Setting B: `optimize/setting_b/setting_b.json` (`python -m ctdr.eval_setting_b`)
- Mapping audit: `optimize/mapping_audit/` (`validate_vo_mapping` must pass)
- Legacy contrasts: `optimize/graph_value/graph_value.json`
- Entrypoints: `eval_table1_lock`; `eval_after_vo_fix`; `export_mechanism_example`; `eval_setting_b`; `validate_vo_mapping`; `apply_vo_corrections_upstream`
- Demo: `osean_neo4j/app_xy_isg.py` (Design tab)

### S9. Prototype Design interface (screenshot)

![Figure S9. OSEAN-CTDR Design tab](figS_demo_design.png)

**Figure S9.** Prototype Design view: query inputs, predicted design fields, and OntSim-ranked supporting NCT ids. Decision support only.

### S10. Mapping audit + primary-investigational corrections

**Primary-investigational mapping (locked).** `arch_vo` = primary investigational vaccine only; `vo_platform` = that vaccine’s technology platform.

**Rubric.** `correct` / `incorrect` / `ambiguous` for Material and platform separately; `unknown` platform = unlabeled.

**Audit sample (`mapping_audit_sample_reviewed.csv`, n=80).** Disposition by reviewer tag (not interchangeable with expert correctness):
- `advisor_review`: **19** (human advisor seed)
- `ctgov_final22`: **22** (CT.gov-checked final-22 set)
- `auto_remediated_verified`: **39** (auto-remediation + programmatic verification)

Judgment counts describe the mappings at the time of review, before application of the final corrections. Material: correct 60 / incorrect 17 / ambiguous 2 / blank 1. Platform: correct 59 / incorrect 12 / ambiguous 4 / blank 5. These support “80 rows have disposition records,” not “80/80 expert-verified correct.” `validate_vo_mapping` PASS confirms regression checks (overlay presence, intentional unknowns, rule integrity)—not independent semantic accuracy for every CTDR row.

**Expanded remediation (620 NCTs, ~44% of CTDR).** Targets in `vo_corrections_primary.csv` applied **upstream** (`disease_vo_map` + `study_context_profile`) and as an idempotent CTDR overlay. Hard checks: `python -m ctdr.validate_vo_mapping` (must pass). Rebuild: `build_vo_corrections_bulk` → `apply_vo_corrections_upstream` → `validate_vo_mapping` → `build_table` → `eval_after_vo_fix` / `eval_table1_lock` / `eval_setting_b`.

**Cohort mapping how / platform (CTDR n=1,418).** Pre-remediation snapshot: how platform 801 / product 320 / both 297; `vo_platform=unknown` 241. Post-remediation (analysis table): how platform 539 / product 101 / both 778; unknown **24**.

**Post-remediation neighbor metrics.** `after_vo_fix_eval.json` + `table1_lock.json` + `setting_b/setting_b.json`. Chrono exact=IC-sib 0.661 / role 0.664 / HistGBDT 0.664; pooled ~0.645 (n=888, 23 classes); LOMO class-equal exact 0.633 / OntSim+role 0.646 / IC-sib 0.654 / HistGBDT 0.651 / frequency 0.645.

### S11. Val-tuned weights (not a fair ablation)

On 2019–2021 validation, retuning purpose/platform weights for IC-sib OntSim (purpose 1.0, platform 0.5; other weights as in §3.4) yielded chronological field-mean **0.670** and pooled trial-mean **0.638** when scored once on test/pooled. Because weights no longer match the exact-match control, this variant is exploratory only and is not used in Tables 1–4.

---

## Funding / Data & Code Availability

**Funding.** *[Corresponding author to insert funding statement before submission.]*  
**Author contributions.** J. Zhu: analysis design, implementation, evaluation, and drafting. F.-Y. Yeh: VO/VaxCT mapping resources. J. Zheng: *[role to confirm]*. Y. He: supervision and critical revision. All authors approved the submitted version.  
**Conflict of interest.** None declared.  
**Data and code.** Code: https://github.com/zhujiadongok/osean-ctdr. Supporting data: https://doi.org/10.5281/zenodo.23773934 (DOI reserved; publish at submission).  
**AI assistance.** Language and code editing assistance was used; authors verified all scientific content and results.

---

## Figure legends

**Figure 1.** Pipeline: AACT → SEA-CDM v1.4 (vaccine remapped) → VO grounding → CTDR / AE analytics. Instance graphs organize entities; OntSim uses the VO class graph.

**Figure 2.** VO grounding of platform/product strings; RNA/mRNA platform identifiers are normalized to `VO_0000186`.

**Figure 3.** Entity types and relations in the OSEAN resource graph. CTDR uses typed inputs plus VO class relations for OntSim; design targets are not retrieval edges.

**Figure 4.** Validation field-mean accuracy (2019–2021, n=244); zero-baseline bars.

**Figure 5.** Per-target test accuracy: HistGBDT vs phase-frequency (n=283).

**Figure 6.** Unseen Materials after primary-investigational mapping remediation: (A) pooled trial-mean ranks (exact, primary OntSim+role, IC-sib exploratory, frequency, HistGBDT; zero baseline); (B) LOMO per-class Δ(**primary OntSim+role − exact**). IC-sib LOMO numbers are in Table 4 / S4.

**Figure S9.** Prototype Design interface. Supplement only.

---

## References

1. Zarin DA, Tse T, Williams RJ, Califf RM, Ide NC. The ClinicalTrials.gov results database—update and key issues. *N Engl J Med*. 2011;364:852–860.
2. Tasneem A, Aberle L, Ananthakrishnan S, et al. The database for Aggregate Analysis of ClinicalTrials.gov (AACT) and subsequent regrouping by clinical specialty. *PLoS One*. 2012;7:e33677.
3. Chen Z, Peng B, Ioannidis VN, et al. A knowledge graph of clinical trials (CTKG). *Sci Rep*. 2022;12:4724.
4. He Y, Cowell L, Diehl AD, et al. VO: Vaccine Ontology. *Nature Precedings*. 2009.
5. Lin Y, He Y. Ontology representation and analysis of vaccine formulation and administration and their effects on vaccine immune responses. *J Biomed Semantics*. 2012;3:17.
6. Xiang Z, Todd T, Ku KP, et al. VIOLIN: vaccine investigation and online information network. *Nucleic Acids Res*. 2008;36:D923–D928.
7. Huffman A, Yeh F-Y, Hur J, et al. SEA CDM: Study-Experiment-Assay Common Data Model and Databases for Cross-Domain Data Integration and Analysis. *Sci Data*. 2026;13:238. doi:10.1038/s41597-026-06558-z
8. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *JMLR*. 2011;12:2825–2830.

---

## Notes for coauthors

- **Still need humans:** funding statement; J. Zheng CRediT role; publish Zenodo DOI; submit cover letter.  
- **Locked for submission:** primary-investigational mapping; 620 remediations; audit sample dispositions 19/22/39 (not 80/80 expert-correct); `validate_vo_mapping` PASS; Table 1 / Setting B / LOMO refreshed (`table1_lock.json`, `setting_b.json`, `after_vo_fix_eval.json`).  
- AACT snapshot **2026-04-12** (folder mtime + `studies.txt` timestamps); IC-sib is exploratory same-weight sensitivity (S11 weight retune separate).
