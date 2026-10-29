# OSEAN-CTDR (Clinical Trial Design Recommendation)

Highest-standard PoC on **SEA-CDM v1.4** + advisor **`vo.owl`**.

## Task

Given disease, Material/platform VO, phase, purpose (+ route/dose/eligibility when available),
**recommend design variables**:

- `allocation`
- `masking`
- `intervention_model`
- `enrollment_bucket`

## Methods compared

| Method | Role |
|--------|------|
| Frequency by phase | Baseline |
| Logistic / HistGBDT | Classical regression |
| Ontology-weighted kNN | Advisor-style similarity |
| TransE-kNN | Graph embedding |
| CompGCN-kNN | Graph embedding |

## Leakage controls

- Chronological split (`start_date < 2022` train / `≥ 2022` test).
- Targets **never** appear as graph edges.
- kNN neighbors are drawn from the **train** set only.

## Run

```bash
cd <repo>
python -m ctdr.run_experiment
```

Outputs under `osean_kg/ckpts/sea_ml_v14/ctdr/`.
