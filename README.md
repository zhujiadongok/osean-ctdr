# OSEAN-CTDR

**OSEAN-CTDR** = **O**ntology-based **SEA** **N**etwork **C**linical **T**rial **D**esign **R**ecommendation.

Reproduction package for the JAMIA draft: *OSEAN-CTDR: Ontology-based SEA Network Clinical Trial Design Recommendation for SEA-CDM–based vaccine trial evidence integration*.

This repository contains analysis code, feature sidecars used by CTDR, frozen result artifacts cited in the paper, and manuscript helpers. It does **not** include the full AACT dump (too large; snapshot date was not retained locally—see paper Table S8). Supporting data Zenodo draft: https://doi.org/10.5281/zenodo.23773934

## Layout

```
osean_kg/ctdr/                 # CTDR package (eval / recommend / regress)
osean_kg/vo_architecture.py    # VO class load + PROFILE_TO_OFFICIAL roll-up
osean_kg/ckpts/sea_ml_v14/     # disease_vo_map, AE metrics, optimize/*.json
data/sea_cdm_v14/              # study_context_profile + intervention profile sidecars
paper/                         # draft + figure label helpers
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set data roots (defaults assume this repo layout):

```bash
# optional overrides
set OSEAN_CTDR_ROOT=%CD%
set SEA_CDM_V14=%CD%\data\sea_cdm_v14
```

Place or symlink Vaccine Ontology release **2026-06-12** where `vo_architecture.py` expects, or set:

```bash
set VO_OWL=path\to\vo.owl
```

Official release: https://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl

## Main entrypoints

From repo root (with `osean_kg` on `PYTHONPATH`):

```bash
set PYTHONPATH=%CD%\osean_kg;%PYTHONPATH%
python -u -m ctdr.eval_onto_graph_knn
python -u -m ctdr.eval_graph_value
python -u -m ctdr.eval_seacdm_linkage
```

Frozen paper numbers live under `osean_kg/ckpts/sea_ml_v14/ctdr/optimize/` (`graph_value`, `onto_graph_knn`, `seacdm_linkage`, `push08`).

## Citation

Manuscript draft: `paper/OSEAN_JAMIA_draft.md`. Archival DOI / Zenodo link will be added at journal submission for any remaining large assets.
