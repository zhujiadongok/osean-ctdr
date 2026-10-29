"""Serve locked CTDR recommendations for the Streamlit app."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ctdr import config as cfg
from ctdr.embeddings import inductive_study_vectors, study_matrix
from ctdr.graph import build_input_graph
from ctdr.regress import regress_predict

OPT = cfg.CTDR_DIR / "optimize"
BEST_PATH = OPT / "best_config.json"
STUDIES_PATH = OPT / "ctdr_studies.csv"
EMB_PATH = OPT / "embeddings_train.npz"


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    studies = pd.read_csv(STUDIES_PATH, dtype=str).fillna("")
    train = studies[studies["split"] == "train"].reset_index(drop=True)
    best = json.loads(BEST_PATH.read_text(encoding="utf-8")) if BEST_PATH.exists() else {}
    emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    g_train = build_input_graph(train)
    # prefer CompGCN-128 (structure lock); fall back
    vec_key = "compgcn_128" if emb is not None and "compgcn_128" in emb.files else None
    if vec_key is None and emb is not None and "transe_128" in emb.files:
        vec_key = "transe_128"
    train_vec_map = {}
    if emb is not None and vec_key is not None:
        mat, ids = study_matrix(emb[vec_key], g_train)
        train_vec_map = {n: mat[i] for i, n in enumerate(ids)}
    return {
        "studies": studies,
        "train": train,
        "best": best,
        "graph": g_train,
        "emb": emb,
        "vec_key": vec_key,
        "train_vecs": train_vec_map,
    }


def recommend_design(
    *,
    disease: str,
    arch_vo: str,
    vo_platform: str,
    phase: str,
    primary_purpose: str = "PREVENTION",
    route: str = "unknown",
    dose: str = "unknown",
    sex: str = "SEX_ALL",
    older: str = "unknown",
    child: str = "unknown",
    sponsor_class: str = "unknown",
    k: int = 10,
    alpha: float = 0.75,
) -> dict:
    """Return design recommendations + top supporting NCT ids."""
    b = load_bundle()
    qid = "QUERY_USER"
    row = {
        "nct_id": qid,
        "split": "test",
        "disease": disease,
        "arch_vo": arch_vo,
        "vo_platform": vo_platform,
        "phase": phase,
        "primary_purpose": primary_purpose,
        "route": route,
        "dose": dose,
        "sex": sex,
        "older": older,
        "child": child,
        "sponsor_class": sponsor_class,
        # placeholders so columns align
        "allocation": "unknown",
        "masking": "unknown",
        "intervention_model": "unknown",
        "enrollment_bucket": "unknown",
        "primary_outcome_family": "unknown",
    }
    studies = pd.concat([b["studies"], pd.DataFrame([row])], ignore_index=True)
    train_ids = list(b["train"]["nct_id"])
    query = pd.DataFrame([row])
    # Confirmatory lock: HistGBDT chosen on 2019–2021 validation, not on the 2022+ test grid.
    preds = regress_predict(b["train"], query, kind="gbdt", targets=cfg.TARGETS[:4])
    method = "hist_gbdt"

    # supporting neighbors via ontology rank (explainable NCT list)
    from ctdr.recommend import ontology_similarity_matrix

    ranked = ontology_similarity_matrix(studies, train_ids, [qid]).get(qid, [])[:k]
    support = [{"nct_id": n, "score": float(s)} for n, s in ranked]
    return {
        "method": method,
        "recommendation": preds.get(qid, {}),
        "support": support,
        "best_lock": b["best"],
    }
