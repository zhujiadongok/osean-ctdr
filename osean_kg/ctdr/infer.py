"""Serve CTDR recommendations for the Streamlit demo (aligned with paper)."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ctdr import config as cfg
from ctdr.embeddings import study_matrix
from ctdr.graph import build_input_graph
from ctdr.push08_lock import COARSE, add_coarse
from ctdr.recommend import ontology_knn_recommend, ontology_similarity_matrix
from ctdr.regress import frequency_predict, regress_predict

OPT = cfg.CTDR_DIR / "optimize"
BEST_PATH = OPT / "best_config.json"
STUDIES_PATH = OPT / "ctdr_studies.csv"
EMB_PATH = OPT / "embeddings_train.npz"

FINE = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)

# Paper Table 1 / Table 2 (Setting A / B temporal test), for UI captions.
PAPER_METRICS = {
    "A": {
        "hist_gbdt": 0.658,
        "logistic": 0.661,
        "frequency": 0.633,
        "exact_knn": 0.658,
        "ontsim": 0.656,
    },
    "B": {
        "hist_gbdt": 0.821,
        "logistic": 0.815,
        "frequency": 0.802,
        "exact_knn": 0.797,
        "ontsim": 0.800,
    },
}


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    studies = pd.read_csv(STUDIES_PATH, dtype=str).fillna("")
    studies = add_coarse(studies)
    train = studies[studies["split"] == "train"].reset_index(drop=True)
    best = json.loads(BEST_PATH.read_text(encoding="utf-8")) if BEST_PATH.exists() else {}
    emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    g_train = build_input_graph(train)
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
    k: int = 20,
    method: str = "hist_gbdt",
    label_setting: str = "A",
) -> dict:
    """Return design recommendations + OntSim-ranked supporting NCT ids.

    method: hist_gbdt | logistic | frequency | exact_knn | ontsim_knn
            | ontsim_ic_sep | ontsim_gated_ic_sep | ontsim_gated_ic_sib
            | ontsim_gated_ic_sib_perf
    label_setting: A (fine Setting A targets) | B (coarse Setting B targets)
    """
    b = load_bundle()
    targets = FINE if label_setting == "A" else COARSE
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
        "allocation": "unknown",
        "masking": "unknown",
        "intervention_model": "unknown",
        "enrollment_bucket": "unknown",
        "primary_outcome_family": "unknown",
        "masking_binary": "unknown",
        "intervention_binary": "unknown",
        "enrollment_binary": "unknown",
    }
    studies = pd.concat([b["studies"], pd.DataFrame([row])], ignore_index=True)
    train = b["train"]
    train_ids = list(train["nct_id"])
    query = pd.DataFrame([row])

    knn_mode = None
    if method == "hist_gbdt":
        preds = regress_predict(train, query, kind="gbdt", targets=targets)
    elif method == "logistic":
        preds = regress_predict(train, query, kind="lr", targets=targets)
    elif method == "frequency":
        preds = frequency_predict(train, query, targets=targets)
    elif method == "exact_knn":
        knn_mode = "exact"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="exact"
        )
    elif method == "ontsim_knn":
        knn_mode = "graph"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="graph"
        )
    elif method == "ontsim_ic_sep":
        knn_mode = "graph_ic_sep"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="graph_ic_sep"
        )
    elif method == "ontsim_gated_ic_sep":
        knn_mode = "graph_gated_ic_sep"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="graph_gated_ic_sep"
        )
    elif method == "ontsim_gated_ic_sib":
        knn_mode = "graph_gated_ic_sib"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="graph_gated_ic_sib"
        )
    elif method == "ontsim_gated_ic_sib_perf":
        knn_mode = "graph_gated_ic_sib_perf"
        preds = ontology_knn_recommend(
            studies, train_ids, [qid], targets=targets, k=k, mode="graph_gated_ic_sib_perf"
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Supporting neighbors: improved OntSim list when using improved methods.
    support_mode = knn_mode if knn_mode and knn_mode != "exact" else "graph_gated_ic_sib"
    ranked = ontology_similarity_matrix(
        studies, train_ids, [qid], mode=support_mode if support_mode else "graph"
    ).get(qid, [])[:k]
    support = [{"nct_id": n, "ontsim_score": float(s)} for n, s in ranked]

    paper = PAPER_METRICS[label_setting]
    metric_key = {
        "hist_gbdt": "hist_gbdt",
        "logistic": "logistic",
        "frequency": "frequency",
        "exact_knn": "exact_knn",
        "ontsim_knn": "ontsim",
        "ontsim_ic_sep": "ontsim",
        "ontsim_gated_ic_sep": "ontsim",
        "ontsim_gated_ic_sib": "ontsim",
        "ontsim_gated_ic_sib_perf": "ontsim",
    }[method]

    return {
        "method": method,
        "label_setting": label_setting,
        "targets": list(targets),
        "knn_mode": knn_mode,
        "k": k,
        "recommendation": preds.get(qid, {}),
        "support": support,
        "paper_field_mean_acc": paper[metric_key],
        "paper_metrics": paper,
        "best_lock": b["best"],
    }
