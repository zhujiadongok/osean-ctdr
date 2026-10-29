"""Validation-only improvement sweep for CTDR, then one test evaluation.

Goals
-----
Improve CompGCN + fusion under the *same* lock protocol as lock_eval.py:
  fit < 2019, val 2019-2021, test >= 2022.
Test labels are never used to choose hyperparameters.

  python -u -m ctdr.improve_lock
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.embeddings import (  # noqa: E402
    inductive_study_vectors,
    study_matrix,
    train_compgcn_v2,
)
from ctdr.graph import build_input_graph  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import (  # noqa: E402
    hybrid_knn_recommend,
    ontology_knn_recommend,
    ontology_similarity_matrix,
    triple_hybrid_recommend,
)
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.text_baselines import trial_text  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
VAL_START = "2019-01-01"
OUT = cfg.CTDR_DIR / "optimize" / "improve_lock"


def _struct(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "per_target": {t: rep["targets"][t] for t in STRUCT},
    }


def _tfidf_rank(
    train: pd.DataFrame, query: pd.DataFrame
) -> dict[str, list[tuple[str, float]]]:
    train_ids = list(train["nct_id"])
    docs = [trial_text(r) for _, r in train.iterrows()]
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    Xtr = vec.fit_transform(docs)
    out: dict[str, list[tuple[str, float]]] = {}
    for _, row in query.iterrows():
        qv = vec.transform([trial_text(row)])
        sims = cosine_similarity(qv, Xtr).ravel()
        order = np.argsort(-sims)
        out[row.nct_id] = [(train_ids[j], float(sims[j])) for j in order]
    return out


def _emb_feature_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_vecs: dict[str, np.ndarray],
    test_vecs: dict[str, np.ndarray],
    *,
    kind: str = "gbdt",
    emb_dim_use: int = 32,
) -> dict[str, dict[str, str]]:
    """Classical model on categorical X plus CompGCN inductive vector."""
    cats = list(cfg.INPUT_CATS)
    # fixed projection of embedding to keep feature count modest
    rng = np.random.default_rng(cfg.SEED)
    dim = next(iter(train_vecs.values())).shape[0]
    proj = rng.normal(size=(dim, emb_dim_use))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

    def pack(df: pd.DataFrame, vecs: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
        Xcat = df[cats].fillna("unknown").astype(str).reset_index(drop=True)
        E = np.stack(
            [
                (vecs[n] @ proj) if n in vecs else np.zeros(emb_dim_use)
                for n in df["nct_id"]
            ]
        )
        return Xcat, E

    out: dict[str, dict[str, str]] = {n: {} for n in test["nct_id"]}
    Xtr_cat, Etr = pack(train, train_vecs)
    Xte_cat, Ete = pack(test, test_vecs)
    emb_cols = [f"e{i}" for i in range(emb_dim_use)]
    Xtr = Xtr_cat.copy()
    Xte = Xte_cat.copy()
    for i, c in enumerate(emb_cols):
        Xtr[c] = Etr[:, i]
        Xte[c] = Ete[:, i]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("emb", StandardScaler(), emb_cols),
        ]
    )
    for t in STRUCT:
        ytr = train[t].astype(str).to_numpy()
        if kind == "lr":
            clf = LogisticRegression(max_iter=cfg.REG_MAX_ITER, solver="liblinear")
            if len(set(ytr)) > 2:
                pipe = Pipeline([("prep", pre), ("clf", OneVsRestClassifier(clf))])
            else:
                pipe = Pipeline([("prep", pre), ("clf", clf)])
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=cfg.GBDT_MAX_DEPTH,
                learning_rate=cfg.GBDT_LR,
                max_iter=cfg.GBDT_ITER,
                random_state=cfg.SEED,
            )
            pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        for nct, p in zip(test["nct_id"], pred):
            out[nct][t] = str(p)
    return out


def _train_emb(fit: pd.DataFrame, dim: int, epochs: int) -> tuple[np.ndarray, dict]:
    g = build_input_graph(fit)
    print(
        f"CompGCN-v2 dim={dim} nodes={len(g['nodes'])} triples={len(g['triples'])} epochs={epochs}",
        flush=True,
    )
    emb = train_compgcn_v2(
        len(g["nodes"]),
        g["n_rel"],
        g["triples"],
        dim=dim,
        epochs=epochs,
        n_layers=3,
        n_neg=8,
    )
    return emb, g


def _knn_bundle(
    both: pd.DataFrame,
    train: pd.DataFrame,
    query: pd.DataFrame,
    emb: np.ndarray,
    graph: dict,
) -> dict[str, dict]:
    train_ids = list(train["nct_id"])
    qids = list(query["nct_id"])
    mat, ids = study_matrix(emb, graph)
    train_map = {n: mat[i] for i, n in enumerate(ids)}
    qvecs = inductive_study_vectors(emb, graph, both, qids)
    text_rank = _tfidf_rank(train, query)

    out: dict[str, dict] = {}
    # pure embedding
    for k in (5, 10, 15, 20):
        out[f"compgcn_v2_ind_k{k}"] = hybrid_knn_recommend(
            both, train_ids, qids, qvecs, train_map, alpha=0.0, k=k, targets=STRUCT
        )
    # ontology-embedding hybrids
    for alpha in (0.25, 0.5, 0.65, 0.75, 0.85):
        for k in (10, 15, 20):
            out[f"hybrid_v2_a{alpha}_k{k}"] = hybrid_knn_recommend(
                both,
                train_ids,
                qids,
                qvecs,
                train_map,
                alpha=alpha,
                k=k,
                targets=STRUCT,
            )
    # triple hybrids (onto + emb + text)
    for w_onto, w_emb, w_text in (
        (0.5, 0.4, 0.1),
        (0.45, 0.4, 0.15),
        (0.4, 0.4, 0.2),
        (0.35, 0.45, 0.2),
        (0.5, 0.3, 0.2),
        (0.3, 0.5, 0.2),
    ):
        for k in (10, 20):
            name = f"triple_o{w_onto}_e{w_emb}_t{w_text}_k{k}"
            out[name] = triple_hybrid_recommend(
                both,
                train_ids,
                qids,
                qvecs,
                train_map,
                text_rank,
                w_onto=w_onto,
                w_emb=w_emb,
                w_text=w_text,
                k=k,
                targets=STRUCT,
            )
    # GNN vector as features for classical models
    out["gbdt_plus_compgcn"] = _emb_feature_predict(
        train, query, train_map, qvecs, kind="gbdt"
    )
    out["logistic_plus_compgcn"] = _emb_feature_predict(
        train, query, train_map, qvecs, kind="lr"
    )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_ctdr_table()
    table["start_date_parsed"] = pd.to_datetime(table["start_date"], errors="coerce")
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    fit = train[train["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train[train["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    print(f"fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    # --- validation search ---
    print("=== validation sweep ===", flush=True)
    both_val = pd.concat([fit, val], ignore_index=True)
    classical = {
        "frequency_by_phase": frequency_predict(fit, val, STRUCT),
        "logistic": regress_predict(fit, val, "lr", STRUCT),
        "hist_gbdt": regress_predict(fit, val, "gbdt", STRUCT),
        "ontology_knn_k10": ontology_knn_recommend(
            both_val, list(fit["nct_id"]), list(val["nct_id"]), STRUCT, k=10
        ),
        "ontology_knn_k20": ontology_knn_recommend(
            both_val, list(fit["nct_id"]), list(val["nct_id"]), STRUCT, k=20
        ),
    }
    val_scores = {n: _struct(score_predictions(val, p, STRUCT)) for n, p in classical.items()}

    for dim, epochs in ((64, 400), (128, 500), (128, 700)):
        key = f"d{dim}_e{epochs}"
        emb, g = _train_emb(fit, dim, epochs)
        knn = _knn_bundle(both_val, fit, val, emb, g)
        for name, pred in knn.items():
            full = f"{key}__{name}"
            sc = _struct(score_predictions(val, pred, STRUCT))
            val_scores[full] = sc
            print(
                f"  VAL {sc['mean_accuracy']:.4f} f1={sc['mean_macro_f1']:.4f}  {full}",
                flush=True,
            )

    # pick best by accuracy then macro-F1
    best_name = max(
        val_scores,
        key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]),
    )
    print(f"LOCKED ON VAL: {best_name} acc={val_scores[best_name]['mean_accuracy']:.4f}", flush=True)

    # --- refit on full train; score test once ---
    print("=== refit full train; score test once ===", flush=True)
    both_te = pd.concat([train, test], ignore_index=True)
    test_preds: dict[str, dict] = {
        "frequency_by_phase": frequency_predict(train, test, STRUCT),
        "logistic": regress_predict(train, test, "lr", STRUCT),
        "hist_gbdt": regress_predict(train, test, "gbdt", STRUCT),
        "ontology_knn_k20": ontology_knn_recommend(
            both_te, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20
        ),
    }

    # parse locked name
    if "__" in best_name:
        emb_key, method = best_name.split("__", 1)
        # retrain matching dim/epochs on full train
        dim = int(emb_key.split("_")[0][1:])
        epochs = int(emb_key.split("_")[1][1:])
        emb, g = _train_emb(train, dim, epochs)
        bundle = _knn_bundle(both_te, train, test, emb, g)
        if method not in bundle:
            raise KeyError(f"method {method} missing after refit; have {list(bundle)[:10]}")
        test_preds[best_name] = bundle[method]
        # also score a few strong graph candidates for reporting
        for extra in (
            "hybrid_v2_a0.75_k10",
            "hybrid_v2_a0.65_k20",
            "triple_o0.4_e0.4_t0.2_k20",
            "gbdt_plus_compgcn",
            "compgcn_v2_ind_k10",
        ):
            if extra in bundle:
                test_preds[f"{emb_key}__{extra}"] = bundle[extra]
    else:
        # classical lock
        if best_name == "hist_gbdt":
            test_preds[best_name] = test_preds["hist_gbdt"]
        elif best_name == "logistic":
            test_preds[best_name] = test_preds["logistic"]
        elif best_name.startswith("ontology"):
            k = 20 if "k20" in best_name else 10
            test_preds[best_name] = ontology_knn_recommend(
                both_te, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=k
            )
        else:
            test_preds[best_name] = test_preds.get(best_name) or frequency_predict(
                train, test, STRUCT
            )

    test_scores = {n: _struct(score_predictions(test, p, STRUCT)) for n, p in test_preds.items()}
    locked_test = test_scores[best_name]

    # leaderboards
    val_board = sorted(val_scores.items(), key=lambda kv: (-kv[1]["mean_accuracy"], -kv[1]["mean_macro_f1"]))
    test_board = sorted(test_scores.items(), key=lambda kv: (-kv[1]["mean_accuracy"], -kv[1]["mean_macro_f1"]))

    payload = {
        "protocol": {
            "fit": f"start < {VAL_START}",
            "val": f"{VAL_START} <= start < {cfg.SPLIT_DATE}",
            "test": f"start >= {cfg.SPLIT_DATE}",
            "n_fit": int(len(fit)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "test_used_for_selection": False,
            "improvements": [
                "CompGCN-v2: 3 layers, self-loops, hard negatives, residual",
                "same_context study-study edges (disease+platform+phase)",
                "weighted inductive vectors",
                "hybrid alpha/k grid + triple onto/emb/text fusion",
                "gbdt/logistic + CompGCN vector features",
            ],
        },
        "locked_on_validation": best_name,
        "val_score_locked": val_scores[best_name],
        "test_score_locked": locked_test,
        "val_top15": [
            {"name": n, "mean_accuracy": s["mean_accuracy"], "mean_macro_f1": s["mean_macro_f1"]}
            for n, s in val_board[:15]
        ],
        "test_scores": {
            n: {"mean_accuracy": s["mean_accuracy"], "mean_macro_f1": s["mean_macro_f1"]}
            for n, s in test_board
        },
        "gnn_involved": ("compgcn" in best_name.lower())
        or ("hybrid" in best_name.lower())
        or ("triple" in best_name.lower())
        or ("plus_compgcn" in best_name.lower()),
    }
    (OUT / "improve_lock.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "CTDR IMPROVE LOCK",
        f"locked={best_name}",
        f"val_acc={val_scores[best_name]['mean_accuracy']:.4f}",
        f"test_acc={locked_test['mean_accuracy']:.4f}",
        f"gnn_involved={payload['gnn_involved']}",
        "",
        "## VAL TOP 15",
    ]
    for n, s in val_board[:15]:
        lines.append(f"  {s['mean_accuracy']:.4f}  f1={s['mean_macro_f1']:.4f}  {n}")
    lines.append("")
    lines.append("## TEST (scored methods)")
    for n, s in test_board:
        lines.append(f"  {s['mean_accuracy']:.4f}  f1={s['mean_macro_f1']:.4f}  {n}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
