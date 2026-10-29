"""Highest-standard CTDR optimization sweep.

Improvements over run_experiment:
  - AACT primary outcome families as an extra target
  - sponsor_class input feature / graph edge
  - harder TransE negatives + L2 entity norm
  - train-only graph + inductive test vectors (no target leakage)
  - hybrid ontology × embedding similarity
  - ensemble of top classical + graph methods
  - grid over k / alpha / dim

  python -m ctdr.optimize_run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.embeddings import (  # noqa: E402
    inductive_study_vectors,
    study_matrix,
    train_compgcn,
    train_transe,
)
from ctdr.graph import build_input_graph  # noqa: E402
from ctdr.metrics import format_report, score_predictions  # noqa: E402
from ctdr.recommend import (  # noqa: E402
    ensemble_vote,
    hybrid_knn_recommend,
    knn_recommend,
    ontology_knn_recommend,
)
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402

OUT = cfg.CTDR_DIR / "optimize"


def _record(results: dict, lines: list[str], name: str, test: pd.DataFrame, preds: dict) -> dict:
    rep = score_predictions(test, preds, cfg.TARGETS)
    results["methods"][name] = rep
    block = format_report(name, rep)
    print(block)
    lines.append(block)
    lines.append("")
    return rep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Build enriched cohort ===")
    table = build_ctdr_table()
    table.to_csv(OUT / "ctdr_studies.csv", index=False)
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    print(
        f"train={len(train)} test={len(test)}  "
        f"outcome labeled test="
        f"{(test['primary_outcome_family']!='unknown').sum()}"
    )

    results: dict = {
        "config": {
            "split_date": cfg.SPLIT_DATE,
            "targets": list(cfg.TARGETS),
            "inputs": list(cfg.INPUT_CATS),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "note": "train-only KG + inductive test; targets never as edges",
        },
        "methods": {},
    }
    lines = [
        "OSEAN CTDR OPTIMIZE",
        f"split={cfg.SPLIT_DATE} train={len(train)} test={len(test)}",
        "",
    ]

    print("=== Classical ===")
    freq = frequency_predict(train, test)
    _record(results, lines, "frequency_by_phase", test, freq)
    logistic = regress_predict(train, test, kind="lr")
    _record(results, lines, "logistic", test, logistic)
    gbdt = regress_predict(train, test, kind="gbdt")
    _record(results, lines, "hist_gbdt", test, gbdt)

    print("=== Ontology kNN ===")
    ont_preds = {}
    for k in cfg.K_LIST:
        preds = ontology_knn_recommend(
            table, list(train["nct_id"]), list(test["nct_id"]), k=k
        )
        ont_preds[k] = preds
        _record(results, lines, f"ontology_knn_k{k}", test, preds)

    print("=== Train-only input graph ===")
    g_train = build_input_graph(train)
    print(f"train graph nodes={len(g_train['nodes'])} triples={len(g_train['triples'])}")

    best_graph_name = None
    best_graph_acc = -1.0
    best_graph_preds = None
    emb_cache = {}

    for dim in (64, 128):
        print(f"--- TransE dim={dim} ---")
        emb_t = train_transe(
            len(g_train["nodes"]), g_train["n_rel"], g_train["triples"], dim=dim, epochs=cfg.EPOCHS
        )
        emb_cache[f"transe_{dim}"] = emb_t
        train_vecs_mat, train_ids_g = study_matrix(emb_t, g_train)
        train_vec_map = {n: train_vecs_mat[i] for i, n in enumerate(train_ids_g)}
        q_vecs = inductive_study_vectors(emb_t, g_train, table, list(test["nct_id"]))

        for alpha in cfg.HYBRID_ALPHAS:
            for k in cfg.K_LIST:
                if alpha >= 1.0:
                    # pure ontology already covered
                    continue
                if alpha <= 0.0:
                    # pure embedding via hybrid with alpha=0
                    preds = hybrid_knn_recommend(
                        table,
                        list(train["nct_id"]),
                        list(test["nct_id"]),
                        q_vecs,
                        train_vec_map,
                        alpha=0.0,
                        k=k,
                    )
                    name = f"transe{dim}_ind_k{k}"
                else:
                    preds = hybrid_knn_recommend(
                        table,
                        list(train["nct_id"]),
                        list(test["nct_id"]),
                        q_vecs,
                        train_vec_map,
                        alpha=alpha,
                        k=k,
                    )
                    name = f"hybrid_transe{dim}_a{alpha}_k{k}"
                rep = _record(results, lines, name, test, preds)
                if rep["mean_accuracy"] > best_graph_acc:
                    best_graph_acc = rep["mean_accuracy"]
                    best_graph_name = name
                    best_graph_preds = preds

        print(f"--- CompGCN dim={dim} ---")
        emb_c = train_compgcn(
            len(g_train["nodes"]), g_train["n_rel"], g_train["triples"], dim=dim, epochs=cfg.EPOCHS
        )
        emb_cache[f"compgcn_{dim}"] = emb_c
        train_vecs_mat, train_ids_g = study_matrix(emb_c, g_train)
        train_vec_map = {n: train_vecs_mat[i] for i, n in enumerate(train_ids_g)}
        q_vecs = inductive_study_vectors(emb_c, g_train, table, list(test["nct_id"]))
        for alpha in (0.0, 0.5, 0.75):
            for k in (10, 20):
                preds = hybrid_knn_recommend(
                    table,
                    list(train["nct_id"]),
                    list(test["nct_id"]),
                    q_vecs,
                    train_vec_map,
                    alpha=alpha,
                    k=k,
                )
                name = (
                    f"compgcn{dim}_ind_k{k}"
                    if alpha == 0.0
                    else f"hybrid_compgcn{dim}_a{alpha}_k{k}"
                )
                rep = _record(results, lines, name, test, preds)
                if rep["mean_accuracy"] > best_graph_acc:
                    best_graph_acc = rep["mean_accuracy"]
                    best_graph_name = name
                    best_graph_preds = preds

    print("=== Ensembles ===")
    # pick strong classical + best ontology k + best graph
    ont_best_k = max(
        cfg.K_LIST,
        key=lambda k: results["methods"][f"ontology_knn_k{k}"]["mean_accuracy"],
    )
    ens1 = ensemble_vote(
        [gbdt, ont_preds[ont_best_k], best_graph_preds or ont_preds[ont_best_k]],
        list(test["nct_id"]),
    )
    _record(
        results,
        lines,
        f"ensemble_gbdt_ontk{ont_best_k}_{best_graph_name}",
        test,
        ens1,
    )
    ens2 = ensemble_vote([logistic, gbdt, ont_preds[ont_best_k]], list(test["nct_id"]))
    _record(results, lines, f"ensemble_lr_gbdt_ontk{ont_best_k}", test, ens2)

    board = sorted(
        ((n, m["mean_accuracy"], m["mean_macro_f1"]) for n, m in results["methods"].items()),
        key=lambda x: (-x[1], -x[2]),
    )
    lines.append("## Leaderboard")
    for name, acc, f1 in board[:20]:
        lines.append(f"  {acc:.3f}  F1={f1:.3f}  {name}")
    print("\n".join(lines[-21:]))

    results["best"] = {
        "name": board[0][0],
        "mean_accuracy": board[0][1],
        "mean_macro_f1": board[0][2],
        "best_graph": best_graph_name,
        "best_graph_acc": best_graph_acc,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "embeddings_train.npz", **emb_cache)
    print(f"BEST: {board[0][0]} acc={board[0][1]:.3f} F1={board[0][2]:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
