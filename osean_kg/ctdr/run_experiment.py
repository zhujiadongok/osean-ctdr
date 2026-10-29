"""End-to-end CTDR experiment: build table → graph → baselines → embeddings → metrics.

  python -m ctdr.run_experiment
  # or: python osean_kg/ctdr/run_experiment.py
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
from ctdr.embeddings import study_matrix, train_compgcn, train_transe  # noqa: E402
from ctdr.graph import build_input_graph  # noqa: E402
from ctdr.metrics import format_report, score_predictions  # noqa: E402
from ctdr.recommend import knn_recommend, ontology_knn_recommend  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402


def main() -> None:
    cfg.CTDR_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 1. Build cohort (chrono split, mapped interventional) ===")
    table = build_ctdr_table()
    table.to_csv(cfg.CTDR_DIR / "ctdr_studies.csv", index=False)
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train)}  test={len(test)}  split_date={cfg.SPLIT_DATE}")
    print(f"targets={cfg.TARGETS}")

    results: dict = {
        "config": {
            "split_date": cfg.SPLIT_DATE,
            "dim": cfg.DIM,
            "epochs": cfg.EPOCHS,
            "k_list": list(cfg.K_LIST),
            "targets": list(cfg.TARGETS),
            "inputs": list(cfg.INPUT_CATS),
            "n_train": len(train),
            "n_test": len(test),
        },
        "methods": {},
    }
    lines = [
        "OSEAN CTDR — design variable recommendation",
        f"SEA-CDM v1.4 + vo.owl | chrono split at {cfg.SPLIT_DATE}",
        f"train={len(train)} test={len(test)}",
        "Targets never used as graph edges. Neighbors = train only.",
        "",
    ]

    print("=== 2. Frequency + regression baselines ===")
    for name, preds in (
        ("frequency_by_phase", frequency_predict(train, test)),
        ("logistic", regress_predict(train, test, kind="lr")),
        ("hist_gbdt", regress_predict(train, test, kind="gbdt")),
    ):
        rep = score_predictions(test, preds, cfg.TARGETS)
        results["methods"][name] = rep
        block = format_report(name, rep)
        print(block)
        lines.append(block)
        lines.append("")

    print("=== 3. Ontology-weighted kNN ===")
    for k in cfg.K_LIST:
        preds = ontology_knn_recommend(
            table, list(train["nct_id"]), list(test["nct_id"]), k=k
        )
        name = f"ontology_knn_k{k}"
        rep = score_predictions(test, preds, cfg.TARGETS)
        results["methods"][name] = rep
        block = format_report(name, rep)
        print(block)
        lines.append(block)
        lines.append("")

    print("=== 4. Input-only graph + TransE / CompGCN ===")
    # Transductive on INPUT edges only (all studies); labels from train neighbors only.
    graph = build_input_graph(table)
    print(
        f"graph nodes={len(graph['nodes'])} triples={len(graph['triples'])} "
        f"studies={len(graph['study_ids'])}"
    )
    (cfg.CTDR_DIR / "graph_meta.json").write_text(
        json.dumps(
            {
                "n_nodes": len(graph["nodes"]),
                "n_triples": len(graph["triples"]),
                "n_studies": len(graph["study_ids"]),
                "n_rel": graph["n_rel"],
                "note": "No allocation/masking/model/enrollment edges",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training TransE…")
    emb_t = train_transe(len(graph["nodes"]), graph["n_rel"], graph["triples"])
    print("Training CompGCN…")
    emb_c = train_compgcn(len(graph["nodes"]), graph["n_rel"], graph["triples"])
    np.savez_compressed(cfg.CTDR_DIR / "embeddings.npz", transe=emb_t, compgcn=emb_c)

    train_ids = list(train["nct_id"])
    test_ids = list(test["nct_id"])
    for tag, emb in (("transe", emb_t), ("compgcn", emb_c)):
        vecs, ids = study_matrix(emb, graph)
        id_to_i = {n: i for i, n in enumerate(ids)}
        for k in cfg.K_LIST:
            preds = knn_recommend(test_ids, train_ids, vecs, id_to_i, table, k=k)
            name = f"{tag}_knn_k{k}"
            rep = score_predictions(test, preds, cfg.TARGETS)
            results["methods"][name] = rep
            block = format_report(name, rep)
            print(block)
            lines.append(block)
            lines.append("")

    # leaderboard
    board = sorted(
        (
            (name, m["mean_accuracy"], m["mean_macro_f1"])
            for name, m in results["methods"].items()
        ),
        key=lambda x: (-x[1], -x[2]),
    )
    lines.append("## Leaderboard (by mean accuracy)")
    for name, acc, f1 in board:
        lines.append(f"  {acc:.3f}  F1={f1:.3f}  {name}")
    print("\n".join(lines[-len(board) - 1 :]))

    (cfg.CTDR_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (cfg.CTDR_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {cfg.CTDR_DIR}")


if __name__ == "__main__":
    main()
