"""Add BM25 + TF-IDF baselines into optimize metrics (advisor B2/B3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.metrics import format_report, score_predictions  # noqa: E402
from ctdr.text_baselines import bm25_recommend, tfidf_recommend  # noqa: E402

OPT = cfg.CTDR_DIR / "optimize"
STRUCT = ("allocation", "masking", "intervention_model", "enrollment_bucket")


def structure_mean(rep: dict) -> tuple[float, float]:
    acc = [rep["targets"][t]["accuracy"] for t in STRUCT]
    f1 = [rep["targets"][t]["macro_f1"] for t in STRUCT]
    return float(sum(acc) / 4), float(sum(f1) / 4)


def main() -> None:
    studies = pd.read_csv(OPT / "ctdr_studies.csv", dtype=str).fillna("")
    train = studies[studies["split"] == "train"]
    test = studies[studies["split"] == "test"]
    metrics_path = OPT / "metrics.json"
    results = json.loads(metrics_path.read_text(encoding="utf-8"))

    lines = ["## Text baselines (advisor B2/B3)", ""]
    for k in (5, 10, 20):
        for name, fn in (("bm25", bm25_recommend), ("tfidf", tfidf_recommend)):
            preds = fn(train, test, k=k)
            key = f"{name}_knn_k{k}"
            rep = score_predictions(test, preds, cfg.TARGETS)
            s_acc, s_f1 = structure_mean(rep)
            rep["structure_mean_accuracy"] = s_acc
            rep["structure_mean_macro_f1"] = s_f1
            results["methods"][key] = rep
            block = format_report(key, rep)
            print(block)
            print(f"  structure_mean_acc={s_acc:.3f} F1={s_f1:.3f}")
            lines.append(block)
            lines.append(f"  structure_mean_acc={s_acc:.3f} F1={s_f1:.3f}")
            lines.append("")

    # refresh structure leaderboard including text methods
    board = []
    for name, m in results["methods"].items():
        if "structure_mean_accuracy" in m:
            board.append((m["structure_mean_accuracy"], m.get("structure_mean_macro_f1", 0), name))
        else:
            accs = [m["targets"][t]["accuracy"] for t in STRUCT if t in m.get("targets", {})]
            f1s = [m["targets"][t]["macro_f1"] for t in STRUCT if t in m.get("targets", {})]
            if len(accs) == 4:
                board.append((sum(accs) / 4, sum(f1s) / 4, name))
    board.sort(reverse=True)
    lines.append("## STRUCTURE-ONLY leaderboard (with text baselines)")
    for a, f, name in board[:25]:
        lines.append(f"  {a:.3f}  F1={f:.3f}  {name}")
        print(f"  {a:.3f}  F1={f:.3f}  {name}")

    results["best_structure"] = {
        "name": board[0][2],
        "structure_mean_acc": board[0][0],
        "structure_mean_macro_f1": board[0][1],
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OPT / "best_config.json").write_text(
        json.dumps(
            {
                "best_structure_method": board[0][2],
                "structure_mean_acc": board[0][0],
                "structure_mean_macro_f1": board[0][1],
                "structure_targets": list(STRUCT),
                "text_baselines": "bm25 + tfidf (B2/B3)",
                "note": "primary_outcome_family reported separately; chrono-test n small",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rep_path = OPT / "report.txt"
    prev = rep_path.read_text(encoding="utf-8") if rep_path.exists() else ""
    rep_path.write_text(prev + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("BEST structure", board[0])
    print("Wrote", OPT)


if __name__ == "__main__":
    main()
