"""Evaluation metrics for CTDR design-variable recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def score_predictions(
    gold: pd.DataFrame,
    preds: dict[str, dict[str, str]],
    targets: tuple[str, ...],
    *,
    skip_unknown_gold: bool = True,
) -> dict:
    g = gold.set_index("nct_id")
    report: dict = {"n": 0, "targets": {}}
    ncts = [n for n in preds if n in g.index]
    report["n"] = len(ncts)
    for t in targets:
        pairs = []
        for n in ncts:
            yt = str(g.loc[n, t])
            if skip_unknown_gold and yt in {"unknown", "", "nan"}:
                continue
            pairs.append((yt, str(preds[n].get(t, "unknown"))))
        if not pairs:
            report["targets"][t] = {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "n_classes_gold": 0,
                "n_eval": 0,
            }
            continue
        y_true, y_pred = zip(*pairs)
        report["targets"][t] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n_classes_gold": int(len(set(y_true))),
            "n_eval": int(len(y_true)),
        }
    accs = [report["targets"][t]["accuracy"] for t in targets if report["targets"][t].get("n_eval", 0) > 0]
    f1s = [report["targets"][t]["macro_f1"] for t in targets if report["targets"][t].get("n_eval", 0) > 0]
    report["mean_accuracy"] = float(np.mean(accs)) if accs else 0.0
    report["mean_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return report


def format_report(name: str, report: dict) -> str:
    lines = [
        f"## {name}",
        f"n_test={report['n']}  mean_acc={report['mean_accuracy']:.3f}  "
        f"mean_macro_f1={report['mean_macro_f1']:.3f}",
    ]
    for t, m in report["targets"].items():
        lines.append(
            f"  {t:22}  acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
            f"classes={m['n_classes_gold']}"
        )
    return "\n".join(lines)
