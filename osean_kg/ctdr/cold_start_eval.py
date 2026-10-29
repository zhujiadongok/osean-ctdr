"""Cold-start slice: rare Material VO in train. Does GNN win there?"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.push08_lock import (  # noqa: E402
    COARSE,
    _mean,
    add_coarse,
    build_ctdr_table,
    classical_predict,
    frequency_predict,
    ontology_knn,
    train_rgcn,
)


def main() -> None:
    table = add_coarse(build_ctdr_table())
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    vc = train["arch_vo"].value_counts()
    for thr in (2, 5, 10):
        rare = set(vc[vc <= thr].index)
        cold = test[test["arch_vo"].isin(rare)].reset_index(drop=True)
        print(f"threshold<={thr} cold n={len(cold)}", flush=True)
        if len(cold) < 15:
            continue
        both = pd.concat([train, cold], ignore_index=True)
        preds = {
            "frequency_by_phase": frequency_predict(train, cold),
            "logistic": classical_predict(train, cold, "lr"),
            "hist_gbdt": classical_predict(train, cold, "gbdt"),
            "ontology_knn_k20": ontology_knn(train, cold, 20),
            "rgcn_rich": train_rgcn(train, cold, both, dim=128, epochs=400, lr=1e-2),
        }
        for n, p in preds.items():
            s = _mean(score_predictions(cold, p, COARSE), COARSE)
            print(f"  COLD {s['mean_accuracy']:.4f}  {n}", flush=True)
        # keep going for larger thresholds too


if __name__ == "__main__":
    main()
