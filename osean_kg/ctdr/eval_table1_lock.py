"""Refresh Setting A Table 1 on the current CTDR table (acc + macro-F1).

Models: frequency, exact kNN@20, OntSim subclass / +role, gated OntSim,
logistic, HistGBDT. Writes optimize/table1_lock.json.
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
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import ontology_knn_recommend, ontology_knn_recommend_cfg  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.vo_graph_sim import OntSimConfig  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "table1_lock.json"


def _mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_target": {t: rep["targets"].get(t, {}) for t in STRUCT},
    }


def main() -> None:
    df = build_ctdr_table()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    tr, te = list(train["nct_id"]), list(test["nct_id"])
    print(f"Table1 train={len(train)} test={len(test)}", flush=True)

    gate = OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True)
    print("classical…", flush=True)
    preds = {
        "frequency_by_phase": frequency_predict(train, test, STRUCT),
        "logistic": regress_predict(train, test, "lr", STRUCT),
        "hist_gbdt": regress_predict(train, test, "gbdt", STRUCT),
    }
    print("kNN…", flush=True)
    preds["exact_knn_k20"] = ontology_knn_recommend(
        both, tr, te, STRUCT, k=20, mode="exact"
    )
    preds["ontsim_subclass_k20"] = ontology_knn_recommend(
        both, tr, te, STRUCT, k=20, mode="graph_subclass"
    )
    preds["ontsim_subclass_role_k20"] = ontology_knn_recommend(
        both, tr, te, STRUCT, k=20, mode="graph"
    )
    preds["gated_ic_sib_k20"] = ontology_knn_recommend_cfg(
        both, tr, te, STRUCT, k=20, sim_cfg=gate
    )

    scores = {n: _mean(score_predictions(test, p, STRUCT)) for n, p in preds.items()}
    for n, s in sorted(scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(
            f"  {s['mean_accuracy']:.4f}  F1 {s['mean_macro_f1']:.4f}  {n}",
            flush=True,
        )

    corr_n = 0
    if Path(cfg.VO_CORRECTIONS).is_file():
        corr = pd.read_csv(cfg.VO_CORRECTIONS, dtype=str)
        corr_n = int(len(set(df["nct_id"]) & set(corr["nct_id"].astype(str).str.upper())))

    payload = {
        "protocol": "Setting A Table 1 lock after primary-investigational mapping remediation",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_corrections_in_cohort": corr_n,
        "test_scores": scores,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
