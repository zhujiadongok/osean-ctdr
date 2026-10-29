"""Setting B: same models/protocol as Setting A Table 1, coarse Y.

Coarse targets (unknown gold preserved, excluded from train loss / eval):
  allocation           unchanged
  masking_binary       OPEN / BLINDED
  intervention_binary  PARALLEL / OTHER
  enrollment_binary    SMALL (<150) / LARGE (>=150)

Models match Setting A: frequency, exact kNN@20, OntSim subclass / +role,
logistic, HistGBDT. Shared missing-ID rules via recommend.py.

  python -u -m ctdr.eval_setting_b
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
from ctdr.push08_lock import add_coarse  # noqa: E402
from ctdr.recommend import ontology_knn_recommend  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402

VAL_START = "2019-01-01"
COARSE = (
    "allocation",
    "masking_binary",
    "intervention_binary",
    "enrollment_binary",
)
OUT = cfg.CTDR_DIR / "optimize" / "setting_b"
N_BOOT = 1500


def _mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in COARSE if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in COARSE if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_target": {t: rep["targets"][t] for t in COARSE},
    }


def _trial_means(gold: pd.DataFrame, preds: dict) -> np.ndarray:
    g = gold.set_index("nct_id")
    rows = []
    for n in gold["nct_id"]:
        hits = []
        for t in COARSE:
            yt = str(g.loc[n, t])
            if yt in {"unknown", "", "nan", "None", "none"}:
                continue
            hits.append(float(str(preds[n].get(t, "unknown")) == yt))
        rows.append(float(np.mean(hits)) if hits else np.nan)
    return np.asarray(rows, dtype=float)


def _bootstrap_delta(gold: pd.DataFrame, a: dict, b: dict, n_boot: int = N_BOOT) -> dict:
    ha = _trial_means(gold, a)
    hb = _trial_means(gold, b)
    mask = np.isfinite(ha) & np.isfinite(hb)
    ha, hb = ha[mask], hb[mask]
    diff = ha - hb
    rng = np.random.default_rng(7)
    boots = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(n_boot)]
    arr = np.sort(np.asarray(boots))
    return {
        "mean_diff_trial": float(diff.mean()),
        "ci95_lo": float(arr[int(0.025 * n_boot)]),
        "ci95_hi": float(arr[int(0.975 * n_boot)]),
        "n_trials": int(len(diff)),
        "n_boot": n_boot,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = add_coarse(build_ctdr_table())
    table["start_date_parsed"] = pd.to_datetime(table["start_date"], errors="coerce")
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    fit = train[train["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train[train["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    print(f"Setting B fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    both_te = pd.concat([train, test], ignore_index=True)
    tr_ids, te_ids = list(train["nct_id"]), list(test["nct_id"])

    print("classical…", flush=True)
    preds = {
        "frequency_by_phase": frequency_predict(train, test, COARSE),
        "logistic": regress_predict(train, test, "lr", COARSE),
        "hist_gbdt": regress_predict(train, test, "gbdt", COARSE),
    }
    print("kNN exact / OntSim…", flush=True)
    preds["exact_knn_k20"] = ontology_knn_recommend(
        both_te, tr_ids, te_ids, COARSE, k=20, mode="exact"
    )
    preds["ontsim_subclass_k20"] = ontology_knn_recommend(
        both_te, tr_ids, te_ids, COARSE, k=20, mode="graph_subclass"
    )
    preds["ontsim_subclass_role_k20"] = ontology_knn_recommend(
        both_te, tr_ids, te_ids, COARSE, k=20, mode="graph"
    )

    scores = {n: _mean(score_predictions(test, p, COARSE)) for n, p in preds.items()}
    for n, s in sorted(scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(
            f"  TEST {s['mean_accuracy']:.4f}  F1 {s['mean_macro_f1']:.4f}  {n}",
            flush=True,
        )

    boot = {
        "ontsim_role_minus_exact": _bootstrap_delta(
            test, preds["ontsim_subclass_role_k20"], preds["exact_knn_k20"]
        ),
        "ontsim_subclass_minus_exact": _bootstrap_delta(
            test, preds["ontsim_subclass_k20"], preds["exact_knn_k20"]
        ),
    }
    print("bootstrap OntSim+role − exact:", boot["ontsim_role_minus_exact"], flush=True)

    # Validation classical (informational; same chrono lock as A)
    val_scores = {
        "frequency_by_phase": _mean(
            score_predictions(val, frequency_predict(fit, val, COARSE), COARSE)
        ),
        "logistic": _mean(score_predictions(val, regress_predict(fit, val, "lr", COARSE), COARSE)),
        "hist_gbdt": _mean(
            score_predictions(val, regress_predict(fit, val, "gbdt", COARSE), COARSE)
        ),
    }

    payload = {
        "protocol": {
            "name": "Setting B (coarse labels)",
            "targets": list(COARSE),
            "split": "train start <2022; test >=2022; inner fit <2019 / val 2019-2021",
            "models": "Same families as Setting A Table 1; shared missing-ID / OntSim weights",
            "note": (
                "Post-hoc coarsening of Y; not a lift from Setting A. "
                "Replaces historical push08 classical numbers for paper Table."
            ),
            "n_fit": int(len(fit)),
            "n_val": int(len(val)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "test_scores": scores,
        "val_classical": val_scores,
        "bootstrap_ontsim_vs_exact": boot,
        "legacy_push08": {
            "path": "optimize/push08/push08.json",
            "note": "Predates missing-ID alignment; keep for R-GCN only if not re-run",
        },
    }
    (OUT / "setting_b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'setting_b.json'}", flush=True)


if __name__ == "__main__":
    main()
