"""Fair ablation: exact-match vs OntSim with matched Material/platform weights.

Primary weights: Material=0.5, Platform=1.0 for ALL modes.
Sensitivity: Material=1.0 for ALL modes.
Also compares graph (subclass+role) vs graph_subclass (subclass only).

  python -u -m ctdr.eval_onto_graph_knn
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
from ctdr.recommend import (  # noqa: E402
    W_MATERIAL,
    W_PLATFORM,
    ontology_knn_recommend,
)
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402

VAL_START = "2019-01-01"
STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "onto_graph_knn"
N_BOOT = 2000


def _mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(sum(accs) / len(accs)) if accs else 0.0,
        "mean_macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "per_target": {t: rep["targets"][t] for t in STRUCT},
    }


def _trial_means(gold: pd.DataFrame, preds: dict) -> np.ndarray:
    g = gold.set_index("nct_id")
    rows = []
    for n in gold["nct_id"]:
        hits = []
        for t in STRUCT:
            yt = str(g.loc[n, t])
            if yt in {"unknown", "", "nan"}:
                continue
            hits.append(float(str(preds[n].get(t, "unknown")) == yt))
        rows.append(float(np.mean(hits)) if hits else np.nan)
    return np.asarray(rows, dtype=float)


def _bootstrap_delta(gold: pd.DataFrame, a: dict, b: dict, n_boot: int = N_BOOT) -> dict:
    """NCT-paired bootstrap of mean(trial_acc_a - trial_acc_b)."""
    ha = _trial_means(gold, a)
    hb = _trial_means(gold, b)
    mask = np.isfinite(ha) & np.isfinite(hb)
    ha, hb = ha[mask], hb[mask]
    diff = ha - hb
    rng = np.random.default_rng(7)
    boots = []
    n = len(diff)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        boots.append(float(diff[ix].mean()))
    arr = np.sort(np.asarray(boots))
    return {
        "mean_diff_trial": float(diff.mean()),
        "ci95_lo": float(arr[int(0.025 * n_boot)]),
        "ci95_hi": float(arr[int(0.975 * n_boot)]),
        "n_trials": int(n),
        "n_boot": n_boot,
        "approx_extra_field_hits": float(diff.mean() * n * len(STRUCT)),
    }


def _run_block(fit, query, both, w_mat: float, w_plat: float, tag: str) -> tuple[dict, dict]:
    train_ids = list(fit["nct_id"])
    qids = list(query["nct_id"])
    preds = {
        f"exact_k20_{tag}": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=20, mode="exact",
            w_material=w_mat, w_platform=w_plat,
        ),
        f"graph_subclass_k20_{tag}": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=20, mode="graph_subclass",
            w_material=w_mat, w_platform=w_plat,
        ),
        f"graph_subclass_role_k20_{tag}": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=20, mode="graph",
            w_material=w_mat, w_platform=w_plat,
        ),
    }
    scores = {n: _mean(score_predictions(query, p, STRUCT)) for n, p in preds.items()}
    return preds, scores


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_ctdr_table()
    table["start_date_parsed"] = pd.to_datetime(table["start_date"], errors="coerce")
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    fit = train[train["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train[train["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    print(f"fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    both_val = pd.concat([fit, val], ignore_index=True)
    both_te = pd.concat([train, test], ignore_index=True)

    # Classical refs (unchanged)
    val_ref = {
        "frequency_by_phase": _mean(score_predictions(val, frequency_predict(fit, val, STRUCT), STRUCT)),
        "logistic": _mean(score_predictions(val, regress_predict(fit, val, "lr", STRUCT), STRUCT)),
        "hist_gbdt": _mean(score_predictions(val, regress_predict(fit, val, "gbdt", STRUCT), STRUCT)),
    }
    te_ref = {
        "frequency_by_phase": _mean(score_predictions(test, frequency_predict(train, test, STRUCT), STRUCT)),
        "logistic": _mean(score_predictions(test, regress_predict(train, test, "lr", STRUCT), STRUCT)),
        "hist_gbdt": _mean(score_predictions(test, regress_predict(train, test, "gbdt", STRUCT), STRUCT)),
    }

    # Primary fair weights
    print("\n=== Primary weights Material=0.5 Platform=1.0 ===", flush=True)
    _, val_p = _run_block(fit, val, both_val, 0.5, 1.0, "wM05")
    te_preds_p, te_p = _run_block(train, test, both_te, 0.5, 1.0, "wM05")
    for n, s in sorted({**val_ref, **val_p}.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  VAL  {s['mean_accuracy']:.4f}  {n}", flush=True)
    for n, s in sorted({**te_ref, **te_p}.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  TEST {s['mean_accuracy']:.4f}  {n}", flush=True)

    boot_graph_vs_exact = _bootstrap_delta(
        test,
        te_preds_p["graph_subclass_role_k20_wM05"],
        te_preds_p["exact_k20_wM05"],
    )
    boot_sub_vs_exact = _bootstrap_delta(
        test,
        te_preds_p["graph_subclass_k20_wM05"],
        te_preds_p["exact_k20_wM05"],
    )
    print("bootstrap graph+role − exact:", boot_graph_vs_exact, flush=True)
    print("bootstrap subclass-only − exact:", boot_sub_vs_exact, flush=True)

    # Sensitivity Material=1.0 both modes
    print("\n=== Sensitivity Material=1.0 Platform=1.0 ===", flush=True)
    _, val_s = _run_block(fit, val, both_val, 1.0, 1.0, "wM10")
    te_preds_s, te_s = _run_block(train, test, both_te, 1.0, 1.0, "wM10")
    for n, s in sorted(te_s.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  TEST {s['mean_accuracy']:.4f}  {n}", flush=True)
    boot_s = _bootstrap_delta(
        test,
        te_preds_s["graph_subclass_role_k20_wM10"],
        te_preds_s["exact_k20_wM10"],
    )
    print("bootstrap sens graph+role − exact:", boot_s, flush=True)

    payload = {
        "note": (
            "Fair ablation: exact vs OntSim use identical Material/platform weights. "
            "OntSim = Jaccard over relation-expanded neighborhoods "
            "(self + subclass superclasses; optional reachable roles)."
        ),
        "primary_weights": {"material": 0.5, "platform": 1.0, "disease": 1.0, "phase": 1.0, "purpose": 0.5},
        "sensitivity_weights": {"material": 1.0, "platform": 1.0},
        "val_ref": val_ref,
        "test_ref": te_ref,
        "val_primary": val_p,
        "test_primary": te_p,
        "val_sensitivity_wM1": val_s,
        "test_sensitivity_wM1": te_s,
        "bootstrap_graph_role_minus_exact_primary": boot_graph_vs_exact,
        "bootstrap_subclass_minus_exact_primary": boot_sub_vs_exact,
        "bootstrap_graph_role_minus_exact_wM1": boot_s,
        "defaults": {
            "W_MATERIAL": W_MATERIAL,
            "W_PLATFORM": W_PLATFORM,
            "recommend_default_mode": "graph",
        },
        "setting_b_ontology_knn_0.800": (
            "Historical push08 figure used exact-match ontology_knn before graph OntSim; "
            "do not relabel as ontology-graph without re-running push08 with mode=graph."
        ),
    }
    (OUT / "onto_graph_knn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "FAIR OntSim ABLATION (matched weights)",
        "Primary: Material=0.5 Platform=1.0",
        "",
        "## TEST primary",
    ]
    for n, s in sorted({**te_ref, **te_p}.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        lines.append(f"  {s['mean_accuracy']:.4f}  f1={s['mean_macro_f1']:.4f}  {n}")
    lines += [
        "",
        f"bootstrap graph+role - exact: {boot_graph_vs_exact['mean_diff_trial']:+.4f} "
        f"CI [{boot_graph_vs_exact['ci95_lo']:+.4f}, {boot_graph_vs_exact['ci95_hi']:+.4f}]",
        f"bootstrap subclass - exact: {boot_sub_vs_exact['mean_diff_trial']:+.4f} "
        f"CI [{boot_sub_vs_exact['ci95_lo']:+.4f}, {boot_sub_vs_exact['ci95_hi']:+.4f}]",
        "",
        "## TEST sensitivity Material=1.0",
    ]
    for n, s in sorted(te_s.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        lines.append(f"  {s['mean_accuracy']:.4f}  {n}")
    lines.append(
        f"bootstrap sens: {boot_s['mean_diff_trial']:+.4f} "
        f"CI [{boot_s['ci95_lo']:+.4f}, {boot_s['ci95_hi']:+.4f}]"
    )
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
