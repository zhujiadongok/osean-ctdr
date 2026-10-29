"""Lock CTDR model choice on an inner chronological split, then score the test set once.

Train studies start before 2022-01-01. Of those:
  fit  = start < 2019-01-01
  val  = 2019-01-01 <= start < 2022-01-01
The 2022+ test set is not used to choose a method or a per-target method.

  python -m ctdr.lock_eval
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
from ctdr.embeddings import inductive_study_vectors, study_matrix, train_compgcn  # noqa: E402
from ctdr.graph import build_input_graph  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import hybrid_knn_recommend, ontology_knn_recommend  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
VAL_START = "2019-01-01"
OUT = cfg.CTDR_DIR / "optimize" / "locked"
N_BOOT = 2000


def _struct(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "per_target": {t: rep["targets"][t] for t in STRUCT},
    }


def _candidates(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    both: pd.DataFrame,
    emb: np.ndarray | None,
    graph: dict | None,
) -> dict[str, dict]:
    train_ids = list(fit["nct_id"])
    qids = list(query["nct_id"])
    out = {
        "frequency_by_phase": frequency_predict(fit, query, STRUCT),
        "logistic": regress_predict(fit, query, "lr", STRUCT),
        "logistic_balanced": regress_predict(fit, query, "lr_bal", STRUCT),
        "hist_gbdt": regress_predict(fit, query, "gbdt", STRUCT),
        "hist_gbdt_balanced": regress_predict(fit, query, "gbdt_bal", STRUCT),
        # Explicit modes: graph = subclass+role OntSim; weights matched to exact.
        "ontology_knn_k10": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=10, mode="graph"
        ),
        "ontology_knn_k20": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=20, mode="graph"
        ),
        "feature_exact_knn_k20": ontology_knn_recommend(
            both, train_ids, qids, STRUCT, k=20, mode="exact"
        ),
    }
    if emb is not None and graph is not None:
        mat, ids = study_matrix(emb, graph)
        train_map = {n: mat[i] for i, n in enumerate(ids)}
        qvecs = inductive_study_vectors(emb, graph, both, qids)
        out["hybrid_compgcn128_a0.75_k10"] = hybrid_knn_recommend(
            both,
            train_ids,
            qids,
            qvecs,
            train_map,
            alpha=0.75,
            k=10,
            targets=STRUCT,
        )
    return out


def _train_hybrid_emb(fit: pd.DataFrame) -> tuple[np.ndarray, dict]:
    g = build_input_graph(fit)
    print(f"CompGCN-128 nodes={len(g['nodes'])} triples={len(g['triples'])}", flush=True)
    emb = train_compgcn(len(g["nodes"]), g["n_rel"], g["triples"], dim=128, epochs=cfg.EPOCHS)
    return emb, g


def _mixture(parts: dict[str, dict], choice: dict[str, str], qids: list[str]) -> dict:
    out: dict[str, dict[str, str]] = {}
    for q in qids:
        out[q] = {}
        for t, name in choice.items():
            out[q][t] = parts[name][q][t]
    return out


def _trial_hits(gold: pd.DataFrame, preds: dict, ncts: list[str]) -> np.ndarray:
    """Per-trial mean of four target hits. Shape (n,)."""
    g = gold.set_index("nct_id")
    rows = []
    for n in ncts:
        hits = []
        for t in STRUCT:
            yt = str(g.loc[n, t])
            if yt in {"unknown", "", "nan"}:
                continue
            hits.append(float(str(preds[n].get(t, "unknown")) == yt))
        rows.append(float(np.mean(hits)) if hits else np.nan)
    return np.asarray(rows, dtype=float)


def _bootstrap(gold: pd.DataFrame, a: dict, b: dict, ncts: list[str]) -> dict:
    ha = _trial_hits(gold, a, ncts)
    hb = _trial_hits(gold, b, ncts)
    mask = np.isfinite(ha) & np.isfinite(hb)
    ha, hb = ha[mask], hb[mask]
    rng = np.random.default_rng(cfg.SEED)
    diffs = []
    n = len(ha)
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, n)
        diffs.append(float(ha[ix].mean() - hb[ix].mean()))
    arr = np.sort(np.asarray(diffs))
    return {
        "mean_diff": float(ha.mean() - hb.mean()),
        "ci95_lo": float(arr[int(0.025 * N_BOOT)]),
        "ci95_hi": float(arr[int(0.975 * N_BOOT)]),
        "n_boot": N_BOOT,
        "n": int(n),
    }


def _by_phase(gold: pd.DataFrame, preds: dict) -> dict:
    out = {}
    for phase, g in gold.groupby("phase"):
        sub = {n: preds[n] for n in g["nct_id"] if n in preds}
        if len(sub) < 15:
            continue
        rep = _struct(score_predictions(g, sub, STRUCT))
        out[str(phase)] = {"n": int(len(sub)), **{k: rep[k] for k in ("mean_accuracy", "mean_macro_f1")}}
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

    print("=== validation candidates (fit graph only) ===", flush=True)
    emb_fit, g_fit = _train_hybrid_emb(fit)
    both_val = pd.concat([fit, val], ignore_index=True)
    val_preds = _candidates(fit, val, both_val, emb_fit, g_fit)
    val_scores = {name: _struct(score_predictions(val, pred, STRUCT)) for name, pred in val_preds.items()}
    for name, sc in sorted(val_scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  VAL {sc['mean_accuracy']:.4f} f1={sc['mean_macro_f1']:.4f}  {name}", flush=True)

    best_single = max(val_scores, key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]))
    per_target = {
        t: max(val_scores, key=lambda n: (val_scores[n]["per_target"][t]["accuracy"], val_scores[n]["per_target"][t]["macro_f1"]))
        for t in STRUCT
    }
    print("locked single:", best_single, flush=True)
    print("locked per target:", per_target, flush=True)

    print("=== refit on full train; score test once ===", flush=True)
    need = set(per_target.values()) | {best_single, "logistic", "frequency_by_phase"}
    emb_full = g_full = None
    if "hybrid_compgcn128_a0.75_k10" in need:
        emb_full, g_full = _train_hybrid_emb(train)
    both_te = pd.concat([train, test], ignore_index=True)
    test_preds = _candidates(
        train,
        test,
        both_te,
        emb_full,
        g_full,
    )
    # _candidates always builds the hybrid only if emb is present. If hybrid was not needed
    # it is absent; if it was needed, emb_full is set. Good.
    test_scores = {name: _struct(score_predictions(test, pred, STRUCT)) for name, pred in test_preds.items()}
    mix = _mixture(test_preds, per_target, list(test["nct_id"]))
    mix_score = _struct(score_predictions(test, mix, STRUCT))
    single_score = test_scores[best_single]

    qids = list(test["nct_id"])
    boot = {
        "single_minus_logistic": _bootstrap(test, test_preds[best_single], test_preds["logistic"], qids),
        "single_minus_frequency": _bootstrap(test, test_preds[best_single], test_preds["frequency_by_phase"], qids),
        "mixture_minus_logistic": _bootstrap(test, mix, test_preds["logistic"], qids),
        "mixture_minus_single": _bootstrap(test, mix, test_preds[best_single], qids),
    }
    phase = _by_phase(test, test_preds[best_single])

    # persist per-trial predictions for the locked single method
    rows = []
    gold = test.set_index("nct_id")
    for n in qids:
        row = {"nct_id": n, "phase": gold.loc[n, "phase"]}
        for t in STRUCT:
            row[f"{t}_gold"] = gold.loc[n, t]
            row[f"{t}_pred"] = test_preds[best_single][n][t]
            row[f"{t}_mix"] = mix[n][t]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "test_predictions.csv", index=False)

    payload = {
        "protocol": {
            "fit": f"start < {VAL_START}",
            "val": f"{VAL_START} <= start < {cfg.SPLIT_DATE}",
            "test": f"start >= {cfg.SPLIT_DATE}",
            "n_fit": int(len(fit)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "selection": "max structure mean accuracy on val; ties broken by macro-F1",
            "test_used_for_selection": False,
        },
        "val_scores": val_scores,
        "locked_single": best_single,
        "locked_per_target": per_target,
        "test_scores": test_scores,
        "test_locked_single": single_score,
        "test_per_target_mixture": mix_score,
        "bootstrap_trial_mean": boot,
        "test_by_phase_locked_single": phase,
    }
    (OUT / "locked_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "locked_single": best_single,
        "test_single": single_score["mean_accuracy"],
        "test_mixture": mix_score["mean_accuracy"],
        "per_target": per_target,
        "boot_single_minus_logistic": boot["single_minus_logistic"],
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
