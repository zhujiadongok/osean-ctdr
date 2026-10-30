"""Compare paper OntSim vs improved variants (chrono Setting A + pooled unseen).

Does not overwrite paper artifacts. Writes:
  ckpts/sea_ml_v14/ctdr/optimize/ontsim_improve/ontsim_improve.json

  python -u -m ctdr.eval_ontsim_improve
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
from ctdr.eval_graph_value import (  # noqa: E402
    _bootstrap_delta,
    _eligible_ontology_connected,
    _score_bundle,
)
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import SimMode, ontology_knn_recommend  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.vo_graph_sim import ont_sim, ont_sim_ic_sep, ont_sim_sep  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "ontsim_improve"

MODES: list[tuple[str, SimMode]] = [
    ("exact", "exact"),
    ("ontsim_union_role_paper", "graph"),
    ("ontsim_subclass_paper", "graph_subclass"),
    ("ontsim_sep_role", "graph_sep"),
    ("ontsim_ic", "graph_ic"),
    ("ontsim_ic_sep", "graph_ic_sep"),
    ("ontsim_gated_sep", "graph_gated_sep"),
    ("ontsim_gated_ic_sep", "graph_gated_ic_sep"),
]


def _field_mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _knn(both: pd.DataFrame, train_ids: list, test_ids: list, mode: SimMode) -> dict:
    return ontology_knn_recommend(both, train_ids, test_ids, STRUCT, k=20, mode=mode)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    tr_ids, te_ids = list(train["nct_id"]), list(test["nct_id"])
    print(f"chrono train={len(train)} test={len(test)}", flush=True)

    # toy sanity on the paper example pair
    a, b = "VO_0005198", "VO_0004908"
    sanity = {
        "pair": [a, b],
        "paper_union_role": ont_sim(a, b, use_roles=True),
        "paper_subclass": ont_sim(a, b, use_roles=False),
        "sep_role": ont_sim_sep(a, b),
        "ic_sep": ont_sim_ic_sep(a, b),
    }
    print("sanity", sanity, flush=True)

    chrono: dict[str, dict] = {}
    chrono_preds: dict[str, dict] = {}
    for name, mode in MODES:
        print(f"chrono {name}…", flush=True)
        preds = _knn(both, tr_ids, te_ids, mode)
        chrono_preds[name] = preds
        chrono[name] = _field_mean(score_predictions(test, preds, STRUCT))
        print(
            f"  {chrono[name]['mean_accuracy']:.4f}  F1 {chrono[name]['mean_macro_f1']:.4f}",
            flush=True,
        )
    for name in ["frequency_by_phase", "hist_gbdt", "logistic"]:
        if name == "frequency_by_phase":
            preds = frequency_predict(train, test, STRUCT)
        elif name == "hist_gbdt":
            preds = regress_predict(train, test, "gbdt", STRUCT)
        else:
            preds = regress_predict(train, test, "lr", STRUCT)
        chrono[name] = _field_mean(score_predictions(test, preds, STRUCT))

    boot = {}
    for name in [
        "ontsim_union_role_paper",
        "ontsim_sep_role",
        "ontsim_ic_sep",
        "ontsim_gated_sep",
        "ontsim_gated_ic_sep",
    ]:
        boot[f"{name}_minus_exact"] = _bootstrap_delta(
            test, chrono_preds[name], chrono_preds["exact"]
        )

    # pooled unseen (same eligibility as paper)
    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)
    print(f"pooled unseen train={len(train_u)} test={len(test_u)} classes={len(eligible)}", flush=True)
    pooled: dict[str, dict] = {}
    pooled_preds: dict[str, dict] = {}
    for name, mode in MODES:
        print(f"pooled {name}…", flush=True)
        preds = _knn(both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), mode)
        pooled_preds[name] = preds
        pooled[name] = _score_bundle(test_u, preds)
        print(
            f"  trial={pooled[name]['trial_mean_accuracy']:.4f} "
            f"field={pooled[name]['field_mean_accuracy']:.4f}",
            flush=True,
        )
    pooled["frequency_by_phase"] = _score_bundle(
        test_u, frequency_predict(train_u, test_u, STRUCT)
    )
    pooled["hist_gbdt"] = _score_bundle(
        test_u, regress_predict(train_u, test_u, "gbdt", STRUCT)
    )

    boot_p = {}
    for name in [
        "ontsim_union_role_paper",
        "ontsim_sep_role",
        "ontsim_ic_sep",
        "ontsim_gated_sep",
        "ontsim_gated_ic_sep",
    ]:
        boot_p[f"{name}_minus_exact"] = _bootstrap_delta(
            test_u, pooled_preds[name], pooled_preds["exact"]
        )

    payload = {
        "note": (
            "Improved OntSim variants vs paper baseline. "
            "Paper numbers remain those from onto_graph_knn / graph_value."
        ),
        "sanity_pair_jaccard": sanity,
        "chrono_setting_a": chrono,
        "chrono_bootstrap_vs_exact": boot,
        "pooled_unseen": {
            "n_train": int(len(train_u)),
            "n_test": int(len(test_u)),
            "n_classes": int(len(eligible)),
            "scores": {
                n: {
                    "trial_mean_accuracy": s["trial_mean_accuracy"],
                    "field_mean_accuracy": s["field_mean_accuracy"],
                }
                for n, s in pooled.items()
            },
            "bootstrap_vs_exact": boot_p,
        },
    }
    (OUT / "ontsim_improve.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'ontsim_improve.json'}", flush=True)

    print("\n=== CHRONO field-mean ===", flush=True)
    for n, s in sorted(chrono.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  {s['mean_accuracy']:.4f}  {n}", flush=True)
    print("\n=== POOLED trial-mean ===", flush=True)
    for n, s in sorted(
        pooled.items(),
        key=lambda kv: -kv[1].get("trial_mean_accuracy", kv[1].get("mean_accuracy", 0)),
    ):
        tm = s.get("trial_mean_accuracy", s.get("mean_accuracy", 0))
        print(f"  {tm:.4f}  {n}", flush=True)


if __name__ == "__main__":
    main()
