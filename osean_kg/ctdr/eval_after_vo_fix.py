"""Re-score exact / primary OntSim / IC-sib gated / HistGBDT / frequency after primary-investigational mapping."""

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
    _trial_mean_accuracy,
)
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
OUT = cfg.CTDR_DIR / "optimize" / "mapping_audit"

# Exploratory IC-weighted subclass-and-sibling retrieval (gate = exact fallback when
# query Material already appears in training). Same field weights / k as primary ablation.
IC_SIB = OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True)


def _field(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {"mean_accuracy": float(np.mean(accs)) if accs else 0.0}


def _lomo_summarize(rows: list[dict], keys: list[str]) -> dict:
    n_tot = sum(x["n"] for x in rows)
    out: dict[str, float] = {}
    for k in keys:
        vals = [x[k] for x in rows]
        out[f"{k}_class"] = float(np.mean(vals))
        out[f"{k}_trial"] = float(sum(x[k] * x["n"] for x in rows) / n_tot)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    corr = pd.read_csv(cfg.VO_CORRECTIONS, dtype=str)
    present = set(df["nct_id"]) & set(corr["nct_id"].str.upper())
    print(f"cohort={len(df)} corrections_in_cohort={len(present)}", flush=True)

    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)
    print(f"chrono train={len(train)} test={len(test)}", flush=True)
    print(f"pooled train={len(train_u)} test={len(test_u)} classes={len(eligible)}", flush=True)

    modes = {
        "exact": lambda b, tr, te: ontology_knn_recommend(
            b, list(tr["nct_id"]), list(te["nct_id"]), STRUCT, k=20, mode="exact"
        ),
        "ontsim_role": lambda b, tr, te: ontology_knn_recommend(
            b, list(tr["nct_id"]), list(te["nct_id"]), STRUCT, k=20, mode="graph"
        ),
        "gated_ic_sib": lambda b, tr, te: ontology_knn_recommend_cfg(
            b, list(tr["nct_id"]), list(te["nct_id"]), STRUCT, k=20, sim_cfg=IC_SIB
        ),
        "frequency_by_phase": lambda b, tr, te: frequency_predict(tr, te, STRUCT),
        "hist_gbdt": lambda b, tr, te: regress_predict(tr, te, "gbdt", STRUCT),
    }
    neighbor_names = ["exact", "ontsim_role", "gated_ic_sib"]
    all_names = neighbor_names + ["frequency_by_phase", "hist_gbdt"]

    chrono, pooled, preds_c, preds_p = {}, {}, {}, {}
    for name in all_names:
        fn = modes[name]
        print(f"eval {name}…", flush=True)
        pc = fn(both, train, test)
        pp = fn(both_u, train_u, test_u)
        chrono[name] = _field(score_predictions(test, pc, STRUCT))
        pooled[name] = {
            "trial_mean_accuracy": _score_bundle(test_u, pp)["trial_mean_accuracy"],
            "field_mean_accuracy": _score_bundle(test_u, pp)["field_mean_accuracy"],
        }
        preds_c[name] = pc
        preds_p[name] = pp
        print(
            f"  chrono={chrono[name]['mean_accuracy']:.4f} "
            f"pooled={pooled[name]['trial_mean_accuracy']:.4f}",
            flush=True,
        )

    lomo = []
    for m in eligible:
        te = df[df["arch_vo"].astype(str) == m].reset_index(drop=True)
        tr = df[df["arch_vo"].astype(str) != m].reset_index(drop=True)
        both_m = pd.concat([tr, te], ignore_index=True)
        row = {"vo": m, "n": int(len(te))}
        bits = []
        for name in all_names:
            preds = modes[name](both_m, tr, te)
            row[name] = _trial_mean_accuracy(te, preds)
            bits.append(f"{name}={row[name]:.3f}")
        lomo.append(row)
        print(f"  LOMO {m} n={len(te)} " + " ".join(bits), flush=True)

    lomo_sum = _lomo_summarize(lomo, all_names)
    lomo_sum["delta_role_minus_exact_class"] = (
        lomo_sum["ontsim_role_class"] - lomo_sum["exact_class"]
    )
    lomo_sum["delta_role_minus_exact_trial"] = (
        lomo_sum["ontsim_role_trial"] - lomo_sum["exact_trial"]
    )
    lomo_sum["delta_gated_minus_exact_class"] = (
        lomo_sum["gated_ic_sib_class"] - lomo_sum["exact_class"]
    )
    lomo_sum["delta_gated_minus_exact_trial"] = (
        lomo_sum["gated_ic_sib_trial"] - lomo_sum["exact_trial"]
    )

    payload = {
        "policy": "primary_investigational",
        "corrections_file": str(cfg.VO_CORRECTIONS),
        "n_corrections_in_cohort": len(present),
        "corrected_ncts": sorted(present),
        "methods": {
            "exact": "exact VO Material/platform match",
            "ontsim_role": "primary OntSim: subclass + role union Jaccard",
            "gated_ic_sib": (
                "exploratory IC-weighted subclass-and-sibling retrieval; "
                "exact fallback when query Material id is seen in training"
            ),
            "frequency_by_phase": "majority label among same-phase training trials",
            "hist_gbdt": "HistGradientBoosting on SEA-CDM/VO categorical columns",
        },
        "chrono": chrono,
        "pooled": pooled,
        "bootstrap_gated_minus_exact_chrono": _bootstrap_delta(
            test, preds_c["gated_ic_sib"], preds_c["exact"]
        ),
        "bootstrap_gated_minus_exact_pooled": _bootstrap_delta(
            test_u, preds_p["gated_ic_sib"], preds_p["exact"]
        ),
        "bootstrap_role_minus_exact_pooled": _bootstrap_delta(
            test_u, preds_p["ontsim_role"], preds_p["exact"]
        ),
        "bootstrap_role_minus_exact_chrono": _bootstrap_delta(
            test, preds_c["ontsim_role"], preds_c["exact"]
        ),
        "lomo_summary": lomo_sum,
        "lomo_per_class": lomo,
        "eligible_classes": eligible,
    }
    path = OUT / "after_vo_fix_eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"chrono": chrono, "pooled": pooled, "lomo": lomo_sum}, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
