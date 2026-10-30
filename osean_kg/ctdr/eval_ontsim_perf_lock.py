"""Lock top perf candidates on chrono + pooled (once)."""

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
from ctdr.recommend import (  # noqa: E402
    ontology_knn_recommend,
    ontology_knn_recommend_cfg,
    soft_ensemble,
)
from ctdr.regress import regress_predict  # noqa: E402
from ctdr.vo_graph_sim import OntSimConfig  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "ontsim_perf"


def _field(rep):
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)

    cands = []
    # baselines
    cands.append(("exact_k20", None, dict(w_disease=1, w_phase=1, w_purpose=0.5, w_platform=1, w_material=0.5, w_route=0), 20, "exact"))
    cands.append(
        (
            "paper_union_k20",
            OntSimConfig(role_mode="union"),
            dict(w_disease=1, w_phase=1, w_purpose=0.5, w_platform=1, w_material=0.5, w_route=0),
            20,
            "ont",
        )
    )
    cands.append(
        (
            "gated_ic_sib_k20",
            OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True),
            dict(w_disease=1, w_phase=1, w_purpose=0.5, w_platform=1, w_material=0.5, w_route=0),
            20,
            "ont",
        )
    )
    # best pure-ont from val (phase1 #1)
    cands.append(
        (
            "ont_ic_sib_ungated_tuned",
            OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=False),
            dict(w_disease=1.0, w_phase=1.0, w_purpose=1.5, w_platform=1.5, w_material=1.0, w_route=0.25),
            25,
            "ont",
        )
    )
    # best gated tuned
    cands.append(
        (
            "ont_gated_ic_sib_tuned",
            OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True),
            dict(w_disease=1.5, w_phase=1.0, w_purpose=1.5, w_platform=0.5, w_material=0.5, w_route=0.0),
            30,
            "ont",
        )
    )
    # gated resnik
    cands.append(
        (
            "ont_gated_resnik",
            OntSimConfig(metric="resnik", gated_query=True, role_mode="none"),
            dict(w_disease=2.0, w_phase=1.5, w_purpose=1.0, w_platform=0.5, w_material=0.5, w_route=0.0),
            30,
            "ont",
        )
    )
    # prior tuned weights winner
    cands.append(
        (
            "ont_gated_ic_sib_w_purpose1_plat05",
            OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True),
            dict(w_disease=1.0, w_phase=1.0, w_purpose=1.0, w_platform=0.5, w_material=0.5, w_route=0.0),
            20,
            "ont",
        )
    )

    rows = []
    preds_c = {}
    preds_p = {}
    for name, sc, w, k, kind in cands:
        print(f"{name}…", flush=True)
        if kind == "exact":
            pc = ontology_knn_recommend(both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=k, mode="exact")
            pp = ontology_knn_recommend(both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=k, mode="exact")
        else:
            pc = ontology_knn_recommend_cfg(
                both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=k, sim_cfg=sc, **w
            )
            pp = ontology_knn_recommend_cfg(
                both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=k, sim_cfg=sc, **w
            )
        cs = _field(score_predictions(test, pc, STRUCT))
        ps = _score_bundle(test_u, pp)
        # ensemble 50/50 with gbdt
        gbdt_c = regress_predict(train, test, "gbdt", STRUCT)
        gbdt_p = regress_predict(train_u, test_u, "gbdt", STRUCT)
        ens_c = soft_ensemble([(pc, 0.5), (gbdt_c, 0.5)], list(test["nct_id"]), STRUCT)
        ens_p = soft_ensemble([(pp, 0.5), (gbdt_p, 0.5)], list(test_u["nct_id"]), STRUCT)
        ecs = _field(score_predictions(test, ens_c, STRUCT))
        eps = _score_bundle(test_u, ens_p)
        row = {
            "name": name,
            "chrono": cs["mean_accuracy"],
            "chrono_f1": cs["mean_macro_f1"],
            "pooled": ps["trial_mean_accuracy"],
            "ens_chrono": ecs["mean_accuracy"],
            "ens_pooled": eps["trial_mean_accuracy"],
            "cfg": None if sc is None else sc.to_dict(),
            "weights": w,
            "k": k,
        }
        rows.append(row)
        preds_c[name] = pc
        preds_p[name] = pp
        preds_c[name + "+gbdt"] = ens_c
        preds_p[name + "+gbdt"] = ens_p
        print(
            f"  chrono={cs['mean_accuracy']:.4f} pooled={ps['trial_mean_accuracy']:.4f} | "
            f"ens {ecs['mean_accuracy']:.4f} / {eps['trial_mean_accuracy']:.4f}",
            flush=True,
        )

    exact_c = next(r["chrono"] for r in rows if r["name"] == "exact_k20")
    exact_p = next(r["pooled"] for r in rows if r["name"] == "exact_k20")
    for r in rows:
        r["chrono_delta"] = r["chrono"] - exact_c
        r["pooled_delta"] = r["pooled"] - exact_p
        r["ens_chrono_delta"] = r["ens_chrono"] - exact_c
        r["ens_pooled_delta"] = r["ens_pooled"] - exact_p

    # pick max performance: prefer ens if better on combined
    flat = []
    for r in rows:
        flat.append(
            {
                "name": r["name"],
                "chrono": r["chrono"],
                "pooled": r["pooled"],
                "score": (r["chrono"] - exact_c) + 0.75 * (r["pooled"] - exact_p),
                "mode": "pure",
                "meta": r,
            }
        )
        flat.append(
            {
                "name": r["name"] + "+gbdt50",
                "chrono": r["ens_chrono"],
                "pooled": r["ens_pooled"],
                "score": (r["ens_chrono"] - exact_c) + 0.75 * (r["ens_pooled"] - exact_p),
                "mode": "ens",
                "meta": r,
            }
        )
    flat.sort(key=lambda x: (-x["score"], -x["chrono"], -x["pooled"]))

    payload = {
        "rows": rows,
        "ranking": flat,
        "best": flat[0],
        "bootstrap_best_pure_chrono": None,
    }
    # bootstrap best pure ont vs exact
    best_pure = max(
        (r for r in rows if r["name"] != "exact_k20"),
        key=lambda r: (r["chrono"] - exact_c) + 0.75 * (r["pooled"] - exact_p),
    )
    payload["best_pure"] = best_pure
    payload["bootstrap_best_pure_chrono"] = _bootstrap_delta(
        test, preds_c[best_pure["name"]], preds_c["exact_k20"]
    )
    payload["bootstrap_best_pure_pooled"] = _bootstrap_delta(
        test_u, preds_p[best_pure["name"]], preds_p["exact_k20"]
    )

    path = OUT / "perf_lock.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== RANKING ===", flush=True)
    for x in flat:
        print(
            f"  score {x['score']:+.4f}  chrono {x['chrono']:.4f}  pooled {x['pooled']:.4f}  {x['name']}",
            flush=True,
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
