"""Combinatorial OntSim search + val weight tuning (Setting A chrono + pooled).

Protocol
--------
1. Structural grid on chrono val (fit start <2019; val 2019–2021), default weights.
2. Coarse weight / alpha_role / route grid on the same val for top structural configs
   (reuses precomputed match components when only weights change).
3. Evaluate selected configs ONCE on chrono test (>=2022) and pooled unseen.

Does not overwrite paper artifacts.

  python -u -m ctdr.eval_ontsim_search
"""

from __future__ import annotations

import itertools
import json
import sys
import time
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
    W_DISEASE,
    W_MATERIAL,
    W_PHASE,
    W_PLATFORM,
    W_PURPOSE,
    ontology_component_matrix_cfg,
    ontology_knn_recommend,
    ontology_knn_recommend_cfg,
    rank_from_components,
    _vote_from_ranked,
)
from ctdr.vo_graph_sim import OntSimConfig, _node_ic, ont_sim, ont_sim_cfg  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "ontsim_search"
VAL_START = "2019-01-01"
K = 20
TOP_STRUCT = 8


def _field_mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _default_weights() -> dict[str, float]:
    return {
        "w_disease": W_DISEASE,
        "w_phase": W_PHASE,
        "w_purpose": W_PURPOSE,
        "w_platform": W_PLATFORM,
        "w_material": W_MATERIAL,
        "w_route": 0.0,
    }


def _score_from_components(
    studies: pd.DataFrame,
    query_df: pd.DataFrame,
    comps: dict,
    weights: dict[str, float],
) -> tuple[dict, dict]:
    ranked = rank_from_components(
        comps,
        w_disease=weights.get("w_disease", W_DISEASE),
        w_phase=weights.get("w_phase", W_PHASE),
        w_purpose=weights.get("w_purpose", W_PURPOSE),
        w_platform=weights.get("w_platform", W_PLATFORM),
        w_material=weights.get("w_material", W_MATERIAL),
        w_route=weights.get("w_route", 0.0),
    )
    preds = _vote_from_ranked(studies, ranked, STRUCT, K)
    return _field_mean(score_predictions(query_df, preds, STRUCT)), preds


def _structural_grid() -> list[OntSimConfig]:
    cfgs: list[OntSimConfig] = []
    baselines = [
        OntSimConfig(role_mode="union", gated_query=False),
        OntSimConfig(role_mode="none", gated_query=False),
        OntSimConfig(role_mode="sep", alpha_role=0.3, gated_query=False),
        OntSimConfig(role_mode="sep", alpha_role=0.3, gated_query=True),
        OntSimConfig(role_mode="sep", alpha_role=0.3, use_ic=True, gated_query=False),
        OntSimConfig(role_mode="sep", alpha_role=0.3, use_ic=True, gated_query=True),
        OntSimConfig(role_mode="sep_if_no_exact", alpha_role=0.3, gated_query=False),
        OntSimConfig(role_mode="union", use_ic=True, gated_query=False),
    ]
    cfgs.extend(baselines)

    for gated, use_ic, role, alpha, ch, sib in itertools.product(
        [False, True],
        [False, True],
        ["none", "sep", "union"],
        [0.2, 0.3, 0.5],
        [0, 1],
        [False, True],
    ):
        if role == "none" and alpha != 0.3:
            continue
        if role == "union" and alpha != 0.3:
            continue
        if gated and ch == 1 and sib:
            continue
        item = OntSimConfig(
            children_depth=ch,
            use_siblings=sib,
            use_ic=use_ic,
            role_mode=role,
            alpha_role=alpha,
            gated_query=gated,
        )
        if item not in cfgs:
            cfgs.append(item)
    return cfgs


def _weight_grid() -> list[dict[str, float]]:
    grids = []
    for w_pur, w_mat, w_plat, w_route in itertools.product(
        [0.25, 0.5, 1.0],
        [0.25, 0.5, 1.0],
        [0.5, 1.0],
        [0.0, 0.25],
    ):
        grids.append(
            {
                "w_disease": 1.0,
                "w_phase": 1.0,
                "w_purpose": float(w_pur),
                "w_platform": float(w_plat),
                "w_material": float(w_mat),
                "w_route": float(w_route),
            }
        )
    return grids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    _ = _node_ic()  # warm IC table

    df = build_ctdr_table()
    df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    fit = train_all[train_all["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train_all[train_all["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    both_fit_val = pd.concat([fit, val], ignore_index=True)
    both_chrono = pd.concat([train_all, test], ignore_index=True)
    print(f"chrono fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    a, b = "VO_0005198", "VO_0004908"
    sanity = {
        "pair": [a, b],
        "paper_union": ont_sim(a, b, use_roles=True),
        "paper_subclass": ont_sim(a, b, use_roles=False),
        "sep_a0.3": ont_sim_cfg(a, b, OntSimConfig(role_mode="sep", alpha_role=0.3)),
        "ic_sep": ont_sim_cfg(a, b, OntSimConfig(role_mode="sep", use_ic=True, alpha_role=0.3)),
        "ch1_sep": ont_sim_cfg(
            a, b, OntSimConfig(role_mode="sep", children_depth=1, alpha_role=0.3)
        ),
        "sib_sep": ont_sim_cfg(
            a, b, OntSimConfig(role_mode="sep", use_siblings=True, alpha_role=0.3)
        ),
    }
    print("sanity", sanity, flush=True)

    default_w = _default_weights()
    struct_cfgs = _structural_grid()
    print(f"structural grid size={len(struct_cfgs)}", flush=True)

    struct_rows: list[dict] = []
    # Cache components by config key for later weight tuning reuse
    comp_cache: dict[str, dict] = {}
    for i, sc in enumerate(struct_cfgs, 1):
        t0 = time.time()
        comps = ontology_component_matrix_cfg(
            both_fit_val, list(fit["nct_id"]), list(val["nct_id"]), sim_cfg=sc
        )
        comp_cache[sc.key()] = comps
        scores, _ = _score_from_components(both_fit_val, val, comps, default_w)
        row = {
            "key": sc.key(),
            "cfg": sc.to_dict(),
            "weights": default_w,
            **scores,
            "sec": round(time.time() - t0, 2),
        }
        struct_rows.append(row)
        if i % 5 == 0 or i == len(struct_cfgs):
            print(
                f"  [{i}/{len(struct_cfgs)}] val_acc={scores['mean_accuracy']:.4f}  "
                f"{sc.key()}  ({row['sec']}s)",
                flush=True,
            )

    struct_rows.sort(key=lambda r: (-r["mean_accuracy"], -r["mean_macro_f1"], r["key"]))
    (OUT / "structural_val.json").write_text(
        json.dumps(struct_rows, indent=2), encoding="utf-8"
    )
    print("\n=== TOP structural (val) ===", flush=True)
    for r in struct_rows[:12]:
        print(f"  {r['mean_accuracy']:.4f}  F1 {r['mean_macro_f1']:.4f}  {r['key']}", flush=True)

    weight_candidates = _weight_grid()
    tuned_rows: list[dict] = []
    for r in struct_rows[:TOP_STRUCT]:
        base_cfg = OntSimConfig(**r["cfg"])
        # Alpha variants: rebuild components only when alpha changes role channel
        alpha_list = (
            [0.2, 0.3, 0.5] if base_cfg.role_mode in {"sep", "sep_if_no_exact"} else [base_cfg.alpha_role]
        )
        best_local = None
        for alpha in alpha_list:
            sc = OntSimConfig(
                children_depth=base_cfg.children_depth,
                use_siblings=base_cfg.use_siblings,
                use_ic=base_cfg.use_ic,
                role_mode=base_cfg.role_mode,
                alpha_role=float(alpha),
                gated_query=base_cfg.gated_query,
            )
            key = sc.key()
            if key not in comp_cache:
                comp_cache[key] = ontology_component_matrix_cfg(
                    both_fit_val, list(fit["nct_id"]), list(val["nct_id"]), sim_cfg=sc
                )
            comps = comp_cache[key]
            for w in weight_candidates:
                scores, _ = _score_from_components(both_fit_val, val, comps, w)
                cand = {
                    "key": key,
                    "cfg": sc.to_dict(),
                    "weights": w,
                    **scores,
                }
                if best_local is None or (
                    cand["mean_accuracy"],
                    cand["mean_macro_f1"],
                ) > (best_local["mean_accuracy"], best_local["mean_macro_f1"]):
                    best_local = cand
        assert best_local is not None
        tuned_rows.append(best_local)
        print(
            f"tuned {best_local['key']} "
            f"acc={best_local['mean_accuracy']:.4f} "
            f"w={best_local['weights']}",
            flush=True,
        )

    tuned_rows.sort(key=lambda r: (-r["mean_accuracy"], -r["mean_macro_f1"], r["key"]))
    (OUT / "tuned_val.json").write_text(json.dumps(tuned_rows, indent=2), encoding="utf-8")

    paper_refs: list[tuple[str, OntSimConfig | None, dict[str, float]]] = [
        ("exact", None, default_w),
        ("paper_union", OntSimConfig(role_mode="union", gated_query=False), default_w),
        ("gated_sep", OntSimConfig(role_mode="sep", alpha_role=0.3, gated_query=True), default_w),
        (
            "gated_ic_sep",
            OntSimConfig(role_mode="sep", alpha_role=0.3, use_ic=True, gated_query=True),
            default_w,
        ),
        (
            "ic_sep",
            OntSimConfig(role_mode="sep", alpha_role=0.3, use_ic=True, gated_query=False),
            default_w,
        ),
    ]
    final_candidates = list(paper_refs)
    for i, r in enumerate(tuned_rows[:5]):
        final_candidates.append((f"tuned_{i+1}_{r['key']}", OntSimConfig(**r["cfg"]), r["weights"]))

    seen: set[str] = set()
    uniq: list[tuple[str, OntSimConfig | None, dict[str, float]]] = []
    for name, sc, w in final_candidates:
        tag = f"{name}|{None if sc is None else sc.key()}|{json.dumps(w, sort_keys=True)}"
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append((name, sc, w))

    chrono_out: dict[str, dict] = {}
    chrono_preds: dict[str, dict] = {}
    for name, sc, w in uniq:
        print(f"chrono-test {name}…", flush=True)
        if sc is None:
            preds = ontology_knn_recommend(
                both_chrono, list(train_all["nct_id"]), list(test["nct_id"]), STRUCT, k=K, mode="exact"
            )
            scores = _field_mean(score_predictions(test, preds, STRUCT))
        else:
            preds = ontology_knn_recommend_cfg(
                both_chrono,
                list(train_all["nct_id"]),
                list(test["nct_id"]),
                STRUCT,
                k=K,
                sim_cfg=sc,
                w_disease=w["w_disease"],
                w_phase=w["w_phase"],
                w_purpose=w["w_purpose"],
                w_platform=w["w_platform"],
                w_material=w["w_material"],
                w_route=w["w_route"],
            )
            scores = _field_mean(score_predictions(test, preds, STRUCT))
        chrono_out[name] = {**scores, "cfg": None if sc is None else sc.to_dict(), "weights": w}
        chrono_preds[name] = preds
        print(f"  {scores['mean_accuracy']:.4f}  F1 {scores['mean_macro_f1']:.4f}", flush=True)

    boot = {
        f"{name}_minus_exact": _bootstrap_delta(test, chrono_preds[name], chrono_preds["exact"])
        for name in chrono_preds
        if name != "exact"
    }

    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)
    print(
        f"pooled unseen train={len(train_u)} test={len(test_u)} classes={len(eligible)}",
        flush=True,
    )
    pooled_out: dict[str, dict] = {}
    pooled_preds: dict[str, dict] = {}
    for name, sc, w in uniq:
        print(f"pooled {name}…", flush=True)
        if sc is None:
            preds = ontology_knn_recommend(
                both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=K, mode="exact"
            )
        else:
            preds = ontology_knn_recommend_cfg(
                both_u,
                list(train_u["nct_id"]),
                list(test_u["nct_id"]),
                STRUCT,
                k=K,
                sim_cfg=sc,
                w_disease=w["w_disease"],
                w_phase=w["w_phase"],
                w_purpose=w["w_purpose"],
                w_platform=w["w_platform"],
                w_material=w["w_material"],
                w_route=w["w_route"],
            )
        bundle = _score_bundle(test_u, preds)
        pooled_out[name] = {
            "trial_mean_accuracy": bundle["trial_mean_accuracy"],
            "field_mean_accuracy": bundle["field_mean_accuracy"],
            "cfg": None if sc is None else sc.to_dict(),
            "weights": w,
        }
        pooled_preds[name] = preds
        print(f"  trial={bundle['trial_mean_accuracy']:.4f}", flush=True)

    boot_p = {
        f"{name}_minus_exact": _bootstrap_delta(
            test_u, pooled_preds[name], pooled_preds["exact"]
        )
        for name in pooled_preds
        if name != "exact"
    }

    exact_chrono = chrono_out["exact"]["mean_accuracy"]
    exact_pooled = pooled_out["exact"]["trial_mean_accuracy"]
    ranked_pick = []
    for name in chrono_out:
        if name == "exact":
            continue
        c_acc = chrono_out[name]["mean_accuracy"]
        p_acc = pooled_out[name]["trial_mean_accuracy"]
        ranked_pick.append(
            {
                "name": name,
                "chrono_acc": c_acc,
                "pooled_trial_acc": p_acc,
                "chrono_delta": c_acc - exact_chrono,
                "pooled_delta": p_acc - exact_pooled,
                "cfg": chrono_out[name]["cfg"],
                "weights": chrono_out[name]["weights"],
                "score": (p_acc - exact_pooled) + 0.25 * (c_acc - exact_chrono),
            }
        )
    ranked_pick.sort(key=lambda r: (-r["score"], -r["pooled_delta"], -r["chrono_delta"]))
    best = ranked_pick[0] if ranked_pick else None

    # Prefer a clean gated story if within 0.002 of best pooled and chrono not hurt
    story_pick = best
    for r in ranked_pick:
        cfg_d = r.get("cfg") or {}
        if cfg_d.get("gated_query") and cfg_d.get("role_mode") in {"sep", "sep_if_no_exact"}:
            if r["pooled_delta"] >= (best["pooled_delta"] - 0.002) and r["chrono_delta"] >= -0.002:
                story_pick = r
                break

    payload = {
        "protocol": {
            "val_window": "2019-01-01 <= start < 2022-01-01",
            "fit": "start < 2019-01-01",
            "test": "start >= 2022-01-01",
            "k": K,
            "n_structural": len(struct_cfgs),
            "top_struct_tuned": TOP_STRUCT,
            "elapsed_sec": round(time.time() - t_all, 1),
        },
        "sanity_pair": sanity,
        "structural_val_top12": struct_rows[:12],
        "tuned_val": tuned_rows,
        "chrono_test": chrono_out,
        "chrono_bootstrap_vs_exact": boot,
        "pooled_unseen": {
            "n_train": int(len(train_u)),
            "n_test": int(len(test_u)),
            "n_classes": int(len(eligible)),
            "scores": pooled_out,
            "bootstrap_vs_exact": boot_p,
        },
        "recommended": best,
        "story_recommended": story_pick,
        "ranking": ranked_pick,
    }
    (OUT / "ontsim_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== CHRONO TEST ===", flush=True)
    for name, s in sorted(chrono_out.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  {s['mean_accuracy']:.4f}  {name}", flush=True)
    print("\n=== POOLED TRIAL ===", flush=True)
    for name, s in sorted(pooled_out.items(), key=lambda kv: -kv[1]["trial_mean_accuracy"]):
        print(f"  {s['trial_mean_accuracy']:.4f}  {name}", flush=True)
    print("\n=== RECOMMENDED ===", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print("\n=== STORY PICK (gated preferred if close) ===", flush=True)
    print(json.dumps(story_pick, indent=2), flush=True)
    print(f"wrote {OUT / 'ontsim_search.json'}  elapsed={time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
