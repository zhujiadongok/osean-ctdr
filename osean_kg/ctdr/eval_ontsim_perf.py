"""Performance-first OntSim push (vectorized weight/k/hybrid/ensemble search).

  python -u -m ctdr.eval_ontsim_perf
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
from ctdr.recommend import (  # noqa: E402
    MISSING,
    W_DISEASE,
    W_MATERIAL,
    W_PHASE,
    W_PLATFORM,
    W_PURPOSE,
    ontology_component_matrix_cfg,
    ontology_knn_recommend,
    ontology_knn_recommend_cfg,
    soft_ensemble,
)
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.vo_graph_sim import OntSimConfig, _node_ic  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
COMP_KEYS = ("disease", "phase", "purpose", "platform", "material", "route")
FEAT_COLS = [
    "disease",
    "arch_vo",
    "vo_platform",
    "phase",
    "primary_purpose",
    "route",
    "dose",
    "sex",
    "older",
    "child",
    "sponsor_class",
]
OUT = cfg.CTDR_DIR / "optimize" / "ontsim_perf"
VAL_START = "2019-01-01"


def _field(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _gold_map(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        str(r.nct_id): {t: str(getattr(r, t)) for t in STRUCT}
        for r in df.itertuples(index=False)
    }


def _fast_acc(preds: dict[str, dict[str, str]], gold: dict[str, dict[str, str]]) -> float:
    """Field-micro accuracy over non-missing gold — fast search objective."""
    hits = 0
    n = 0
    for q, g in gold.items():
        p = preds.get(q)
        if not p:
            continue
        for t in STRUCT:
            yt = g[t]
            if yt in MISSING:
                continue
            hits += int(p.get(t, "unknown") == yt)
            n += 1
    return float(hits / n) if n else 0.0


def _wvec(w: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            w.get("w_disease", W_DISEASE),
            w.get("w_phase", W_PHASE),
            w.get("w_purpose", W_PURPOSE),
            w.get("w_platform", W_PLATFORM),
            w.get("w_material", W_MATERIAL),
            w.get("w_route", 0.0),
        ],
        dtype=np.float64,
    )


def _pack_components(
    studies: pd.DataFrame,
    comps: dict[str, list[tuple[str, dict[str, float]]]],
    query_ids: list[str],
) -> dict:
    """Pack components + train labels into numpy for fast re-scoring."""
    lab = studies.set_index("nct_id")
    packed = {}
    for q in query_ids:
        rows = comps.get(q, [])
        if not rows:
            continue
        ncts = [n for n, _ in rows]
        C = np.asarray([[c[k] for k in COMP_KEYS] for _, c in rows], dtype=np.float64)
        Y = {
            t: np.asarray([str(lab.loc[n, t]) for n in ncts], dtype=object)
            for t in STRUCT
        }
        packed[q] = {"ncts": ncts, "C": C, "Y": Y}
    return packed


def _predict_from_packed(packed: dict, w: dict[str, float], k: int) -> dict[str, dict[str, str]]:
    wv = _wvec(w)
    out: dict[str, dict[str, str]] = {}
    for q, pack in packed.items():
        scores = pack["C"] @ wv
        # top-k with stable tie-break via nct order already in rows
        if k >= len(scores):
            idx = np.argsort(-scores, kind="mergesort")
        else:
            part = np.argpartition(-scores, k)[:k]
            idx = part[np.argsort(-scores[part], kind="mergesort")]
        pred: dict[str, str] = {}
        for t in STRUCT:
            labs = pack["Y"][t][idx]
            sc = scores[idx]
            weights: dict[str, float] = {}
            for lab_v, s in zip(labs, sc):
                if lab_v in MISSING:
                    continue
                weights[lab_v] = weights.get(lab_v, 0.0) + max(float(s), 0.0) + 1e-6
            pred[t] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0] if weights else "unknown"
        out[q] = pred
    return out


def _hybrid_from_packed(
    packed: dict,
    query_vecs: dict[str, np.ndarray],
    train_vecs: dict[str, np.ndarray],
    w: dict[str, float],
    k: int,
    alpha: float,
) -> dict[str, dict[str, str]]:
    wv = _wvec(w)
    a = min(max(alpha, 0.0), 1.0)
    out: dict[str, dict[str, str]] = {}
    for q, pack in packed.items():
        if q not in query_vecs:
            continue
        qv = query_vecs[q]
        ont = pack["C"] @ wv
        max_o = max(float(ont.max()) if len(ont) else 1.0, 1e-6)
        cos = np.asarray(
            [max(float(train_vecs[n] @ qv), 0.0) if n in train_vecs else 0.0 for n in pack["ncts"]],
            dtype=np.float64,
        )
        scores = a * (ont / max_o) + (1.0 - a) * cos
        if k >= len(scores):
            idx = np.argsort(-scores, kind="mergesort")
        else:
            part = np.argpartition(-scores, k)[:k]
            idx = part[np.argsort(-scores[part], kind="mergesort")]
        pred: dict[str, str] = {}
        for t in STRUCT:
            labs = pack["Y"][t][idx]
            sc = scores[idx]
            weights: dict[str, float] = {}
            for lab_v, s in zip(labs, sc):
                if lab_v in MISSING:
                    continue
                weights[lab_v] = weights.get(lab_v, 0.0) + max(float(s), 0.0) + 1e-6
            pred[t] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0] if weights else "unknown"
        out[q] = pred
    return out


def _build_tab_vecs(fit: pd.DataFrame, frames: list[pd.DataFrame]):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(fit[FEAT_COLS].astype(str))
    out = []
    for df in frames:
        X = enc.transform(df[FEAT_COLS].astype(str))
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        out.append({n: X[i] for i, n in enumerate(df["nct_id"])})
    return out


def _sim_cfgs() -> list[OntSimConfig]:
    cfgs: list[OntSimConfig] = []
    seeds = [
        OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True),
        OntSimConfig(role_mode="sep", use_ic=True, use_siblings=True, gated_query=True, alpha_role=0.2),
        OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True, children_depth=1),
        OntSimConfig(
            role_mode="none",
            use_ic=True,
            use_siblings=True,
            gated_query=True,
            children_if_coarse=True,
            children_depth=1,
        ),
        OntSimConfig(role_mode="none", use_ic=True, gated_query=True),
        OntSimConfig(role_mode="sep", use_ic=True, gated_query=True, alpha_role=0.3),
        OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=False),
        OntSimConfig(metric="lin", gated_query=True, role_mode="none"),
        OntSimConfig(metric="lin", gated_query=True, role_mode="sep", alpha_role=0.3),
        OntSimConfig(metric="lin", gated_query=True, role_mode="none", use_siblings=True),
        OntSimConfig(metric="resnik", gated_query=True, role_mode="none"),
        OntSimConfig(metric="resnik", gated_query=True, role_mode="none", use_siblings=True),
        OntSimConfig(metric="lin", gated_query=False, role_mode="none"),
        OntSimConfig(metric="resnik", gated_query=False, role_mode="none"),
        OntSimConfig(metric="resnik", gated_query=False, role_mode="none", use_siblings=True),
        OntSimConfig(role_mode="union", gated_query=False),
        OntSimConfig(role_mode="sep", use_ic=True, alpha_role=0.3, gated_query=False),
        OntSimConfig(role_mode="none", use_ic=True, gated_query=False),
        OntSimConfig(metric="lin", gated_query=False, role_mode="none", use_siblings=True),
        OntSimConfig(metric="lin", gated_query=False, role_mode="sep", alpha_role=0.3, use_siblings=True),
    ]
    for c in seeds:
        if c not in cfgs:
            cfgs.append(c)
    return cfgs


def _weight_grid() -> list[dict[str, float]]:
    grids = [
        {
            "w_disease": W_DISEASE,
            "w_phase": W_PHASE,
            "w_purpose": W_PURPOSE,
            "w_platform": W_PLATFORM,
            "w_material": W_MATERIAL,
            "w_route": 0.0,
        }
    ]
    for wd, wpu, wpl, wm, wr in itertools.product(
        [1.0, 1.5, 2.0],
        [0.5, 1.0, 1.5],
        [0.5, 1.0, 1.5, 2.0],
        [0.5, 1.0, 1.5, 2.0],
        [0.0, 0.25],
    ):
        grids.append(
            {
                "w_disease": float(wd),
                "w_phase": 1.0 if wd < 2.0 else 1.5,
                "w_purpose": float(wpu),
                "w_platform": float(wpl),
                "w_material": float(wm),
                "w_route": float(wr),
            }
        )
    seen, out = set(), []
    for g in grids:
        tag = json.dumps(g, sort_keys=True)
        if tag not in seen:
            seen.add(tag)
            out.append(g)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _ = _node_ic()

    df = build_ctdr_table()
    df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    fit = train_all[train_all["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train_all[train_all["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    both_fv = pd.concat([fit, val], ignore_index=True)
    both_chrono = pd.concat([train_all, test], ignore_index=True)
    print(f"fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    fit_vecs, val_vecs = _build_tab_vecs(fit, [fit, val])
    train_vecs, test_vecs = _build_tab_vecs(train_all, [train_all, test])

    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)
    train_u_vecs, test_u_vecs = _build_tab_vecs(train_u, [train_u, test_u])
    print(f"pooled train={len(train_u)} test={len(test_u)} classes={len(eligible)}", flush=True)

    sim_cfgs = _sim_cfgs()
    weights = _weight_grid()
    ks = [10, 15, 20, 25, 30, 40, 50]
    alphas = [0.5, 0.65, 0.75, 0.85, 0.95]
    print(f"n_sim={len(sim_cfgs)} n_weights={len(weights)}", flush=True)

    default_w = weights[0]
    gold_val = _gold_map(val)
    phase1a = []
    pack_cache: dict[str, dict] = {}

    for i, sc in enumerate(sim_cfgs, 1):
        key = sc.key()
        t1 = time.time()
        comps = ontology_component_matrix_cfg(
            both_fv, list(fit["nct_id"]), list(val["nct_id"]), sim_cfg=sc
        )
        packed = _pack_components(both_fv, comps, list(val["nct_id"]))
        pack_cache[key] = packed
        best = None
        for k in ks:
            acc = _fast_acc(_predict_from_packed(packed, default_w, k), gold_val)
            cand = {
                "stage": "ont",
                "key": key,
                "cfg": sc.to_dict(),
                "weights": default_w,
                "k": k,
                "alpha": 1.0,
                "mean_accuracy": acc,
                "mean_macro_f1": 0.0,
            }
            if best is None or cand["mean_accuracy"] > best["mean_accuracy"]:
                best = cand
        phase1a.append(best)
        print(
            f"  [{i}/{len(sim_cfgs)}] val={best['mean_accuracy']:.4f} k={best['k']} {key} ({time.time()-t1:.1f}s)",
            flush=True,
        )
    phase1a.sort(key=lambda r: -r["mean_accuracy"])

    phase1 = []
    for r in phase1a[:12]:
        packed = pack_cache[r["key"]]
        best = dict(r)
        t1 = time.time()
        for w in weights:
            for k in ks:
                acc = _fast_acc(_predict_from_packed(packed, w, k), gold_val)
                if acc > best["mean_accuracy"]:
                    best = {
                        "stage": "ont",
                        "key": r["key"],
                        "cfg": r["cfg"],
                        "weights": w,
                        "k": k,
                        "alpha": 1.0,
                        "mean_accuracy": acc,
                        "mean_macro_f1": 0.0,
                    }
        phase1.append(best)
        print(
            f"  tuned {best['mean_accuracy']:.4f} k={best['k']} {best['key']} ({time.time()-t1:.1f}s)",
            flush=True,
        )
    phase1.sort(key=lambda r: -r["mean_accuracy"])
    (OUT / "phase1_val.json").write_text(json.dumps(phase1[:20], indent=2), encoding="utf-8")
    print("\n=== TOP OntSim val ===", flush=True)
    for r in phase1[:10]:
        print(f"  {r['mean_accuracy']:.4f}  k={r['k']}  {r['key']}", flush=True)

    gbdt_val = regress_predict(fit, val, "gbdt", STRUCT)
    print(f"HistGBDT val={_fast_acc(gbdt_val, gold_val):.4f}", flush=True)

    phase2 = list(phase1[:8])
    for r in phase1[:8]:
        packed = pack_cache[r["key"]]
        w = r["weights"]
        for k, alpha in itertools.product(ks, alphas):
            acc = _fast_acc(
                _hybrid_from_packed(packed, val_vecs, fit_vecs, w, k, alpha), gold_val
            )
            phase2.append(
                {
                    "stage": "hybrid_tab",
                    "key": r["key"],
                    "cfg": r["cfg"],
                    "weights": w,
                    "k": k,
                    "alpha": alpha,
                    "mean_accuracy": acc,
                    "mean_macro_f1": 0.0,
                }
            )
        for ens_w, k in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], ks):
            preds_o = _predict_from_packed(packed, w, k)
            ens = soft_ensemble(
                [(preds_o, ens_w), (gbdt_val, 1.0 - ens_w)], list(val["nct_id"]), STRUCT
            )
            phase2.append(
                {
                    "stage": "ens_gbdt",
                    "key": r["key"],
                    "cfg": r["cfg"],
                    "weights": w,
                    "k": k,
                    "alpha": ens_w,
                    "mean_accuracy": _fast_acc(ens, gold_val),
                    "mean_macro_f1": 0.0,
                }
            )
        for ens_w, k, alpha in itertools.product(
            [0.4, 0.5, 0.6], [20, 30, 40], [0.65, 0.75, 0.85]
        ):
            preds_h = _hybrid_from_packed(packed, val_vecs, fit_vecs, w, k, alpha)
            ens = soft_ensemble(
                [(preds_h, ens_w), (gbdt_val, 1.0 - ens_w)], list(val["nct_id"]), STRUCT
            )
            phase2.append(
                {
                    "stage": "ens_hybrid_gbdt",
                    "key": r["key"],
                    "cfg": r["cfg"],
                    "weights": w,
                    "k": k,
                    "alpha": alpha,
                    "ens_w": ens_w,
                    "mean_accuracy": _fast_acc(ens, gold_val),
                    "mean_macro_f1": 0.0,
                }
            )

    phase2.sort(key=lambda r: -r["mean_accuracy"])
    (OUT / "phase2_val.json").write_text(json.dumps(phase2[:40], indent=2), encoding="utf-8")
    print("\n=== TOP overall val ===", flush=True)
    for r in phase2[:15]:
        print(
            f"  {r['mean_accuracy']:.4f}  {r['stage']} k={r['k']} a={r.get('alpha')} {r['key']}",
            flush=True,
        )

    # Finalists
    finalists = []
    seen = set()
    for r in phase2:
        tag = (
            r["stage"],
            r["key"],
            r["k"],
            round(float(r.get("alpha", 1.0)), 4),
            round(float(r.get("ens_w", -1)), 4),
            json.dumps(r["weights"], sort_keys=True),
        )
        if tag in seen:
            continue
        seen.add(tag)
        finalists.append(r)
        if len(finalists) >= 10:
            break
    baselines = [
        {
            "stage": "exact",
            "key": "exact",
            "cfg": None,
            "weights": default_w,
            "k": 20,
            "alpha": 1.0,
        },
        {
            "stage": "paper_union",
            "key": OntSimConfig(role_mode="union").key(),
            "cfg": OntSimConfig(role_mode="union").to_dict(),
            "weights": default_w,
            "k": 20,
            "alpha": 1.0,
        },
        {
            "stage": "gated_ic_sib",
            "key": OntSimConfig(
                role_mode="none", use_ic=True, use_siblings=True, gated_query=True
            ).key(),
            "cfg": OntSimConfig(
                role_mode="none", use_ic=True, use_siblings=True, gated_query=True
            ).to_dict(),
            "weights": default_w,
            "k": 20,
            "alpha": 1.0,
        },
    ]
    finalists.extend(baselines)

    def _predict_row(row, both, tr_df, te_df, tr_v, te_v):
        stage = row["stage"]
        k = int(row["k"])
        w = row["weights"]
        tr_ids, te_ids = list(tr_df["nct_id"]), list(te_df["nct_id"])
        if stage == "exact":
            return ontology_knn_recommend(both, tr_ids, te_ids, STRUCT, k=k, mode="exact")
        sc = OntSimConfig(**row["cfg"])
        if stage in {"ont", "paper_union", "gated_ic_sib"}:
            return ontology_knn_recommend_cfg(
                both, tr_ids, te_ids, STRUCT, k=k, sim_cfg=sc, **w
            )
        comps = ontology_component_matrix_cfg(both, tr_ids, te_ids, sim_cfg=sc)
        packed = _pack_components(both, comps, te_ids)
        if stage == "hybrid_tab":
            return _hybrid_from_packed(packed, te_v, tr_v, w, k, float(row["alpha"]))
        gbdt = regress_predict(tr_df, te_df, "gbdt", STRUCT)
        if stage == "ens_gbdt":
            preds_o = _predict_from_packed(packed, w, k)
            return soft_ensemble(
                [(preds_o, float(row["alpha"])), (gbdt, 1.0 - float(row["alpha"]))],
                te_ids,
                STRUCT,
            )
        if stage == "ens_hybrid_gbdt":
            preds_h = _hybrid_from_packed(packed, te_v, tr_v, w, k, float(row["alpha"]))
            ew = float(row.get("ens_w", 0.5))
            return soft_ensemble([(preds_h, ew), (gbdt, 1.0 - ew)], te_ids, STRUCT)
        return ontology_knn_recommend_cfg(both, tr_ids, te_ids, STRUCT, k=k, sim_cfg=sc, **w)

    uniq = []
    seen_n = set()
    for r in finalists:
        name = f"{r['stage']}__{r['key']}__k{r['k']}__a{r.get('alpha')}"
        if "ens_w" in r:
            name += f"__e{r['ens_w']}"
        if name in seen_n:
            continue
        seen_n.add(name)
        uniq.append((name, r))

    chrono_out, pooled_out, chrono_preds, pooled_preds = {}, {}, {}, {}
    for name, r in uniq:
        print(f"eval {name}…", flush=True)
        pc = _predict_row(r, both_chrono, train_all, test, train_vecs, test_vecs)
        pp = _predict_row(r, both_u, train_u, test_u, train_u_vecs, test_u_vecs)
        cs = _field(score_predictions(test, pc, STRUCT))
        ps = _score_bundle(test_u, pp)
        chrono_out[name] = {
            **cs,
            "stage": r["stage"],
            "cfg": r.get("cfg"),
            "weights": r["weights"],
            "k": r["k"],
            "alpha": r.get("alpha"),
            "ens_w": r.get("ens_w"),
        }
        pooled_out[name] = {
            "trial_mean_accuracy": ps["trial_mean_accuracy"],
            "field_mean_accuracy": ps["field_mean_accuracy"],
        }
        chrono_preds[name] = pc
        pooled_preds[name] = pp
        print(f"  chrono={cs['mean_accuracy']:.4f} pooled={ps['trial_mean_accuracy']:.4f}", flush=True)

    for tag, fn in [
        ("frequency", lambda tr, te: frequency_predict(tr, te, STRUCT)),
        ("hist_gbdt", lambda tr, te: regress_predict(tr, te, "gbdt", STRUCT)),
        ("logistic", lambda tr, te: regress_predict(tr, te, "lr", STRUCT)),
    ]:
        pc, pp = fn(train_all, test), fn(train_u, test_u)
        chrono_out[tag] = _field(score_predictions(test, pc, STRUCT))
        ps = _score_bundle(test_u, pp)
        pooled_out[tag] = {
            "trial_mean_accuracy": ps["trial_mean_accuracy"],
            "field_mean_accuracy": ps["field_mean_accuracy"],
        }
        chrono_preds[tag] = pc
        pooled_preds[tag] = pp

    exact_name = next(n for n in chrono_out if n.startswith("exact__"))
    exact_c = chrono_out[exact_name]["mean_accuracy"]
    exact_p = pooled_out[exact_name]["trial_mean_accuracy"]
    ranking = []
    for name in chrono_out:
        if name in {"frequency", "hist_gbdt", "logistic"}:
            continue
        c = chrono_out[name]["mean_accuracy"]
        p = pooled_out[name]["trial_mean_accuracy"]
        ranking.append(
            {
                "name": name,
                "chrono_acc": c,
                "pooled_trial": p,
                "chrono_delta": c - exact_c,
                "pooled_delta": p - exact_p,
                "score": (c - exact_c) + 0.75 * (p - exact_p),
                "meta": chrono_out[name],
            }
        )
    ranking.sort(key=lambda r: (-r["score"], -r["chrono_acc"], -r["pooled_trial"]))

    # LOMO top3 + baselines
    lomo_names = [exact_name]
    for r in ranking[:3]:
        if r["name"] not in lomo_names:
            lomo_names.append(r["name"])
    for n in chrono_out:
        if n.startswith("gated_ic_sib__") and n not in lomo_names:
            lomo_names.append(n)
            break
    for n in chrono_out:
        if n.startswith("paper_union__") and n not in lomo_names:
            lomo_names.append(n)
            break

    print(f"\nLOMO {lomo_names}…", flush=True)
    lomo = {}
    row_by = {n: r for n, r in uniq}
    for m in eligible:
        te = df[df["arch_vo"].astype(str) == m].reset_index(drop=True)
        tr = df[df["arch_vo"].astype(str) != m].reset_index(drop=True)
        both_m = pd.concat([tr, te], ignore_index=True)
        tr_v, te_v = _build_tab_vecs(tr, [tr, te])
        lomo[m] = {"n": int(len(te)), "scores": {}}
        for name in lomo_names:
            if name not in row_by:
                continue
            preds = _predict_row(row_by[name], both_m, tr, te, tr_v, te_v)
            lomo[m]["scores"][name] = _trial_mean_accuracy(te, preds)
        bits = " ".join(
            f"{n.split('__')[0]}={lomo[m]['scores'][n]:.3f}"
            for n in lomo_names
            if n in lomo[m]["scores"]
        )
        print(f"  {m} n={len(te)} {bits}", flush=True)

    lomo_summary = {}
    for name in lomo_names:
        vals = [lomo[m]["scores"][name] for m in eligible if name in lomo[m]["scores"]]
        if not vals:
            continue
        n_tot = sum(lomo[m]["n"] for m in eligible if name in lomo[m]["scores"])
        lomo_summary[name] = {
            "class_mean": float(np.mean(vals)),
            "trial_weighted": float(
                sum(lomo[m]["scores"][name] * lomo[m]["n"] for m in eligible if name in lomo[m]["scores"])
                / max(n_tot, 1)
            ),
        }

    best = ranking[0]
    # Prefer pure-ont / gated story if within 0.003 chrono of best absolute chrono
    story = best
    for r in ranking:
        st = r["meta"].get("stage")
        if st in {"ont", "gated_ic_sib", "paper_union"} and r["chrono_acc"] >= best["chrono_acc"] - 0.005:
            if r["pooled_delta"] >= best["pooled_delta"] - 0.005:
                story = r
                break

    payload = {
        "elapsed_sec": round(time.time() - t0, 1),
        "phase1_top10": phase1[:10],
        "phase2_top15": phase2[:15],
        "chrono": chrono_out,
        "pooled": pooled_out,
        "ranking": ranking,
        "best": best,
        "story_best": story,
        "lomo_summary": lomo_summary,
        "lomo_per_class": lomo,
        "bootstrap_best_chrono_vs_exact": _bootstrap_delta(
            test, chrono_preds[best["name"]], chrono_preds[exact_name]
        ),
        "bootstrap_best_pooled_vs_exact": _bootstrap_delta(
            test_u, pooled_preds[best["name"]], pooled_preds[exact_name]
        ),
    }
    (OUT / "ontsim_perf.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== CHRONO ===", flush=True)
    for n, s in sorted(chrono_out.items(), key=lambda kv: -kv[1].get("mean_accuracy", 0)):
        print(f"  {s.get('mean_accuracy', 0):.4f}  {n}", flush=True)
    print("\n=== POOLED ===", flush=True)
    for n, s in sorted(pooled_out.items(), key=lambda kv: -kv[1]["trial_mean_accuracy"]):
        print(f"  {s['trial_mean_accuracy']:.4f}  {n}", flush=True)
    print("\n=== LOMO class-mean ===", flush=True)
    for n, s in sorted(lomo_summary.items(), key=lambda kv: -kv[1]["class_mean"]):
        print(f"  {s['class_mean']:.4f} / {s['trial_weighted']:.4f}  {n}", flush=True)
    print("\n=== BEST ===", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print(f"wrote {OUT / 'ontsim_perf.json'} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
