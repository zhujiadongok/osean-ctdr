"""Graph-value experiments for OSEAN-CTDR (Setting A targets).

1) Ontology-connected unseen-Material evaluation (exploratory, not chronological):
   Eligible = Material VO classes with 12–80 trials that share a superclass with ≥1
   other cohort Material. Fixed rule — report ALL eligible classes (no post-hoc
   picking of “graph-friendly” subsets).
   - Leave-one-Material-out (LOMO): class-mean of trial-mean accuracy (equal class weight).
   - Pooled simultaneous holdout of all eligible: trial-level bootstrap OntSim−exact.
   Hypothesis: OntSim improves neighbor transfer vs exact-match when Material id is unseen.

2) Matched-information ontology ablation on chronological Setting A test:
   same inputs/weights; only VO relation expansion changes.

  python -u -m ctdr.eval_graph_value
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
from ctdr.recommend import ontology_knn_recommend  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.vo_graph_sim import ont_sim, subclass_closure  # noqa: E402

OUT = cfg.CTDR_DIR / "optimize" / "graph_value"
STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
CATS_FULL = list(cfg.INPUT_CATS)
CATS_NO_MATERIAL = [c for c in CATS_FULL if c != "arch_vo"]
CATS_NO_VO = [c for c in CATS_FULL if c not in {"arch_vo", "vo_platform"}]
MIN_HELD_OUT_CLASS_N = 12
MAX_CLASS_N = 80
N_BOOT = 1500
KNN_METHODS = (
    ("exact_knn_k20", "exact"),
    ("ontsim_subclass_k20", "graph_subclass"),
    ("ontsim_subclass_role_k20", "graph"),
)


def _field_mean(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "field_mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "field_mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
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


def _trial_mean_accuracy(gold: pd.DataFrame, preds: dict) -> float:
    arr = _trial_means(gold, preds)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else 0.0


def _score_bundle(gold: pd.DataFrame, preds: dict) -> dict:
    field = _field_mean(score_predictions(gold, preds, STRUCT))
    return {
        **field,
        "trial_mean_accuracy": _trial_mean_accuracy(gold, preds),
        "n_trials": int(len(gold)),
    }


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
        "statistic": "trial_mean_accuracy",
        "mean_diff_trial": float(diff.mean()),
        "ci95_lo": float(arr[int(0.025 * n_boot)]),
        "ci95_hi": float(arr[int(0.975 * n_boot)]),
        "n_trials": int(len(diff)),
        "n_boot": n_boot,
        "note": (
            "Resamples trials, not Material classes. "
            "Does not alone support inference to new Material classes."
        ),
    }


def _knn(both: pd.DataFrame, train_ids: list, test_ids: list, mode: str) -> dict:
    return ontology_knn_recommend(both, train_ids, test_ids, STRUCT, k=20, mode=mode)


def _eligible_ontology_connected(df: pd.DataFrame) -> list[str]:
    """Fixed eligibility: mid-frequency + shares a superclass with another Material."""
    vc = df["arch_vo"].astype(str).value_counts()
    cands = [
        m
        for m, n in vc.items()
        if str(m).startswith("VO_") and MIN_HELD_OUT_CLASS_N <= int(n) <= MAX_CLASS_N
    ]
    out = []
    for m in cands:
        anc = subclass_closure(m)
        connected = False
        for other in vc.index:
            if other == m or not str(other).startswith("VO_"):
                continue
            if anc & subclass_closure(str(other)):
                connected = True
                break
        if connected:
            out.append(m)
    # Deterministic: larger classes first (transparent; not post-hoc by OntSim lift).
    out.sort(key=lambda m: (-int(vc[m]), m))
    return out


def _run_methods(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict]:
    both = pd.concat([train, test], ignore_index=True)
    tr_ids, te_ids = list(train["nct_id"]), list(test["nct_id"])
    preds: dict[str, dict] = {
        "frequency_by_phase": frequency_predict(train, test, STRUCT),
        "hist_gbdt_full": regress_predict(train, test, "gbdt", STRUCT, cats=CATS_FULL),
        "hist_gbdt_no_material": regress_predict(
            train, test, "gbdt", STRUCT, cats=CATS_NO_MATERIAL
        ),
        "hist_gbdt_no_vo": regress_predict(train, test, "gbdt", STRUCT, cats=CATS_NO_VO),
    }
    for name, mode in KNN_METHODS:
        preds[name] = _knn(both, tr_ids, te_ids, mode)
    return preds


def unseen_material_eval(df: pd.DataFrame) -> dict:
    eligible = _eligible_ontology_connected(df)
    vc = df["arch_vo"].astype(str).value_counts()

    # --- LOMO over ALL eligible classes (class-equal summary) ---
    per_class: dict[str, dict] = {}
    for m in eligible:
        test = df[df["arch_vo"].astype(str) == m].reset_index(drop=True)
        train = df[df["arch_vo"].astype(str) != m].reset_index(drop=True)
        preds = _run_methods(train, test)
        scores = {n: _score_bundle(test, p) for n, p in preds.items()}
        per_class[m] = {
            "n_test": int(len(test)),
            "has_positive_ontsim_to_train": bool(
                any(
                    ont_sim(m, tm, use_roles=False) > 0
                    for tm in train["arch_vo"].astype(str).unique()
                    if str(tm).startswith("VO_")
                )
            ),
            "scores": {
                n: {
                    "trial_mean_accuracy": s["trial_mean_accuracy"],
                    "field_mean_accuracy": s["field_mean_accuracy"],
                }
                for n, s in scores.items()
            },
            "delta_ontsim_role_minus_exact_trial": (
                scores["ontsim_subclass_role_k20"]["trial_mean_accuracy"]
                - scores["exact_knn_k20"]["trial_mean_accuracy"]
            ),
            "delta_ontsim_subclass_minus_exact_trial": (
                scores["ontsim_subclass_k20"]["trial_mean_accuracy"]
                - scores["exact_knn_k20"]["trial_mean_accuracy"]
            ),
        }
        print(
            f"  LOMO {m} n={len(test)} "
            f"exact={scores['exact_knn_k20']['trial_mean_accuracy']:.3f} "
            f"role={scores['ontsim_subclass_role_k20']['trial_mean_accuracy']:.3f} "
            f"Δ={per_class[m]['delta_ontsim_role_minus_exact_trial']:+.3f}",
            flush=True,
        )

    method_names = list(next(iter(per_class.values()))["scores"].keys())
    # Class-equal: mean of per-class trial means. Trial-equal: size-weighted by n_test.
    n_by = {m: per_class[m]["n_test"] for m in eligible}
    n_tot = float(sum(n_by.values()))
    class_mean = {}
    trial_weighted = {}
    for n in method_names:
        class_vals = [per_class[m]["scores"][n]["trial_mean_accuracy"] for m in eligible]
        class_mean[n] = {
            "class_mean_trial_accuracy": float(np.mean(class_vals)),
            "class_mean_field_accuracy": float(
                np.mean([per_class[m]["scores"][n]["field_mean_accuracy"] for m in eligible])
            ),
        }
        trial_weighted[n] = {
            "trial_weighted_mean_accuracy": float(
                sum(per_class[m]["scores"][n]["trial_mean_accuracy"] * n_by[m] for m in eligible)
                / n_tot
            )
        }
    class_mean_deltas = {
        "ontsim_role_minus_exact": float(
            np.mean([per_class[m]["delta_ontsim_role_minus_exact_trial"] for m in eligible])
        ),
        "ontsim_subclass_minus_exact": float(
            np.mean([per_class[m]["delta_ontsim_subclass_minus_exact_trial"] for m in eligible])
        ),
    }
    trial_weighted_deltas = {
        "ontsim_role_minus_exact": float(
            trial_weighted["ontsim_subclass_role_k20"]["trial_weighted_mean_accuracy"]
            - trial_weighted["exact_knn_k20"]["trial_weighted_mean_accuracy"]
        ),
        "ontsim_subclass_minus_exact": float(
            trial_weighted["ontsim_subclass_k20"]["trial_weighted_mean_accuracy"]
            - trial_weighted["exact_knn_k20"]["trial_weighted_mean_accuracy"]
        ),
    }

    # --- Pooled simultaneous holdout of ALL eligible (trial-level bootstrap) ---
    test_pool = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_pool = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    preds_pool = _run_methods(train_pool, test_pool)
    scores_pool = {n: _score_bundle(test_pool, p) for n, p in preds_pool.items()}
    # Class-equal summary under the same pooled predictions (same train set as pooled).
    pooled_per_class = {}
    for m in eligible:
        sub = test_pool[test_pool["arch_vo"].astype(str) == m].reset_index(drop=True)
        sub_preds = {n: {nid: preds_pool[n][nid] for nid in sub["nct_id"]} for n in preds_pool}
        pooled_per_class[m] = {
            "n_test": int(len(sub)),
            "scores": {
                n: {
                    "trial_mean_accuracy": _trial_mean_accuracy(sub, sub_preds[n]),
                }
                for n in preds_pool
            },
            "delta_ontsim_role_minus_exact_trial": float(
                _trial_mean_accuracy(sub, sub_preds["ontsim_subclass_role_k20"])
                - _trial_mean_accuracy(sub, sub_preds["exact_knn_k20"])
            ),
        }
    pooled_class_mean = {
        n: {
            "class_mean_trial_accuracy": float(
                np.mean(
                    [pooled_per_class[m]["scores"][n]["trial_mean_accuracy"] for m in eligible]
                )
            )
        }
        for n in preds_pool
    }
    boots = {
        "ontsim_subclass_minus_exact": _bootstrap_delta(
            test_pool, preds_pool["ontsim_subclass_k20"], preds_pool["exact_knn_k20"]
        ),
        "ontsim_role_minus_exact": _bootstrap_delta(
            test_pool, preds_pool["ontsim_subclass_role_k20"], preds_pool["exact_knn_k20"]
        ),
        "ontsim_role_minus_hist_gbdt_full": _bootstrap_delta(
            test_pool, preds_pool["ontsim_subclass_role_k20"], preds_pool["hist_gbdt_full"]
        ),
        "ontsim_subclass_minus_gbdt_no_mat": _bootstrap_delta(
            test_pool, preds_pool["ontsim_subclass_k20"], preds_pool["hist_gbdt_no_material"]
        ),
        "ontsim_subclass_minus_freq": _bootstrap_delta(
            test_pool, preds_pool["ontsim_subclass_k20"], preds_pool["frequency_by_phase"]
        ),
    }

    return {
        "protocol": {
            "name": "ontology_connected_unseen_material",
            "display_name": "Ontology-connected unseen Material evaluation",
            "eligibility": (
                f"Material VO classes with {MIN_HELD_OUT_CLASS_N}–{MAX_CLASS_N} trials "
                "that share a superclass with ≥1 other cohort Material"
            ),
            "eligible_materials": eligible,
            "eligible_class_counts": {m: int(vc[m]) for m in eligible},
            "n_eligible_classes": len(eligible),
            "time_split": False,
            "note": (
                "Not a chronological holdout. Exploratory generalization analysis "
                "specified after chronological Setting A. Evaluates ontology-connected "
                "unseen Materials only — not all possible unseen vaccines. "
                "All eligible classes reported; selection fixed before scoring. "
                "LOMO and pooled use different training sets; both report trial-equal "
                "and class-equal summaries."
            ),
            "pooled_n_train": int(len(train_pool)),
            "pooled_n_test": int(len(test_pool)),
        },
        "lomo": {
            "n_classes": len(eligible),
            "class_equal": {"scores": class_mean, "mean_delta_trial": class_mean_deltas},
            "trial_equal": {
                "scores": trial_weighted,
                "mean_delta_trial": trial_weighted_deltas,
            },
            "per_class": per_class,
        },
        # backward-compatible aliases used by older report readers
        "lomo_class_mean": {
            "n_classes": len(eligible),
            "scores": class_mean,
            "mean_delta_trial": class_mean_deltas,
        },
        "lomo_per_class": per_class,
        "pooled_holdout": {
            "scores": scores_pool,
            "class_equal": {"scores": pooled_class_mean},
            "per_class": pooled_per_class,
            "bootstrap_trial_level": boots,
        },
    }


def matched_ontology_ablation(df: pd.DataFrame) -> dict:
    """Chronological split; same weights; only VO relation expansion changes."""
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    preds = _run_methods(train, test)
    # Keep only ablation-relevant methods in primary table.
    keep = [
        "exact_knn_k20",
        "ontsim_subclass_k20",
        "ontsim_subclass_role_k20",
        "frequency_by_phase",
        "hist_gbdt_full",
        "hist_gbdt_no_vo",
    ]
    scores = {n: _score_bundle(test, preds[n]) for n in keep}
    boots = {
        "subclass_minus_exact": _bootstrap_delta(
            test, preds["ontsim_subclass_k20"], preds["exact_knn_k20"]
        ),
        "role_minus_exact": _bootstrap_delta(
            test, preds["ontsim_subclass_role_k20"], preds["exact_knn_k20"]
        ),
        "role_minus_subclass": _bootstrap_delta(
            test, preds["ontsim_subclass_role_k20"], preds["ontsim_subclass_k20"]
        ),
        "full_minus_no_vo_gbdt": _bootstrap_delta(
            test, preds["hist_gbdt_full"], preds["hist_gbdt_no_vo"]
        ),
    }
    return {
        "protocol": {
            "name": "matched_info_ontology_ablation",
            "split": "chronological train<2022 / test>=2022",
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "weights": {
                "material": 0.5,
                "platform": 1.0,
                "disease": 1.0,
                "phase": 1.0,
                "purpose": 0.5,
            },
            "hist_gbdt_entry": "ctdr.regress.regress_predict (locked hyperparams)",
            "metric_note": (
                "field_mean_accuracy = mean of per-field accuracies; "
                "trial_mean_accuracy = mean over trials of per-trial field hit-rate. "
                "Bootstrap uses trial_mean_accuracy."
            ),
        },
        "scores": scores,
        "bootstrap": boots,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    print(f"cohort n={len(df)} materials={df['arch_vo'].nunique()}", flush=True)

    print("\n=== Ontology-connected unseen Material (LOMO + pooled) ===", flush=True)
    unseen = unseen_material_eval(df)
    proto = unseen["protocol"]
    print(
        f"eligible={proto['n_eligible_classes']} "
        f"pooled train={proto['pooled_n_train']} test={proto['pooled_n_test']}",
        flush=True,
    )
    print("eligible:", proto["eligible_materials"], flush=True)
    print("LOMO class-equal / trial-equal:", flush=True)
    cm = unseen["lomo"]["class_equal"]["scores"]
    tw = unseen["lomo"]["trial_equal"]["scores"]
    for n, s in sorted(cm.items(), key=lambda kv: -kv[1]["class_mean_trial_accuracy"]):
        print(
            f"  class={s['class_mean_trial_accuracy']:.4f}  "
            f"trial={tw[n]['trial_weighted_mean_accuracy']:.4f}  {n}",
            flush=True,
        )
    md = unseen["lomo"]["class_equal"]["mean_delta_trial"]
    md_t = unseen["lomo"]["trial_equal"]["mean_delta_trial"]
    print(
        f"  class-eq Δ role−exact: {md['ontsim_role_minus_exact']:+.4f}; "
        f"trial-eq Δ: {md_t['ontsim_role_minus_exact']:+.4f}",
        flush=True,
    )
    pcm = unseen["pooled_holdout"]["class_equal"]["scores"]
    print(
        f"  pooled class-eq role="
        f"{pcm['ontsim_subclass_role_k20']['class_mean_trial_accuracy']:.4f} "
        f"exact={pcm['exact_knn_k20']['class_mean_trial_accuracy']:.4f}",
        flush=True,
    )
    print("Pooled holdout (trial-mean / field-mean):", flush=True)
    for n, s in sorted(
        unseen["pooled_holdout"]["scores"].items(),
        key=lambda kv: -kv[1]["trial_mean_accuracy"],
    ):
        print(
            f"  trial={s['trial_mean_accuracy']:.4f}  "
            f"field={s['field_mean_accuracy']:.4f}  {n}",
            flush=True,
        )
    for k, b in unseen["pooled_holdout"]["bootstrap_trial_level"].items():
        print(
            f"  boot {k}: {b['mean_diff_trial']:+.4f} "
            f"CI[{b['ci95_lo']:+.4f},{b['ci95_hi']:+.4f}]",
            flush=True,
        )

    print("\n=== Matched-info ontology ablation (chrono test) ===", flush=True)
    matched = matched_ontology_ablation(df)
    for n, s in sorted(matched["scores"].items(), key=lambda kv: -kv[1]["trial_mean_accuracy"]):
        print(
            f"  trial={s['trial_mean_accuracy']:.4f}  "
            f"field={s['field_mean_accuracy']:.4f}  {n}",
            flush=True,
        )
    for k, b in matched["bootstrap"].items():
        print(
            f"  boot {k}: {b['mean_diff_trial']:+.4f} "
            f"CI[{b['ci95_lo']:+.4f},{b['ci95_hi']:+.4f}]",
            flush=True,
        )

    payload = {
        "note": (
            "Graph-value experiments. Ontology-connected unseen Material tests semantic "
            "transfer (exploratory). Matched ablation isolates VO relations."
        ),
        "unseen_material": unseen,
        "matched_ontology_ablation": matched,
    }
    (OUT / "graph_value.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "GRAPH VALUE EXPERIMENTS",
        "",
        "## Ontology-connected unseen Material",
        f"eligible={proto['n_eligible_classes']} "
        f"pooled_train={proto['pooled_n_train']} pooled_test={proto['pooled_n_test']}",
        "LOMO class-mean trial accuracy:",
    ]
    for n, s in sorted(cm.items(), key=lambda kv: -kv[1]["class_mean_trial_accuracy"]):
        lines.append(f"  {s['class_mean_trial_accuracy']:.4f}  {n}")
    lines += ["", "Pooled trial-mean accuracy:"]
    for n, s in sorted(
        unseen["pooled_holdout"]["scores"].items(),
        key=lambda kv: -kv[1]["trial_mean_accuracy"],
    ):
        lines.append(f"  {s['trial_mean_accuracy']:.4f}  {n}")
    lines += ["", "## Matched ontology ablation (chrono, trial-mean)"]
    for n, s in sorted(matched["scores"].items(), key=lambda kv: -kv[1]["trial_mean_accuracy"]):
        lines.append(f"  {s['trial_mean_accuracy']:.4f}  {n}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
