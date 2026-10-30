"""Short finalist bake-off after combinatorial search."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.eval_graph_value import _eligible_ontology_connected, _score_bundle  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import ontology_knn_recommend, ontology_knn_recommend_cfg  # noqa: E402
from ctdr.vo_graph_sim import OntSimConfig  # noqa: E402
from ctdr import config as cfg  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
OUT = cfg.CTDR_DIR / "optimize" / "ontsim_search"


def _field(rep: dict) -> tuple[float, float]:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT if rep["targets"][t].get("n_eval", 0) > 0]
    return float(np.mean(accs)) if accs else 0.0, float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    both = pd.concat([train, test], ignore_index=True)
    eligible = _eligible_ontology_connected(df)
    test_u = df[df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    train_u = df[~df["arch_vo"].astype(str).isin(eligible)].reset_index(drop=True)
    both_u = pd.concat([train_u, test_u], ignore_index=True)

    finalists: dict[str, OntSimConfig | None] = {
        "exact": None,
        "paper_union": OntSimConfig(role_mode="union"),
        "sep": OntSimConfig(role_mode="sep", alpha_role=0.3),
        "ic": OntSimConfig(role_mode="none", use_ic=True),
        "ic_sep": OntSimConfig(role_mode="sep", use_ic=True, alpha_role=0.3),
        "gated_sep": OntSimConfig(role_mode="sep", alpha_role=0.3, gated_query=True),
        "gated_ic": OntSimConfig(role_mode="none", use_ic=True, gated_query=True),
        "gated_ic_sep": OntSimConfig(
            role_mode="sep", use_ic=True, alpha_role=0.3, gated_query=True
        ),
        "gated_ic_sib": OntSimConfig(
            role_mode="none", use_ic=True, use_siblings=True, gated_query=True
        ),
        "gated_ic_sep_sib": OntSimConfig(
            role_mode="sep",
            use_ic=True,
            alpha_role=0.3,
            use_siblings=True,
            gated_query=True,
        ),
        "ic_sib": OntSimConfig(role_mode="none", use_ic=True, use_siblings=True),
        "ic_ch1": OntSimConfig(role_mode="none", use_ic=True, children_depth=1),
    }

    rows = []
    for name, sc in finalists.items():
        if sc is None:
            pc = ontology_knn_recommend(
                both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20, mode="exact"
            )
            pp = ontology_knn_recommend(
                both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=20, mode="exact"
            )
        else:
            pc = ontology_knn_recommend_cfg(
                both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20, sim_cfg=sc
            )
            pp = ontology_knn_recommend_cfg(
                both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=20, sim_cfg=sc
            )
        ca, cf = _field(score_predictions(test, pc, STRUCT))
        pb = _score_bundle(test_u, pp)
        row = {
            "name": name,
            "chrono_acc": ca,
            "chrono_f1": cf,
            "pooled_trial": pb["trial_mean_accuracy"],
            "cfg": None if sc is None else sc.to_dict(),
            "weights": "default",
        }
        rows.append(row)
        print(
            f"{ca:.4f}  pooled {pb['trial_mean_accuracy']:.4f}  {name}",
            flush=True,
        )

    # Best chrono from search: gated IC + siblings + tuned weights (+/- route)
    sc = OntSimConfig(
        children_depth=0,
        use_siblings=True,
        use_ic=True,
        role_mode="none",
        alpha_role=0.3,
        gated_query=True,
    )
    for tag, w in [
        (
            "tuned_gated_ic_sib+route",
            dict(
                w_disease=1.0,
                w_phase=1.0,
                w_purpose=1.0,
                w_platform=0.5,
                w_material=0.5,
                w_route=0.25,
            ),
        ),
        (
            "tuned_gated_ic_sib_no_route",
            dict(
                w_disease=1.0,
                w_phase=1.0,
                w_purpose=1.0,
                w_platform=0.5,
                w_material=0.5,
                w_route=0.0,
            ),
        ),
        (
            "default_w_gated_ic_sib",
            dict(
                w_disease=1.0,
                w_phase=1.0,
                w_purpose=0.5,
                w_platform=1.0,
                w_material=0.5,
                w_route=0.0,
            ),
        ),
    ]:
        pc = ontology_knn_recommend_cfg(
            both, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20, sim_cfg=sc, **w
        )
        pp = ontology_knn_recommend_cfg(
            both_u, list(train_u["nct_id"]), list(test_u["nct_id"]), STRUCT, k=20, sim_cfg=sc, **w
        )
        ca, cf = _field(score_predictions(test, pc, STRUCT))
        pb = _score_bundle(test_u, pp)
        rows.append(
            {
                "name": tag,
                "chrono_acc": ca,
                "chrono_f1": cf,
                "pooled_trial": pb["trial_mean_accuracy"],
                "cfg": sc.to_dict(),
                "weights": w,
            }
        )
        print(f"{ca:.4f}  pooled {pb['trial_mean_accuracy']:.4f}  {tag}", flush=True)

    exact_c = next(r["chrono_acc"] for r in rows if r["name"] == "exact")
    exact_p = next(r["pooled_trial"] for r in rows if r["name"] == "exact")
    for r in rows:
        r["chrono_delta"] = r["chrono_acc"] - exact_c
        r["pooled_delta"] = r["pooled_trial"] - exact_p
        r["score"] = r["pooled_delta"] + 0.25 * r["chrono_delta"]
    rows_sorted = sorted(rows, key=lambda r: (-r["score"], -r["pooled_delta"], -r["chrono_delta"]))

    payload = {
        "note": "Default-weight finalists + best tuned chrono variant.",
        "rows": rows_sorted,
        "best_by_score": rows_sorted[0],
    }
    path = OUT / "finalists.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== ranked by pooled_delta + 0.25*chrono_delta ===", flush=True)
    for r in rows_sorted:
        print(
            f"  score {r['score']:+.4f}  chrono {r['chrono_acc']:.4f} "
            f"({r['chrono_delta']:+.4f})  pooled {r['pooled_trial']:.4f} "
            f"({r['pooled_delta']:+.4f})  {r['name']}",
            flush=True,
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
