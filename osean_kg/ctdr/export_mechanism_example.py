"""Export one post-remediation LOMO query: exact vs primary OntSim neighbors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.recommend import (  # noqa: E402
    ontology_knn_recommend,
    ontology_similarity_matrix,
    ontology_similarity_matrix_cfg,
)
from ctdr.vo_graph_sim import OntSimConfig  # noqa: E402
from vo_architecture import load_classes  # noqa: E402

OUT = cfg.CTDR_DIR / "optimize" / "mapping_audit"
STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)

# Post-remediation LOMO class with clear primary OntSim gain.
QUERY_VO = "VO_0007319"  # autologous tumor cell vaccine
QUERY_NCT = "NCT00085397"
TOP_N = 5


def _label(vid: str, classes: dict) -> str:
    meta = classes.get(vid) or {}
    return str(meta.get("label") or meta.get("name") or vid)


def _top(ranked: dict, q: str, n: int) -> list[dict]:
    rows = ranked.get(q, [])[:n]
    return [{"nct_id": nct, "score": float(s)} for nct, s in rows]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    classes = load_classes()
    te = df[df["arch_vo"].astype(str) == QUERY_VO].sort_values("start_date").reset_index(drop=True)
    if te.empty:
        raise SystemExit(f"no trials for {QUERY_VO}")
    q = QUERY_NCT or str(te.iloc[0]["nct_id"])
    if q not in set(te["nct_id"]):
        raise SystemExit(f"{q} not in held-out class {QUERY_VO}")
    tr = df[df["arch_vo"].astype(str) != QUERY_VO].reset_index(drop=True)
    both = pd.concat([tr, te], ignore_index=True)
    train_ids = list(tr["nct_id"])
    query_ids = [q]

    exact_rank = ontology_similarity_matrix(
        both, train_ids, query_ids, mode="exact"
    )
    role_rank = ontology_similarity_matrix(
        both, train_ids, query_ids, mode="graph"
    )
    ic_sib = OntSimConfig(role_mode="none", use_ic=True, use_siblings=True, gated_query=True)
    icsib_rank = ontology_similarity_matrix_cfg(
        both, train_ids, query_ids, sim_cfg=ic_sib
    )

    pred_exact = ontology_knn_recommend(
        both, train_ids, query_ids, STRUCT, k=20, mode="exact"
    )[q]
    pred_role = ontology_knn_recommend(
        both, train_ids, query_ids, STRUCT, k=20, mode="graph"
    )[q]

    gold = {t: str(te.loc[te["nct_id"] == q, t].iloc[0]) for t in STRUCT}
    qrow = te.loc[te["nct_id"] == q].iloc[0]

    def enrich(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            hit = both.loc[both["nct_id"] == r["nct_id"]].iloc[0]
            out.append(
                {
                    **r,
                    "arch_vo": str(hit["arch_vo"]),
                    "arch_label": _label(str(hit["arch_vo"]), classes),
                    "disease": str(hit["disease"]),
                    "phase": str(hit["phase"]),
                    "vo_platform": str(hit["vo_platform"]),
                }
            )
        return out

    payload = {
        "protocol": "LOMO hold-out of query Material class",
        "query_nct": q,
        "query_material": QUERY_VO,
        "query_material_label": _label(QUERY_VO, classes),
        "query_disease": str(qrow["disease"]),
        "query_phase": str(qrow["phase"]),
        "query_platform": str(qrow["vo_platform"]),
        "gold": gold,
        "pred_exact": pred_exact,
        "pred_ontsim_role": pred_role,
        "neighbors_exact_top5": enrich(_top(exact_rank, q, TOP_N)),
        "neighbors_ontsim_role_top5": enrich(_top(role_rank, q, TOP_N)),
        "neighbors_ic_sib_top5": enrich(_top(icsib_rank, q, TOP_N)),
        "note": (
            "Primary OntSim = subclass+role Jaccard (paper ablation). "
            "IC-sib = exploratory IC-weighted subclass-and-sibling variant "
            "(gate opens for all LOMO queries because Material is unseen)."
        ),
    }
    path = OUT / "mechanism_example_NCT_neighbors.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
