"""Input-only SEA-CDM + VO graph for CTDR (targets never appear as edges)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from vo_architecture import load_classes  # noqa: E402

# Relation ids — no allocation / masking / model / enrollment
REL = {
    "typed_as": 0,
    "has_disease": 1,
    "has_phase": 2,
    "has_purpose": 3,
    "has_platform": 4,
    "has_route": 5,
    "has_dose": 6,
    "has_sex": 7,
    "has_older": 8,
    "has_child": 9,
    "has_sponsor": 10,
    "subClassOf": 11,
    "has_role": 12,
    # input-only study proximity: same disease + platform (+ optional phase)
    "same_context": 13,
}


def _vo_closure(arch_ids: set[str]) -> set[str]:
    classes = load_classes()
    keep = set(arch_ids) & set(classes)
    changed = True
    while changed:
        changed = False
        for vid in list(keep):
            for parent in classes.get(vid, {}).get("parents", []):
                if parent in classes and not classes[parent].get("deprecated") and parent not in keep:
                    keep.add(parent)
                    changed = True
            for role in classes.get(vid, {}).get("roles", []):
                if role in classes and not classes[role].get("deprecated") and role not in keep:
                    keep.add(role)
                    changed = True
                    for parent in classes[role].get("parents", []):
                        if parent in classes and not classes[parent].get("deprecated"):
                            keep.add(parent)
    return keep


def build_input_graph(studies: pd.DataFrame) -> dict:
    """Build triples using only CTDR inputs (never design targets)."""
    classes = load_classes()
    arch_ids = {v for v in studies["arch_vo"].astype(str) if v.startswith("VO_")}
    vo_nodes = sorted(_vo_closure(arch_ids))

    box_specs = [
        ("disease", "DISEASE"),
        ("phase", "PHASE"),
        ("primary_purpose", "PURPOSE"),
        ("vo_platform", "PLATFORM"),
        ("route", "ROUTE"),
        ("dose", "DOSE"),
        ("sex", "SEX"),
        ("older", "OLDER"),
        ("child", "CHILD"),
        ("sponsor_class", "SPONSOR"),
    ]
    box_nodes: list[str] = []
    for col, prefix in box_specs:
        for val in sorted(studies[col].astype(str).unique()):
            if val and val != "nan":
                box_nodes.append(f"{prefix}:{val}")

    study_nodes = [f"STUDY:{n}" for n in studies["nct_id"]]
    nodes = study_nodes + vo_nodes + box_nodes
    index = {n: i for i, n in enumerate(nodes)}

    triples: list[tuple[int, int, int]] = []
    for row in studies.itertuples(index=False):
        s = index[f"STUDY:{row.nct_id}"]
        if str(row.arch_vo).startswith("VO_") and row.arch_vo in index:
            triples.append((s, REL["typed_as"], index[row.arch_vo]))
        for col, rel, prefix in (
            ("disease", "has_disease", "DISEASE"),
            ("phase", "has_phase", "PHASE"),
            ("primary_purpose", "has_purpose", "PURPOSE"),
            ("vo_platform", "has_platform", "PLATFORM"),
            ("route", "has_route", "ROUTE"),
            ("dose", "has_dose", "DOSE"),
            ("sex", "has_sex", "SEX"),
            ("older", "has_older", "OLDER"),
            ("child", "has_child", "CHILD"),
            ("sponsor_class", "has_sponsor", "SPONSOR"),
        ):
            key = f"{prefix}:{getattr(row, col)}"
            if key in index:
                triples.append((s, REL[rel], index[key]))

    for vid in vo_nodes:
        for parent in classes.get(vid, {}).get("parents", []):
            if parent in index:
                triples.append((index[vid], REL["subClassOf"], index[parent]))
        for role in classes.get(vid, {}).get("roles", []):
            if role in index:
                triples.append((index[vid], REL["has_role"], index[role]))

    # Connect studies that share disease+platform+phase (inputs only; no design targets).
    # Cap degree so the graph does not densify into a complete clique.
    buckets: dict[tuple[str, str, str], list[str]] = {}
    for row in studies.itertuples(index=False):
        key = (str(row.disease), str(row.vo_platform), str(row.phase))
        buckets.setdefault(key, []).append(str(row.nct_id))
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        # star + ring keeps local neighborhoods without O(n^2) edges
        hub = ids[0]
        hs = index[f"STUDY:{hub}"]
        for other in ids[1:]:
            os_ = index[f"STUDY:{other}"]
            triples.append((hs, REL["same_context"], os_))
            triples.append((os_, REL["same_context"], hs))
        for a, b in zip(ids, ids[1:]):
            ia, ib = index[f"STUDY:{a}"], index[f"STUDY:{b}"]
            triples.append((ia, REL["same_context"], ib))
            triples.append((ib, REL["same_context"], ia))

    return {
        "nodes": nodes,
        "index": index,
        "triples": triples,
        "n_rel": len(REL),
        "study_ids": list(studies["nct_id"]),
        "rel_names": {v: k for k, v in REL.items()},
    }
