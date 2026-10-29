"""Neighbor aggregation for design-variable recommendation."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

from ctdr import config as cfg
from ctdr.vo_graph_sim import _canonical, ont_sim

SimMode = Literal["graph", "exact", "graph_subclass"]

# Matched field weights for exact vs OntSim (fair ablation).
# Sensitivity can set w_material=1.0 for BOTH modes.
W_DISEASE = 1.0
W_PHASE = 1.0
W_PURPOSE = 0.5
W_PLATFORM = 1.0
W_MATERIAL = 0.5

# Shared missing / non-evidence values for exact and OntSim ablations.
# unknown==unknown is NOT counted as a positive match (same as OntSim→0).
MISSING = frozenset({"unknown", "nan", "", "None", "none", "NaN"})


def _usable_vo(vid: str) -> str:
    """Canonical VO id, or empty if missing / non-VO / deprecated alias unresolved."""
    return _canonical(str(vid or "").strip())


def _exact_cat_match(a: object, b: object) -> float:
    """Indicator equality; missing on either side → 0 (no evidence)."""
    sa, sb = str(a), str(b)
    if sa in MISSING or sb in MISSING:
        return 0.0
    return 1.0 if sa == sb else 0.0


def _exact_vo_match(a: object, b: object) -> float:
    """Exact match on canonical VO ids; missing/non-VO on either side → 0."""
    ca, cb = _usable_vo(str(a)), _usable_vo(str(b))
    if not ca or not cb:
        return 0.0
    return 1.0 if ca == cb else 0.0


def _ont_vo_sim(a: object, b: object, *, use_roles: bool) -> float:
    """OntSim; missing/non-VO → 0 (aligned with exact missing rule)."""
    return float(ont_sim(str(a), str(b), use_roles=use_roles))


def knn_recommend(
    query_ids: list[str],
    train_ids: list[str],
    vecs: np.ndarray,
    id_to_i: dict[str, int],
    labels: pd.DataFrame,
    targets: tuple[str, ...] = cfg.TARGETS,
    k: int = cfg.DEFAULT_K,
) -> dict[str, dict[str, str]]:
    """For each query, vote target labels from top-k *train* neighbors by cosine."""
    train_idx = np.array([id_to_i[n] for n in train_ids if n in id_to_i])
    train_ids_ok = [n for n in train_ids if n in id_to_i]
    train_vecs = vecs[train_idx]
    out: dict[str, dict[str, str]] = {}
    lab = labels.set_index("nct_id")
    for q in query_ids:
        if q not in id_to_i:
            continue
        qv = vecs[id_to_i[q]]
        scores = train_vecs @ qv
        order = np.argsort(-scores)
        picks: list[str] = []
        pick_scores: list[float] = []
        for j in order:
            nct = train_ids_ok[j]
            if nct == q:
                continue
            picks.append(nct)
            pick_scores.append(float(scores[j]))
            if len(picks) >= k:
                break
        pred: dict[str, str] = {}
        for t in targets:
            weights: dict[str, float] = defaultdict(float)
            for nct, w in zip(picks, pick_scores):
                val = str(lab.loc[nct, t])
                weights[val] += max(w, 0.0) + 1e-6
            pred[t] = max(weights, key=weights.get) if weights else "unknown"
        out[q] = pred
    return out


def ontology_similarity_matrix(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    *,
    mode: SimMode = "graph",
    w_disease: float = W_DISEASE,
    w_phase: float = W_PHASE,
    w_purpose: float = W_PURPOSE,
    w_platform: float = W_PLATFORM,
    w_material: float = W_MATERIAL,
) -> dict[str, list[tuple[str, float]]]:
    """Rank training trials for each query with matched field weights.

    mode=\"exact\": indicator equality on platform / Material.
    mode=\"graph\": OntSim with subclass + role-expanded neighborhoods.
    mode=\"graph_subclass\": OntSim with subclass closures only (no roles).

    Platform and Material use the SAME weights in exact and OntSim modes.
    """
    use_roles = mode == "graph"
    meta = studies.set_index("nct_id")
    ranked: dict[str, list[tuple[str, float]]] = {}
    for q in query_ids:
        if q not in meta.index:
            continue
        qd = meta.loc[q, "disease"]
        qt = str(meta.loc[q, "vo_platform"])
        qp = meta.loc[q, "phase"]
        qpu = meta.loc[q, "primary_purpose"]
        qa = str(meta.loc[q, "arch_vo"])
        rows: list[tuple[str, float]] = []
        for j in train_ids:
            if j == q or j not in meta.index:
                continue
            s = 0.0
            s += w_disease * _exact_cat_match(meta.loc[j, "disease"], qd)
            s += w_phase * _exact_cat_match(meta.loc[j, "phase"], qp)
            s += w_purpose * _exact_cat_match(meta.loc[j, "primary_purpose"], qpu)
            # Platform: OntSim only when both sides are usable VO ids; otherwise the
            # same categorical exact rule (missing≠match) for exact and OntSim modes.
            jp = str(meta.loc[j, "vo_platform"])
            c_qt, c_jp = _usable_vo(qt), _usable_vo(jp)
            if c_qt and c_jp:
                if mode == "exact":
                    s += w_platform * (1.0 if c_qt == c_jp else 0.0)
                else:
                    s += w_platform * _ont_vo_sim(qt, jp, use_roles=use_roles)
            else:
                s += w_platform * _exact_cat_match(jp, qt)
            # Material: always VO-canonical (CTDR cohort requires arch_vo).
            ja = str(meta.loc[j, "arch_vo"])
            if mode == "exact":
                s += w_material * _exact_vo_match(ja, qa)
            else:
                s += w_material * _ont_vo_sim(qa, ja, use_roles=use_roles)
            rows.append((j, s))
        rows.sort(key=lambda x: (-x[1], x[0]))  # stable ties by nct_id
        ranked[q] = rows
    return ranked


def ontology_knn_recommend(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    targets: tuple[str, ...] = cfg.TARGETS,
    k: int = cfg.DEFAULT_K,
    *,
    mode: SimMode = "graph",
    w_material: float = W_MATERIAL,
    w_platform: float = W_PLATFORM,
) -> dict[str, dict[str, str]]:
    ranked = ontology_similarity_matrix(
        studies,
        train_ids,
        query_ids,
        mode=mode,
        w_material=w_material,
        w_platform=w_platform,
    )
    lab = studies.set_index("nct_id")
    out: dict[str, dict[str, str]] = {}
    for q, rows in ranked.items():
        picks = rows[:k]
        pred: dict[str, str] = {}
        for t in targets:
            weights: dict[str, float] = defaultdict(float)
            for nct, s in picks:
                lab_v = str(lab.loc[nct, t])
                if lab_v in MISSING:
                    continue
                weights[lab_v] += max(s, 0.0) + 1e-6
            if not weights:
                pred[t] = "unknown"
            else:
                # Deterministic: highest weight, then label string.
                pred[t] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
        out[q] = pred
    return out


def hybrid_knn_recommend(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    query_vecs: dict[str, np.ndarray],
    train_vecs: dict[str, np.ndarray],
    *,
    alpha: float = 0.5,
    k: int = cfg.DEFAULT_K,
    targets: tuple[str, ...] = cfg.TARGETS,
    mode: SimMode = "graph",
) -> dict[str, dict[str, str]]:
    """alpha * ontology_sim_norm + (1-alpha) * cosine."""
    ont = ontology_similarity_matrix(studies, train_ids, query_ids, mode=mode)
    lab = studies.set_index("nct_id")
    out: dict[str, dict[str, str]] = {}
    for q in query_ids:
        if q not in ont or q not in query_vecs:
            continue
        qv = query_vecs[q]
        # ontology scores already listed; normalize to [0,1] by max
        ont_map = {n: s for n, s in ont[q]}
        max_o = max(ont_map.values()) if ont_map else 1.0
        max_o = max(max_o, 1e-6)
        scored: list[tuple[str, float]] = []
        for j in train_ids:
            if j == q or j not in train_vecs:
                continue
            cos = float(train_vecs[j] @ qv)
            o = ont_map.get(j, 0.0) / max_o
            scored.append((j, alpha * o + (1.0 - alpha) * max(cos, 0.0)))
        scored.sort(key=lambda x: -x[1])
        picks = scored[:k]
        pred: dict[str, str] = {}
        for t in targets:
            weights: dict[str, float] = defaultdict(float)
            for nct, s in picks:
                weights[str(lab.loc[nct, t])] += max(s, 0.0) + 1e-6
            pred[t] = max(weights, key=weights.get) if weights else "unknown"
        out[q] = pred
    return out


def triple_hybrid_recommend(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    query_vecs: dict[str, np.ndarray],
    train_vecs: dict[str, np.ndarray],
    text_rank: dict[str, list[tuple[str, float]]],
    *,
    w_onto: float = 0.5,
    w_emb: float = 0.3,
    w_text: float = 0.2,
    k: int = cfg.DEFAULT_K,
    targets: tuple[str, ...] = cfg.TARGETS,
    mode: SimMode = "graph",
) -> dict[str, dict[str, str]]:
    """Weighted fusion of ontology, embedding cosine, and text similarity."""
    ssum = max(w_onto + w_emb + w_text, 1e-8)
    w_onto, w_emb, w_text = w_onto / ssum, w_emb / ssum, w_text / ssum
    ont = ontology_similarity_matrix(studies, train_ids, query_ids, mode=mode)
    lab = studies.set_index("nct_id")
    out: dict[str, dict[str, str]] = {}
    for q in query_ids:
        if q not in ont or q not in query_vecs:
            continue
        qv = query_vecs[q]
        ont_map = {n: s for n, s in ont[q]}
        max_o = max(max(ont_map.values()) if ont_map else 1.0, 1e-6)
        text_map = {n: s for n, s in text_rank.get(q, [])}
        max_t = max(max(text_map.values()) if text_map else 1.0, 1e-6)
        scored: list[tuple[str, float]] = []
        for j in train_ids:
            if j == q or j not in train_vecs:
                continue
            cos = max(float(train_vecs[j] @ qv), 0.0)
            o = ont_map.get(j, 0.0) / max_o
            t = max(text_map.get(j, 0.0), 0.0) / max_t
            scored.append((j, w_onto * o + w_emb * cos + w_text * t))
        scored.sort(key=lambda x: -x[1])
        picks = scored[:k]
        pred: dict[str, str] = {}
        for tname in targets:
            weights: dict[str, float] = defaultdict(float)
            for nct, s in picks:
                weights[str(lab.loc[nct, tname])] += max(s, 0.0) + 1e-6
            pred[tname] = max(weights, key=weights.get) if weights else "unknown"
        out[q] = pred
    return out


def ensemble_vote(
    methods: list[dict[str, dict[str, str]]],
    query_ids: list[str],
    targets: tuple[str, ...] = cfg.TARGETS,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for q in query_ids:
        pred: dict[str, str] = {}
        for t in targets:
            votes: dict[str, int] = defaultdict(int)
            for m in methods:
                if q in m and t in m[q]:
                    votes[m[q][t]] += 1
            pred[t] = max(votes, key=votes.get) if votes else "unknown"
        out[q] = pred
    return out
