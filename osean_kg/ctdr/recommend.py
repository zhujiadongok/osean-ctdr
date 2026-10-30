"""Neighbor aggregation for design-variable recommendation."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

from ctdr import config as cfg
from ctdr.vo_graph_sim import (  # noqa: E402
    OntSimConfig,
    _canonical,
    ont_sim,
    ont_sim_cfg,
    ont_sim_ic,
    ont_sim_ic_sep,
    ont_sim_sep,
)

SimMode = Literal[
    "graph",
    "exact",
    "graph_subclass",
    "graph_sep",
    "graph_ic",
    "graph_ic_sep",
    "graph_gated_sep",
    "graph_gated_ic_sep",
    "graph_gated_ic_sib",  # default-weight: gated + IC + siblings
    "graph_gated_ic_sib_perf",  # val-tuned weights; max chrono+pooled lock
]

# Matched field weights for exact vs OntSim (fair ablation).
# Sensitivity can set w_material=1.0 for BOTH modes.
W_DISEASE = 1.0
W_PHASE = 1.0
W_PURPOSE = 0.5
W_PLATFORM = 1.0
W_MATERIAL = 0.5
DEFAULT_ALPHA_ROLE = 0.3

# Val-tuned weights for max Setting A chrono + pooled (gated_ic_sib structure).
# Fair paper ablation still uses the defaults above for exact vs OntSim.
W_PERF = {
    "w_disease": 1.0,
    "w_phase": 1.0,
    "w_purpose": 1.0,
    "w_platform": 0.5,
    "w_material": 0.5,
    "w_route": 0.0,
}

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
    """Paper baseline OntSim; missing/non-VO → 0 (aligned with exact missing rule)."""
    return float(ont_sim(str(a), str(b), use_roles=use_roles))


def _vo_sim_for_mode(a: object, b: object, mode: SimMode) -> float:
    """Material/platform VO similarity dispatcher (non-exact modes)."""
    if mode == "graph":
        return _ont_vo_sim(a, b, use_roles=True)
    if mode == "graph_subclass":
        return _ont_vo_sim(a, b, use_roles=False)
    if mode in {"graph_sep", "graph_gated_sep"}:
        return float(ont_sim_sep(str(a), str(b), alpha_role=DEFAULT_ALPHA_ROLE))
    if mode == "graph_ic":
        return float(ont_sim_ic(str(a), str(b)))
    if mode in {"graph_ic_sep", "graph_gated_ic_sep"}:
        return float(ont_sim_ic_sep(str(a), str(b), alpha_role=DEFAULT_ALPHA_ROLE))
    if mode in {"graph_gated_ic_sib", "graph_gated_ic_sib_perf"}:
        return float(
            ont_sim_cfg(
                str(a),
                str(b),
                OntSimConfig(
                    role_mode="none",
                    use_ic=True,
                    use_siblings=True,
                    gated_query=True,
                ),
            )
        )
    raise ValueError(f"Unknown OntSim mode: {mode}")

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

    Modes:
      exact              — VO id equality on platform / Material
      graph              — paper OntSim (subclass ∪ role Jaccard)
      graph_subclass     — subclass Jaccard only
      graph_sep          — subclass Jaccard + separate role channel
      graph_ic           — IC-weighted subclass Jaccard
      graph_ic_sep       — IC subclass + separate role
      graph_gated_sep    — if query Material id appears in train → exact VO;
                           else graph_sep (same rule for platform when both VO)
      graph_gated_ic_sep — gated with graph_ic_sep fallback
      graph_gated_ic_sib — gated; IC-weighted subclass + sibling neighborhood
      graph_gated_ic_sib_perf — same OntSim as gated_ic_sib but val-tuned
                           field weights (W_PERF); not a fair exact ablation

    Platform and Material use the SAME weights across exact / OntSim modes
    unless mode is graph_gated_ic_sib_perf (uses W_PERF).
    """
    if mode == "graph_gated_ic_sib_perf":
        w_disease = float(W_PERF["w_disease"])
        w_phase = float(W_PERF["w_phase"])
        w_purpose = float(W_PERF["w_purpose"])
        w_platform = float(W_PERF["w_platform"])
        w_material = float(W_PERF["w_material"])
    meta = studies.set_index("nct_id")
    train_materials = {
        _usable_vo(str(meta.loc[n, "arch_vo"]))
        for n in train_ids
        if n in meta.index and _usable_vo(str(meta.loc[n, "arch_vo"]))
    }
    gated = mode in {
        "graph_gated_sep",
        "graph_gated_ic_sep",
        "graph_gated_ic_sib",
        "graph_gated_ic_sib_perf",
    }
    ranked: dict[str, list[tuple[str, float]]] = {}
    for q in query_ids:
        if q not in meta.index:
            continue
        qd = meta.loc[q, "disease"]
        qt = str(meta.loc[q, "vo_platform"])
        qp = meta.loc[q, "phase"]
        qpu = meta.loc[q, "primary_purpose"]
        qa = str(meta.loc[q, "arch_vo"])
        qa_c = _usable_vo(qa)
        # Query-level gate: seen Material id → exact VO matching; else improved OntSim.
        use_exact_vo = mode == "exact" or (gated and qa_c in train_materials)
        rows: list[tuple[str, float]] = []
        for j in train_ids:
            if j == q or j not in meta.index:
                continue
            s = 0.0
            s += w_disease * _exact_cat_match(meta.loc[j, "disease"], qd)
            s += w_phase * _exact_cat_match(meta.loc[j, "phase"], qp)
            s += w_purpose * _exact_cat_match(meta.loc[j, "primary_purpose"], qpu)
            jp = str(meta.loc[j, "vo_platform"])
            c_qt, c_jp = _usable_vo(qt), _usable_vo(jp)
            if c_qt and c_jp:
                if use_exact_vo:
                    s += w_platform * (1.0 if c_qt == c_jp else 0.0)
                else:
                    s += w_platform * _vo_sim_for_mode(qt, jp, mode)
            else:
                s += w_platform * _exact_cat_match(jp, qt)
            ja = str(meta.loc[j, "arch_vo"])
            if use_exact_vo:
                s += w_material * _exact_vo_match(ja, qa)
            else:
                s += w_material * _vo_sim_for_mode(qa, ja, mode)
            rows.append((j, s))
        rows.sort(key=lambda x: (-x[1], x[0]))  # stable ties by nct_id
        ranked[q] = rows
    return ranked


def _vote_from_ranked(
    studies: pd.DataFrame,
    ranked: dict[str, list[tuple[str, float]]],
    targets: tuple[str, ...],
    k: int,
) -> dict[str, dict[str, str]]:
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
                pred[t] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0]
        out[q] = pred
    return out


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
    return _vote_from_ranked(studies, ranked, targets, k)


def ontology_component_matrix_cfg(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    *,
    sim_cfg: OntSimConfig,
) -> dict[str, list[tuple[str, dict[str, float]]]]:
    """Per-(query, train) match components for fast weight re-scoring.

    Components: disease, phase, purpose, platform, material, route ∈ [0, 1].
    """
    meta = studies.set_index("nct_id")
    train_materials = {
        _usable_vo(str(meta.loc[n, "arch_vo"]))
        for n in train_ids
        if n in meta.index and _usable_vo(str(meta.loc[n, "arch_vo"]))
    }
    has_route = "route" in meta.columns
    vo_cache: dict[tuple[str, str], float] = {}

    def _vo_sim(a: str, b: str) -> float:
        ca, cb = _usable_vo(a), _usable_vo(b)
        if not ca or not cb:
            return 0.0
        if ca == cb:
            return 1.0
        key = (ca, cb) if ca <= cb else (cb, ca)
        hit = vo_cache.get(key)
        if hit is None:
            hit = float(ont_sim_cfg(ca, cb, sim_cfg))
            vo_cache[key] = hit
        return hit

    # Pre-extract train rows once
    train_rows: list[tuple[str, object, object, object, str, str, str]] = []
    for j in train_ids:
        if j not in meta.index:
            continue
        train_rows.append(
            (
                j,
                meta.loc[j, "disease"],
                meta.loc[j, "phase"],
                meta.loc[j, "primary_purpose"],
                str(meta.loc[j, "vo_platform"]),
                str(meta.loc[j, "arch_vo"]),
                str(meta.loc[j, "route"]) if has_route else "",
            )
        )

    out: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for q in query_ids:
        if q not in meta.index:
            continue
        qd = meta.loc[q, "disease"]
        qt = str(meta.loc[q, "vo_platform"])
        qp = meta.loc[q, "phase"]
        qpu = meta.loc[q, "primary_purpose"]
        qa = str(meta.loc[q, "arch_vo"])
        qa_c = _usable_vo(qa)
        qr = str(meta.loc[q, "route"]) if has_route else ""
        use_exact_vo = bool(sim_cfg.gated_query and qa_c in train_materials)
        c_qt = _usable_vo(qt)
        rows: list[tuple[str, dict[str, float]]] = []
        for j, jd, jph, jpu, jp, ja, jr in train_rows:
            if j == q:
                continue
            c_jp = _usable_vo(jp)
            if c_qt and c_jp:
                if use_exact_vo:
                    plat = 1.0 if c_qt == c_jp else 0.0
                else:
                    plat = _vo_sim(qt, jp)
            else:
                plat = _exact_cat_match(jp, qt)
            if use_exact_vo:
                mat = _exact_vo_match(ja, qa)
            else:
                mat = _vo_sim(qa, ja)
            rows.append(
                (
                    j,
                    {
                        "disease": _exact_cat_match(jd, qd),
                        "phase": _exact_cat_match(jph, qp),
                        "purpose": _exact_cat_match(jpu, qpu),
                        "platform": float(plat),
                        "material": float(mat),
                        "route": _exact_cat_match(jr, qr) if has_route else 0.0,
                    },
                )
            )
        out[q] = rows
    return out


def rank_from_components(
    components: dict[str, list[tuple[str, dict[str, float]]]],
    *,
    w_disease: float = W_DISEASE,
    w_phase: float = W_PHASE,
    w_purpose: float = W_PURPOSE,
    w_platform: float = W_PLATFORM,
    w_material: float = W_MATERIAL,
    w_route: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    ranked: dict[str, list[tuple[str, float]]] = {}
    for q, rows in components.items():
        scored = [
            (
                j,
                (
                    w_disease * c["disease"]
                    + w_phase * c["phase"]
                    + w_purpose * c["purpose"]
                    + w_platform * c["platform"]
                    + w_material * c["material"]
                    + w_route * c["route"]
                ),
            )
            for j, c in rows
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        ranked[q] = scored
    return ranked


def ontology_similarity_matrix_cfg(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    *,
    sim_cfg: OntSimConfig,
    w_disease: float = W_DISEASE,
    w_phase: float = W_PHASE,
    w_purpose: float = W_PURPOSE,
    w_platform: float = W_PLATFORM,
    w_material: float = W_MATERIAL,
    w_route: float = 0.0,
) -> dict[str, list[tuple[str, float]]]:
    """Configurable OntSim ranking for combinatorial search / weight tuning."""
    comps = ontology_component_matrix_cfg(
        studies, train_ids, query_ids, sim_cfg=sim_cfg
    )
    return rank_from_components(
        comps,
        w_disease=w_disease,
        w_phase=w_phase,
        w_purpose=w_purpose,
        w_platform=w_platform,
        w_material=w_material,
        w_route=w_route,
    )


def ontology_knn_recommend_cfg(
    studies: pd.DataFrame,
    train_ids: list[str],
    query_ids: list[str],
    targets: tuple[str, ...] = cfg.TARGETS,
    k: int = cfg.DEFAULT_K,
    *,
    sim_cfg: OntSimConfig,
    w_disease: float = W_DISEASE,
    w_phase: float = W_PHASE,
    w_purpose: float = W_PURPOSE,
    w_platform: float = W_PLATFORM,
    w_material: float = W_MATERIAL,
    w_route: float = 0.0,
) -> dict[str, dict[str, str]]:
    ranked = ontology_similarity_matrix_cfg(
        studies,
        train_ids,
        query_ids,
        sim_cfg=sim_cfg,
        w_disease=w_disease,
        w_phase=w_phase,
        w_purpose=w_purpose,
        w_platform=w_platform,
        w_material=w_material,
        w_route=w_route,
    )
    return _vote_from_ranked(studies, ranked, targets, k)


def hybrid_from_components(
    studies: pd.DataFrame,
    components: dict[str, list[tuple[str, dict[str, float]]]],
    query_vecs: dict[str, np.ndarray],
    train_vecs: dict[str, np.ndarray],
    train_ids: list[str],
    targets: tuple[str, ...],
    *,
    k: int,
    alpha: float,
    w_disease: float = W_DISEASE,
    w_phase: float = W_PHASE,
    w_purpose: float = W_PURPOSE,
    w_platform: float = W_PLATFORM,
    w_material: float = W_MATERIAL,
    w_route: float = 0.0,
) -> dict[str, dict[str, str]]:
    """alpha * ont_score_norm + (1-alpha) * cosine, from precomputed OntSim components."""
    ranked_ont = rank_from_components(
        components,
        w_disease=w_disease,
        w_phase=w_phase,
        w_purpose=w_purpose,
        w_platform=w_platform,
        w_material=w_material,
        w_route=w_route,
    )
    lab = studies.set_index("nct_id")
    out: dict[str, dict[str, str]] = {}
    a = min(max(alpha, 0.0), 1.0)
    for q, ont_rows in ranked_ont.items():
        if q not in query_vecs:
            continue
        qv = query_vecs[q]
        ont_map = {n: s for n, s in ont_rows}
        max_o = max(max(ont_map.values()) if ont_map else 1.0, 1e-6)
        scored: list[tuple[str, float]] = []
        for j in train_ids:
            if j == q or j not in train_vecs:
                continue
            cos = max(float(train_vecs[j] @ qv), 0.0)
            o = ont_map.get(j, 0.0) / max_o
            scored.append((j, a * o + (1.0 - a) * cos))
        scored.sort(key=lambda x: (-x[1], x[0]))
        picks = scored[:k]
        pred: dict[str, str] = {}
        for t in targets:
            weights: dict[str, float] = defaultdict(float)
            for nct, s in picks:
                lab_v = str(lab.loc[nct, t])
                if lab_v in MISSING:
                    continue
                weights[lab_v] += max(s, 0.0) + 1e-6
            pred[t] = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0] if weights else "unknown"
        out[q] = pred
    return out


def soft_ensemble(
    methods: list[tuple[dict[str, dict[str, str]], float]],
    query_ids: list[str],
    targets: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    """Weighted plurality over method predictions (deterministic tie-break by label)."""
    out: dict[str, dict[str, str]] = {}
    for q in query_ids:
        pred: dict[str, str] = {}
        for t in targets:
            scores: dict[str, float] = defaultdict(float)
            for preds, w in methods:
                if q in preds and t in preds[q]:
                    scores[str(preds[q][t])] += float(w)
            if not scores:
                pred[t] = "unknown"
            else:
                pred[t] = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
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
