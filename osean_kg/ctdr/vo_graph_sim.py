"""VO ontology-graph similarity via relation-expanded neighborhoods.

Paper baseline OntSim (`ont_sim`):
  Jaccard over subclass closure, optionally unioned with role closure.

Search / improved APIs:
  - separated role channel
  - IC-weighted subclass Jaccard
  - optional children (depth-limited) and sibling expansion
  - configurable OntSimConfig used by combinatorial search
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

from vo_architecture import PROFILE_TO_OFFICIAL, load_classes


def _canonical(vid: str) -> str:
    v = str(vid or "").strip()
    if not v.startswith("VO_"):
        return ""
    return PROFILE_TO_OFFICIAL.get(v, v)


@lru_cache(maxsize=1)
def _classes() -> dict:
    return load_classes()


@lru_cache(maxsize=1)
def _children_index() -> dict[str, frozenset[str]]:
    classes = _classes()
    kids: dict[str, set[str]] = {}
    for cid, meta in classes.items():
        if meta.get("deprecated"):
            continue
        for parent in meta.get("parents", []):
            kids.setdefault(parent, set()).add(cid)
    return {k: frozenset(v) for k, v in kids.items()}


@lru_cache(maxsize=8192)
def subclass_closure(vid: str) -> frozenset[str]:
    """Self + superclass nodes along subClassOf only."""
    root = _canonical(vid)
    if not root:
        return frozenset()
    classes = _classes()
    if root not in classes or classes[root].get("deprecated"):
        return frozenset({root})
    out = {root}
    seen = {root}
    frontier = list(classes[root].get("parents", []))
    while frontier:
        nxt = frontier.pop()
        if nxt in seen or nxt not in classes:
            continue
        if classes[nxt].get("deprecated"):
            continue
        seen.add(nxt)
        out.add(nxt)
        frontier.extend(classes[nxt].get("parents", []))
    return frozenset(out)


@lru_cache(maxsize=8192)
def children_closure(vid: str, depth: int = 1) -> frozenset[str]:
    """Descendants up to `depth` (not including self)."""
    root = _canonical(vid)
    if not root or depth <= 0:
        return frozenset()
    kids_idx = _children_index()
    out: set[str] = set()
    frontier = {root}
    for _ in range(depth):
        nxt_layer: set[str] = set()
        for node in frontier:
            for ch in kids_idx.get(node, frozenset()):
                if ch not in out and ch != root:
                    out.add(ch)
                    nxt_layer.add(ch)
        frontier = nxt_layer
        if not frontier:
            break
    return frozenset(out)


@lru_cache(maxsize=8192)
def sibling_nodes(vid: str) -> frozenset[str]:
    """Other classes sharing a direct parent (excluding self)."""
    root = _canonical(vid)
    if not root:
        return frozenset()
    classes = _classes()
    kids_idx = _children_index()
    if root not in classes or classes[root].get("deprecated"):
        return frozenset()
    out: set[str] = set()
    for parent in classes[root].get("parents", []):
        for sib in kids_idx.get(parent, frozenset()):
            if sib != root and sib in classes and not classes[sib].get("deprecated"):
                out.add(sib)
    return frozenset(out)


@lru_cache(maxsize=8192)
def role_closure(vid: str) -> frozenset[str]:
    """Reachable role nodes and their superclasses."""
    root = _canonical(vid)
    if not root:
        return frozenset()
    classes = _classes()
    if root not in classes or classes[root].get("deprecated"):
        return frozenset()
    roles: set[str] = set()
    frontier_nodes = [root]
    visited_nodes: set[str] = set()
    while frontier_nodes:
        node = frontier_nodes.pop()
        if node in visited_nodes or node not in classes:
            continue
        visited_nodes.add(node)
        for role in classes[node].get("roles", []):
            if role in classes and not classes[role].get("deprecated"):
                roles.add(role)
        frontier_nodes.extend(classes[node].get("parents", []))
    out = set(roles)
    frontier = list(roles)
    seen_r = set(roles)
    while frontier:
        nxt = frontier.pop()
        if nxt not in classes:
            continue
        for parent in classes[nxt].get("parents", []):
            if parent in seen_r or parent not in classes:
                continue
            if classes[parent].get("deprecated"):
                continue
            seen_r.add(parent)
            out.add(parent)
            frontier.append(parent)
    return frozenset(out)


@lru_cache(maxsize=8192)
def relation_neighborhood(vid: str, *, use_roles: bool = True) -> frozenset[str]:
    base = subclass_closure(vid)
    if not use_roles:
        return base
    return frozenset(base | role_closure(vid))


def material_neighborhood(
    vid: str,
    *,
    children_depth: int = 0,
    use_siblings: bool = False,
) -> frozenset[str]:
    """Subclass-up neighborhood, optionally plus children and/or siblings."""
    root = _canonical(vid)
    if not root:
        return frozenset()
    out = set(subclass_closure(root))
    if children_depth > 0:
        out |= children_closure(root, depth=children_depth)
    if use_siblings:
        out |= sibling_nodes(root)
    return frozenset(out)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


@lru_cache(maxsize=1)
def _node_ic() -> dict[str, float]:
    classes = _classes()
    ids = [c for c, meta in classes.items() if not meta.get("deprecated")]
    if not ids:
        return {}
    freq: dict[str, int] = {}
    for cid in ids:
        for node in subclass_closure(cid):
            freq[node] = freq.get(node, 0) + 1
    n = float(len(ids))
    return {node: -math.log(max(cnt, 1) / n) for node, cnt in freq.items()}


def _weighted_jaccard(a: frozenset[str], b: frozenset[str], weights: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    w_inter = sum(weights.get(x, 0.0) for x in inter)
    w_union = sum(weights.get(x, 0.0) for x in union)
    if w_union <= 0:
        return _jaccard(a, b)
    return w_inter / w_union


@dataclass(frozen=True)
class OntSimConfig:
    """Configurable OntSim for combinatorial search."""

    children_depth: int = 0
    use_siblings: bool = False
    use_ic: bool = False
    role_mode: str = "none"  # none | union | sep | sep_if_no_exact
    alpha_role: float = 0.3
    gated_query: bool = False  # handled in recommend layer; kept for logging
    # metric: jaccard (default) | lin | resnik — applied on subclass / material neighborhood
    metric: str = "jaccard"
    # if True: add children only for the coarser endpoint (lower node IC)
    children_if_coarse: bool = False

    def key(self) -> str:
        return (
            f"ch{self.children_depth}"
            f"_sib{int(self.use_siblings)}"
            f"_ic{int(self.use_ic)}"
            f"_role{self.role_mode}"
            f"_a{self.alpha_role:.2f}"
            f"_gate{int(self.gated_query)}"
            f"_m{self.metric}"
            f"_cfc{int(self.children_if_coarse)}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _mica_ic(a: str, b: str) -> float:
    """Information content of the most informative common ancestor."""
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    inter = subclass_closure(ca) & subclass_closure(cb)
    if not inter:
        return 0.0
    ic = _node_ic()
    return max(ic.get(x, 0.0) for x in inter)


def ont_sim_lin(a: str, b: str) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    mica = _mica_ic(ca, cb)
    if mica <= 0:
        return 0.0
    ic = _node_ic()
    den = ic.get(ca, 0.0) + ic.get(cb, 0.0)
    if den <= 0:
        return 0.0
    return min(1.0, 2.0 * mica / den)


def ont_sim_resnik(a: str, b: str) -> float:
    """Resnik IC(MICA), scaled by max IC in the ontology to [0,1]."""
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    mica = _mica_ic(ca, cb)
    if mica <= 0:
        return 0.0
    ic = _node_ic()
    max_ic = max(ic.values()) if ic else 1.0
    return min(1.0, mica / max(max_ic, 1e-8))


def ont_sim(a: str, b: str, *, use_roles: bool = True) -> float:
    """Paper baseline: Jaccard over relation-expanded neighborhoods (roles unioned)."""
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    A = relation_neighborhood(ca, use_roles=use_roles)
    B = relation_neighborhood(cb, use_roles=use_roles)
    return _jaccard(A, B)


def ont_sim_sep(a: str, b: str, *, alpha_role: float = 0.3) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    j_sub = _jaccard(subclass_closure(ca), subclass_closure(cb))
    ra, rb = role_closure(ca), role_closure(cb)
    if not ra or not rb:
        return j_sub
    j_role = _jaccard(ra, rb)
    a_role = min(max(alpha_role, 0.0), 1.0)
    return (1.0 - a_role) * j_sub + a_role * j_role


def ont_sim_ic(a: str, b: str) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return _weighted_jaccard(subclass_closure(ca), subclass_closure(cb), _node_ic())


def ont_sim_ic_sep(a: str, b: str, *, alpha_role: float = 0.3) -> float:
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    j_sub = _weighted_jaccard(subclass_closure(ca), subclass_closure(cb), _node_ic())
    ra, rb = role_closure(ca), role_closure(cb)
    if not ra or not rb:
        return j_sub
    j_role = _jaccard(ra, rb)
    a_role = min(max(alpha_role, 0.0), 1.0)
    return (1.0 - a_role) * j_sub + a_role * j_role


def ont_sim_cfg(a: str, b: str, cfg: OntSimConfig) -> float:
    """General OntSim used by combinatorial search."""
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0

    if cfg.metric == "lin":
        base = ont_sim_lin(ca, cb)
        if cfg.role_mode in {"sep", "sep_if_no_exact"}:
            ra, rb = role_closure(ca), role_closure(cb)
            if ra and rb:
                a_role = min(max(cfg.alpha_role, 0.0), 1.0)
                return (1.0 - a_role) * base + a_role * _jaccard(ra, rb)
        return base
    if cfg.metric == "resnik":
        base = ont_sim_resnik(ca, cb)
        if cfg.role_mode in {"sep", "sep_if_no_exact"}:
            ra, rb = role_closure(ca), role_closure(cb)
            if ra and rb:
                a_role = min(max(cfg.alpha_role, 0.0), 1.0)
                return (1.0 - a_role) * base + a_role * _jaccard(ra, rb)
        return base

    if (
        cfg.role_mode == "union"
        and cfg.children_depth == 0
        and not cfg.use_siblings
        and not cfg.use_ic
        and not cfg.children_if_coarse
        and cfg.metric == "jaccard"
    ):
        return ont_sim(ca, cb, use_roles=True)

    ch_a = cfg.children_depth
    ch_b = cfg.children_depth
    if cfg.children_if_coarse:
        ic = _node_ic()
        # Lower IC → coarser class → expand children on that side only.
        ica, icb = ic.get(ca, 0.0), ic.get(cb, 0.0)
        ch_a = max(cfg.children_depth, 1) if ica <= icb else 0
        ch_b = max(cfg.children_depth, 1) if icb < ica else 0

    A = material_neighborhood(ca, children_depth=ch_a, use_siblings=cfg.use_siblings)
    B = material_neighborhood(cb, children_depth=ch_b, use_siblings=cfg.use_siblings)
    if cfg.use_ic:
        j_sub = _weighted_jaccard(A, B, _node_ic())
    else:
        j_sub = _jaccard(A, B)

    if cfg.role_mode == "none":
        return j_sub
    if cfg.role_mode == "union":
        Au = frozenset(A | role_closure(ca))
        Bu = frozenset(B | role_closure(cb))
        return _jaccard(Au, Bu) if not cfg.use_ic else _weighted_jaccard(Au, Bu, _node_ic())
    if cfg.role_mode in {"sep", "sep_if_no_exact"}:
        ra, rb = role_closure(ca), role_closure(cb)
        if not ra or not rb:
            return j_sub
        j_role = _jaccard(ra, rb)
        a_role = min(max(cfg.alpha_role, 0.0), 1.0)
        return (1.0 - a_role) * j_sub + a_role * j_role
    raise ValueError(f"Unknown role_mode: {cfg.role_mode}")


def clear_caches() -> None:
    subclass_closure.cache_clear()
    children_closure.cache_clear()
    sibling_nodes.cache_clear()
    role_closure.cache_clear()
    relation_neighborhood.cache_clear()
    _children_index.cache_clear()
    _node_ic.cache_clear()
    _classes.cache_clear()
