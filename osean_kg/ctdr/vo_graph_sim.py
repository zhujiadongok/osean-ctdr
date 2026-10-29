"""VO ontology-graph similarity via relation-expanded neighborhoods.

OntSim = Jaccard over a neighborhood that can include:
  - the class itself
  - superclass nodes along subClassOf
  - reachable role nodes (has_role) and their superclasses (optional)

Roles are NOT called \"ancestors\"; they are a separate relation expansion.
"""

from __future__ import annotations

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
def role_closure(vid: str) -> frozenset[str]:
    """Reachable role nodes and their superclasses (not material subclass ancestors)."""
    root = _canonical(vid)
    if not root:
        return frozenset()
    classes = _classes()
    if root not in classes or classes[root].get("deprecated"):
        return frozenset()
    roles: set[str] = set()
    seen = {root}
    # Direct and inherited roles via walking subclass parents collecting roles.
    frontier_nodes = [root]
    visited_nodes = set()
    while frontier_nodes:
        node = frontier_nodes.pop()
        if node in visited_nodes or node not in classes:
            continue
        visited_nodes.add(node)
        for role in classes[node].get("roles", []):
            if role in classes and not classes[role].get("deprecated"):
                roles.add(role)
        frontier_nodes.extend(classes[node].get("parents", []))
    # Expand each role along its own subclass parents.
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
    """Class itself + superclass nodes (+ optional role-expanded nodes)."""
    base = subclass_closure(vid)
    if not use_roles:
        return base
    return frozenset(base | role_closure(vid))


def ont_sim(a: str, b: str, *, use_roles: bool = True) -> float:
    """Jaccard over relation-expanded neighborhoods."""
    ca, cb = _canonical(a), _canonical(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    A = relation_neighborhood(ca, use_roles=use_roles)
    B = relation_neighborhood(cb, use_roles=use_roles)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    if inter == 0:
        return 0.0
    return inter / len(A | B)


def clear_caches() -> None:
    subclass_closure.cache_clear()
    role_closure.cache_clear()
    relation_neighborhood.cache_clear()
    _classes.cache_clear()
