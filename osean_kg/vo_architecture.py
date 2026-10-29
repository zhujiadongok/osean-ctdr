"""Vaccine architecture from the advisor Vaccine Ontology file (vo.owl).

The SEA-CDM profile stores an older platform ID. Several of those IDs are a
different class in current VO. This module reads FOR_ADVISOR/vo.owl and follows
the crosswalk onto the official class so the apps can show parents and roles
without changing trial counts.
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path

import os

_ADVISOR = (
    Path(__file__).resolve().parents[1]
    / "SEA-CDM AACT vaccine remapped"
    / "research_factors"
    / "vo_mapping"
    / "FOR_ADVISOR"
)
# Prefer VO_OWL env, then local advisor copy, then data/vo.owl in the repo.
_OWL_CANDIDATES = [
    Path(p)
    for p in (
        os.environ.get("VO_OWL"),
        str(_ADVISOR / "vo.owl"),
        str(Path(__file__).resolve().parents[1] / "data" / "vo.owl"),
        str(Path(__file__).resolve().parents[2] / "data" / "vo.owl"),
    )
    if p
]
OWL_FILES = tuple(p for p in _OWL_CANDIDATES if p.exists()) or (_ADVISOR / "vo.owl",)

OBO = "http://purl.obolibrary.org/obo/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
HAS_ROLE = "RO_0000087"

# Profile ID used by the tables and the webpage -> official class in vo.owl.
# Counts stay on the profile ID. Only the architecture text uses the official class.
PROFILE_TO_OFFICIAL = {
    "VO_0000186": "VO_0000186",  # RNA vaccine
    "VO_0000369": "VO_0000367",  # live attenuated vaccine
    "VO_0000006": "VO_0000484",  # recombinant vector vaccine
    "VO_0000003": "VO_0000534",  # subunit vaccine
    "VO_0001482": "VO_0000195",  # conjugate vaccine
    "VO_0000741": "VO_0000315",  # inactivated vaccine
    "VO_0000011": "VO_0000032",  # DNA vaccine
    "VO_0000729": "VO_0000561",  # toxoid vaccine
}

MISSING_FROM_OWL: dict[str, str] = {}


def _local(iri: str | None) -> str | None:
    if not iri or not iri.startswith(OBO):
        return None
    token = iri[len(OBO) :]
    return token if token.startswith("VO_") else None


def _parse(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    classes: dict[str, dict] = {}
    for node in root.findall(f"{{{OWL_NS}}}Class"):
        vid = _local(node.get(f"{{{RDF}}}about"))
        if not vid:
            continue
        labels = [t.text.strip() for t in node.findall(f"{{{RDFS}}}label") if t.text]
        defin = [
            t.text.strip()
            for t in node.findall(f"{{{OBO}}}IAO_0000115")
            if t.text
        ]
        deprecated = any(
            (t.text or "").strip().lower() == "true"
            for t in node.findall(f"{{{OWL_NS}}}deprecated")
        )
        parents = []
        roles = []
        for sub in node.findall(f"{{{RDFS}}}subClassOf"):
            parent = _local(sub.get(f"{{{RDF}}}resource"))
            if parent:
                parents.append(parent)
            restriction = sub.find(f"{{{OWL_NS}}}Restriction")
            if restriction is None:
                continue
            prop = restriction.find(f"{{{OWL_NS}}}onProperty")
            filler = restriction.find(f"{{{OWL_NS}}}someValuesFrom")
            prop_iri = prop.get(f"{{{RDF}}}resource") if prop is not None else ""
            role = _local(filler.get(f"{{{RDF}}}resource") if filler is not None else None)
            if prop_iri.endswith(HAS_ROLE) and role:
                roles.append(role)
        classes[vid] = {
            "id": vid,
            "label": labels[0] if labels else vid,
            "definition": defin[0] if defin else "",
            "deprecated": deprecated,
            "parents": parents,
            "roles": roles,
        }
    return classes


@lru_cache(maxsize=1)
def load_classes(paths: tuple[str, ...] | None = None) -> dict[str, dict]:
    files = [Path(p) for p in paths] if paths else list(OWL_FILES)
    merged: dict[str, dict] = {}
    for owl in files:
        if owl.exists():
            merged.update(_parse(owl))
    return merged


def _chain(classes: dict[str, dict], vid: str, limit: int = 6) -> list[str]:
    out = []
    seen = {vid}
    frontier = list(classes.get(vid, {}).get("parents", []))
    while frontier and len(out) < limit:
        nxt = frontier.pop(0)
        if nxt in seen or nxt not in classes:
            continue
        seen.add(nxt)
        if classes[nxt].get("deprecated"):
            continue
        out.append(f"{classes[nxt]['label']} ({nxt})")
        frontier.extend(classes[nxt].get("parents", []))
    return out


def architecture_for_profile(profile_vo: str) -> dict | None:
    """Official class, parents, and roles for a platform ID stored on the profile."""
    if not profile_vo or profile_vo in {"ALL", "(any)"}:
        return None
    if profile_vo in MISSING_FROM_OWL:
        return {"profile_id": profile_vo, "missing": MISSING_FROM_OWL[profile_vo]}
    official = PROFILE_TO_OFFICIAL.get(profile_vo)
    if not official:
        return None
    classes = load_classes()
    node = classes.get(official)
    if not node or node.get("deprecated"):
        return {"profile_id": profile_vo, "missing": f"{official} is missing or deprecated in vo.owl"}
    roles = []
    for rid in node.get("roles", []):
        role = classes.get(rid)
        if role and not role.get("deprecated"):
            roles.append(f"{role['label']} ({rid})")
    return {
        "profile_id": profile_vo,
        "official_id": official,
        "label": node["label"],
        "definition": node["definition"],
        "parents": _chain(classes, official),
        "roles": roles,
        "id_differs": official != profile_vo,
    }


def architecture_text(profile_vo: str) -> str | None:
    info = architecture_for_profile(profile_vo)
    if not info:
        return None
    if info.get("missing"):
        return f"Ontology architecture: {info['missing']}."
    bits = [f"Ontology architecture: {info['label']} ({info['official_id']})"]
    if info.get("roles"):
        bits.append("role " + "; ".join(info["roles"]))
    if info.get("parents"):
        bits.append("under " + " → ".join(info["parents"][:4]))
    if info.get("definition"):
        bits.append(info["definition"])
    if info.get("id_differs"):
        bits.append(
            f"Trial rows keep the stored tag {info['profile_id']}. "
            "The class above is the matching class in the Vaccine Ontology import."
        )
    return ". ".join(bits)


@lru_cache(maxsize=1)
def _product_rows() -> tuple[tuple[str, str, str], ...]:
    path = Path(__file__).resolve().parent / "ckpts" / "sea_ml_v14" / "product_vo_map.csv"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(
            (row["profile_vo"], row["product_vo_id"], row["product_label"])
            for row in csv.DictReader(handle)
        )


@lru_cache(maxsize=1)
def browsable_ontology() -> dict:
    """Parents and children among the platform and product classes we actually use.

    The full import has hundreds of classes. This keeps the classes on a path
    from vaccine down to a mapped platform or product, so the page can show
    one level of buttons at a time.
    """
    classes = load_classes()
    keep = set(PROFILE_TO_OFFICIAL.values()) | {row[1] for row in _product_rows()} | {"VO_0000001"}
    changed = True
    while changed:
        changed = False
        for vid in list(keep):
            for parent in classes.get(vid, {}).get("parents", []):
                if parent in classes and not classes[parent].get("deprecated") and parent not in keep:
                    keep.add(parent)
                    changed = True
    children: dict[str, list[str]] = {vid: [] for vid in keep}
    for vid in keep:
        for parent in classes.get(vid, {}).get("parents", []):
            if parent in keep and vid not in children[parent]:
                children[parent].append(vid)
    for vids in children.values():
        vids.sort(key=lambda item: classes.get(item, {}).get("label", item).lower())
    counts = Counter(row[1] for row in _product_rows())
    return {"children": children, "counts": counts}


def ontology_children(vid: str) -> list[dict]:
    graph = browsable_ontology()
    classes = load_classes()
    official_to_profile = {official: profile for profile, official in PROFILE_TO_OFFICIAL.items()}
    out = []
    for child in graph["children"].get(vid, []):
        node = classes.get(child)
        if not node:
            continue
        out.append(
            {
                "id": child,
                "label": node["label"],
                "n_trials": int(graph["counts"].get(child, 0)),
                "profile_id": official_to_profile.get(child, ""),
                "n_children": len(graph["children"].get(child, [])),
            }
        )
    return out


def ancestors_of(classes: dict, node_id: str) -> set[str]:
    seen = {node_id}
    frontier = list(classes.get(node_id, {}).get("parents", []))
    frontier += list(classes.get(node_id, {}).get("roles", []))
    while frontier:
        nxt = frontier.pop()
        if nxt in seen or nxt not in classes or classes[nxt].get("deprecated"):
            continue
        seen.add(nxt)
        frontier.extend(classes[nxt].get("parents", []))
    return seen


def similar_classes(vid: str, k: int = 5) -> list[dict]:
    """Other classes that share the most parents and roles with this one."""
    classes = load_classes()
    graph = browsable_ontology()
    nodes = [item for item in graph["children"] if item in classes and item != vid]
    platform_sets = []
    for official in PROFILE_TO_OFFICIAL.values():
        if official in classes:
            platform_sets.append(ancestors_of(classes, official))
    common = set.intersection(*platform_sets) if platform_sets else set()

    mine = ancestors_of(classes, vid) - common
    scored = []
    for other in nodes:
        theirs = ancestors_of(classes, other) - common
        union = len(mine | theirs) or 1
        scored.append((len(mine & theirs) / union, other))
    scored.sort(key=lambda item: (-item[0], item[1]))
    out = []
    for score, other in scored[:k]:
        if score <= 0:
            continue
        out.append({"id": other, "label": classes[other]["label"], "jaccard": round(score, 2)})
    return out


def shared_labels(left: str, right: str) -> list[str]:
    """Parents and roles that two classes share, after dropping nodes every platform has."""
    classes = load_classes()
    platform_sets = [
        ancestors_of(classes, official)
        for official in PROFILE_TO_OFFICIAL.values()
        if official in classes
    ]
    common = set.intersection(*platform_sets) if platform_sets else set()
    shared = (ancestors_of(classes, left) & ancestors_of(classes, right)) - common - {left, right}
    labels = [classes[vid]["label"] for vid in shared if vid in classes]
    labels.sort()
    return labels


@lru_cache(maxsize=1)
def _roles_shared_by_every_platform() -> frozenset[str]:
    """Roles reached from every stored platform, such as vaccine role itself."""
    classes = load_classes()
    sets = []
    for official in PROFILE_TO_OFFICIAL.values():
        node = classes.get(official)
        if not node:
            continue
        reached: set[str] = set()
        frontier = list(node.get("roles", []))
        seen: set[str] = set()
        while frontier:
            role = frontier.pop()
            if role in seen or role not in classes or classes[role].get("deprecated"):
                continue
            seen.add(role)
            reached.add(role)
            frontier.extend(classes[role].get("parents", []))
        if reached:
            sets.append(reached)
    if not sets:
        return frozenset()
    return frozenset(set.intersection(*sets))


def inferred_role_records(vid: str) -> list[dict]:
    """Roles a class gets by relation composition, not by its own has-role line.

    Two compositions: a parent class states the role, or a stated role has a
    parent role. Both are subclass steps.
    """
    classes = load_classes()
    node = classes.get(vid)
    if not node or node.get("deprecated"):
        return []

    def climb(role_id: str) -> list[str]:
        found = []
        frontier = [role_id]
        seen: set[str] = set()
        while frontier:
            role = frontier.pop()
            if role in seen or role not in classes or classes[role].get("deprecated"):
                continue
            seen.add(role)
            found.append(role)
            frontier.extend(classes[role].get("parents", []))
        return found

    direct = set(node.get("roles", []))
    records = []
    seen_roles: set[str] = set()

    def add(role_id: str, via_id: str, kind: str) -> None:
        if role_id in direct or role_id in seen_roles or role_id not in classes:
            return
        if classes[role_id].get("deprecated"):
            return
        seen_roles.add(role_id)
        records.append(
            {
                "role_id": role_id,
                "role_label": classes[role_id]["label"],
                "via_id": via_id,
                "via_label": classes.get(via_id, {}).get("label", via_id),
                "kind": kind,
            }
        )

    for role in list(direct):
        for parent in climb(role)[1:]:
            add(parent, role, "role parent")
    seen_classes = {vid}
    frontier = list(node.get("parents", []))
    while frontier:
        parent = frontier.pop()
        if parent in seen_classes or parent not in classes or classes[parent].get("deprecated"):
            continue
        seen_classes.add(parent)
        for role in classes[parent].get("roles", []):
            chain = climb(role)
            if chain:
                add(chain[0], parent, "parent class")
            for higher in chain[1:]:
                add(higher, chain[0], "role parent")
        frontier.extend(classes[parent].get("parents", []))
    shared = _roles_shared_by_every_platform()
    return [row for row in records if row["role_id"] not in shared]


def inferred_roles(vid: str) -> list[str]:
    """Short text for each inferred role."""
    return [
        f"{row['role_label']} via {row['via_label']}"
        for row in inferred_role_records(vid)
    ]


def ontology_node(vid: str) -> dict | None:
    classes = load_classes()
    node = classes.get(vid)
    if not node or node.get("deprecated"):
        return None
    graph = browsable_ontology()
    official_to_profile = {official: profile for profile, official in PROFILE_TO_OFFICIAL.items()}
    roles = []
    for rid in node.get("roles", []):
        role = classes.get(rid)
        if role and not role.get("deprecated"):
            roles.append(f"{role['label']} ({rid})")
    return {
        "id": vid,
        "label": node["label"],
        "definition": node["definition"],
        "roles": roles,
        "parents": [pid for pid in node.get("parents", []) if pid in graph["children"]],
        "n_trials": int(graph["counts"].get(vid, 0)),
        "profile_id": official_to_profile.get(vid, ""),
    }


def product_coverage_text(profile_vo: str) -> str | None:
    """How many trials of this stored platform have a product class, and which classes."""
    if not profile_vo or profile_vo in {"ALL", "(any)"}:
        return None
    hit = [row for row in _product_rows() if row[0] == profile_vo]
    if not hit:
        return None
    counts = Counter((vid, label) for _, vid, label in hit)
    bits = [f"{n} as {label} ({vid})" for (vid, label), n in counts.most_common(3)]
    return (
        f"Product classes matched inside this platform: {len(hit)} trials. "
        + "; ".join(bits)
        + "."
    )
