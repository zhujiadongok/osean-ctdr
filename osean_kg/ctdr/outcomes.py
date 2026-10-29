"""Attach AACT primary-outcome families (supplement; not a SEA-CDM rebuild)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AACT_OUTCOMES = (
    Path(__file__).resolve().parents[2]
    / "AACT Database"
    / "vaccine_nct_subset"
    / "outcomes.txt"
)


def categorize_outcome(title: str, description: str = "") -> str:
    t = f"{title} {description}".lower()
    rules = (
        (
            "safety",
            (
                "adverse",
                "safety",
                "reactogen",
                "sae",
                "tolerab",
                "solicited",
                "injection site",
                "serious adverse",
            ),
        ),
        (
            "immunogenicity",
            (
                "antibody",
                "igg",
                "igna",
                "neutraliz",
                "immunogen",
                "titer",
                "gmt",
                "seroconvers",
                "seropositiv",
                "elisa",
                "t-cell",
                "t cell",
                "cellular immun",
                "hi titer",
                "hemagglutination",
            ),
        ),
        (
            "clinical_efficacy",
            (
                "efficacy",
                "effectiveness",
                "infection",
                "incidence",
                "hospital",
                "death",
                "mortality",
                "lab-confirmed",
                "laboratory-confirmed",
                "pcr-confirmed",
                "disease rate",
                "cases of",
            ),
        ),
    )
    for lab, keys in rules:
        if any(k in t for k in keys):
            return lab
    return "other"


def primary_outcome_family_by_nct() -> pd.Series:
    """One family per NCT: majority vote over PRIMARY outcomes; tie → safety>immuno>clinical>other."""
    if not AACT_OUTCOMES.exists():
        return pd.Series(dtype=str)
    out = pd.read_csv(AACT_OUTCOMES, sep="|", dtype=str).fillna("")
    out["nct_id"] = out["nct_id"].astype(str).str.upper()
    prim = out[out["outcome_type"].str.upper() == "PRIMARY"].copy()
    if prim.empty:
        return pd.Series(dtype=str)
    prim["family"] = [
        categorize_outcome(a, b) for a, b in zip(prim["title"], prim["description"])
    ]
    prefer = {"safety": 0, "immunogenicity": 1, "clinical_efficacy": 2, "other": 3}

    def _pick(s: pd.Series) -> str:
        counts = s.value_counts()
        best = counts.max()
        cands = [k for k, v in counts.items() if v == best]
        return sorted(cands, key=lambda x: prefer.get(x, 9))[0]

    return prim.groupby("nct_id")["family"].agg(_pick)
