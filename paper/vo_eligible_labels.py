"""Canonical VO ID → display label for paper tables/figures (matches S7 / vo.owl)."""

from __future__ import annotations

# Labels from vo.owl rdfs:label (2026-06-12 release); keep in sync with draft Table S7.
VO_ELIGIBLE_LABELS: dict[str, str] = {
    "VO_0000561": "toxoid vaccine",
    "VO_0004673": "quadrivalent influenza vaccine",
    "VO_0001178": "live attenuated influenza vaccine",
    "VO_0000642": "influenza virus vaccine",
    "VO_0005198": "COVID-19 RNA vaccine",
    "VO_0000761": "malaria vaccine",
    "VO_0000047": "Fluzone",
    "VO_0007062": "Quadrivalent HPV (6/11/16/18) recombinant vaccine",
    "VO_0007063": "Recombinant HPV nonavalent vaccine",
}

# Slightly shorter y-tick forms for Figure 6B (same classes; not product names).
VO_ELIGIBLE_LABELS_SHORT: dict[str, str] = {
    "VO_0000561": "toxoid vaccine",
    "VO_0004673": "quadrivalent influenza vaccine",
    "VO_0001178": "live attenuated influenza vaccine",
    "VO_0000642": "influenza virus vaccine",
    "VO_0005198": "COVID-19 RNA vaccine",
    "VO_0000761": "malaria vaccine",
    "VO_0000047": "Fluzone",
    "VO_0007062": "HPV 4-valent recombinant",
    "VO_0007063": "HPV 9-valent recombinant",
}
