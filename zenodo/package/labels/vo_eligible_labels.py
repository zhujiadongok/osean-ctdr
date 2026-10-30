"""Canonical VO ID → display label for paper tables/figures (matches S7 / vo.owl)."""

from __future__ import annotations

# Labels from vo.owl rdfs:label (2026-06-12 release); keep in sync with draft Table S7.
# Post primary-investigational mapping remediation eligible set (23 ontology-connected classes).
VO_ELIGIBLE_LABELS: dict[str, str] = {
    "VO_0000534": "subunit vaccine",
    "VO_0000561": "toxoid vaccine",
    "VO_0000032": "DNA vaccine",
    "VO_0004673": "quadrivalent influenza vaccine",
    "VO_0000367": "live attenuated vaccine",
    "VO_0001178": "live attenuated influenza vaccine",
    "VO_0003150": "Hepatitis B surface antigen vaccine",
    "VO_0000484": "recombinant vector vaccine",
    "VO_0000642": "influenza virus vaccine",
    "VO_0000771": "BCG vaccine",
    "VO_0000761": "malaria vaccine",
    "VO_0006034": "meningococcal vaccine (unknown formulation/serogroups)",
    "VO_0000047": "Fluzone",
    "VO_0000577": "peptide vaccine",
    "VO_0003075": "Gardasil 9 vaccine",
    "VO_0007062": "Quadrivalent HPV (6/11/16/18) recombinant vaccine",
    "VO_0004246": "Anthrax vaccine adsorbed with squalene adjuvant",
    "VO_0007319": "autologous tumor cell vaccine",
    "VO_0005158": "Oxford-AstraZeneca COVID-19 vaccine",
    "VO_0007063": "Recombinant HPV nonavalent vaccine",
    "VO_0000186": "RNA vaccine",
    "VO_0003464": "Hepatitis A vaccine, inactivated",
    "VO_0003834": "poliovirus vaccine inactivated",
}

# Slightly shorter y-tick forms for Figure 6B (same classes; not product names).
VO_ELIGIBLE_LABELS_SHORT: dict[str, str] = {
    "VO_0000534": "subunit vaccine",
    "VO_0000561": "toxoid vaccine",
    "VO_0000032": "DNA vaccine",
    "VO_0004673": "quadrivalent influenza",
    "VO_0000367": "live attenuated vaccine",
    "VO_0001178": "live attenuated influenza",
    "VO_0003150": "HBsAg vaccine",
    "VO_0000484": "recombinant vector",
    "VO_0000642": "influenza virus vaccine",
    "VO_0000771": "BCG vaccine",
    "VO_0000761": "malaria vaccine",
    "VO_0006034": "meningococcal (unk. form.)",
    "VO_0000047": "Fluzone",
    "VO_0000577": "peptide vaccine",
    "VO_0003075": "Gardasil 9",
    "VO_0007062": "HPV 4-valent recombinant",
    "VO_0004246": "Anthrax adsorbed + squalene",
    "VO_0007319": "autologous tumor cell",
    "VO_0005158": "Oxford-AstraZeneca COVID-19",
    "VO_0007063": "HPV 9-valent recombinant",
    "VO_0000186": "RNA vaccine",
    "VO_0003464": "Hepatitis A inactivated",
    "VO_0003834": "IPV (inactivated polio)",
}
