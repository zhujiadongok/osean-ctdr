"""Build expanded primary-investigational VO corrections from name patterns + advisor overlay.

Merges hand-audited rows with high-confidence corpus rules based on
specific_vaccine_name / disease. Does not invent product IDs when unsure —
uses platform/material classes already used in vo_corrections_primary.csv.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from vo_architecture import load_classes  # noqa: E402

AUDIT = cfg.CTDR_DIR / "optimize" / "mapping_audit"
ADVISOR_CORR = AUDIT / "vo_corrections_primary.csv"
OUT_CORR = AUDIT / "vo_corrections_primary.csv"
OUT_BULK_LOG = AUDIT / "bulk_corrections_log.csv"
OUT_SUMMARY = AUDIT / "bulk_corrections_summary.json"

# Classes already used in advisor primary-investigational fixes
VO_COVID = "VO_0004908"
VO_BCG = "VO_0000771"
VO_HBSAG = "VO_0003150"
VO_HEPE = "VO_0006017"
VO_GARDASIL9 = "VO_0003075"  # Gardasil-9 lineage used in advisor overlay
VO_VLP = "VO_0004914"
VO_PROTEIN = "VO_0000483"  # recombinant protein / subunit (official crosswalk target)
VO_INACTIVATED = "VO_0000315"
VO_LIVE = "VO_0000367"
VO_VECTOR = "VO_0000484"
VO_DNA = "VO_0000032"
VO_RNA = "VO_0000186"
VO_PEPTIDE = "VO_0000577"
VO_IPV = "VO_0003834"  # poliovirus vaccine inactivated
VO_HEPA = "VO_0003464"  # Hepatitis A Vaccine, Inactivated
VO_VZV = "VO_0000669"  # Varicella-zoster virus vaccine
VO_SHINGRIX = "VO_0003317"
VO_GVAX_PANC = "VO_0007610"
VO_TUMOR_AUTO = "VO_0007319"
VO_DC_OVARIAN_MRNA = "VO_0007132"
VO_PORCINE_ROTA = "VO_0001495"
VO_JE = "VO_0000665"
VO_FLU_INACT = "VO_0001176"  # inactivated influenza vaccine (class)
VO_MEN_CONJ = "VO_0003441"
VO_MEN_UNK = "VO_0006034"
VO_OXFORD_AZ = "VO_0005158"
VO_HYPERACUTE_MEL = "VO_0007676"
VO_CONJUGATE = "VO_0000195"
VO_DC = "VO_0001029"
VO_ANTHRAX = None  # resolve from OWL if possible


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _resolve_anthrax(classes: dict) -> str | None:
    for vid, meta in classes.items():
        lab = _norm(meta.get("label", ""))
        if "anthrax vaccine adsorbed" in lab or lab == "anthrax vaccine":
            return vid
    for vid, meta in classes.items():
        lab = _norm(meta.get("label", ""))
        if "anthrax" in lab and "vaccine" in lab and not meta.get("deprecated"):
            return vid
    return None


def _resolve_label(classes: dict, *needles: str) -> str | None:
    needles_l = [n.lower() for n in needles]
    for vid, meta in classes.items():
        if meta.get("deprecated"):
            continue
        lab = _norm(meta.get("label", ""))
        if all(n in lab for n in needles_l):
            return vid
    return None


def main() -> None:
    classes = load_classes()
    anthrax = _resolve_anthrax(classes) or "VO_0000534"  # fallback: keep prior if missing
    cervarix = _resolve_label(classes, "cervarix") or VO_VLP
    recombivax = _resolve_label(classes, "recombivax") or VO_HBSAG
    engerix = _resolve_label(classes, "engerix") or VO_HBSAG

    # Baseline = raw mapped labels BEFORE overlay (so re-runs stay stable)
    from vo_architecture import PROFILE_TO_OFFICIAL  # noqa: E402

    prof = pd.read_csv(cfg.PROFILE, dtype=str).fillna("")
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()
    link = pd.read_csv(cfg.DISEASE_VO_MAP, dtype=str).fillna("")
    link["nct_id"] = link["nct_id"].astype(str).str.upper()
    link = link.drop_duplicates("nct_id")
    base = prof.merge(link[["nct_id", "arch_vo"]], on="nct_id", how="left")
    base["vo_platform"] = base["vaccine_type_vo_id"].map(
        lambda v: PROFILE_TO_OFFICIAL.get(
            str(v), str(v) if str(v).startswith("VO_") else "unknown"
        )
    )
    base["vo_platform"] = base["vo_platform"].replace({"": "unknown", "nan": "unknown"})
    arch = base["arch_vo"].astype(str)
    base["arch_vo"] = np.where(arch.str.startswith("VO_"), arch, base["vo_platform"])
    base["arch_vo"] = base["arch_vo"].replace({"": "unknown", "nan": "unknown"})

    studies = pd.read_csv(cfg.CTDR_DIR / "ctdr_studies.csv", dtype=str).fillna("")
    # attach baseline labels onto cohort rows (names/disease from studies)
    bmap = base.set_index("nct_id")[["arch_vo", "vo_platform"]]
    studies = studies.copy()
    studies["base_arch"] = studies["nct_id"].map(
        lambda x: bmap.at[x, "arch_vo"] if x in bmap.index else ""
    )
    studies["base_plat"] = studies["nct_id"].map(
        lambda x: bmap.at[x, "vo_platform"] if x in bmap.index else ""
    )
    rows: dict[str, dict[str, str]] = {}

    # Seed ALL prior corrections so re-runs after upstream apply stay complete.
    ADVISOR_NCTS = {
        "NCT04617483",
        "NCT05266898",
        "NCT05985681",
        "NCT01226758",
        "NCT00231816",
        "NCT00197301",
        "NCT01073293",
        "NCT01014845",
        "NCT03718130",
        "NCT04362124",
        "NCT05091307",
        "NCT05519839",
        "NCT06279871",
        "NCT06800131",
        "NCT05749926",
        "NCT02506933",
    }
    seed_path = AUDIT / "bulk_corrections_log.csv"
    if not seed_path.is_file():
        seed_path = ADVISOR_CORR
    if seed_path.is_file():
        seed = pd.read_csv(seed_path, dtype=str).fillna("")
        for r in seed.itertuples(index=False):
            nct = str(r.nct_id).upper()
            src = str(getattr(r, "source", "") or "")
            if not src:
                src = "advisor" if nct in ADVISOR_NCTS else "seed"
            if nct in ADVISOR_NCTS:
                src = "advisor"
            rows[nct] = {
                "nct_id": nct,
                "arch_vo": str(r.arch_vo),
                "vo_platform": str(r.vo_platform),
                "action": "fix",
                "rationale": str(getattr(r, "rationale", "prior")),
                "source": src,
            }

    def propose(nct: str, arch: str, plat: str, rationale: str, source: str) -> None:
        nct = nct.upper()
        if nct in rows and rows[nct]["source"] in {
            "advisor",
            "manual_ctgov",
            "manual_ctgov_final22",
        }:
            return  # never override hand fixes
        cur = studies.loc[studies["nct_id"] == nct]
        if cur.empty:
            return
        cur = cur.iloc[0]
        # Skip if already at target in baseline AND already seeded
        if str(cur["base_arch"]) == arch and str(cur["base_plat"]) == plat:
            if nct not in rows:
                # still record so upstream re-apply stays complete after prior upstream write
                rows[nct] = {
                    "nct_id": nct,
                    "arch_vo": arch,
                    "vo_platform": plat,
                    "action": "fix",
                    "rationale": rationale,
                    "source": source,
                }
            return
        rows[nct] = {
            "nct_id": nct,
            "arch_vo": arch,
            "vo_platform": plat,
            "action": "fix",
            "rationale": rationale,
            "source": source,
        }

    for r in studies.itertuples(index=False):
        nct = str(r.nct_id).upper()
        name = _norm(r.specific_vaccine_name)
        disease = str(r.disease)
        arch = str(r.base_arch)
        plat = str(r.base_plat)
        conds = _norm(r.disease_conditions)

        # --- high-confidence name → Material/platform ---

        # 1) BCG as primary (skip MVA85A prime / BCG boost → MVA is primary)
        if re.search(r"\bbcg\b|tice bcg|rbcg|bacillus calmette", name):
            if "mva85a" in name or re.search(r"\bmva\d*", name):
                # MVA prime / BCG boost: primary = vectored TB candidate, not BCG Material
                if plat != VO_VECTOR:
                    propose(
                        nct,
                        arch if arch.startswith("VO_") and arch not in {VO_LIVE, VO_COVID, VO_BCG} else VO_VECTOR,
                        VO_VECTOR,
                        "MVA±BCG regimen; primary vectored candidate not BCG-only",
                        "rule_mva_bcg",
                    )
                continue
            if arch != VO_BCG or plat not in {VO_LIVE, "unknown"}:
                propose(
                    nct,
                    VO_BCG,
                    VO_LIVE,
                    "BCG primary by name; Material=BCG, platform=live attenuated",
                    "rule_bcg",
                )
                continue

        # 2) Inactivated SARS-CoV / COVID with live-attenuated platform
        if (
            ("inactiv" in name and ("sars" in name or "covid" in name or disease == "covid"))
            or name.startswith("uv inactivated sars")
        ) and plat == VO_LIVE:
            mat = VO_COVID
            propose(
                nct,
                mat,
                VO_INACTIVATED,
                "Inactivated SARS/COVID; platform inactivated not live attenuated",
                "rule_inact_covid",
            )
            continue

        # 3) IPV / inactivated poliovirus
        if re.search(
            r"inactivated polio|inactivated poliovirus|poliovirus vaccine inactivated|\bipol\b|inactivated poliomyelitis",
            name,
        ):
            combo = bool(
                re.search(
                    r"diphtheria|tetanus|pertussis|haemophilus|hepatitis b|dtap|pentacel|infanrix",
                    name,
                )
            )
            if combo:
                # Combo schedule vaccine: fix false live/DNA platform only; keep Material
                if plat in {VO_LIVE, VO_DNA, VO_VECTOR}:
                    propose(
                        nct,
                        arch,
                        VO_INACTIVATED,
                        "Combo includes IPV; platform inactivated (Material unchanged)",
                        "rule_ipv_combo_plat",
                    )
                continue
            if arch != VO_IPV or plat != VO_INACTIVATED:
                propose(
                    nct,
                    VO_IPV,
                    VO_INACTIVATED,
                    "Inactivated poliovirus vaccine; not live/DNA mislabel",
                    "rule_ipv",
                )
                continue

        # 4) Hepatitis A (inactivated)
        if re.search(r"hepatitis a vaccine|\bhavrix\b|\bvaqta\b", name) and "dna" not in name:
            if arch != VO_HEPA or plat != VO_INACTIVATED:
                propose(
                    nct,
                    VO_HEPA,
                    VO_INACTIVATED,
                    "Hepatitis A inactivated vaccine",
                    "rule_hepa",
                )
                continue

        # 5) Standard Hepatitis B vaccine name → HBsAg + protein subunit
        if name in {
            "hepatitis b vaccine",
            "hepatitis b",
            "recombivax hb",
            "engerix-b",
            "engerix b",
        } or (name.startswith("hepatitis b vaccine") and "dna" not in name):
            if "dna" in name:
                pass
            else:
                mat = VO_HBSAG
                if "recombivax" in name:
                    mat = recombivax
                elif "engerix" in name:
                    mat = engerix
                if arch != mat or plat != VO_PROTEIN:
                    propose(
                        nct,
                        mat,
                        VO_PROTEIN,
                        "HepB recombinant protein / HBsAg; not vector/peptide/DNA mislabel",
                        "rule_hepb",
                    )
                    continue

        # 6) Recombinant Hepatitis E → HepE protein
        if "hepatitis e" in name and (
            "recombinant" in name or "protein" in name or arch == VO_VECTOR
        ):
            propose(
                nct,
                VO_HEPE,
                VO_PROTEIN,
                "Recombinant HepE protein; not vector",
                "rule_hepe",
            )
            continue

        # 7) HPV L1 VLP / Cervarix / Gardasil
        if any(
            k in name
            for k in (
                "virus-like particle",
                "virus like particle",
                "gardasil",
                "cervarix",
                "v503",
                "hpv 16/18 l1",
                "hpv-16/18 l1",
            )
        ):
            if "dna" in name or "e7" in name:
                pass
            else:
                if "gardasil" in name or "v503" in name or "9-valent" in name or "nonavalent" in name:
                    mat = VO_GARDASIL9
                elif "cervarix" in name or "16/18 l1" in name or "as04" in name:
                    mat = cervarix
                else:
                    mat = VO_VLP
                if arch != mat or plat != VO_VLP:
                    propose(
                        nct,
                        mat,
                        VO_VLP,
                        "HPV L1 VLP / licensed HPV vaccine; VLP not recombinant vector",
                        "rule_hpv_vlp",
                    )
                    continue

        # 8) Shingrix / heat-treated VZV (V212)
        if "shingrix" in name:
            propose(nct, VO_SHINGRIX, VO_PROTEIN, "Shingrix gE subunit", "rule_shingrix")
            continue
        if "heat-treated varicella" in name or "v212" in name:
            propose(
                nct,
                VO_VZV if arch in {VO_VZV, VO_DNA, VO_LIVE} else arch,
                VO_INACTIVATED,
                "Heat-treated VZV V212; inactivated platform not DNA",
                "rule_v212",
            )
            continue

        # 9) Anthrax AVA with influenza Material
        if ("anthrax" in name or name == "ava" or "anthrax vaccine adsorbed" in name) and (
            disease == "influenza"
            or arch in {"VO_0000642", "VO_0001236", "VO_0004673", "VO_0000047"}
        ):
            propose(
                nct,
                anthrax if str(anthrax).startswith("VO_") else arch,
                VO_INACTIVATED,
                "Anthrax Vaccine Adsorbed primary; not influenza Material",
                "rule_ava",
            )
            continue

        # 10) Explicit mRNA / RNA product names → RNA platform
        if re.search(
            r"mrnas?\b|mrna-|modmrna|sa.?rna|self-amplif|cv7201|bnt162|mrna\s*-?\s*\d+",
            name,
        ):
            # dendritic-cell mRNA vaccines: RNA platform + DC product class when known
            if "dendritic" in name or "langerhans" in name:
                if "ovarian" in name or "htert" in name:
                    mat = VO_DC_OVARIAN_MRNA
                elif arch not in {VO_VECTOR, VO_DNA, VO_LIVE}:
                    mat = arch
                else:
                    mat = "VO_0001029"  # autologous dendritic cell melanoma vaccine (DC class)
                propose(
                    nct,
                    mat,
                    VO_RNA,
                    "mRNA-loaded DC vaccine; RNA platform not vector/DNA",
                    "rule_mrna_dc",
                )
                continue

            if disease == "covid" or "covid" in conds or "sars" in name:
                mat = VO_COVID
            elif arch not in {VO_VECTOR, VO_DNA, VO_LIVE, "VO_0000642", "VO_0001236", "VO_0004673"}:
                mat = arch
            else:
                mat = VO_RNA  # product class unknown; RNA platform class as Material fallback
            if mat != arch or plat != VO_RNA:
                propose(
                    nct,
                    mat,
                    VO_RNA,
                    "Named mRNA/RNA vaccine; RNA Material/platform (not vector/DNA/flu mislabel)",
                    "rule_mrna",
                )
                continue

        # 11a) GM-CSF / tumor cell vaccines before false-DNA catch-all
        if re.search(r"gm-csf.*(cell|myeloma|breast|secreting)|allogeneic gm-csf", name):
            if "vaccinia" not in name and (plat in {VO_DNA, VO_VECTOR} or arch in {VO_DNA, VO_VECTOR}):
                propose(
                    nct,
                    VO_TUMOR_AUTO,
                    VO_PROTEIN,
                    "GM-CSF tumor/cell vaccine; not DNA/vector",
                    "rule_gmcsf_cell",
                )
                continue

        # 11) False DNA platform (name is not a DNA vaccine)
        if plat == VO_DNA and "dna" not in name:
            if "as03" in name and "influenza" in name:
                propose(
                    nct,
                    arch if arch.startswith("VO_") else VO_FLU_INACT,
                    VO_INACTIVATED,
                    "AS03 pandemic influenza; inactivated not DNA",
                    "rule_false_dna",
                )
                continue
            if "adenovirus" in name or re.search(r"\bad\d+\b", name) or "alvac" in name:
                propose(
                    nct,
                    arch if arch != VO_DNA else VO_VECTOR,
                    VO_VECTOR,
                    "Adenoviral/ALVAC vector; not DNA platform",
                    "rule_false_dna",
                )
                continue
            if "hepatitis a" in name:
                propose(nct, VO_HEPA, VO_INACTIVATED, "HepA; not DNA", "rule_false_dna")
                continue
            if "polio" in name:
                propose(nct, VO_IPV, VO_INACTIVATED, "IPV; not DNA", "rule_false_dna")
                continue
            # generic: drop false DNA → unknown platform, keep Material if non-DNA
            if arch == VO_DNA:
                continue  # both DNA with no dna in name — leave for manual
            propose(
                nct,
                arch,
                "unknown",
                "Platform was DNA but name has no DNA evidence; unlabel platform",
                "rule_false_dna",
            )
            continue

        # 12) GVAX / tumor / GM-CSF cell vaccines mislabeled as vector/DNA
        if "gvax" in name:
            mat = VO_GVAX_PANC if "pancreatic" in name else "VO_0007467"
            propose(nct, mat, VO_PROTEIN, "GVAX tumor-cell vaccine; not vector", "rule_gvax")
            continue
        if "autologous tumor cell" in name and plat == VO_VECTOR:
            propose(
                nct,
                VO_TUMOR_AUTO,
                VO_PROTEIN,
                "Autologous tumor cell vaccine; not vector",
                "rule_tumor",
            )
            continue
        if re.search(r"gm-csf.*(cell|myeloma|breast|secreting)|allogeneic gm-csf", name):
            if "vaccinia" in name:
                propose(nct, arch if arch.startswith("VO_") else VO_VECTOR, VO_VECTOR, "Vaccinia-GM-CSF vector", "rule_gmcsf_vector")
                continue
            propose(
                nct,
                VO_TUMOR_AUTO,
                VO_PROTEIN,
                "GM-CSF tumor/cell vaccine; not DNA/vector",
                "rule_gmcsf_cell",
            )
            continue
        if "peptide" in name and "gm-csf" in name and plat == VO_VECTOR:
            propose(nct, VO_PEPTIDE, "VO_0000534", "Peptide/GM-CSF; peptide not vector", "rule_peptide")
            continue

        # 12b) Chimeric HBV core particle (VLP-like protein) ≠ recombinant vector
        if "chimaeric hbv core" in name or "chimeric hbv core" in name:
            propose(
                nct,
                VO_VLP,
                VO_VLP,
                "Chimeric HBV core particle vaccine; VLP/protein not vector",
                "rule_hbv_core_vlp",
            )
            continue

        # 13) Porcine rotavirus marked vector
        if "porcine rotavirus" in name and plat == VO_VECTOR:
            propose(
                nct,
                VO_PORCINE_ROTA,
                VO_LIVE,
                "Porcine rotavirus MLV; not recombinant vector",
                "rule_rota",
            )
            continue

        # 14) Japanese encephalitis (non-DNA) marked vector
        if "japanese encephalitis" in name and "dna" not in name and plat == VO_VECTOR:
            propose(
                nct,
                VO_JE,
                VO_INACTIVATED if "inactiv" in name else "unknown",
                "JE vaccine; not recombinant vector",
                "rule_je",
            )
            continue

        # 15) Named DNA vaccine stuck on vector (pure DNA, not Ad/ALVAC hybrid)
        if "dna vaccine" in name and plat == VO_VECTOR:
            if "alvac" in name or re.search(r"\bad5\b|\bad35\b", name):
                propose(
                    nct,
                    arch if arch.startswith("VO_") else VO_VECTOR,
                    VO_VECTOR,
                    "DNA prime / viral-vector product; keep vector platform",
                    "rule_dna_vector_hybrid",
                )
                continue
            propose(
                nct,
                arch if arch != VO_VECTOR else VO_DNA,
                VO_DNA,
                "Named DNA vaccine; DNA platform not vector",
                "rule_dna",
            )
            continue

        # 16) Oxford–AstraZeneca / ChAdOx COVID named product
        if re.search(r"oxford-?astrazeneca|chadox1|vaxzevria|azd1222", name):
            propose(
                nct,
                VO_OXFORD_AZ,
                VO_VECTOR,
                "Oxford-AstraZeneca / ChAdOx COVID; vectored COVID product",
                "rule_oxford",
            )
            continue

        # 17) Anthrax Vaccine Adsorbed (any wrong Material/platform)
        if "anthrax vaccine adsorbed" in name or name.strip() in {"ava", "bioThrax".lower()}:
            propose(
                nct,
                anthrax if str(anthrax).startswith("VO_") else arch,
                VO_INACTIVATED,
                "AVA: anthrax Material + inactivated platform",
                "rule_ava_all",
            )
            continue

        # 18) Standard inactivated influenza products with unlabeled/wrong platform
        if re.search(
            r"trivalent influenza|quadrivalent influenza|\bfluzone\b|\bfluarix\b|"
            r"\bflulaval\b|\bafluria\b|\bfluvirin\b|influenza vaccine hd|"
            r"\d{4}.\d{4} trivalent influenza",
            name,
        ):
            if re.search(r"live attenuated|flumist|\blaiv\b|cold.adapted", name):
                pass
            else:
                mat = arch
                if mat not in {
                    "VO_0001236",
                    "VO_0004673",
                    "VO_0000047",
                    "VO_0000642",
                    VO_FLU_INACT,
                    "VO_0001176",
                }:
                    # keep disease-flu arches; otherwise leave Material
                    if disease == "influenza" or "influenza" in conds:
                        if "quadrivalent" in name or name.endswith(" hd") or " vaccine hd" in name:
                            mat = "VO_0004673"
                        elif "fluzone" in name:
                            mat = "VO_0000047"
                        elif "trivalent" in name:
                            mat = "VO_0001236"
                        else:
                            mat = VO_FLU_INACT
                if plat in {"unknown", "", VO_DNA, VO_LIVE, VO_VECTOR} or mat != arch:
                    propose(
                        nct,
                        mat,
                        VO_INACTIVATED,
                        "Licensed inactivated influenza product; platform inactivated",
                        "rule_flu_inact_plat",
                    )
                    continue

        # 19) Generic "Meningococcal Vaccine" marked live attenuated
        if name.strip() in {"meningococcal vaccine", "meningococcal conjugate vaccine", "meningococcal conjugate vaccine mcv4"} or (
            "meningococcal" in name and "conjugate" in name
        ):
            if plat == VO_LIVE or arch == VO_LIVE:
                mat = VO_MEN_CONJ if "conjugate" in name or "mcv4" in name else VO_MEN_UNK
                propose(
                    nct,
                    mat,
                    VO_CONJUGATE,
                    "Meningococcal conjugate/unknown formulation; not live attenuated",
                    "rule_men_not_live",
                )
                continue

        # 20) Dendritic-cell vaccines without viral-vector tokens → not recombinant vector
        if "dendritic" in name and plat == VO_VECTOR:
            if re.search(
                r"adenovirus|\bad5\b|\bad35\b|\bad26\b|vaccinia|fowlpox|alvac|mva|lentivirus",
                name,
            ):
                pass
            else:
                mat = VO_DC
                if "peptide" in name:
                    propose(nct, VO_PEPTIDE, "VO_0000534", "DC-peptide vaccine; peptide not vector", "rule_dc")
                elif "lysate" in name or "fusion" in name or "tumor" in name:
                    propose(nct, VO_TUMOR_AUTO, VO_PROTEIN, "DC-tumor vaccine; not vector", "rule_dc")
                else:
                    propose(nct, mat, VO_PROTEIN, "Dendritic-cell vaccine; not recombinant vector", "rule_dc")
                continue

        # 21) Synthetic / multi-epitope peptide marked vector
        if "peptide" in name and plat == VO_VECTOR and "dendritic" not in name:
            propose(
                nct,
                VO_PEPTIDE,
                "VO_0000534",
                "Peptide vaccine; not recombinant vector",
                "rule_peptide_vec",
            )
            continue

        # 22) HyperAcute melanoma cell vaccine
        if "hyperacute" in name:
            mat = VO_HYPERACUTE_MEL if "melanoma" in name else arch
            propose(
                nct,
                mat if str(mat).startswith("VO_") else VO_HYPERACUTE_MEL,
                VO_PROTEIN,
                "HyperAcute cell vaccine; not generic recombinant vector",
                "rule_hyperacute",
            )
            continue

        # 23) Measles Virus Vaccine VVN (live measles product name) marked vector
        if "measles virus vaccine vvn" in name and plat == VO_VECTOR:
            propose(
                nct,
                arch if arch != VO_VECTOR else VO_LIVE,
                VO_LIVE,
                "Measles Virus Vaccine VVN; live attenuated not vector",
                "rule_measles_vvn",
            )
            continue

    # COVID+flu co-admin with flu-only extracted names (CT.gov checked)
    MANUAL = {
        "NCT04848467": (
            VO_COVID,
            VO_RNA,
            "CVnCoV mRNA + QIV co-admin; primary=CVnCoV RNA",
        ),
        "NCT06831786": (
            VO_COVID,
            VO_RNA,
            "BNT162b2 + QIV co-admin; primary=COVID mRNA",
        ),
        "NCT06097273": (
            VO_COVID,
            VO_RNA,
            "mRNA-1083 (SARS-CoV-2+influenza) primary; RNA platform",
        ),
        "NCT06864143": (
            VO_COVID,
            VO_RNA,
            "mRNA-1083 (influenza+COVID) primary; RNA platform",
        ),
        "NCT06091410": (
            VO_COVID,
            "unknown",
            "COVID + influenza co-admin; primary=COVID; platform multi→unknown",
        ),
        "NCT05827926": (
            VO_COVID,
            VO_RNA,
            "mRNA-1083 multi-component; not influenza DNA Material/platform",
        ),
    }
    # Final 22 sample leftovers — CT.gov primary-investigational pass (2026-10-30)
    FINAL22 = {
        "NCT00131833": ("VO_0003443", "VO_0000534", "Typhoid Vi polysaccharide; not live attenuated"),
        "NCT01531530": ("VO_0000668", "VO_0000367", "CVD 1208S live oral Shigella"),
        "NCT00806195": ("VO_0003441", "VO_0000195", "Primary MenACWY-CRM; ActHIB was coadmin mis-extract"),
        "NCT03537508": ("VO_0003441", "VO_0000195", "MenACYW conjugate primary"),
        "NCT01056705": ("VO_0003834", "VO_0000315", "Sabin IPV primary; not DTaP-IPV-Hib name extract"),
        "NCT01943825": ("VO_0010342", "VO_0000367", "CYD/Dengvaxia dengue primary; not JE DNA"),
        "NCT01909752": ("VO_0007319", "VO_0000483", "DRibble tumor vaccine primary; HPV secondary"),
        "NCT00707148": ("VO_0003180", "VO_0000561", "Tdap in pregnancy; not tetanus-toxoid-only"),
        "NCT01546389": ("VO_0005808", "VO_0000367", "PfSPZ Sanaria"),
        "NCT03163121": ("VO_0005808", "VO_0000367", "PfSPZ Sanaria"),
        "NCT00384059": ("VO_0001244", "VO_0000195", "PCV13 (Prevnar 13)"),
        "NCT02309515": ("VO_0001244", "VO_0000195", "PCV13"),
        "NCT01359449": ("VO_0000071", "VO_0000195", "Menactra MCV4"),
        "NCT00705640": ("VO_0000577", "VO_0000534", "Melanoma multi-epitope peptide"),
        "NCT00373217": ("VO_0000577", "VO_0000534", "Ovarian peptide vaccine"),
        "NCT05111353": ("VO_0000577", "VO_0000534", "Synthetic MUC1 peptide"),
        "NCT03412786": ("VO_0000577", "VO_0000534", "Bcl-Xl peptide vaccine"),
        "NCT00640861": ("VO_0000577", "VO_0000534", "HER2/MUC1 peptide vaccines"),
        "NCT01355393": ("VO_0000577", "VO_0000534", "HER2 peptide vaccine"),
        "NCT01260727": ("VO_0000032", "VO_0000032", "PENNVAX DNA primary (+MVA boost)"),
        "NCT00704938": ("VO_0000484", "VO_0000484", "DC-adenovirus p53; vector OK"),
        "NCT01349647": ("VO_0000195", "VO_0000195", "KLH conjugate cocktail; conjugate OK"),
    }
    MANUAL.update(FINAL22)
    for nct, (arch, plat, why) in MANUAL.items():
        if nct in rows and rows[nct]["source"] == "advisor":
            continue
        src = "manual_ctgov_final22" if nct in FINAL22 else "manual_ctgov"
        rows[nct] = {
            "nct_id": nct,
            "arch_vo": arch,
            "vo_platform": plat,
            "action": "fix",
            "rationale": why,
            "source": src,
        }

    out = pd.DataFrame(rows.values())
    if out.empty:
        raise SystemExit("no corrections")
    # stable column order
    out = out[["nct_id", "arch_vo", "vo_platform", "action", "rationale", "source"]]
    out = out.sort_values(["source", "nct_id"]).reset_index(drop=True)

    # write corrections without source col for build_table
    corr = out[["nct_id", "arch_vo", "vo_platform", "action", "rationale"]].copy()
    corr.to_csv(OUT_CORR, index=False)
    out.to_csv(OUT_BULK_LOG, index=False)

    summary = {
        "n_corrections": int(len(out)),
        "by_source": out["source"].value_counts().to_dict(),
        "n_advisor": int((out["source"] == "advisor").sum()),
        "n_rules": int((out["source"] != "advisor").sum()),
        "labels": {
            "anthrax": anthrax,
            "cervarix": cervarix,
            "recombivax": recombivax,
            "engerix": engerix,
            "VO_0003150": classes.get(VO_HBSAG, {}).get("label"),
            "VO_0000483": classes.get(VO_PROTEIN, {}).get("label"),
            "VO_0000315": classes.get(VO_INACTIVATED, {}).get("label"),
            "VO_0000771": classes.get(VO_BCG, {}).get("label"),
            "VO_0004914": classes.get(VO_VLP, {}).get("label"),
            "VO_0003075": classes.get(VO_GARDASIL9, {}).get("label"),
            "VO_0006017": classes.get(VO_HEPE, {}).get("label"),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", OUT_CORR, "n=", len(corr))


if __name__ == "__main__":
    main()
