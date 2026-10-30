"""Apply advisor VO mapping review into mapping_audit_sample.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "mapping_audit_sample.csv"
OUT_CSV = HERE / "mapping_audit_sample_reviewed.csv"
REVIEWER = "advisor_review_2026-10-30"

# material_judgment / platform_judgment:
#   correct | incorrect | ambiguous | (blank = not yet reviewed)
# vo_platform == unknown is NOT coded incorrect unless reviewed as wrong.

ROWS: dict[str, dict[str, str]] = {
    # --- confirmed errors (section 1) ---
    "NCT04617483": {
        "material_judgment": "correct",  # COVID-19 vaccine class; platform is the issue
        "platform_judgment": "incorrect",
        "notes": "Inactivated SARS-CoV-2 vaccine; VO_0000367 is live attenuated. Suggest inactivated platform class.",
        "suggested_vo": "platform:inactivated (not VO_0000367)",
    },
    "NCT05266898": {
        "material_judgment": "correct",
        "platform_judgment": "incorrect",
        "notes": "Gardasil-9 (HPV L1 VLP); VO_0000484 recombinant vector is wrong platform. Candidate platform VO_0004914 (VLP).",
        "suggested_vo": "platform:VO_0004914?",
    },
    "NCT05985681": {
        "material_judgment": "incorrect",
        "platform_judgment": "incorrect",
        "notes": "RG1-VLP candidate, not Gardasil-9 (VO_0007063); not recombinant vector platform.",
        "suggested_vo": "material:RG1-VLP class; platform:VLP/protein",
    },
    "NCT01226758": {
        "material_judgment": "incorrect",
        "platform_judgment": "",
        "notes": "FLU-v is conserved T-cell epitope candidate; not trivalent influenza vaccine (VO_0001236). platform was unknown (unlabeled).",
        "suggested_vo": "material:peptide/epitope vaccine class",
    },
    "NCT00231816": {
        "material_judgment": "ambiguous",
        "platform_judgment": "",
        "notes": "Zostavax + inactivated influenza. If arch_vo may be any study vaccine, flu label may be OK; if primary investigational only, use Zostavax. Policy TBD.",
        "suggested_vo": "policy-dependent: Zostavax vs influenza",
    },
    # --- further errors / high-risk ---
    "NCT00197301": {
        "material_judgment": "",
        "platform_judgment": "incorrect",
        "notes": "Liposomal nasal influenza vaccine; not DNA vaccine (VO_0000032).",
        "suggested_vo": "platform:not VO_0000032",
    },
    "NCT01073293": {
        "material_judgment": "incorrect",
        "platform_judgment": "incorrect",
        "notes": "Primary vaccine V503 / HPV L1 VLP (Gardasil-9 lineage); not recombinant vector for material or platform.",
        "suggested_vo": "material:Gardasil-9/V503; platform:VO_0004914?",
    },
    "NCT01014845": {
        "material_judgment": "incorrect",
        "platform_judgment": "incorrect",
        "notes": "E. coli–expressed recombinant Hepatitis E protein vaccine ≠ recombinant vector delivery platform.",
        "suggested_vo": "platform:recombinant protein (not VO_0000484)",
    },
    "NCT03718130": {
        "material_judgment": "incorrect",
        "platform_judgment": "incorrect",
        "notes": "HTNV/PUUV DNA vaccine; consider VO_0000032 for platform; arch should be DNA hantavirus vaccine class.",
        "suggested_vo": "platform:VO_0000032",
    },
    "NCT04362124": {
        "material_judgment": "incorrect",
        "platform_judgment": "correct",
        "notes": "BCG, not COVID-19 vaccine (VO_0004908). Live-attenuated platform VO_0000367 OK for BCG.",
        "suggested_vo": "material:BCG/tuberculosis vaccine",
    },
    "NCT05091307": {
        "material_judgment": "incorrect",
        "platform_judgment": "",
        "notes": "Ad26.COV2.S + quadrivalent influenza coadministration; not trivalent influenza (VO_0001236). platform unknown=unlabeled.",
        "suggested_vo": "material:multi-vaccine / Ad26 COVID or QIV",
    },
    "NCT05519839": {
        "material_judgment": "incorrect",
        "platform_judgment": "",
        "notes": "SARS-CoV-2 + quadrivalent influenza nanoparticle combination; not trivalent influenza.",
        "suggested_vo": "material:combination / QIV+COVID",
    },
    "NCT06279871": {
        "material_judgment": "incorrect",
        "platform_judgment": "",
        "notes": "Self-amplifying RNA COVID-19 + quadrivalent influenza; not trivalent influenza.",
        "suggested_vo": "material:saRNA COVID and/or QIV",
    },
    "NCT06800131": {
        "material_judgment": "incorrect",
        "platform_judgment": "incorrect",
        "notes": "HBsAg recombinant protein + microneedle patch; not recombinant vector vaccine.",
        "suggested_vo": "platform:recombinant protein",
    },
    "NCT05749926": {
        "material_judgment": "ambiguous",
        "platform_judgment": "ambiguous",
        "notes": "Compares recombinant protein COVID booster vs mRNA booster; single RNA platform label cannot describe all arms.",
        "suggested_vo": "per-arm platforms",
    },
    # --- retain / mostly OK ---
    "NCT04751682": {
        "material_judgment": "correct",
        "platform_judgment": "correct",
        "notes": "BBV154 adenovirus-vectored COVID-19; VO_0004908 + VO_0000484 consistent.",
        "suggested_vo": "",
    },
    "NCT02506933": {
        "material_judgment": "correct",
        "platform_judgment": "ambiguous",
        "notes": "VO_0007460 CMV-MVA Triplex OK; platform unknown—consider recombinant vector if labeling platforms.",
        "suggested_vo": "platform:recombinant vector?",
    },
    "NCT03460743": {
        "material_judgment": "correct",
        "platform_judgment": "ambiguous",
        "notes": "Quadrivalent influenza study includes recombinant HA and inactivated arms; single platform insufficient.",
        "suggested_vo": "per-arm platforms",
    },
    "NCT05827978": {
        "material_judgment": "correct",
        "platform_judgment": "ambiguous",
        "notes": "Includes QIV; experimental mRNA vs inactivated comparator need separate platforms. primary_purpose=TREATMENT is registry value—keep separate from VO judgment.",
        "suggested_vo": "per-arm platforms",
    },
}


def main() -> None:
    df = pd.read_csv(CSV, dtype=str).fillna("")
    for nct, upd in ROWS.items():
        m = df["nct_id"] == nct
        if not m.any():
            raise SystemExit(f"missing {nct}")
        df.loc[m, "reviewer"] = REVIEWER
        for k, v in upd.items():
            df.loc[m, k] = v

    try:
        df.to_csv(CSV, index=False)
        written = CSV
    except PermissionError:
        df.to_csv(OUT_CSV, index=False)
        written = OUT_CSV
        print(f"WARNING: {CSV.name} locked; wrote {OUT_CSV.name}", flush=True)

    # summary among reviewed rows
    rev = df[df["reviewer"] == REVIEWER]
    def counts(col: str) -> dict:
        s = rev[col].replace("", pd.NA).dropna()
        return s.value_counts().to_dict()

    summary = {
        "n_sample": int(len(df)),
        "n_reviewed": int(len(rev)),
        "material_judgment": counts("material_judgment"),
        "platform_judgment": counts("platform_judgment"),
        "n_material_incorrect": int((rev["material_judgment"] == "incorrect").sum()),
        "n_platform_incorrect": int((rev["platform_judgment"] == "incorrect").sum()),
        "n_either_incorrect": int(
            (
                (rev["material_judgment"] == "incorrect")
                | (rev["platform_judgment"] == "incorrect")
            ).sum()
        ),
        "n_unreviewed": int((df["reviewer"] == "").sum()),
    }
    (HERE / "advisor_review_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print("wrote", written)
    print(summary)


if __name__ == "__main__":
    main()
