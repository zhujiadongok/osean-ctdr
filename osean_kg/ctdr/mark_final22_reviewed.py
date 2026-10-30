"""Mark the final 22 sample leftovers as CT.gov-reviewed in the audit worksheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "ckpts" / "sea_ml_v14" / "ctdr" / "optimize" / "mapping_audit"
CSV = HERE / "mapping_audit_sample_reviewed.csv"
REVIEWER = "ctgov_final22_2026-10-30"

NOTES = {
    "NCT00131833": ("incorrect", "incorrect", "Typhoid Vi polysaccharide ≠ live attenuated"),
    "NCT01531530": ("correct", "correct", "CVD 1208S live Shigella; refined to VO_0000668"),
    "NCT00806195": ("incorrect", "correct", "Primary MenACWY; ActHIB was coadministered name"),
    "NCT03537508": ("incorrect", "correct", "MenACYW conjugate; name was generic conjugate"),
    "NCT01056705": ("incorrect", "incorrect", "Sabin IPV study; name extract was DTaP-IPV-Hib"),
    "NCT01943825": ("incorrect", "incorrect", "CYD dengue primary; not JE DNA"),
    "NCT01909752": ("incorrect", "incorrect", "DRibble primary; HPV DNA secondary"),
    "NCT00707148": ("incorrect", "correct", "Tdap pregnancy study; not TT-only"),
    "NCT01546389": ("correct", "correct", "PfSPZ; refined product id"),
    "NCT03163121": ("correct", "correct", "PfSPZ; refined product id"),
    "NCT00384059": ("correct", "correct", "PCV13; refined to Prevnar 13"),
    "NCT02309515": ("correct", "correct", "PCV13; refined to Prevnar 13"),
    "NCT01359449": ("correct", "correct", "Menactra; refined product id"),
    "NCT00705640": ("correct", "correct", "Peptide; Material→peptide class"),
    "NCT00373217": ("correct", "correct", "Peptide; Material→peptide class"),
    "NCT05111353": ("correct", "correct", "Peptide; Material→peptide class"),
    "NCT03412786": ("correct", "correct", "Peptide; Material→peptide class"),
    "NCT00640861": ("correct", "correct", "Peptide; set subunit platform"),
    "NCT01355393": ("correct", "correct", "Peptide; set subunit platform"),
    "NCT01260727": ("correct", "correct", "PENNVAX DNA primary retained"),
    "NCT00704938": ("correct", "correct", "DC-adenovirus vector retained"),
    "NCT01349647": ("correct", "correct", "KLH conjugate retained"),
}


def main() -> None:
    df = pd.read_csv(CSV, dtype=str).fillna("")
    for nct, (mj, pj, note) in NOTES.items():
        m = df["nct_id"] == nct
        if not m.any():
            raise SystemExit(f"missing {nct}")
        df.loc[m, "reviewer"] = REVIEWER
        df.loc[m, "material_judgment"] = mj
        df.loc[m, "platform_judgment"] = pj
        df.loc[m, "notes"] = note
    df.to_csv(CSV, index=False)
    # also try unlocked sample path
    alt = HERE / "mapping_audit_sample.csv"
    try:
        df.to_csv(alt, index=False)
    except PermissionError:
        pass
    print(f"marked {len(NOTES)} rows reviewer={REVIEWER}")


if __name__ == "__main__":
    main()
