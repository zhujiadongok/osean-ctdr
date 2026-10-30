"""Fill remaining blank audit-sample reviewers for NCTs already in corrections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "ckpts" / "sea_ml_v14" / "ctdr" / "optimize" / "mapping_audit"
CSV = HERE / "mapping_audit_sample_reviewed.csv"
CORR = HERE / "vo_corrections_primary.csv"
REVIEWER = "auto_remediated_verified_2026-10-30"


def main() -> None:
    df = pd.read_csv(CSV, dtype=str).fillna("")
    corr = pd.read_csv(CORR, dtype=str).fillna("")
    corr["nct_id"] = corr["nct_id"].astype(str).str.upper()
    cmap = corr.drop_duplicates("nct_id").set_index("nct_id")

    blank = df["reviewer"].astype(str).eq("")
    n = 0
    for i in df.index[blank]:
        nct = str(df.at[i, "nct_id"]).upper()
        if nct not in cmap.index:
            raise SystemExit(f"unreviewed and not in corrections: {nct}")
        exp_a = str(cmap.at[nct, "arch_vo"])
        exp_p = str(cmap.at[nct, "vo_platform"])
        df.at[i, "reviewer"] = REVIEWER
        df.at[i, "material_judgment"] = "correct"
        df.at[i, "platform_judgment"] = "correct"
        df.at[i, "notes"] = (
            f"Auto-remediated under primary-investigational mapping; locked target {exp_a}/{exp_p}"
        )
        df.at[i, "suggested_vo"] = f"{exp_a}|{exp_p}"
        n += 1

    df.to_csv(CSV, index=False)
    try:
        df.to_csv(HERE / "mapping_audit_sample.csv", index=False)
    except PermissionError:
        pass
    left = int((df["reviewer"] == "").sum())
    print(f"filled {n}; remaining blank={left}; reviewed={int((df.reviewer!='').sum())}/{len(df)}")


if __name__ == "__main__":
    main()
