"""Build the CTDR study table + chronological split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.outcomes import primary_outcome_family_by_nct  # noqa: E402
from sea_ml import _canon_route, disease_bucket  # noqa: E402
from vaxct_ae_analysis import load_intervention_profile, load_subject_demographics  # noqa: E402
from vo_architecture import PROFILE_TO_OFFICIAL  # noqa: E402


def enrollment_bucket(val) -> str:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return "unknown"
    if np.isnan(n):
        return "unknown"
    a, b, c = cfg.ENROLL_EDGES
    if n < a:
        return "lt_50"
    if n < b:
        return "50_149"
    if n < c:
        return "150_499"
    return "ge_500"


def _yn_flag(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.lower()
    return np.where(
        raw.isin(["1", "true", "yes"]),
        "yes",
        np.where(raw.isin(["0", "false", "no"]), "no", "unknown"),
    )


def build_ctdr_table() -> pd.DataFrame:
    prof = pd.read_csv(cfg.PROFILE, dtype=str).fillna("")
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()
    link = pd.read_csv(cfg.DISEASE_VO_MAP, dtype=str).fillna("")
    link["nct_id"] = link["nct_id"].astype(str).str.upper()
    link = link.drop_duplicates("nct_id")

    demo = load_subject_demographics()
    demo["nct_id"] = demo["nct_id"].astype(str).str.upper()
    demo = demo.drop_duplicates("nct_id")
    iv = load_intervention_profile()
    iv["nct_id"] = iv["nct_id"].astype(str).str.upper()
    iv = iv.drop_duplicates("nct_id")

    df = prof.merge(link[["nct_id", "arch_vo", "how", "official"]], on="nct_id", how="left")
    df = df.merge(
        demo[
            [
                "nct_id",
                "sex_category_id",
                "min_age_years",
                "includes_adult",
                "includes_child",
                "includes_older_adult",
            ]
        ],
        on="nct_id",
        how="left",
    )
    df = df.merge(
        iv[["nct_id", "intervention_route_label", "dose_bucket"]],
        on="nct_id",
        how="left",
    )

    df["disease"] = df["disease_conditions"].map(disease_bucket)
    df["vo_platform"] = df["vaccine_type_vo_id"].map(
        lambda v: PROFILE_TO_OFFICIAL.get(str(v), str(v) if str(v).startswith("VO_") else "unknown")
    )
    df["vo_platform"] = df["vo_platform"].replace({"": "unknown", "nan": "unknown"})
    # Material: prefer mapped arch_vo, else official platform
    arch = df["arch_vo"].astype(str)
    df["arch_vo"] = np.where(arch.str.startswith("VO_"), arch, df["vo_platform"])
    df["arch_vo"] = df["arch_vo"].replace({"": "unknown", "nan": "unknown"})

    # Primary-investigational audit overlay (primary investigational vaccine; see mapping_audit/POLICY.md)
    corr_path = getattr(cfg, "VO_CORRECTIONS", None)
    if corr_path is not None and Path(corr_path).is_file():
        corr = pd.read_csv(corr_path, dtype=str).fillna("")
        corr["nct_id"] = corr["nct_id"].astype(str).str.upper()
        corr = corr.drop_duplicates("nct_id")
        before = df.set_index("nct_id")
        n_fix = 0
        for row in corr.itertuples(index=False):
            nct = row.nct_id
            if nct not in before.index:
                continue
            if str(row.arch_vo).startswith("VO_"):
                before.at[nct, "arch_vo"] = str(row.arch_vo)
            plat = str(row.vo_platform)
            if plat.startswith("VO_") or plat == "unknown":
                before.at[nct, "vo_platform"] = plat
            n_fix += 1
        df = before.reset_index()
        df.attrs["vo_corrections_applied"] = n_fix

    df["route"] = df["intervention_route_label"].map(_canon_route)
    df["dose"] = df["dose_bucket"].replace({"": "unknown", "nan": "unknown"}).fillna("unknown")
    df["sex"] = df["sex_category_id"].replace({"": "unknown", "nan": "unknown"}).fillna("unknown")
    df["older"] = _yn_flag(df["includes_older_adult"])
    df["child"] = _yn_flag(df["includes_child"])
    df["adult"] = _yn_flag(df["includes_adult"])
    df["min_age"] = pd.to_numeric(df["min_age_years"], errors="coerce")
    df["enrollment_num"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["enrollment_bucket"] = df["enrollment_num"].map(enrollment_bucket)
    df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")

    for col in ("phase", "allocation", "masking", "intervention_model", "primary_purpose"):
        df[col] = df[col].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})

    df["sponsor_class"] = (
        df.get("lead_sponsor_class", pd.Series("unknown", index=df.index))
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown"})
    )
    fam = primary_outcome_family_by_nct()
    df["primary_outcome_family"] = df["nct_id"].map(fam).fillna("unknown")

    mask = pd.Series(True, index=df.index)
    if cfg.REQUIRE_INTERVENTIONAL:
        mask &= df["study_type"].str.upper() == "INTERVENTIONAL"
    if cfg.REQUIRE_PHASE:
        mask &= df["phase"] != "unknown"
    if cfg.REQUIRE_ALLOCATION:
        mask &= df["allocation"] != "unknown"
    if cfg.REQUIRE_ARCH_VO:
        mask &= df["arch_vo"].astype(str).str.startswith("VO_")

    out = df.loc[mask].copy()
    out["split"] = np.where(
        out["start_date_parsed"] >= pd.Timestamp(cfg.SPLIT_DATE),
        "test",
        np.where(out["start_date_parsed"].notna(), "train", "drop"),
    )
    out = out[out["split"].isin(["train", "test"])].reset_index(drop=True)

    keep = [
        "nct_id",
        "split",
        "start_date",
        "study_type",
        "phase",
        "primary_purpose",
        "disease",
        "arch_vo",
        "vo_platform",
        "how",
        "route",
        "dose",
        "sex",
        "older",
        "child",
        "adult",
        "min_age",
        "allocation",
        "masking",
        "intervention_model",
        "enrollment_num",
        "enrollment_bucket",
        "primary_outcome_family",
        "sponsor_class",
        "specific_vaccine_name",
        "disease_conditions",
    ]
    return out[keep]


def main() -> None:
    cfg.CTDR_DIR.mkdir(parents=True, exist_ok=True)
    table = build_ctdr_table()
    path = cfg.CTDR_DIR / "ctdr_studies.csv"
    table.to_csv(path, index=False)
    print(f"wrote {path}  n={len(table)}")
    print(table["split"].value_counts().to_string())
    for t in cfg.TARGETS:
        print(f"\n{t}:")
        print(table[t].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
