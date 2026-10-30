"""Hard validation after primary-investigational mapping remediation.

Fails loudly if:
  1) Advisor-flagged incorrect NCTs do not match correction targets
  2) Known residual error patterns still fire (HepB=vector, BCG≠BCG, etc.)
  3) Upstream sidecars disagree with CTDR table on corrected NCTs
  4) Overlay vs no-overlay CTDR labels differ on corrected NCTs

Exit code 0 = all checks passed.
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
from ctdr.build_table import build_ctdr_table  # noqa: E402
from vo_architecture import PROFILE_TO_OFFICIAL  # noqa: E402

AUDIT = cfg.CTDR_DIR / "optimize" / "mapping_audit"
OUT = AUDIT / "validation_report.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _official_plat(v: str) -> str:
    v = str(v)
    if not v or v in {"nan", "none"}:
        return "unknown"
    return PROFILE_TO_OFFICIAL.get(v, v if v.startswith("VO_") else "unknown")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    corr = pd.read_csv(AUDIT / "vo_corrections_primary.csv", dtype=str).fillna("")
    corr["nct_id"] = corr["nct_id"].astype(str).str.upper()
    corr = corr.drop_duplicates("nct_id").set_index("nct_id")

    reviewed = pd.read_csv(
        AUDIT / "mapping_audit_sample_reviewed.csv", dtype=str
    ).fillna("")
    reviewed["nct_id"] = reviewed["nct_id"].astype(str).str.upper()
    rev = reviewed[reviewed["reviewer"].astype(str) != ""].copy()

    # --- 1) Reviewed incorrect rows must match overlay targets ---
    bad = rev[
        (rev["material_judgment"] == "incorrect")
        | (rev["platform_judgment"] == "incorrect")
    ]
    advisor_checks = []
    for nct in bad["nct_id"]:
        if nct not in corr.index:
            errors.append(f"reviewed-incorrect {nct} missing from corrections")
            continue
        advisor_checks.append(nct)

    # Entire audit sample should now be reviewed
    still_blank = reviewed[reviewed["reviewer"].astype(str) == ""]
    if len(still_blank):
        errors.append(f"audit sample still unreviewed n={len(still_blank)}")


    # Build CTDR (with overlay)
    df = build_ctdr_table()
    df = df.set_index("nct_id")

    for nct in advisor_checks:
        exp_a = str(corr.at[nct, "arch_vo"])
        exp_p = str(corr.at[nct, "vo_platform"])
        if nct not in df.index:
            errors.append(f"advisor-incorrect {nct} not in CTDR cohort")
            continue
        got_a = str(df.at[nct, "arch_vo"])
        got_p = str(df.at[nct, "vo_platform"])
        if exp_a.startswith("VO_") and got_a != exp_a:
            errors.append(f"{nct} Material want {exp_a} got {got_a}")
        if exp_p.startswith("VO_") or exp_p == "unknown":
            if got_p != exp_p:
                errors.append(f"{nct} Platform want {exp_p} got {got_p}")

    # Previously "correct" advisor rows must not be wrecked into known-bad patterns
    ok = rev[
        (rev["material_judgment"] == "correct")
        & (
            (rev["platform_judgment"] == "correct")
            | (rev["platform_judgment"] == "")
        )
    ]
    for nct in ok["nct_id"]:
        if nct not in df.index:
            continue
        # NCT04751682 BBV154: COVID + vector must remain
        if nct == "NCT04751682":
            if str(df.at[nct, "arch_vo"]) != "VO_0004908" or str(
                df.at[nct, "vo_platform"]
            ) != "VO_0000484":
                errors.append(f"{nct} previously-correct COVID vector broken")

    # --- 2) Residual pattern scanners on full CTDR ---
    s = df.reset_index()
    name = s["specific_vaccine_name"].map(_norm)

    def count(mask) -> int:
        return int(mask.sum())

    residual = {
        "hepb_still_vector": count(
            (name == "hepatitis b vaccine") & (s["vo_platform"] == "VO_0000484")
        ),
        "bcg_not_bcg_material": count(
            name.str.contains(r"\bbcg\b|tice bcg|rbcg", regex=True)
            & ~name.str.contains("mva", regex=True)
            & (s["arch_vo"] != "VO_0000771")
        ),
        "inact_covid_still_live": count(
            name.str.contains("inactiv")
            & name.str.contains("sars|covid", regex=True)
            & (s["vo_platform"] == "VO_0000367")
        ),
        "gardasil_not_vlp": count(
            name.str.contains(r"gardasil|\bv503\b|cervarix|virus-like", regex=True)
            & ~name.str.contains("dna|e7", regex=True)
            & (s["vo_platform"] != "VO_0004914")
        ),
        "mrna_not_rna_plat": count(
            name.str.contains(r"mrna", regex=True)
            & (s["vo_platform"] != "VO_0000186")
            & (s["vo_platform"] != "unknown")
        ),
        "covid_disease_flu_arch": count(
            (s["disease"] == "covid")
            & s["arch_vo"].isin(["VO_0001236", "VO_0004673", "VO_0000047"])
        ),
        "standalone_ipv_not_inact": count(
            name.str.contains(r"^inactivated polio|^inactivated poliovirus", regex=True)
            & (s["vo_platform"] != "VO_0000315")
        ),
        "ava_not_inact": count(
            name.str.contains("anthrax vaccine adsorbed")
            & (s["vo_platform"] != "VO_0000315")
        ),
        # Flu name + unknown platform is OK when primary-investigational overlay intentionally
        # unmarked multi-arm / mis-extracted names (advisor/manual).
        "licensed_flu_unknown_plat": count(
            name.str.contains(
                r"trivalent influenza|quadrivalent influenza|\bfluzone\b", regex=True
            )
            & ~name.str.contains(r"live attenuated|flumist|\blaiv\b", regex=True)
            & (s["vo_platform"] == "unknown")
            & ~s["nct_id"].isin(
                corr.index[
                    corr["vo_platform"].astype(str).eq("unknown")
                ]
            )
        ),
        "men_generic_live": count(
            (name == "meningococcal vaccine") & (s["vo_platform"] == "VO_0000367")
        ),
        "oxford_az_not_vector_covid": count(
            name.str.contains(r"oxford-?astrazeneca", regex=True)
            & (
                (s["vo_platform"] != "VO_0000484")
                | (~s["arch_vo"].isin(["VO_0005158", "VO_0004908"]))
            )
        ),
    }
    for k, v in residual.items():
        if v > 0:
            errors.append(f"residual pattern {k}={v}")

    # Soft residual: vector==vector count (many legitimate) — warn only if huge jump
    n_vec = count((s["arch_vo"] == "VO_0000484") & (s["vo_platform"] == "VO_0000484"))
    if n_vec > 200:
        warnings.append(f"high arch=plat=vector count {n_vec}")

    # --- 3) Upstream sidecars must match correction targets ---
    link = pd.read_csv(cfg.DISEASE_VO_MAP, dtype=str).fillna("")
    link["nct_id"] = link["nct_id"].astype(str).str.upper()
    link = link.drop_duplicates("nct_id").set_index("nct_id")
    prof = pd.read_csv(cfg.PROFILE, dtype=str).fillna("")
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()
    prof = prof.drop_duplicates("nct_id").set_index("nct_id")

    upstream_mismatch = 0
    for nct, row in corr.iterrows():
        exp_a = str(row.arch_vo)
        exp_p = str(row.vo_platform)
        if nct in link.index and exp_a.startswith("VO_"):
            if str(link.at[nct, "arch_vo"]) != exp_a:
                errors.append(
                    f"upstream disease_vo_map {nct} arch {link.at[nct,'arch_vo']} != {exp_a}"
                )
                upstream_mismatch += 1
        if nct in prof.index and (exp_p.startswith("VO_") or exp_p == "unknown"):
            got = _official_plat(str(prof.at[nct, "vaccine_type_vo_id"]))
            want = exp_p if exp_p != "unknown" else "unknown"
            # empty profile id → unknown
            if str(prof.at[nct, "vaccine_type_vo_id"]) in {"", "nan"}:
                got = "unknown"
            if got != want:
                errors.append(
                    f"upstream profile {nct} platform {got} != {want}"
                )
                upstream_mismatch += 1

    # --- 4) Overlay idempotence: disable overlay rebuild must match ---
    # Simulate raw join like build_table without overlay
    prof2 = pd.read_csv(cfg.PROFILE, dtype=str).fillna("")
    prof2["nct_id"] = prof2["nct_id"].astype(str).str.upper()
    link2 = pd.read_csv(cfg.DISEASE_VO_MAP, dtype=str).fillna("")
    link2["nct_id"] = link2["nct_id"].astype(str).str.upper()
    raw = prof2.merge(link2[["nct_id", "arch_vo"]], on="nct_id", how="left")
    raw["vo_platform"] = raw["vaccine_type_vo_id"].map(_official_plat)
    arch = raw["arch_vo"].astype(str)
    raw["arch_vo"] = np.where(arch.str.startswith("VO_"), arch, raw["vo_platform"])
    raw = raw.set_index("nct_id")

    idem_fail = 0
    for nct, row in corr.iterrows():
        if nct not in df.index or nct not in raw.index:
            continue
        for col, exp in (("arch_vo", row.arch_vo), ("vo_platform", row.vo_platform)):
            if not (str(exp).startswith("VO_") or exp == "unknown"):
                continue
            if str(raw.at[nct, col]) != str(df.at[nct, col]):
                errors.append(
                    f"idempotence {nct} {col}: upstream-raw {raw.at[nct,col]} vs CTDR {df.at[nct,col]}"
                )
                idem_fail += 1

    report = {
        "passed": len(errors) == 0,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "n_corrections": int(len(corr)),
        "n_advisor_incorrect_checked": len(advisor_checks),
        "n_advisor_reviewed": int(len(rev)),
        "residual": residual,
        "n_arch_eq_plat_vector": n_vec,
        "upstream_mismatch": upstream_mismatch,
        "idempotence_fail": idem_fail,
        "errors": errors[:80],
        "warnings": warnings,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        print(f"\nFAILED: {len(errors)} errors — see {OUT}", flush=True)
        return 1
    print(f"\nPASSED all hard checks ({len(advisor_checks)} advisor regressions).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
