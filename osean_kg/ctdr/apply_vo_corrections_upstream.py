"""Push primary-investigational VO corrections into upstream sidecars (root fix).

Updates:
  - osean_kg/ckpts/sea_ml_v14/disease_vo_map.csv  (arch_vo Material)
  - SEA-CDM v1.4 study_context_profile.csv       (vaccine_type_vo_id platform)

Backs up originals once to *.pre_policy_a.bak (historical suffix; never overwrite bak).
Also mirrors disease_vo_map into osean-ctdr/ if present.

CTDR build_table overlay remains as a safety net (idempotent after this).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from vo_architecture import PROFILE_TO_OFFICIAL  # noqa: E402

AUDIT = cfg.CTDR_DIR / "optimize" / "mapping_audit"
CORR = AUDIT / "vo_corrections_primary.csv"
REPORT = AUDIT / "upstream_apply_report.json"

OFFICIAL_TO_PROFILE = {v: k for k, v in PROFILE_TO_OFFICIAL.items()}
# Prefer writing official ids that already appear as profile ids
for oid in list(PROFILE_TO_OFFICIAL.values()):
    OFFICIAL_TO_PROFILE.setdefault(oid, oid)


def _backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".pre_policy_a.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"backup -> {bak}", flush=True)
    else:
        print(f"backup exists (kept): {bak}", flush=True)
    return bak


def _to_profile_platform(official: str) -> str:
    if official == "unknown":
        return ""
    return OFFICIAL_TO_PROFILE.get(official, official)


def main() -> None:
    if not CORR.is_file():
        raise SystemExit(f"missing {CORR}")
    corr = pd.read_csv(CORR, dtype=str).fillna("")
    corr["nct_id"] = corr["nct_id"].astype(str).str.upper()
    corr = corr.drop_duplicates("nct_id")

    link_path = cfg.DISEASE_VO_MAP
    prof_path = cfg.PROFILE
    _backup(link_path)
    _backup(prof_path)

    link = pd.read_csv(link_path, dtype=str).fillna("")
    link["nct_id"] = link["nct_id"].astype(str).str.upper()
    prof = pd.read_csv(prof_path, dtype=str).fillna("")
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()

    link_i = link.set_index("nct_id")
    prof_i = prof.set_index("nct_id")

    n_arch = n_plat = n_miss_link = n_miss_prof = 0
    details = []
    for r in corr.itertuples(index=False):
        nct = r.nct_id
        arch = str(r.arch_vo)
        plat = str(r.vo_platform)
        row = {"nct_id": nct, "arch_vo": arch, "vo_platform": plat}

        if nct in link_i.index and arch.startswith("VO_"):
            before = str(link_i.at[nct, "arch_vo"])
            link_i.at[nct, "arch_vo"] = arch
            # keep official column aligned when used as platform/product display
            if "official" in link_i.columns and plat.startswith("VO_"):
                link_i.at[nct, "official"] = plat
            elif "official" in link_i.columns and arch.startswith("VO_"):
                # product-level material: mark how=product if was platform-only
                pass
            if "how" in link_i.columns:
                how = str(link_i.at[nct, "how"])
                if plat.startswith("VO_") and arch.startswith("VO_") and arch != plat:
                    link_i.at[nct, "how"] = "both"
                elif how == "platform" and arch.startswith("VO_") and arch != before:
                    link_i.at[nct, "how"] = "product"
            n_arch += 1
            row["arch_before"] = before
        else:
            n_miss_link += 1
            row["arch_before"] = None

        if nct in prof_i.index and (plat.startswith("VO_") or plat == "unknown"):
            before_p = str(prof_i.at[nct, "vaccine_type_vo_id"])
            new_p = "" if plat == "unknown" else _to_profile_platform(plat)
            # empty string for unknown platform (unlabeled)
            prof_i.at[nct, "vaccine_type_vo_id"] = new_p
            n_plat += 1
            row["platform_before"] = before_p
            row["platform_written"] = new_p
        else:
            n_miss_prof += 1
            row["platform_before"] = None

        details.append(row)

    link_out = link_i.reset_index()
    prof_out = prof_i.reset_index()
    link_out.to_csv(link_path, index=False)
    prof_out.to_csv(prof_path, index=False)

    # mirror disease_vo_map into packaged copy if present
    mirror = (
        cfg.ROOT / "osean-ctdr" / "osean_kg" / "ckpts" / "sea_ml_v14" / "disease_vo_map.csv"
    )
    if mirror.parent.is_dir():
        shutil.copy2(link_path, mirror)
        print(f"mirrored {mirror}", flush=True)

    report = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_corrections": int(len(corr)),
        "n_arch_updated": n_arch,
        "n_platform_updated": n_plat,
        "n_missing_link": n_miss_link,
        "n_missing_profile": n_miss_prof,
        "disease_vo_map": str(link_path),
        "profile": str(prof_path),
        "corrections_file": str(CORR),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (AUDIT / "upstream_apply_details.csv").write_text(
        pd.DataFrame(details).to_csv(index=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
