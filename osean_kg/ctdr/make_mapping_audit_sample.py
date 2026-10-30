"""Create a stratified Material/platform mapping-audit worksheet for advisor review.

Samples CTDR-eligible trials for manual coding:
  correct | too_coarse | incorrect | unsure

Does not invent gold labels — leaves review columns blank.

  python -u -m ctdr.make_mapping_audit_sample
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402

OUT = cfg.CTDR_DIR / "optimize" / "mapping_audit"
N_TOTAL = 80
SEED = 7


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    # Prefer rows with both product-ish and platform signals when available
    rng = np.random.default_rng(SEED)
    # Stratify by how mapping was obtained if column exists; else by platform unknown
    if "mapping_how" in df.columns:
        strata = df["mapping_how"].astype(str)
    else:
        strata = np.where(
            df["vo_platform"].astype(str).isin(["unknown", "", "nan"]),
            "plat_unknown",
            "plat_known",
        )
        strata = pd.Series(strata, index=df.index)

    parts = []
    groups = list(pd.unique(strata))
    per = max(1, N_TOTAL // max(len(groups), 1))
    for g in groups:
        sub = df.loc[strata == g]
        n = min(len(sub), per)
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(df.loc[idx])
    sample = pd.concat(parts, ignore_index=True)
    # top up if short
    if len(sample) < N_TOTAL:
        rest = df.loc[~df["nct_id"].isin(sample["nct_id"])]
        need = min(N_TOTAL - len(sample), len(rest))
        if need:
            idx = rng.choice(rest.index.to_numpy(), size=need, replace=False)
            sample = pd.concat([sample, df.loc[idx]], ignore_index=True)

    cols = [
        c
        for c in [
            "nct_id",
            "disease",
            "phase",
            "arch_vo",
            "vo_platform",
            "primary_purpose",
            "start_date",
            "mapping_how",
        ]
        if c in sample.columns
    ]
    out = sample[cols].copy()
    out["reviewer"] = ""
    out["material_judgment"] = ""  # correct | too_coarse | incorrect | unsure
    out["platform_judgment"] = ""
    out["notes"] = ""
    out["suggested_vo"] = ""
    path = OUT / "mapping_audit_sample.csv"
    out.to_csv(path, index=False)
    readme = OUT / "README.md"
    readme.write_text(
        (
            "# VO mapping audit worksheet\n\n"
            "Sample of CTDR-eligible trials for expert review before/during JAMIA revision.\n\n"
            "## Coding rubric\n"
            "- **correct**: Material/platform VO matches the registered product at an appropriate grain\n"
            "- **too_coarse**: related but parent/sibling too broad for retrieval (e.g., generic vaccine)\n"
            "- **incorrect**: wrong lineage or unrelated class\n"
            "- **unsure**: needs second reviewer\n\n"
            f"File: `{path.name}` (n={len(out)}, seed={SEED}).\n"
            "Do not impute judgments automatically.\n"
        ),
        encoding="utf-8",
    )
    print(f"wrote {path} n={len(out)}", flush=True)


if __name__ == "__main__":
    main()
