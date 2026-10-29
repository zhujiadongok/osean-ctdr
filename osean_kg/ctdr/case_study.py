"""Explainable held-out case study for CTDR (Figure-style artifact)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.infer import load_bundle, recommend_design  # noqa: E402
from vo_architecture import load_classes  # noqa: E402

OPT = cfg.CTDR_DIR / "optimize"
STRUCT = ("allocation", "masking", "intervention_model", "enrollment_bucket")


def _label(vid: str) -> str:
    node = load_classes().get(vid, {})
    return node.get("label", vid)


def pick_query(test: pd.DataFrame) -> pd.Series:
    """Prefer Phase2 prevention with non-unknown structure labels."""
    cand = test[
        (test["phase"].str.contains("PHASE2", na=False))
        & (test["primary_purpose"] == "PREVENTION")
        & (test["allocation"] != "unknown")
        & (test["masking"] != "unknown")
    ]
    if cand.empty:
        cand = test[test["allocation"] != "unknown"]
    # prefer RNA / covid if present
    for pref in (
        (cand["vo_platform"] == "VO_0000186"),
        (cand["disease"] == "covid"),
        (cand["disease"] == "influenza"),
    ):
        hit = cand[pref]
        if not hit.empty:
            return hit.iloc[0]
    return cand.iloc[0]


def main() -> None:
    studies = pd.read_csv(OPT / "ctdr_studies.csv", dtype=str).fillna("")
    test = studies[studies["split"] == "test"]
    gold = pick_query(test)
    nct = gold["nct_id"]

    out = recommend_design(
        disease=str(gold["disease"]),
        arch_vo=str(gold["arch_vo"]),
        vo_platform=str(gold["vo_platform"]),
        phase=str(gold["phase"]),
        primary_purpose=str(gold["primary_purpose"]),
        route=str(gold.get("route", "unknown")),
        dose=str(gold.get("dose", "unknown")),
        sex=str(gold.get("sex", "unknown")),
        older=str(gold.get("older", "unknown")),
        child=str(gold.get("child", "unknown")),
        sponsor_class=str(gold.get("sponsor_class", "unknown")),
        alpha=0.75,
        k=10,
    )
    rec = out["recommendation"]

    rows = []
    for t in STRUCT:
        rows.append(
            {
                "variable": t,
                "recommended": rec.get(t, ""),
                "actual_registered": gold[t],
                "match": str(rec.get(t, "")) == str(gold[t]),
            }
        )
    cmp = pd.DataFrame(rows)
    support = pd.DataFrame(out["support"])

    def _md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    md = [
        "# OSEAN-CTDR case study (held-out)",
        "",
        f"**Query NCT (hidden design):** `{nct}`",
        f"- Disease: {gold['disease']}",
        f"- Material: {_label(gold['arch_vo'])} (`{gold['arch_vo']}`)",
        f"- Platform: {_label(gold['vo_platform'])} (`{gold['vo_platform']}`)",
        f"- Phase: {gold['phase']}",
        f"- Purpose: {gold['primary_purpose']}",
        f"- Method: `{out['method']}`",
        "",
        "## Recommendation vs registered design",
        "",
        _md_table(cmp),
        "",
        f"**Exact matches:** {int(cmp['match'].sum())} / {len(cmp)}",
        "",
        "## Supporting historical trials (evidence)",
        "",
        _md_table(support) if not support.empty else "_none_",
        "",
        "## Note",
        "",
        "This demonstrates explainable recovery of historical design practice, ",
        "not clinical optimality of the recommendation.",
        "",
    ]
    (OPT / "case_study.csv").write_text(cmp.to_csv(index=False), encoding="utf-8")
    support.to_csv(OPT / "case_study_support.csv", index=False)
    (OPT / "case_study.md").write_text("\n".join(md), encoding="utf-8")
    meta = {
        "nct_id": nct,
        "method": out["method"],
        "matches": int(cmp["match"].sum()),
        "n_vars": int(len(cmp)),
        "inputs": {
            "disease": gold["disease"],
            "arch_vo": gold["arch_vo"],
            "vo_platform": gold["vo_platform"],
            "phase": gold["phase"],
            "primary_purpose": gold["primary_purpose"],
        },
    }
    (OPT / "case_study.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n".join(md))
    print("Wrote", OPT / "case_study.md")


if __name__ == "__main__":
    main()
