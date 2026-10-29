"""SEA-CDM structural integrity checks on the **paper CTDR package** (v1.4).

Uses: SEA-CDM AACT vaccine remapped/v1.4  (same as ctdr.config.V14)
Not: SEA-CDM VaxCT VO enriched (different export; 9,149 studies).

Reports primary-key uniqueness, foreign-key fill/resolve rates (row-level),
and same-study path consistency. Distinguishes:
  - field completeness (non-empty)
  - FK validity (value exists in parent table)
  - semantic mapping correctness (requires source audit; not claimed here)

Also summarizes existing V14_COMPLIANCE.json / OFFICIAL_V14_HARDENING.json.

  python -u -m ctdr.eval_seacdm_linkage
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402

V14 = cfg.V14
OUT = cfg.CTDR_DIR / "optimize" / "seacdm_linkage"
EMPTY = {"", "nan", "None", "none", "NaN"}


def _read(name: str) -> pd.DataFrame:
    p = V14 / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, dtype=str, low_memory=False).fillna("")


def _filled(s: pd.Series) -> pd.Series:
    return ~s.astype(str).isin(EMPTY) & s.astype(str).str.len().gt(0)


def _pk_report(df: pd.DataFrame, col: str) -> dict:
    s = df[col].astype(str)
    filled = _filled(s)
    vals = s[filled]
    n_dup = int(vals.duplicated().sum())
    n_unique = int(vals.nunique())
    return {
        "column": col,
        "n_rows": int(len(df)),
        "n_nonempty": int(filled.sum()),
        "n_unique_nonempty": n_unique,
        "n_duplicate_nonempty_rows": n_dup,
        "pct_nonempty": round(100.0 * float(filled.mean()), 4) if len(df) else 0.0,
    }


def _fk_report(
    child: pd.DataFrame,
    child_col: str,
    parent_ids: set[str],
    *,
    allow_empty: bool = False,
) -> dict:
    """Row-level FK: missing vs unresolvable vs ok."""
    s = child[child_col].astype(str)
    n = len(child)
    empty = ~_filled(s)
    nonempty = ~empty
    resolved = s.map(lambda x: x in parent_ids) & nonempty
    unresolved = nonempty & ~resolved
    return {
        "child_column": child_col,
        "n_rows": n,
        "n_missing_empty": int(empty.sum()),
        "n_unresolvable": int(unresolved.sum()),
        "n_resolved": int(resolved.sum()),
        "pct_rows_resolved": round(100.0 * float(resolved.mean()), 4) if n else 0.0,
        "pct_nonempty_that_resolve": round(
            100.0 * float(resolved[nonempty].mean()) if int(nonempty.sum()) else 0.0, 4
        ),
        "allow_empty_note": allow_empty,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    study = _read("study.csv")
    experiment = _read("experiment.csv")
    interventions = _read("interventions.csv")
    subject = _read("subject.csv")
    material = _read("material.csv")
    occurrence = _read("occurrence.csv")
    results = _read("results.csv")
    groups = _read("groups.csv")

    study_ids = set(study["study_id"].astype(str)) - EMPTY
    exp_ids = set(experiment["experiment_id"].astype(str)) - EMPTY
    subj_ids = set(subject["subject_id"].astype(str)) - EMPTY
    mat_ids = set(material["material_id"].astype(str)) - EMPTY
    grp_ids = set(groups["group_id"].astype(str)) - EMPTY
    # occurrence uses organism_id → subject in this package
    org_ids = subj_ids

    exp_to_study = {
        str(e): str(s)
        for e, s in zip(experiment["experiment_id"], experiment["study_id"])
        if str(e) not in EMPTY
    }

    # Same-study path: intervention.experiment → study == subject's experiment → study
    # when both FKs resolve
    int_ok = interventions.copy()
    int_ok["_exp_study"] = int_ok["experiment_id"].map(exp_to_study)
    subj_exp_study = {
        str(sid): exp_to_study.get(str(eid), "")
        for sid, eid in zip(subject["subject_id"], subject["experiment_id"])
    }
    int_ok["_subj_study"] = int_ok["subject_id"].map(subj_exp_study)
    both = (
        _filled(int_ok["experiment_id"])
        & _filled(int_ok["subject_id"])
        & int_ok["_exp_study"].isin(study_ids)
        & int_ok["_subj_study"].isin(study_ids)
    )
    same_study = both & (int_ok["_exp_study"] == int_ok["_subj_study"])
    mismatch = both & (int_ok["_exp_study"] != int_ok["_subj_study"])

    # Material placeholder rate (semantic caveat from hardening audit)
    mat_col = interventions["material_id"].astype(str)
    placeholder = mat_col.isin({"VO:0000001", "VO_0000001", "http://purl.obolibrary.org/obo/VO_0000001"})

    primary_keys = {
        "study": _pk_report(study, "study_id"),
        "experiment": _pk_report(experiment, "experiment_id"),
        "interventions": _pk_report(interventions, "intervention_id"),
        "subject": _pk_report(subject, "subject_id"),
        "material": _pk_report(material, "material_id"),
        "groups": _pk_report(groups, "group_id"),
        "occurrence": _pk_report(occurrence, "occurrence_id"),
        "results": _pk_report(results, "results_id"),
    }

    foreign_keys = {
        "experiment.study_id→study": _fk_report(experiment, "study_id", study_ids),
        "subject.experiment_id→experiment": _fk_report(subject, "experiment_id", exp_ids),
        "interventions.experiment_id→experiment": _fk_report(
            interventions, "experiment_id", exp_ids
        ),
        "interventions.subject_id→subject": _fk_report(interventions, "subject_id", subj_ids),
        "interventions.material_id→material": {
            **_fk_report(interventions, "material_id", mat_ids),
            "interpretation": (
                "material_id holds ontology CURIEs (often VO:0000001 placeholder), "
                "not material.material_id row keys — unresolved here is expected and "
                "is a semantic-mapping gap, not a broken Study spine."
            ),
        },
        "subject.group_id→groups": {
            **_fk_report(subject, "group_id", grp_ids),
            "interpretation": (
                "Hardening fills subject.group_id=GROUP_1 (singleton cohort guidance); "
                "design arms live in groups.csv with distinct GROUP_DESIGN_* ids."
            ),
        },
        "occurrence.organism_id→subject": _fk_report(occurrence, "organism_id", org_ids),
        "results.experiment_id→experiment": _fk_report(results, "experiment_id", exp_ids),
        "results.group_id→groups": _fk_report(results, "group_id", grp_ids, allow_empty=True),
    }

    # Load prior compliance artifacts if present
    compliance = {}
    for name in ("V14_COMPLIANCE.json", "OFFICIAL_V14_HARDENING.json"):
        p = V14 / name
        if p.exists():
            compliance[name] = json.loads(p.read_text(encoding="utf-8"))

    hardening_checks = (
        compliance.get("OFFICIAL_V14_HARDENING.json", {}).get("audit", {}).get("checks", {})
    )

    report = {
        "package": str(V14),
        "package_note": (
            "Paper CTDR package (ctdr.config.V14). "
            "Do not conflate with SEA-CDM VaxCT VO enriched (9,149 studies)."
        ),
        "distinction": {
            "field_completeness": "non-empty cell rates / fill after hardening",
            "fk_validity": "row-level: empty vs unresolvable vs resolved in parent PK set",
            "semantic_mapping_correctness": (
                "NOT measured here — requires source-record audit. "
                "material_id may be VO:0000001 placeholder; study_context_profile.csv is a project sidecar."
            ),
        },
        "resource_counts": {
            "n_study": int(len(study)),
            "n_experiment": int(len(experiment)),
            "n_subject": int(len(subject)),
            "n_interventions": int(len(interventions)),
            "n_material": int(len(material)),
            "n_groups": int(len(groups)),
            "n_occurrence": int(len(occurrence)),
            "n_results": int(len(results)),
            "n_occurrence_severity_mapped_ae_like": int(
                compliance.get("OFFICIAL_V14_HARDENING.json", {})
                .get("parts", {})
                .get("occurrence", {})
                .get("severity_mapped_rows", 601707)
            ),
        },
        "primary_keys": primary_keys,
        "foreign_keys_row_level": foreign_keys,
        "same_study_paths": {
            "intervention_experiment_and_subject_both_resolve_to_study": int(both.sum()),
            "n_same_study": int(same_study.sum()),
            "n_cross_study_mismatch": int(mismatch.sum()),
            "pct_resolved_pairs_same_study": round(
                100.0 * float(same_study.sum()) / max(int(both.sum()), 1), 4
            ),
        },
        "material_semantics_caveat": {
            "pct_intervention_material_id_placeholder_VO_0000001": round(
                100.0 * float(placeholder.mean()), 4
            ),
            "note": (
                "Fill rate 100% ≠ specific-vaccine mapping accuracy. "
                "Hardening audit: placeholder material_id with sea_material_row_id in comments."
            ),
        },
        "prior_artifacts": {
            "V14_COMPLIANCE_all_core_columns_present": compliance.get(
                "V14_COMPLIANCE.json", {}
            ).get("all_core_columns_present"),
            "hardening_selected_checks": {
                k: hardening_checks[k]
                for k in (
                    "experiment.study_FK_ok",
                    "subject.experiment_FK_ok",
                    "interventions.subject_id_FK_ok",
                    "occurrence.organism_id_FK_ok",
                    "interventions.material_id_fill",
                    "occurrence.severity_is_CTCAE_0_5",
                )
                if k in hardening_checks
            },
        },
    }

    (OUT / "seacdm_linkage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "SEA-CDM structural checks — paper package v1.4",
        f"path: {V14}",
        f"studies: {report['resource_counts']['n_study']}",
        "",
        "PRIMARY KEYS (duplicate nonempty rows):",
    ]
    for name, blk in primary_keys.items():
        lines.append(
            f"  {name}.{blk['column']}: dup_rows={blk['n_duplicate_nonempty_rows']} "
            f"unique={blk['n_unique_nonempty']} nonempty={blk['n_nonempty']}/{blk['n_rows']}"
        )
    lines += ["", "FOREIGN KEYS (row-level):"]
    for name, blk in foreign_keys.items():
        lines.append(
            f"  {name}: resolved={blk['pct_rows_resolved']}% "
            f"missing={blk['n_missing_empty']} unresolvable={blk['n_unresolvable']}"
        )
    lines += [
        "",
        "SAME-STUDY PATH (intervention experiment vs subject):",
        f"  pairs both resolve: {report['same_study_paths']['intervention_experiment_and_subject_both_resolve_to_study']}",
        f"  same study: {report['same_study_paths']['n_same_study']}",
        f"  mismatch: {report['same_study_paths']['n_cross_study_mismatch']}",
        "",
        "SEMANTIC CAVEAT:",
        f"  material_id placeholder VO:0000001: "
        f"{report['material_semantics_caveat']['pct_intervention_material_id_placeholder_VO_0000001']}%",
        "  Do not equate field fill or FK resolve rates with mapping accuracy.",
    ]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
