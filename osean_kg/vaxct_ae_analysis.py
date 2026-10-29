"""
VaxCT adverse-event (AE) analysis for Application D.

Data: SEA-CDM occurrence.csv (AACT reported_events → occurrence_name text).
MedDRA: occurrence_name_id is empty in current ETL; optional BioPortal lookup for PT labels.

Queries:
  - vaccine_type → top reported AEs (raw text or MedDRA PT rollup)
  - AE term → vaccine types / trials (optional MedDRA expansion)
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

NCT_RE = re.compile(r"(NCT\d{8})")
REPORTED_SOURCE = "ClinicalTrials.gov Reported Event"

VO_LABELS = {
    "VO_0000003": "subunit vaccine",
    "VO_0000006": "recombinant vector vaccine",
    "VO_0000011": "DNA vaccine",
    "VO_0000369": "live attenuated vaccine",
    "VO_0000533": "intramuscular route",
    "VO_0000741": "inactivated vaccine",
    "VO_0001482": "conjugate vaccine",
    "VO_0000186": "RNA vaccine",
    "VO_0004908": "COVID-19 vaccine (former project tag for the mRNA cohort)",
    "VO_0005433": "vaccine response",
}


def nct_from_occurrence_id(occurrence_id: str) -> Optional[str]:
    if not occurrence_id:
        return None
    m = NCT_RE.search(str(occurrence_id))
    return m.group(1) if m else None


def norm_ae_term(term: str) -> str:
    t = re.sub(r"\s+", " ", str(term).strip().lower())
    return t


def _parse_count(val) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    try:
        return int(float(str(val).strip().replace(",", "")))
    except (ValueError, TypeError):
        return 0


def parse_comment_field(comments: str, field: str) -> Optional[str]:
    if not comments or pd.isna(comments):
        return None
    for part in str(comments).split(" | "):
        if part.startswith(field + "="):
            return part.split("=", 1)[1].strip()
    return None


def vo_label(vo_id: str) -> str:
    return VO_LABELS.get(vo_id, vo_id)


def coerce_ae_index_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns exist with correct dtypes."""
    out = df.copy()
    if "row_count" in out.columns:
        out["row_count"] = pd.to_numeric(out["row_count"], errors="coerce").fillna(1).astype("int64")
    if "subjects_affected_sum" in out.columns:
        out["subjects_affected_sum"] = (
            pd.to_numeric(out["subjects_affected_sum"], errors="coerce").fillna(0).astype("int64")
        )
    return out


def load_ae_index(
    cache_path: Path,
    occurrence_csv: Path,
    study_vo_csv: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load cached AE index or build from occurrence.csv."""
    if cache_path.exists() and not rebuild:
        return coerce_ae_index_dtypes(pd.read_csv(cache_path, compression="gzip"))
    idx, _ = build_ae_index(occurrence_csv, study_vo_csv)
    idx = coerce_ae_index_dtypes(idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(cache_path, index=False, compression="gzip")
    return idx


def load_study_vo_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    cols = ["nct_id", "vaccine_type_vo_id", "violin_vaccine_entity", "adjuvant_vo_ids", "specific_vaccine_name"]
    keep = [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates(subset=["nct_id"])


def iter_reported_occurrences(
    occurrence_csv: Path,
    chunksize: int = 200_000,
):
    """Yield dict rows for AACT reported group-level adverse events only."""
    for chunk in pd.read_csv(occurrence_csv, chunksize=chunksize, dtype=str, low_memory=False):
        rep = chunk[chunk["reference_source"].fillna("").str.contains("Reported", case=False)]
        for _, r in rep.iterrows():
            nct = nct_from_occurrence_id(r.get("occurrence_id", ""))
            name = r.get("occurrence_name")
            if not nct or pd.isna(name) or not str(name).strip():
                continue
            yield {
                "nct_id": nct,
                "ae_term": str(name).strip(),
                "ae_norm": norm_ae_term(name),
                "organ_system": parse_comment_field(r.get("comments", ""), "organ_system"),
                "subjects_affected": parse_comment_field(r.get("comments", ""), "subjects_affected"),
                "subjects_at_risk": parse_comment_field(r.get("comments", ""), "subjects_at_risk"),
            }


def build_ae_index(
    occurrence_csv: Path,
    study_vo_csv: Path,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ae_by_study: one row per (nct, ae_norm) with counts metadata
      study_meta: nct joined with VO fields
    """
    study_meta = load_study_vo_map(study_vo_csv)
    rows: List[dict] = []
    n = 0
    for rec in iter_reported_occurrences(occurrence_csv):
        rows.append(rec)
        n += 1
        if max_rows and n >= max_rows:
            break
    ae_raw = pd.DataFrame(rows)
    if ae_raw.empty:
        return ae_raw, study_meta

    ae_raw["subjects_affected_n"] = ae_raw["subjects_affected"].map(_parse_count)

    def _mode_or_none(series: pd.Series):
        s = series.dropna()
        s = s[s.astype(str).str.strip() != ""]
        if s.empty:
            return None
        return s.mode().iloc[0]

    agg = (
        ae_raw.groupby(["nct_id", "ae_norm", "ae_term"], dropna=False)
        .agg(
            row_count=("ae_term", "size"),
            organ_system=("organ_system", _mode_or_none),
            subjects_affected_sum=("subjects_affected_n", "sum"),
        )
        .reset_index()
    )
    agg = agg.merge(study_meta, on="nct_id", how="left")
    return coerce_ae_index_dtypes(agg), study_meta


def top_aes_for_vaccine_type(
    ae_index: pd.DataFrame,
    vaccine_type_vo_id: str,
    top_n: int = 20,
    organ_system: Optional[str] = None,
) -> pd.DataFrame:
    """Given VO vaccine class, return frequent AE terms across trials."""
    sub = ae_index[ae_index["vaccine_type_vo_id"] == vaccine_type_vo_id]
    if organ_system and "organ_system" in sub.columns:
        sub = sub[sub["organ_system"].fillna("").str.lower() == organ_system.strip().lower()]
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "ae_term",
                "ae_norm",
                "trial_count",
                "mention_count",
                "subjects_affected_sum",
                "top_organ_system",
            ]
        )

    agg_kw: dict = {
        "trial_count": ("nct_id", "nunique"),
        "mention_count": ("row_count", "sum"),
    }
    if "subjects_affected_sum" in sub.columns:
        agg_kw["subjects_affected_sum"] = ("subjects_affected_sum", "sum")
    if "organ_system" in sub.columns:
        agg_kw["top_organ_system"] = ("organ_system", _first_non_empty)

    trial_hits = sub.groupby(["ae_norm", "ae_term"], dropna=False).agg(**agg_kw).reset_index()
    sort_cols = ["trial_count", "mention_count"]
    if "subjects_affected_sum" in trial_hits.columns:
        sort_cols = ["trial_count", "subjects_affected_sum", "mention_count"]
    trial_hits = trial_hits.sort_values(sort_cols, ascending=False).head(top_n)
    return trial_hits


def _first_non_empty(series: pd.Series):
    for val in series:
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def top_organs_for_vaccine_type(
    ae_index: pd.DataFrame,
    vaccine_type_vo_id: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top MedDRA organ-system categories for a VO vaccine class."""
    sub = ae_index[ae_index["vaccine_type_vo_id"] == vaccine_type_vo_id].copy()
    if "organ_system" not in sub.columns:
        return pd.DataFrame(columns=["organ_system", "trial_count", "mention_count"])
    sub = sub[sub["organ_system"].fillna("").astype(str).str.strip() != ""]
    if sub.empty:
        return pd.DataFrame(columns=["organ_system", "trial_count", "mention_count"])

    agg_kw: dict = {
        "trial_count": ("nct_id", "nunique"),
        "mention_count": ("row_count", "sum"),
    }
    if "subjects_affected_sum" in sub.columns:
        agg_kw["subjects_affected_sum"] = ("subjects_affected_sum", "sum")

    out = sub.groupby("organ_system", dropna=False).agg(**agg_kw).reset_index()
    sort_cols = ["trial_count", "mention_count"]
    if "subjects_affected_sum" in out.columns:
        sort_cols = ["trial_count", "subjects_affected_sum", "mention_count"]
    return out.sort_values(sort_cols, ascending=False).head(top_n)


def compare_vo_types_for_ae(
    ae_index: pd.DataFrame,
    ae_query: str,
    vo_ids: List[str],
) -> pd.DataFrame:
    """Side-by-side trial/mention counts per VO type for one AE query."""
    sub = _filter_ae_index(ae_index, [norm_ae_term(ae_query)], fuzzy=True)
    rows = []
    for vo in vo_ids:
        part = sub[sub["vaccine_type_vo_id"] == vo]
        row = {
            "vaccine_type_vo_id": vo,
            "vaccine_type_label": vo_label(vo),
            "trial_count": part["nct_id"].nunique() if not part.empty else 0,
            "mention_count": int(part["row_count"].sum()) if not part.empty else 0,
        }
        if "subjects_affected_sum" in part.columns and not part.empty:
            row["subjects_affected_sum"] = int(part["subjects_affected_sum"].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trial_count", ascending=False)


def vo_ae_coverage(ae_index: pd.DataFrame) -> pd.DataFrame:
    """Trials with reported AEs per VO vaccine class."""
    sub = ae_index[ae_index["vaccine_type_vo_id"].notna()].copy()
    if sub.empty:
        return pd.DataFrame(columns=["vaccine_type_vo_id", "trials_with_ae", "ae_mentions"])

    out = (
        sub.groupby("vaccine_type_vo_id", dropna=False)
        .agg(
            trials_with_ae=("nct_id", "nunique"),
            ae_mentions=("row_count", "sum"),
        )
        .reset_index()
    )
    out["vaccine_type_label"] = out["vaccine_type_vo_id"].map(lambda x: vo_label(str(x)))
    return out.sort_values("trials_with_ae", ascending=False)


def load_meddra_pt_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_meddra_pt_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _best_meddra_pt(term: str, hits: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    if not hits:
        return None, None
    q = norm_ae_term(term)
    for h in hits:
        label = h.get("prefLabel")
        if label and norm_ae_term(label) == q:
            return label, h.get("classId")
    h0 = hits[0]
    return h0.get("prefLabel"), h0.get("classId")


def ensure_meddra_mappings(
    ae_norms: List[str],
    lookup_terms: Dict[str, str],
    api_key: str,
    cache_path: Optional[Path] = None,
    max_new_lookups: int = 200,
    pause_sec: float = 0.12,
) -> Dict[str, dict]:
    """
    Map ae_norm keys to MedDRA PT via BioPortal; persist JSON cache on disk.
    lookup_terms: ae_norm -> example text to send to BioPortal.
    """
    api_key = (api_key or "").strip()
    cache: Dict[str, dict] = load_meddra_pt_cache(cache_path) if cache_path else {}
    new_lookups = 0

    for ae_norm in ae_norms:
        if ae_norm in cache:
            continue
        if not api_key or new_lookups >= max_new_lookups:
            cache[ae_norm] = {"meddra_pt": None, "classId": None, "source": "skipped"}
            continue
        query = lookup_terms.get(ae_norm) or ae_norm
        hits = lookup_meddra_pt(query, api_key=api_key)
        pt, cid = _best_meddra_pt(query, hits)
        cache[ae_norm] = {
            "meddra_pt": pt,
            "classId": cid,
            "source": "bioportal",
            "lookup_query": query,
        }
        new_lookups += 1
        if pause_sec > 0:
            time.sleep(pause_sec)

    if cache_path:
        save_meddra_pt_cache(cache_path, cache)
    return cache


def top_meddra_pts_for_vaccine_type(
    ae_index: pd.DataFrame,
    vaccine_type_vo_id: str,
    api_key: Optional[str] = None,
    top_n: int = 20,
    cache_path: Optional[Path] = None,
    max_terms_to_map: int = 200,
    pause_sec: float = 0.12,
) -> pd.DataFrame:
    """
    Given VO vaccine class, aggregate reported AEs by MedDRA Preferred Term (PT).

    Maps the most frequent raw AE strings in this vaccine class to MedDRA via BioPortal,
    then rolls up trial_count / mention_count by PT. Unmapped rows use raw ae_norm as key.
    """
    sub = ae_index[ae_index["vaccine_type_vo_id"] == vaccine_type_vo_id].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "meddra_pt",
                "classId",
                "trial_count",
                "mention_count",
                "distinct_ae_texts",
                "example_ae_term",
            ]
        )

    term_stats = (
        sub.groupby(["ae_norm", "ae_term"], dropna=False)
        .agg(
            trial_count=("nct_id", "nunique"),
            mention_count=("row_count", "sum"),
        )
        .reset_index()
    )
    term_stats = term_stats.sort_values(
        ["mention_count", "trial_count"], ascending=[False, False]
    )
    lookup_terms = (
        term_stats.groupby("ae_norm")["ae_term"].first().astype(str).to_dict()
    )
    ae_norms_ranked = term_stats["ae_norm"].drop_duplicates().tolist()
    to_map = ae_norms_ranked[:max_terms_to_map]

    cache = ensure_meddra_mappings(
        to_map,
        lookup_terms,
        api_key or "",
        cache_path=cache_path,
        max_new_lookups=max_terms_to_map,
        pause_sec=pause_sec,
    )

    def _pt_for_row(ae_norm: str) -> str:
        entry = cache.get(ae_norm, {})
        pt = entry.get("meddra_pt")
        if pt:
            return str(pt)
        return f"(unmapped) {ae_norm}"

    sub["meddra_pt"] = sub["ae_norm"].map(_pt_for_row)

    by_pt = (
        sub.groupby("meddra_pt", dropna=False)
        .agg(
            trial_count=("nct_id", "nunique"),
            mention_count=("row_count", "sum"),
            distinct_ae_texts=("ae_norm", "nunique"),
            example_ae_term=("ae_term", "first"),
        )
        .reset_index()
    )

    # attach classId from cache (first ae_norm mapped to this PT)
    pt_to_cid: Dict[str, Optional[str]] = {}
    for ae_norm, entry in cache.items():
        pt = entry.get("meddra_pt")
        if pt and pt not in pt_to_cid:
            pt_to_cid[pt] = entry.get("classId")
    by_pt["classId"] = by_pt["meddra_pt"].map(
        lambda pt: pt_to_cid.get(pt) if not str(pt).startswith("(unmapped)") else None
    )

    cols = [
        "meddra_pt",
        "classId",
        "trial_count",
        "mention_count",
        "distinct_ae_texts",
        "example_ae_term",
    ]
    return by_pt[cols].sort_values(
        ["trial_count", "mention_count"], ascending=[False, False]
    ).head(top_n)


def _filter_ae_index(ae_index: pd.DataFrame, terms: List[str], fuzzy: bool = True) -> pd.DataFrame:
    if not terms:
        return ae_index.iloc[0:0]
    if fuzzy:
        pattern = "|".join(re.escape(t) for t in terms)
        return ae_index[ae_index["ae_norm"].str.contains(pattern, na=False, regex=True)]
    return ae_index[ae_index["ae_norm"].isin(terms)]


def vaccine_types_for_ae(
    ae_index: pd.DataFrame,
    ae_query: str,
    top_n: int = 20,
    fuzzy: bool = True,
) -> pd.DataFrame:
    """Given AE text, return VO vaccine types and trials."""
    sub = _filter_ae_index(ae_index, [norm_ae_term(ae_query)], fuzzy=fuzzy)
    if sub.empty:
        return pd.DataFrame(
            columns=["vaccine_type_vo_id", "trial_count", "mention_count", "example_ae_term"]
        )

    by_vo = (
        sub.groupby("vaccine_type_vo_id", dropna=False)
        .agg(
            trial_count=("nct_id", "nunique"),
            mention_count=("row_count", "sum"),
            example_ae_term=("ae_term", "first"),
        )
        .reset_index()
    )
    by_vo["vaccine_type_label"] = by_vo["vaccine_type_vo_id"].map(
        lambda x: vo_label(str(x)) if pd.notna(x) else ""
    )
    return by_vo.sort_values(["trial_count", "mention_count"], ascending=False).head(top_n)


def trials_for_ae_and_vo(
    ae_index: pd.DataFrame,
    ae_query: str,
    vaccine_type_vo_id: Optional[str] = None,
    top_n: int = 15,
) -> pd.DataFrame:
    sub = _filter_ae_index(ae_index, [norm_ae_term(ae_query)], fuzzy=True)
    if vaccine_type_vo_id:
        sub = sub[sub["vaccine_type_vo_id"] == vaccine_type_vo_id]
    cols = ["nct_id", "ae_term", "vaccine_type_vo_id", "violin_vaccine_entity", "row_count"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].drop_duplicates(subset=["nct_id"]).head(top_n)


# ---------------------------------------------------------------------------
# Subject-aware stratification (SEA-CDM cohort Subject from eligibilities)
# ---------------------------------------------------------------------------

DEFAULT_SUBJECT_DEMOGRAPHICS = Path(__file__).resolve().parent / "subject_demographics_summary.csv"


def load_subject_demographics(path: Optional[Path] = None) -> pd.DataFrame:
    """Load cohort Subject demographics (one row per NCT)."""
    path = Path(path) if path else DEFAULT_SUBJECT_DEMOGRAPHICS
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python osean_kg/build_subject_enrichment.py"
        )
    df = pd.read_csv(path, dtype=str)
    df["nct_id"] = df["nct_id"].astype(str).str.upper()
    for col in ("min_age_years", "max_age_years"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_ae_index_by_subject(
    ae_index: pd.DataFrame,
    subject_demo: pd.DataFrame,
    *,
    sex: Optional[str] = None,
    min_age: Optional[float] = None,
    max_age: Optional[float] = None,
    includes_older_adult: Optional[bool] = None,
    includes_child: Optional[bool] = None,
    includes_adult: Optional[bool] = None,
    healthy_volunteers: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Restrict AE index rows to studies whose cohort Subject matches filters.

    sex: ALL | MALE | FEMALE (or SEX_* ids)
    Age: keep trials whose eligibility [min,max] overlaps the query window.
    """
    if "nct_id" not in ae_index.columns:
        raise ValueError("ae_index must contain nct_id")

    demo = subject_demo.copy()
    demo["nct_id"] = demo["nct_id"].astype(str).str.upper()

    if sex:
        key = str(sex).strip().upper()
        if not key.startswith("SEX_"):
            key = f"SEX_{key}"
        if "sex_category_id" in demo.columns:
            demo = demo[demo["sex_category_id"].astype(str).str.upper() == key]

    def _flag_col(col: str, want: Optional[bool]) -> None:
        nonlocal demo
        if want is None or col not in demo.columns:
            return
        target = "true" if want else "false"
        demo = demo[demo[col].astype(str).str.lower() == target]

    _flag_col("includes_older_adult", includes_older_adult)
    _flag_col("includes_child", includes_child)
    _flag_col("includes_adult", includes_adult)
    _flag_col("healthy_volunteers", healthy_volunteers)

    if min_age is not None or max_age is not None:
        lo = demo["min_age_years"].fillna(0.0) if "min_age_years" in demo.columns else pd.Series(0.0, index=demo.index)
        hi = demo["max_age_years"].fillna(120.0) if "max_age_years" in demo.columns else pd.Series(120.0, index=demo.index)
        q_lo = float(min_age) if min_age is not None else 0.0
        q_hi = float(max_age) if max_age is not None else 120.0
        demo = demo[(lo <= q_hi) & (hi >= q_lo)]

    keep = set(demo["nct_id"])
    return ae_index[ae_index["nct_id"].astype(str).str.upper().isin(keep)].copy()


def top_aes_for_vaccine_type_stratified(
    ae_index: pd.DataFrame,
    subject_demo: pd.DataFrame,
    vaccine_type_vo_id: str,
    top_n: int = 20,
    **subject_filters,
) -> pd.DataFrame:
    """Forward query with Subject demographic filters."""
    filtered = filter_ae_index_by_subject(ae_index, subject_demo, **subject_filters)
    return top_aes_for_vaccine_type(filtered, vaccine_type_vo_id, top_n=top_n)


def compare_strata_for_vo(
    ae_index: pd.DataFrame,
    subject_demo: pd.DataFrame,
    vaccine_type_vo_id: str,
    strata: Dict[str, dict],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Side-by-side top AEs for the same VO under named Subject strata.

    strata example:
      {"all": {}, "older_adult": {"includes_older_adult": True}}
    """
    rows = []
    for name, filters in strata.items():
        scoped = filter_ae_index_by_subject(ae_index, subject_demo, **filters)
        scoped_vo = scoped[scoped["vaccine_type_vo_id"] == vaccine_type_vo_id]
        n_trials = int(scoped_vo["nct_id"].nunique()) if not scoped_vo.empty else 0
        hits = top_aes_for_vaccine_type(scoped, vaccine_type_vo_id, top_n=top_n)
        if hits.empty:
            rows.append(
                {
                    "stratum": name,
                    "n_studies_in_stratum": n_trials,
                    "rank": None,
                    "ae_term": None,
                    "ae_norm": None,
                    "trial_count": 0,
                    "mention_count": 0,
                }
            )
            continue
        for i, r in enumerate(hits.itertuples(index=False), start=1):
            rows.append(
                {
                    "stratum": name,
                    "n_studies_in_stratum": n_trials,
                    "rank": i,
                    "ae_term": getattr(r, "ae_term", None),
                    "ae_norm": getattr(r, "ae_norm", None),
                    "trial_count": int(getattr(r, "trial_count", 0) or 0),
                    "mention_count": int(getattr(r, "mention_count", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Intervention-conditioned filters (route / dose) — X→Y | I, S
# ---------------------------------------------------------------------------

DEFAULT_INTERVENTION_PROFILE = Path(__file__).resolve().parent / "study_intervention_profile.csv"


def load_intervention_profile(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_INTERVENTION_PROFILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python osean_kg/build_intervention_enrichment.py"
        )
    df = pd.read_csv(path, dtype=str)
    df["nct_id"] = df["nct_id"].astype(str).str.upper()
    return df


def filter_ae_index_by_intervention(
    ae_index: pd.DataFrame,
    intervention_profile: pd.DataFrame,
    *,
    route: Optional[str] = None,
    route_vo_id: Optional[str] = None,
    dose_bucket: Optional[str] = None,
    dose_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Restrict AE rows by Intervention settings (study-level profile).

    route: intramuscular | subcutaneous | intradermal | intranasal | oral | ...
    dose_bucket: single_dose | two_dose | three_or_more_doses | unknown
    """
    prof = intervention_profile.copy()
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()

    if route_vo_id:
        vo = str(route_vo_id).strip().upper()
        prof = prof[prof["intervention_route_vo_id"].fillna("").str.upper() == vo]
    elif route:
        key = str(route).strip().lower()
        # accept short aliases
        aliases = {
            "im": "intramuscular",
            "sc": "subcutaneous",
            "id": "intradermal",
            "in": "intranasal",
            "iv": "intravenous",
        }
        key = aliases.get(key, key)
        lab = prof["intervention_route_label"].fillna("").str.lower()
        vo = prof["intervention_route_vo_id"].fillna("").str.upper()
        # map label→vo for matching either
        from_label = lab.str.contains(re.escape(key), regex=True)
        # also match VO id if user passed VO_
        if key.startswith("vo_"):
            from_label = vo == key.upper()
        prof = prof[from_label]

    if dose_bucket:
        prof = prof[prof["dose_bucket"].fillna("unknown").str.lower() == str(dose_bucket).lower()]
    if dose_count is not None:
        dc = pd.to_numeric(prof.get("dose_count_est"), errors="coerce")
        prof = prof[dc == int(dose_count)]

    keep = set(prof["nct_id"])
    return ae_index[ae_index["nct_id"].astype(str).str.upper().isin(keep)].copy()


def filter_ae_index_conditional(
    ae_index: pd.DataFrame,
    *,
    subject_demo: Optional[pd.DataFrame] = None,
    intervention_profile: Optional[pd.DataFrame] = None,
    subject_filters: Optional[dict] = None,
    intervention_filters: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply optional Subject (S) and Intervention (I) filters: X→Y | I,S."""
    out = ae_index
    if subject_demo is not None and subject_filters:
        out = filter_ae_index_by_subject(out, subject_demo, **subject_filters)
    if intervention_profile is not None and intervention_filters:
        out = filter_ae_index_by_intervention(out, intervention_profile, **intervention_filters)
    return out


def top_aes_conditional(
    ae_index: pd.DataFrame,
    vaccine_type_vo_id: str,
    top_n: int = 20,
    *,
    subject_demo: Optional[pd.DataFrame] = None,
    intervention_profile: Optional[pd.DataFrame] = None,
    subject_filters: Optional[dict] = None,
    intervention_filters: Optional[dict] = None,
) -> pd.DataFrame:
    scoped = filter_ae_index_conditional(
        ae_index,
        subject_demo=subject_demo,
        intervention_profile=intervention_profile,
        subject_filters=subject_filters,
        intervention_filters=intervention_filters,
    )
    return top_aes_for_vaccine_type(scoped, vaccine_type_vo_id, top_n=top_n)


def compare_condition_sets_for_vo(
    ae_index: pd.DataFrame,
    vaccine_type_vo_id: str,
    condition_sets: Dict[str, dict],
    *,
    subject_demo: Optional[pd.DataFrame] = None,
    intervention_profile: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compare top AEs under named condition sets.

    condition_sets example:
      {
        "all": {},
        "IM": {"intervention_filters": {"route": "intramuscular"}},
        "older+IM": {
            "subject_filters": {"includes_older_adult": True},
            "intervention_filters": {"route": "intramuscular"},
        },
      }
    """
    rows = []
    for name, cfg in condition_sets.items():
        s_f = cfg.get("subject_filters") or {}
        i_f = cfg.get("intervention_filters") or {}
        scoped = filter_ae_index_conditional(
            ae_index,
            subject_demo=subject_demo,
            intervention_profile=intervention_profile,
            subject_filters=s_f or None,
            intervention_filters=i_f or None,
        )
        scoped_vo = scoped[scoped["vaccine_type_vo_id"] == vaccine_type_vo_id]
        n_trials = int(scoped_vo["nct_id"].nunique()) if not scoped_vo.empty else 0
        hits = top_aes_for_vaccine_type(scoped, vaccine_type_vo_id, top_n=top_n)
        if hits.empty:
            rows.append(
                {
                    "condition": name,
                    "n_studies": n_trials,
                    "rank": None,
                    "ae_term": None,
                    "ae_norm": None,
                    "trial_count": 0,
                    "mention_count": 0,
                }
            )
            continue
        for i, r in enumerate(hits.itertuples(index=False), start=1):
            rows.append(
                {
                    "condition": name,
                    "n_studies": n_trials,
                    "rank": i,
                    "ae_term": getattr(r, "ae_term", None),
                    "ae_norm": getattr(r, "ae_norm", None),
                    "trial_count": int(getattr(r, "trial_count", 0) or 0),
                    "mention_count": int(getattr(r, "mention_count", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def meddra_search_terms(hits: List[dict], original_query: Optional[str] = None) -> List[str]:
    """Unique normalized terms to use for AE index text search after MedDRA lookup."""
    terms: List[str] = []
    seen = set()
    for label in ([original_query] if original_query else []) + [h.get("prefLabel") for h in hits]:
        if not label:
            continue
        n = norm_ae_term(label)
        if n and n not in seen:
            seen.add(n)
            terms.append(n)
    return terms



def vaccine_types_for_meddra(
    ae_index: pd.DataFrame,
    ae_query: str,
    api_key: Optional[str] = None,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Map AE text to MedDRA PT via BioPortal, then aggregate vaccine types over matching rows.
    Returns (summary_table, meddra_hits). Without api_key, falls back to text-only search.
    """
    hits = lookup_meddra_pt(ae_query, api_key=api_key) if api_key else []
    terms = meddra_search_terms(hits, ae_query) if hits else [norm_ae_term(ae_query)]
    sub = _filter_ae_index(ae_index, terms, fuzzy=True)
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "vaccine_type_vo_id",
                "trial_count",
                "mention_count",
                "example_ae_term",
                "vaccine_type_label",
            ]
        ), hits

    by_vo = (
        sub.groupby("vaccine_type_vo_id", dropna=False)
        .agg(
            trial_count=("nct_id", "nunique"),
            mention_count=("row_count", "sum"),
            example_ae_term=("ae_term", "first"),
        )
        .reset_index()
    )
    by_vo["vaccine_type_label"] = by_vo["vaccine_type_vo_id"].map(
        lambda x: vo_label(str(x)) if pd.notna(x) else ""
    )
    if hits:
        by_vo["meddra_pt"] = hits[0].get("prefLabel")
    out = by_vo.sort_values(["trial_count", "mention_count"], ascending=False).head(top_n)
    return out, hits


def _bioportal_request(
    path: str,
    api_key: str,
    params: Optional[dict] = None,
    timeout: int = 30,
) -> Tuple[Optional[dict], Optional[str]]:
    """GET data.bioontology.org/{path}. Returns (json_dict, error_message)."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    api_key = (api_key or "").strip()
    if not api_key:
        return None, "No API key"

    q = dict(params or {})
    q.setdefault("apikey", api_key)
    url = f"https://data.bioontology.org/{path.lstrip('/')}?{urllib.parse.urlencode(q)}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"apikey token={api_key}",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:300]
        except Exception:
            pass
        return None, f"HTTP {exc.code}: {body or exc.reason}"
    except Exception as exc:
        return None, str(exc)


def _unwrap_bioportal_payload(data: dict) -> dict:
    """Unwrap JSON-LD search/annotator payloads (often under owl#Class key)."""
    if not isinstance(data, dict):
        return {}
    if "collection" in data:
        return data
    for key, val in data.items():
        if "owl#Class" in key and isinstance(val, dict):
            return val
    return data


def _label_from_bioportal_item(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract prefLabel and class URI from one search/annotation object."""
    if not isinstance(item, dict):
        return None, None
    label = item.get("prefLabel") or item.get("label")
    if not label and item.get("terms"):
        terms = item["terms"]
        if isinstance(terms, list) and terms:
            t0 = terms[0]
            if isinstance(t0, dict):
                label = t0.get("str") or t0.get("prefLabel")
            elif isinstance(t0, str):
                label = t0
    cid = item.get("@id") or item.get("id")
    class_uri = item.get("class")
    if isinstance(class_uri, str):
        cid = cid or class_uri
    annotated = item.get("annotatedClass")
    if isinstance(annotated, dict):
        alabel, acid = _label_from_bioportal_item(annotated)
        label = label or alabel
        cid = cid or acid
    return label, cid


def _parse_bioportal_annotator(data: dict) -> List[dict]:
    """Parse /annotator JSON into {prefLabel, classId, ontology} rows."""
    out: List[dict] = []
    payload = _unwrap_bioportal_payload(data or {})
    for coll in payload.get("collection", []) or []:
        annotations: List[dict] = []
        if isinstance(coll, list):
            annotations = [a for a in coll if isinstance(a, dict)]
        elif isinstance(coll, dict):
            raw = coll.get("annotations")
            if isinstance(raw, list):
                annotations = [a for a in raw if isinstance(a, dict)]
            else:
                annotations = [coll]
        for ann in annotations:
            label, cid = _label_from_bioportal_item(ann)
            if label or cid:
                out.append({
                    "prefLabel": label,
                    "classId": cid,
                    "ontology": "MEDDRA",
                })
    return out


def _parse_bioportal_search(data: dict) -> List[dict]:
    """Parse /search JSON into {prefLabel, classId, ontology} rows."""
    out: List[dict] = []
    payload = _unwrap_bioportal_payload(data or {})
    for item in payload.get("collection", []) or []:
        if not isinstance(item, dict):
            continue
        ontology = (
            item.get("ontologyName")
            or item.get("ontologyPrefLabel")
            or item.get("ontology")
            or ""
        )
        label, cid = _label_from_bioportal_item(item)
        if label or cid:
            out.append({
                "prefLabel": label,
                "classId": cid,
                "ontology": ontology,
            })
    return out


def check_bioportal_api_key(api_key: Optional[str]) -> dict:
    """Diagnose BioPortal API key (run in notebook before MedDRA)."""
    api_key = (api_key or "").strip()
    if not api_key:
        return {
            "ok": False,
            "message": "BIOPORTAL_API_KEY not set. os.environ['BIOPORTAL_API_KEY'] = '...'",
        }

    _search_params = {
        "include": "prefLabel,synonym",
        "pagesize": "5",
    }
    vo_data, vo_err = _bioportal_request(
        "search",
        api_key,
        {"q": "vaccine", "ontologies": "VO", **_search_params},
    )
    if vo_err:
        return {
            "ok": False,
            "message": f"API key rejected: {vo_err}",
            "hint": "Regenerate at https://bioportal.bioontology.org/account",
        }

    med_data, med_err = _bioportal_request(
        "search",
        api_key,
        {"q": "headache", "ontologies": "MEDDRA", **_search_params},
    )
    med_ann, med_ann_err = _bioportal_request(
        "annotator",
        api_key,
        {"text": "headache", "ontologies": "MEDDRA", "longest_only": "true"},
    )
    if med_err:
        return {
            "ok": False,
            "message": f"VO OK but MEDDRA failed: {med_err}",
        }

    med_hits = _parse_bioportal_search(med_data or {})
    med_ann_hits = _parse_bioportal_annotator(med_ann or {}) if not med_ann_err else []
    if not med_hits and med_ann_hits:
        med_hits = med_ann_hits
    result = {
        "ok": True,
        "vo_hits": len(_parse_bioportal_search(vo_data or {})),
        "meddra_search_hits": len(_parse_bioportal_search(med_data or {})),
        "meddra_annotator_hits": len(med_ann_hits),
        "meddra_hits": len(med_hits),
    }
    if med_hits:
        result["message"] = "API key OK; MedDRA lookup works."
        result["sample_meddra_pt"] = med_hits[0].get("prefLabel")
    else:
        result["message"] = (
            "Auth OK but 0 MedDRA hits parsed. "
            f"annotator_err={med_ann_err!r}"
        )
    return result


def lookup_meddra_pt(term: str, api_key: Optional[str] = None, timeout: int = 30) -> List[dict]:
    """
    Search MedDRA in BioPortal (NCBO). API key: https://bioportal.bioontology.org/account
    Returns list of {prefLabel, classId, ontology} dicts.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return []

    # Annotator works best for free-text AE strings
    ann_data, ann_err = _bioportal_request(
        "annotator",
        api_key,
        {
            "text": term,
            "ontologies": "MEDDRA",
            "longest_only": "true",
            "whole_word_only": "false",
        },
        timeout=timeout,
    )
    out = _parse_bioportal_annotator(ann_data or {})
    if out:
        return out

    data, err = _bioportal_request(
        "search",
        api_key,
        {
            "q": term,
            "ontologies": "MEDDRA",
            "require_exact_match": "false",
            "include": "prefLabel,synonym",
            "pagesize": "10",
        },
        timeout=timeout,
    )
    if err:
        import warnings
        msg = ann_err or err
        warnings.warn(f"BioPortal MedDRA lookup failed: {msg}")
        return []

    return _parse_bioportal_search(data or {})


def summarize_ae_coverage(occurrence_csv: Path) -> dict:
    total_rep = 0
    studies = set()
    terms = Counter()
    for rec in iter_reported_occurrences(occurrence_csv):
        total_rep += 1
        studies.add(rec["nct_id"])
        terms[rec["ae_norm"]] += 1
    return {
        "reported_event_rows": total_rep,
        "studies_with_reported_ae": len(studies),
        "unique_ae_terms": len(terms),
        "top_ae_terms": terms.most_common(10),
    }
