"""
SEA-ML Phase 1 (ClinicalTrials.gov / OSEAN use case).

Long-term target (advisor):
  P(Y | vaccine, C) with C growing over phases.
Phase 1 C_clinical = Intervention + Subject(eligibility) + Group arm.

  Model A:  P(Y | vaccine)                         unconditioned
  Model B:  P(Y | vaccine, C_clinical)             SEA context
  Model C:  P(Y | vaccine, C_clinical, VO family)  + ontology roll-up

App B = condition discovery (which C attributes matter).
App C = condition-aware AE ranking for a chosen (X, C).

Myocarditis is the flagship *question*. It is too rare in AACT for a
supervised head; frequent AEs are the training target. App A still
looks up myocarditis as a coverage result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from group_arm_ae import load_arm_ae_index
from vaxct_ae_analysis import load_intervention_profile, load_subject_demographics, vo_label

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(__file__).resolve().parent / "ckpts" / "sea_ml"
VOCAB_MIN_CONTEXTS = 25
MAX_LABELS = 80
PHASE1_VERSION = "1.1.0"
ROUTE_CANON = (
    "intramuscular",
    "intradermal",
    "intranasal",
    "subcutaneous",
    "intravenous",
    "oral",
    "topical",
    "inhalation",
)


def _canon_route(val: str) -> str:
    t = str(val or "").strip().lower()
    if not t or t in {"nan", "none", "unknown", "unlisted"}:
        return "unknown"
    for name in ROUTE_CANON:
        if name in t:
            return name
    return "unknown"

VO_FAMILY = {
    "VO_0004908": "nucleic_acid",
    "VO_0000186": "nucleic_acid",
    "VO_0000011": "nucleic_acid",
    "VO_0000032": "nucleic_acid",
    "VO_0000006": "vector",
    "VO_0000484": "vector",
    "VO_0000369": "live",
    "VO_0000367": "live",
    "VO_0000741": "inactivated",
    "VO_0000315": "inactivated",
    "VO_0000003": "subunit_like",
    "VO_0000534": "subunit_like",
    "VO_0001482": "subunit_like",
    "VO_0000195": "subunit_like",
    "VO_0000729": "toxoid",
    "VO_0000561": "toxoid",
}

SEA_BUCKET = {
    "vo": "Material / Ontology (X)",
    "vo_platform": "Material / platform roll-up",
    "vo_family": "Ontology roll-up",
    "arm_role": "Group",
    "route": "Intervention",
    "dose": "Intervention",
    "sex": "Subject",
    "older": "Subject",
    "child": "Subject",
    "adult": "Subject",
    "min_age": "Subject",
    "disease": "Study indication",
}

# Model A/B use fine Material (mapped VO class). vo_platform shares strength across products.
CAT_A = ["vo"]
CAT_B = ["vo", "vo_platform", "route", "dose", "sex", "older", "child", "adult", "arm_role", "disease"]
NUM_B = ["min_age"]
CAT_C = CAT_B + ["vo_family"]
NUM_C = NUM_B

_DISEASE_VO_MAP = Path(__file__).resolve().parent / "ckpts" / "sea_ml_v14" / "disease_vo_map.csv"


def _fill_cat(s: pd.Series) -> pd.Series:
    out = s.astype(str).replace({"nan": "unknown", "None": "unknown", "": "unknown"})
    return out.fillna("unknown")


def vo_family(vo: str) -> str:
    return VO_FAMILY.get(str(vo or ""), "other")


def disease_bucket(text: str) -> str:
    """Collapse AACT condition text into a small indication set."""
    t = str(text or "").lower()
    if not t or t in {"nan", "none"}:
        return "unknown"
    rules = (
        ("covid", ("covid", "sars-cov", "coronavirus")),
        ("influenza", ("influenza", "flu")),
        ("pneumococcal", ("pneumococ",)),
        ("meningococcal", ("meningococ",)),
        ("hepatitis", ("hepatitis",)),
        ("hpv", ("papilloma", "hpv")),
        ("rsv", ("respiratory syncytial", "rsv")),
        ("malaria", ("malaria", "plasmodium")),
        ("zoster", ("zoster", "varicella")),
        ("rotavirus", ("rotavirus",)),
    )
    for name, keys in rules:
        if any(k in t for k in keys):
            return name
    return "other"


def _attach_disease(ctx: pd.DataFrame) -> pd.DataFrame:
    profile = (
        Path(__file__).resolve().parents[1]
        / "SEA-CDM AACT vaccine remapped"
        / "v1.4"
        / "study_context_profile.csv"
    )
    if not profile.exists():
        ctx["disease"] = "unknown"
        return ctx
    prof = pd.read_csv(profile, dtype=str, usecols=["nct_id", "disease_conditions"])
    prof["nct_id"] = prof["nct_id"].astype(str).str.upper()
    prof["disease"] = prof["disease_conditions"].map(disease_bucket)
    ctx = ctx.merge(prof[["nct_id", "disease"]], on="nct_id", how="left")
    ctx["disease"] = _fill_cat(ctx["disease"])
    return ctx


def _attach_material_vo(ctx: pd.DataFrame) -> pd.DataFrame:
    """Hierarchical Material: vo = mapped arch_vo (or platform); vo_platform = 7-kind.

    Prediction can then distinguish peptide vs multi-peptide while still borrowing
    strength from the parent platform one-hot.
    """
    try:
        from vo_architecture import PROFILE_TO_OFFICIAL
    except ImportError:
        PROFILE_TO_OFFICIAL = {}

    ctx["vo_platform"] = _fill_cat(ctx["vo"])
    # Normalize profile IDs onto official classes when available (display / family).
    ctx["vo_platform"] = ctx["vo_platform"].map(
        lambda v: PROFILE_TO_OFFICIAL.get(str(v), str(v))
    )

    if _DISEASE_VO_MAP.exists():
        link = pd.read_csv(_DISEASE_VO_MAP, dtype=str).fillna("")
        link["nct_id"] = link["nct_id"].astype(str).str.upper()
        link = link.drop_duplicates("nct_id")
        ctx = ctx.merge(link[["nct_id", "arch_vo"]], on="nct_id", how="left")
        arch = ctx["arch_vo"].astype(str)
        use_arch = arch.str.startswith("VO_")
        ctx["vo"] = np.where(use_arch, arch, ctx["vo_platform"])
    else:
        ctx["vo"] = ctx["vo_platform"]

    ctx["vo"] = _fill_cat(ctx["vo"])
    ctx["vo_platform"] = _fill_cat(ctx["vo_platform"])
    if "arch_vo" in ctx.columns:
        ctx = ctx.drop(columns=["arch_vo"])
    return ctx


def build_phase1_table(
    arm: Optional[pd.DataFrame] = None,
    demo: Optional[pd.DataFrame] = None,
    iv: Optional[pd.DataFrame] = None,
    min_contexts: int = VOCAB_MIN_CONTEXTS,
    max_labels: int = MAX_LABELS,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """One row per (NCT, arm_role) after aggregating same-role result groups.

    Label vocabulary and age imputation are finalized inside train_phase1 /
    robustness_suite on the *training* studies only. A provisional all-data
    vocab is returned for backward-compatible callers; training overwrites it.
    """
    arm = arm if arm is not None else load_arm_ae_index()
    demo = demo if demo is not None else load_subject_demographics()
    iv = iv if iv is not None else load_intervention_profile()

    sub = arm[arm["arm_role"].isin(["vaccine", "control"])].copy()
    sub["nct_id"] = sub["nct_id"].astype(str).str.upper()
    sub["ae_norm"] = sub["ae_norm"].fillna("").astype(str)

    ctx = (
        sub.groupby(["nct_id", "arm_role"], as_index=False)
        .agg(
            vo=("vaccine_type_vo_id", lambda s: next((str(x) for x in s if str(x).startswith("VO_")), "unknown")),
            n_groups=("group_id", "nunique"),
        )
    )
    ae_sets = sub.groupby(["nct_id", "arm_role"])["ae_norm"].apply(lambda s: {x for x in s if x})
    ctx["ae_set"] = list(zip(ctx["nct_id"], ctx["arm_role"]))
    ctx["ae_set"] = ctx["ae_set"].map(lambda k: ae_sets.get(k, set()))

    demo = demo.copy()
    demo["nct_id"] = demo["nct_id"].astype(str).str.upper()
    iv = iv.copy()
    iv["nct_id"] = iv["nct_id"].astype(str).str.upper()
    ctx = ctx.merge(demo, on="nct_id", how="left")
    ctx = ctx.merge(iv, on="nct_id", how="left")

    raw_route = ctx.get("intervention_route_label", pd.Series("unknown", index=ctx.index))
    ctx["route"] = raw_route.map(_canon_route)
    ctx["dose"] = _fill_cat(ctx.get("dose_bucket", pd.Series("unknown", index=ctx.index)))
    ctx["sex"] = _fill_cat(ctx.get("sex_category_id", pd.Series("unknown", index=ctx.index)))
    for src, dst in (
        ("includes_older_adult", "older"),
        ("includes_child", "child"),
        ("includes_adult", "adult"),
    ):
        raw = ctx[src].astype(str).str.lower() if src in ctx.columns else pd.Series("", index=ctx.index)
        ctx[dst] = np.where(raw.isin(["1", "true", "yes"]), "yes", np.where(raw.isin(["0", "false", "no"]), "no", "unknown"))
    ctx["min_age"] = pd.to_numeric(ctx.get("min_age_years"), errors="coerce")
    ctx["vo"] = _fill_cat(ctx["vo"])
    ctx["arm_role"] = _fill_cat(ctx["arm_role"])
    ctx = _attach_disease(ctx)
    ctx = _attach_material_vo(ctx)
    ctx["vo_family"] = ctx["vo_platform"].map(vo_family)

    feat = ctx[
        [
            "nct_id",
            "arm_role",
            "vo",
            "vo_platform",
            "vo_family",
            "route",
            "dose",
            "sex",
            "older",
            "child",
            "adult",
            "min_age",
            "disease",
            "ae_set",
        ]
    ].copy()
    # Provisional vocab for callers; train_phase1 rebuilds from train NCTs only.
    counts = pd.Series([ae for s in feat["ae_set"] for ae in s]).value_counts()
    labels = [a for a, n in counts.items() if n >= min_contexts][:max_labels]
    y = pd.DataFrame({lab: feat["ae_set"].map(lambda s, t=lab: int(t in s)) for lab in labels})
    return feat, labels, y


def _vocab_from_ae_sets(ae_sets: pd.Series, min_contexts: int, max_labels: int) -> list[str]:
    counts = pd.Series([ae for s in ae_sets for ae in s]).value_counts()
    return [a for a, n in counts.items() if n >= min_contexts][:max_labels]


def _y_from_ae_sets(ae_sets: pd.Series, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({lab: ae_sets.map(lambda s, t=lab: int(t in s)) for lab in labels})


def _impute_min_age_train_test(
    feat: pd.DataFrame, tr_i: np.ndarray, te_i: np.ndarray
) -> tuple[pd.DataFrame, float]:
    out = feat.copy()
    med = float(pd.to_numeric(out.iloc[tr_i]["min_age"], errors="coerce").median())
    if np.isnan(med):
        med = 18.0
    out["min_age"] = pd.to_numeric(out["min_age"], errors="coerce").fillna(med)
    return out, med


def _pipeline(cat_cols: list[str], num_cols: list[str], kind: str = "lr") -> Pipeline:
    # Dense one-hot: HistGBDT requires dense; LR is fine at this scale (~2.5k rows).
    blocks = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)]
    if num_cols:
        blocks.append(("num", StandardScaler(), num_cols))
    trans = ColumnTransformer(blocks)
    if kind == "gbdt":
        # Sanity-check that Model B's gain is not an LR artefact.
        base = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=120,
            random_state=7,
        )
        clf = OneVsRestClassifier(base, n_jobs=-1)
    else:
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=400, solver="liblinear"),
            n_jobs=-1,
        )
    return Pipeline([("prep", trans), ("clf", clf)])


def _rank_metrics(proba: np.ndarray, y: np.ndarray, ks=(5, 10)) -> dict:
    """Standard ranking metrics.

    recall@k = |Top-k ∩ Y| / |Y|
    precision@k = |Top-k ∩ Y| / k
    (Previous code used |Top-k ∩ Y| / min(k, |Y|); that is NOT used here.)
    """
    n = len(y)
    rec = {k: [] for k in ks}
    prec = {k: [] for k in ks}
    ap = []
    for i in range(n):
        truth = set(np.flatnonzero(y[i] > 0))
        if not truth:
            continue
        order = np.argsort(-proba[i])
        for k in ks:
            top = set(order[:k])
            hit = len(truth & top)
            rec[k].append(hit / len(truth))
            prec[k].append(hit / k)
        hits = 0.0
        prec_sum = 0.0
        for r, j in enumerate(order, start=1):
            if j in truth:
                hits += 1
                prec_sum += hits / r
        ap.append(prec_sum / len(truth))
    out = {f"recall@{k}": float(np.mean(rec[k]) if rec[k] else 0.0) for k in ks}
    out.update({f"precision@{k}": float(np.mean(prec[k]) if prec[k] else 0.0) for k in ks})
    out["map"] = float(np.mean(ap) if ap else 0.0)
    out["n_eval"] = int(sum(1 for i in range(n) if y[i].sum() > 0))
    return out


def frequency_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    freq = y_train.mean(axis=0)
    return np.tile(freq, (n_test, 1))


def _fit_eval(X_tr, y_tr, X_te, y_te, cat, num, kind: str = "lr") -> tuple[Pipeline, dict, np.ndarray]:
    pipe = _pipeline(cat, num, kind=kind)
    pipe.fit(X_tr[cat + num], y_tr)
    proba = pipe.predict_proba(X_te[cat + num])
    return pipe, _rank_metrics(proba, y_te), proba


def _mean_std(xs: list[float]) -> dict:
    arr = np.asarray(xs, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def _split(feat: pd.DataFrame, y: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    groups = feat["nct_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i, te_i = next(splitter.split(feat, y, groups))
    keep = [lab for lab in y.columns if int(y.iloc[tr_i][lab].sum()) >= 2]
    return tr_i, te_i, keep, y[keep]


def paired_recall10(proba_a: np.ndarray, proba_b: np.ndarray, y: np.ndarray) -> dict:
    """Per-context standard recall@10 for a paired test of B vs A."""
    ra, rb = [], []
    for i in range(len(y)):
        truth = set(np.flatnonzero(y[i] > 0))
        if not truth:
            continue
        oa = np.argsort(-proba_a[i])[:10]
        ob = np.argsort(-proba_b[i])[:10]
        ra.append(len(truth & set(oa)) / len(truth))
        rb.append(len(truth & set(ob)) / len(truth))
    ra, rb = np.asarray(ra), np.asarray(rb)
    diff = rb - ra
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(diff, zero_method="wilcox", alternative="greater")
        p = float(stat.pvalue)
    except Exception:
        p = float("nan")
    return {
        "n": int(len(diff)),
        "mean_delta": float(diff.mean()) if len(diff) else 0.0,
        "frac_B_better": float((diff > 0).mean()) if len(diff) else 0.0,
        "p_wilcoxon_greater": p,
        "metric": "standard_recall@10",
    }


def bootstrap_delta(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 7,
    nct_ids: Optional[np.ndarray] = None,
) -> dict:
    """Bootstrap Δ recall@10. Prefer NCT-clustered resampling when nct_ids given."""
    rng = np.random.default_rng(seed)
    idx = np.asarray([i for i in range(len(y)) if y[i].sum() > 0], dtype=int)
    deltas = []
    mode = "context_row"
    if nct_ids is not None and len(idx):
        mode = "nct_cluster"
        ncts = np.asarray(nct_ids)[idx]
        uniq = np.unique(ncts)
        for _ in range(n_boot):
            draw_n = rng.choice(uniq, size=len(uniq), replace=True)
            draw = np.concatenate([idx[ncts == n] for n in draw_n])
            ma = _rank_metrics(proba_a[draw], y[draw])["recall@10"]
            mb = _rank_metrics(proba_b[draw], y[draw])["recall@10"]
            deltas.append(mb - ma)
    else:
        for _ in range(n_boot):
            draw = rng.choice(idx, size=len(idx), replace=True)
            ma = _rank_metrics(proba_a[draw], y[draw])["recall@10"]
            mb = _rank_metrics(proba_b[draw], y[draw])["recall@10"]
            deltas.append(mb - ma)
    arr = np.asarray(deltas)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "mean": float(arr.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n_boot": n_boot,
        "resample_unit": mode,
        "metric": "standard_recall@10",
    }


def robustness_suite(
    feat: pd.DataFrame,
    labels: list[str],
    y: pd.DataFrame,
    seeds: tuple[int, ...] = (7, 11, 19, 23, 29),
    n_boot: int = 1000,
    min_contexts: int = VOCAB_MIN_CONTEXTS,
    max_labels: int = MAX_LABELS,
) -> dict:
    """Multi-seed, SEA-CDM-class ablation, shuffled-C control, NCT-cluster bootstrap.

    Vocab and age imputation are fit on training NCTs for each seed.
    """
    assert "ae_set" in feat.columns, "feat must include ae_set for train-only vocab"
    c_cols = ["route", "dose", "sex", "older", "child", "adult", "arm_role", "min_age", "disease"]

    ablations = {
        "B minus I": (
            ["vo", "vo_platform", "sex", "older", "child", "adult", "arm_role", "disease"],
            ["min_age"],
        ),
        "B minus S": (["vo", "vo_platform", "route", "dose", "arm_role", "disease"], []),
        "B minus Group": (
            ["vo", "vo_platform", "route", "dose", "sex", "older", "child", "adult", "disease"],
            ["min_age"],
        ),
        "B minus disease": (
            ["vo", "vo_platform", "route", "dose", "sex", "older", "child", "adult", "arm_role"],
            ["min_age"],
        ),
        "B minus platform": (
            ["vo", "route", "dose", "sex", "older", "child", "adult", "arm_role", "disease"],
            ["min_age"],
        ),
        "C only (no VO)": (["route", "dose", "sex", "older", "child", "adult", "arm_role", "disease"], ["min_age"]),
    }

    seed_rows = []
    main = None
    for seed in seeds:
        groups = feat["nct_id"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_i, te_i = next(splitter.split(feat, groups=groups))
        feat_i, age_med = _impute_min_age_train_test(feat, tr_i, te_i)
        keep = _vocab_from_ae_sets(feat_i.iloc[tr_i]["ae_set"], min_contexts, max_labels)
        keep = [lab for lab in keep if int(sum(lab in s for s in feat_i.iloc[tr_i]["ae_set"])) >= 2]
        yk = _y_from_ae_sets(feat_i["ae_set"], keep)
        y_tr, y_te = yk.iloc[tr_i].to_numpy(), yk.iloc[te_i].to_numpy()
        X = feat_i.drop(columns=["nct_id", "ae_set"])
        X_tr, X_te = X.iloc[tr_i], X.iloc[te_i]
        freq = frequency_baseline(y_tr, len(te_i))
        row = {
            "seed": seed,
            "n_test_studies": int(feat.iloc[te_i]["nct_id"].nunique()),
            "n_labels": len(keep),
            "age_median_train": age_med,
        }
        row["freq"] = _rank_metrics(freq, y_te)
        proba = {"freq": freq}
        for name, spec in (("A", (CAT_A, [])), ("B", (CAT_B, NUM_B)), ("C", (CAT_C, NUM_C))):
            cat, num = spec
            _, met, pr = _fit_eval(X_tr, y_tr, X_te, y_te, cat, num)
            row[name] = met
            proba[name] = pr
        row["delta_B_minus_A"] = row["B"]["recall@10"] - row["A"]["recall@10"]
        seed_rows.append(row)
        if seed == seeds[0]:
            main = {
                "train_index": tr_i,
                "test_index": te_i,
                "labels": keep,
                "y_te": y_te,
                "proba": proba,
                "X_tr": X_tr,
                "X_te": X_te,
                "y_tr": y_tr,
                "nct_te": feat.iloc[te_i]["nct_id"].to_numpy(),
            }

    assert main is not None
    ablate_metrics = {}
    for name, (cat, num) in ablations.items():
        _, met, _ = _fit_eval(main["X_tr"], main["y_tr"], main["X_te"], main["y_te"], cat, num)
        ablate_metrics[name] = met

    rng = np.random.default_rng(0)
    X_te_shuf = main["X_te"].copy()
    for col in c_cols:
        X_te_shuf[col] = rng.permutation(X_te_shuf[col].to_numpy())
    pipe_b = _pipeline(CAT_B, NUM_B)
    pipe_b.fit(main["X_tr"][CAT_B + NUM_B], main["y_tr"])
    proba_shuf = pipe_b.predict_proba(X_te_shuf[CAT_B + NUM_B])
    shuffle_metrics = _rank_metrics(proba_shuf, main["y_te"])

    paired = paired_recall10(main["proba"]["A"], main["proba"]["B"], main["y_te"])
    boot = bootstrap_delta(
        main["proba"]["A"],
        main["proba"]["B"],
        main["y_te"],
        n_boot=n_boot,
        seed=seeds[0],
        nct_ids=main["nct_te"],
    )

    def _summarize(key: str, metric: str) -> dict:
        return _mean_std([r[key][metric] for r in seed_rows])

    report = {
        "seeds": list(seeds),
        "per_seed": [
            {
                "seed": r["seed"],
                "n_test_studies": r["n_test_studies"],
                "freq_r10": r["freq"]["recall@10"],
                "A_r10": r["A"]["recall@10"],
                "B_r10": r["B"]["recall@10"],
                "C_r10": r["C"]["recall@10"],
                "delta_B_minus_A": r["delta_B_minus_A"],
            }
            for r in seed_rows
        ],
        "summary": {
            "freq_recall@10": _summarize("freq", "recall@10"),
            "A_recall@10": _summarize("A", "recall@10"),
            "B_recall@10": _summarize("B", "recall@10"),
            "C_recall@10": _summarize("C", "recall@10"),
            "delta_B_minus_A": _mean_std([r["delta_B_minus_A"] for r in seed_rows]),
        },
        "ablation_seed0": {k: {"recall@10": v["recall@10"], "map": v["map"]} for k, v in ablate_metrics.items()},
        "shuffle_C_on_test": {
            "recall@10": shuffle_metrics["recall@10"],
            "map": shuffle_metrics["map"],
        },
        "paired_B_vs_A": paired,
        "bootstrap_delta_B_minus_A": boot,
        "main_seed": seeds[0],
        "main_A": seed_rows[0]["A"],
        "main_B": seed_rows[0]["B"],
    }
    return report


def train_phase1(
    feat: pd.DataFrame,
    labels: list[str],
    y: pd.DataFrame,
    seed: int = 7,
    min_contexts: int = VOCAB_MIN_CONTEXTS,
    max_labels: int = MAX_LABELS,
) -> dict:
    """Train A/B/C with study-grouped split; vocab + age fill from train NCTs only."""
    assert "ae_set" in feat.columns, "feat must include ae_set"
    groups = feat["nct_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i, te_i = next(splitter.split(feat, groups=groups))
    feat, age_med = _impute_min_age_train_test(feat, tr_i, te_i)
    labels = _vocab_from_ae_sets(feat.iloc[tr_i]["ae_set"], min_contexts, max_labels)
    labels = [lab for lab in labels if int(sum(lab in s for s in feat.iloc[tr_i]["ae_set"])) >= 2]
    y = _y_from_ae_sets(feat["ae_set"], labels)
    y_tr, y_te = y.iloc[tr_i].to_numpy(), y.iloc[te_i].to_numpy()
    X = feat.drop(columns=["nct_id", "ae_set"])
    n_eval = int((y_te.sum(axis=1) > 0).sum())
    n_test_ctx = int(len(te_i))

    specs = {
        "A": (CAT_A, [], "lr"),
        "B": (CAT_B, NUM_B, "lr"),
        "C": (CAT_C, NUM_C, "lr"),
        "B_gbdt": (CAT_B, NUM_B, "gbdt"),
    }
    models = {}
    metrics = {
        "phase1_version": PHASE1_VERSION,
        "recall_definition": "standard |Top-k ∩ Y| / |Y|",
        "vocab_fit": "train_ncts_only",
        "age_impute": "train_median",
        "age_median_train": age_med,
        "example_unit": "trial_arm_role_context",
        "n_train_contexts": int(len(tr_i)),
        "n_test_contexts": n_test_ctx,
        "n_eval_contexts": n_eval,
        "n_train_studies": int(feat.iloc[tr_i]["nct_id"].nunique()),
        "n_test_studies": int(feat.iloc[te_i]["nct_id"].nunique()),
        "n_labels": len(labels),
        "models": {},
    }
    freq = frequency_baseline(y_tr, len(te_i))
    metrics["models"]["freq"] = {"name": "frequency baseline", **_rank_metrics(freq, y_te)}

    name_map = {
        "A": "P(Y|Material VO)",
        "B": "P(Y|Material VO, platform, C_clinical)",
        "C": "P(Y|Material VO, platform, C, VO family)",
        "B_gbdt": "P(Y|Material VO, platform, C) HistGBDT sanity",
    }
    for name, (cat, num, kind) in specs.items():
        pipe = _pipeline(cat, num, kind=kind)
        pipe.fit(X.iloc[tr_i][cat + num], y_tr)
        proba = pipe.predict_proba(X.iloc[te_i][cat + num])
        models[name] = pipe
        metrics["models"][name] = {
            "name": name_map[name],
            "features": cat + num,
            "estimator": kind,
            **_rank_metrics(proba, y_te),
        }

    metrics["models"]["B"]["delta_recall@10_vs_A"] = (
        metrics["models"]["B"]["recall@10"] - metrics["models"]["A"]["recall@10"]
    )
    metrics["models"]["C"]["delta_recall@10_vs_B"] = (
        metrics["models"]["C"]["recall@10"] - metrics["models"]["B"]["recall@10"]
    )
    metrics["models"]["B_gbdt"]["delta_recall@10_vs_A"] = (
        metrics["models"]["B_gbdt"]["recall@10"] - metrics["models"]["A"]["recall@10"]
    )
    return {
        "models": models,
        "metrics": metrics,
        "labels": labels,
        "train_index": tr_i,
        "test_index": te_i,
        "y": y,
        "feat": feat,
    }


def condition_importance(model_b: Pipeline, labels: list[str]) -> pd.DataFrame:
    """Mean |logistic weight| per raw feature, tagged by SEA-CDM bucket (App B)."""
    prep = model_b.named_steps["prep"]
    clf = model_b.named_steps["clf"]
    names = prep.get_feature_names_out()
    coefs = np.vstack([est.coef_.ravel() for est in clf.estimators_])  # labels × feats
    mean_abs = np.abs(coefs).mean(axis=0)
    rows = []
    for feat_name, w in zip(names, mean_abs):
        raw = feat_name.split("__", 1)[-1]
        key = raw.split("_", 1)[0] if raw.split("_", 1)[0] in SEA_BUCKET else raw
        for prefix in SEA_BUCKET:
            if raw == prefix or raw.startswith(prefix + "_"):
                key = prefix
                break
        rows.append({"feature": raw, "sea_class": SEA_BUCKET.get(key, "other"), "mean_abs_weight": float(w)})
    out = pd.DataFrame(rows).sort_values("mean_abs_weight", ascending=False)
    return out


def label_condition_weights(model_b: Pipeline, labels: list[str], ae_norm: str) -> pd.DataFrame:
    if ae_norm not in labels:
        return pd.DataFrame()
    prep = model_b.named_steps["prep"]
    est = model_b.named_steps["clf"].estimators_[labels.index(ae_norm)]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    return pd.DataFrame({"feature": names, "weight": est.coef_.ravel()}).sort_values("weight", key=np.abs, ascending=False)


def context_row(
    vo: str,
    arm_role: str = "vaccine",
    route: str = "unknown",
    dose: str = "unknown",
    sex: str = "unknown",
    older: Optional[bool] = None,
    child: Optional[bool] = None,
    adult: Optional[bool] = True,
    min_age: float = 18.0,
    disease: str = "unknown",
    vo_platform: Optional[str] = None,
) -> pd.DataFrame:
    """Build one feature row for inference.

    vo = fine Material (mapped VO class). vo_platform = production-kind roll-up.
    For older/child/adult: True→yes, False→no, None→unknown (caller may
    marginalize over yes/no instead — see predict_aes_query).
    Sex "(any)" / empty → SEX_ALL (training majority), not a fake "unknown".
    """
    sex_raw = str(sex or "").strip()
    if sex_raw in {"", "(any)", "unknown", "any"}:
        sex_val = "SEX_ALL"
    elif sex_raw in {"ALL", "FEMALE", "MALE"}:
        sex_val = f"SEX_{sex_raw}"
    elif sex_raw.startswith("SEX_"):
        sex_val = sex_raw
    else:
        sex_val = "SEX_ALL"

    def _yn(v: Optional[bool]) -> str:
        if v is None:
            return "unknown"
        return "yes" if v else "no"

    material = vo or "unknown"
    platform = vo_platform or material
    return pd.DataFrame(
        [
            {
                "vo": material,
                "vo_platform": platform,
                "vo_family": vo_family(platform),
                "route": route if route and route not in {"(any)", ""} else "unknown",
                "dose": dose if dose and dose not in {"(any)", ""} else "unknown",
                "sex": sex_val,
                "older": _yn(older),
                "child": _yn(child),
                "adult": _yn(adult),
                "arm_role": arm_role if arm_role in {"vaccine", "control"} else "vaccine",
                "min_age": float(min_age),
                "disease": disease if disease and disease not in {"(any)", ""} else "unknown",
            }
        ]
    )


def predict_aes(model: Pipeline, labels: list[str], row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    proba = model.predict_proba(row)[0]
    order = np.argsort(-proba)[:top_n]
    return pd.DataFrame(
        {
            "rank": range(1, len(order) + 1),
            "ae_norm": [labels[j] for j in order],
            "probability": [float(proba[j]) for j in order],
        }
    )


def predict_aes_query(
    model: Pipeline,
    labels: list[str],
    *,
    vo: str,
    arm_role: str = "vaccine",
    route: str = "(any)",
    dose: str = "(any)",
    sex: str = "(any)",
    older: Optional[bool] = None,
    child: Optional[bool] = None,
    disease: str = "(any)",
    top_n: int = 10,
    vo_platform: Optional[str] = None,
) -> pd.DataFrame:
    """Predict AE ranks; when older/child are None ('any'), average yes/no probs.

    vo = fine Material VO class (specific vaccine). vo_platform = production kind.
    """
    grids = [{}]
    if older is None:
        grids = [{**g, "older": v} for g in grids for v in (True, False)]
    else:
        grids = [{**g, "older": older} for g in grids]
    if child is None:
        grids = [{**g, "child": v} for g in grids for v in (True, False)]
    else:
        grids = [{**g, "child": child} for g in grids]
    if not disease or disease in {"(any)", "any"}:
        levels = ("covid", "influenza", "pneumococcal", "meningococcal", "hepatitis", "hpv", "rsv", "malaria", "zoster", "rotavirus", "other")
        grids = [{**g, "disease": v} for g in grids for v in levels]
    else:
        grids = [{**g, "disease": disease} for g in grids]

    probas = []
    for g in grids:
        row = context_row(
            vo=vo,
            vo_platform=vo_platform,
            arm_role=arm_role,
            route=route,
            dose=dose,
            sex=sex,
            older=g["older"],
            child=g["child"],
            disease=g["disease"],
        )
        probas.append(model.predict_proba(row)[0])
    mean_p = np.mean(np.vstack(probas), axis=0)
    order = np.argsort(-mean_p)[:top_n]
    return pd.DataFrame(
        {
            "rank": range(1, len(order) + 1),
            "ae_norm": [labels[j] for j in order],
            "probability": [float(mean_p[j]) for j in order],
        }
    )


def compare_model_predictions(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    top_delta: int = 5,
) -> pd.DataFrame:
    merged = pred_b.merge(pred_a, on="ae_norm", suffixes=("_B", "_A"), how="outer")
    if "probability_B" not in merged.columns or "probability_A" not in merged.columns:
        return pd.DataFrame()
    merged["delta_B_minus_A"] = merged["probability_B"].fillna(0) - merged["probability_A"].fillna(0)
    return merged.nlargest(top_delta, "delta_B_minus_A")[
        ["ae_norm", "probability_B", "probability_A", "delta_B_minus_A"]
    ].reset_index(drop=True)


def build_app_c_report(
    *,
    vo: str,
    arm_role: str,
    route: str,
    dose: str,
    sex: str,
    older: Optional[bool],
    child: Optional[bool],
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    obs: Optional[pd.DataFrame],
    metrics: Optional[dict] = None,
) -> str:
    """Markdown one-pager for trial-planning / advisor demos."""
    older_s = "any" if older is None else ("yes" if older else "no")
    child_s = "any" if child is None else ("yes" if child else "no")
    lines = [
        "# SEA-ML Phase 1 — App C prediction report",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Query (conditions)",
        "",
        f"- **X (platform):** `{vo}` — {vo_label(vo)}",
        f"- **Group:** {arm_role}",
        f"- **I route:** {route or 'any'}",
        f"- **I dose:** {dose or 'any'}",
        f"- **S sex eligibility:** {sex or 'any'}",
        f"- **S older-adult eligibility:** {older_s}",
        f"- **S child-including eligibility:** {child_s}",
        "",
        "## Model B — P(Y | vaccine, C_clinical) top AE",
        "",
        "| rank | ae_norm | probability |",
        "| ---: | --- | ---: |",
    ]
    for r in pred_b.itertuples(index=False):
        lines.append(f"| {r.rank} | {r.ae_norm} | {r.probability:.3f} |")
    lines += [
        "",
        "## Model A — P(Y | vaccine) only (baseline)",
        "",
        "| rank | ae_norm | probability |",
        "| ---: | --- | ---: |",
    ]
    for r in pred_a.itertuples(index=False):
        lines.append(f"| {r.rank} | {r.ae_norm} | {r.probability:.3f} |")
    lines += ["", "## App A observed (same slice, registry counts)", ""]
    if obs is None or obs.empty:
        lines.append("_No observed AE ranks in this slice (narrow filters or sparse platform)._")
    else:
        cols = [c for c in ["ae_term", "trial_count", "subjects_affected_sum"] if c in obs.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in obs[cols].iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    if metrics and "models" in metrics:
        mb = metrics["models"].get("B", {})
        ma = metrics["models"].get("A", {})
        lines += [
            "",
            "## Held-out protocol (training snapshot)",
            "",
            f"- Model B recall@10 = **{mb.get('recall@10', float('nan')):.3f}**",
            f"- Model A recall@10 = **{ma.get('recall@10', float('nan')):.3f}**",
            f"- Δ (B−A) = **{mb.get('delta_recall@10_vs_A', float('nan')):+.3f}**",
            f"- Labels = {metrics.get('n_labels', '?')} frequent AE terms",
        ]
    lines += [
        "",
        "## Disclaimer",
        "",
        "Registry-grain multi-label ranking on public AACT vaccine trials. "
        "**Not** individual patient risk, **not** MedDRA epidemiology, **not** a rare-AE "
        "(e.g. myocarditis) predictor. Use App A to look up rare-term coverage.",
        "",
    ]
    return "\n".join(lines)


def save_bundle(bundle: dict, feat: pd.DataFrame, y: pd.DataFrame, path: Path = CKPT) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": bundle["models"],
        "labels": bundle["labels"],
        "metrics": bundle["metrics"],
        "meta": {
            "phase1_version": PHASE1_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "Registry-grain AE ranking for vaccine-trial planning. "
                "Not individual risk or rare-AE diagnosis."
            ),
        },
    }
    joblib.dump(payload, path / "sea_ml_phase1.joblib")
    (path / "metrics.json").write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    feat_save = feat.drop(columns=["ae_set"], errors="ignore")
    feat_save.to_csv(path / "contexts.csv", index=False)
    y_use = bundle.get("y", y)
    if y_use is not None:
        y_use.to_csv(path / "labels_multihot.csv", index=False)
    return path


def load_bundle(path: Path = CKPT) -> dict:
    return joblib.load(path / "sea_ml_phase1.joblib")
