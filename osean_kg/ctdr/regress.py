"""Classical baselines: frequency, logistic regression, HistGBDT."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ctdr import config as cfg


def _xy(
    df: pd.DataFrame,
    target: str,
    cats: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    use = list(cats) if cats is not None else list(cfg.INPUT_CATS)
    X = df[use].fillna("unknown").astype(str)
    y = df[target].astype(str).to_numpy()
    return X, y, use


def frequency_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    targets: tuple[str, ...] = cfg.TARGETS,
) -> dict[str, dict[str, str]]:
    """Most common label in train studies with the same phase (fallback: global mode)."""
    out: dict[str, dict[str, str]] = {n: {} for n in test["nct_id"]}
    for t in targets:
        global_mode = Counter(train[t].astype(str)).most_common(1)[0][0]
        by_phase = {
            ph: Counter(g[t].astype(str)).most_common(1)[0][0]
            for ph, g in train.groupby("phase")
        }
        for row in test.itertuples(index=False):
            out[row.nct_id][t] = by_phase.get(row.phase, global_mode)
    return out


_UNK = frozenset({"unknown", "nan", "", "None", "none", "NaN"})


def _fit_predict_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    kind: str,
    cats: list[str] | None = None,
) -> dict[str, str]:
    # Drop unknown gold from training (aligned with kNN vote skip / eval exclude).
    keep = ~train[target].astype(str).isin(_UNK)
    train = train.loc[keep]
    Xtr, ytr, cats = _xy(train, target, cats=cats)
    Xte, _, _ = _xy(test, target, cats=cats)
    if len(ytr) == 0:
        return {nct: "unknown" for nct in test["nct_id"]}
    # drop ultra-rare labels in train (<2) by mapping to most common
    counts = Counter(ytr)
    mode = counts.most_common(1)[0][0]
    ytr = np.array([y if counts[y] >= 2 else mode for y in ytr])
    balanced = kind.endswith("_bal")
    base = "lr" if kind.startswith("lr") else "gbdt"

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)]
    )
    if base == "lr":
        clf = LogisticRegression(
            max_iter=cfg.REG_MAX_ITER,
            solver="liblinear",
            class_weight="balanced" if balanced else None,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_depth=cfg.GBDT_MAX_DEPTH,
            learning_rate=cfg.GBDT_LR,
            max_iter=cfg.GBDT_ITER,
            random_state=cfg.SEED,
        )
    # multiclass: if binary-ish still OK; for many classes use OVR for LR stability
    if base == "lr" and len(set(ytr)) > 2:
        pipe = Pipeline([("prep", pre), ("clf", OneVsRestClassifier(clf))])
    else:
        pipe = Pipeline([("prep", pre), ("clf", clf)])
    fit_kw = {}
    if balanced and base == "gbdt":
        freq = Counter(ytr)
        sw = np.array([1.0 / freq[y] for y in ytr], dtype=float)
        sw *= len(sw) / sw.sum()
        fit_kw["clf__sample_weight"] = sw
    pipe.fit(Xtr, ytr, **fit_kw)
    pred = pipe.predict(Xte)
    return {nct: str(p) for nct, p in zip(test["nct_id"], pred)}


def regress_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str = "lr",
    targets: tuple[str, ...] = cfg.TARGETS,
    cats: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {n: {} for n in test["nct_id"]}
    for t in targets:
        one = _fit_predict_one(train, test, t, kind=kind, cats=cats)
        for nct, val in one.items():
            out[nct][t] = val
    return out
