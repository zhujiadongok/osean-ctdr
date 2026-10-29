"""Advisor baselines B2 (BM25) and B3 (TF-IDF cosine) for CTDR."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ctdr import config as cfg

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore


_TOKEN = re.compile(r"[a-z0-9_]+")


def trial_text(row: pd.Series) -> str:
    parts = [
        str(row.get("disease_conditions", "")),
        str(row.get("specific_vaccine_name", "")),
        str(row.get("disease", "")),
        str(row.get("arch_vo", "")),
        str(row.get("vo_platform", "")),
        str(row.get("phase", "")),
        str(row.get("primary_purpose", "")),
        str(row.get("route", "")),
        str(row.get("sponsor_class", "")),
    ]
    return " ".join(p for p in parts if p and p not in {"nan", "unknown", ""})


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def _vote_from_ranked(
    ranked: list[tuple[str, float]],
    labels: pd.DataFrame,
    targets: tuple[str, ...],
    k: int,
) -> dict[str, str]:
    picks = ranked[:k]
    pred: dict[str, str] = {}
    lab = labels.set_index("nct_id")
    for t in targets:
        weights: dict[str, float] = defaultdict(float)
        for nct, s in picks:
            weights[str(lab.loc[nct, t])] += max(float(s), 0.0) + 1e-6
        pred[t] = max(weights, key=weights.get) if weights else "unknown"
    return pred


def bm25_recommend(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k: int = 10,
    targets: tuple[str, ...] = cfg.TARGETS,
) -> dict[str, dict[str, str]]:
    train_ids = list(train["nct_id"])
    docs = [trial_text(r) for _, r in train.iterrows()]
    tokenized = [_tok(d) for d in docs]
    if BM25Okapi is None:
        # fallback: TF-IDF as sparse retrieval
        return tfidf_recommend(train, test, k=k, targets=targets)
    bm25 = BM25Okapi(tokenized)
    out: dict[str, dict[str, str]] = {}
    for _, row in test.iterrows():
        q = _tok(trial_text(row))
        scores = np.asarray(bm25.get_scores(q), dtype=float)
        order = np.argsort(-scores)
        ranked = [(train_ids[j], float(scores[j])) for j in order]
        out[row.nct_id] = _vote_from_ranked(ranked, train, targets, k)
    return out


def tfidf_recommend(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k: int = 10,
    targets: tuple[str, ...] = cfg.TARGETS,
) -> dict[str, dict[str, str]]:
    """B3-style dense-ish baseline: character/word TF-IDF cosine (no GPU)."""
    train_ids = list(train["nct_id"])
    tr_docs = [trial_text(r) for _, r in train.iterrows()]
    te_docs = [trial_text(r) for _, r in test.iterrows()]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=50000,
    )
    Xtr = vec.fit_transform(tr_docs)
    Xte = vec.transform(te_docs)
    sims = cosine_similarity(Xte, Xtr)
    out: dict[str, dict[str, str]] = {}
    for i, nct in enumerate(test["nct_id"]):
        order = np.argsort(-sims[i])
        ranked = [(train_ids[j], float(sims[i, j])) for j in order]
        out[nct] = _vote_from_ranked(ranked, train, targets, k)
    return out
