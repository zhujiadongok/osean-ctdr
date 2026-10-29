"""Try to make a graph method win under the coarse-target protocol.

Ideas
-----
1) Feature-initialized R-GCN: study nodes start from tabular one-hots (projected),
   then message-pass on the enriched OSEAN graph, then classify.
2) HistGBDT / logistic that *must* use R-GCN inductive vectors as extra features.
3) Same coarse Y as push08 (clinically coarsened binaries).
4) Lock on 2019-2021 validation only; score 2022+ once.

  python -u -m ctdr.gnn_first_lock
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.push08_lock import (  # noqa: E402
    COARSE,
    HARD2,
    INPUT_CATS,
    TOKEN,
    add_coarse,
    build_rich_graph,
    classical_predict,
    frequency_predict,
    ontology_knn,
    _mean,
)

VAL_START = "2019-01-01"
OUT = cfg.CTDR_DIR / "optimize" / "gnn_first"


def _maps(train: pd.DataFrame):
    maps = {}
    for t in COARSE:
        labs = sorted(set(train[t].astype(str)))
        maps[t] = {lab: i for i, lab in enumerate(labs)}
    inv = {t: {i: lab for lab, i in m.items()} for t, m in maps.items()}
    return maps, inv


def _study_feature_matrix(studies: pd.DataFrame, fit: pd.DataFrame):
    """One-hot features fit on `fit`, transformed for `studies`."""
    cats = list(INPUT_CATS)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(fit[cats].fillna("unknown").astype(str))
    X = enc.transform(studies[cats].fillna("unknown").astype(str))
    return X.astype(np.float32), enc


class FeatureRGCN(torch.nn.Module):
    """Study nodes = projected tabular features; other nodes free embeddings."""

    def __init__(self, n, n_rel, dim, n_classes, n_study, feat_dim, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.n_study = n_study
        self.feat_proj = torch.nn.Linear(feat_dim, dim)
        self.other = torch.nn.Parameter(torch.empty(n - n_study, dim))
        torch.nn.init.xavier_uniform_(self.other)
        self.w_rel = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(n_rel + 1, dim)) for _ in range(n_layers)]
        )
        self.w_root = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(dim, dim)) for _ in range(n_layers)]
        )
        for w in self.w_rel:
            torch.nn.init.xavier_uniform_(w)
        for w in self.w_root:
            torch.nn.init.xavier_uniform_(w)
        self.heads = torch.nn.ModuleDict(
            {
                t: torch.nn.Sequential(
                    torch.nn.Linear(dim, dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(dim, c),
                )
                for t, c in n_classes.items()
            }
        )

    def node_table(self, study_feat: torch.Tensor) -> torch.Tensor:
        # studies are indices 0..n_study-1 in our builder order
        s = self.feat_proj(study_feat)
        return torch.cat([s, self.other], dim=0)

    def encode(self, study_feat, src, rel, dst, deg):
        h = self.node_table(study_feat)
        for i in range(self.n_layers):
            nbr = h[dst] * self.w_rel[i][rel]
            nbr = nbr / deg[src].unsqueeze(1).clamp(min=1.0)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h @ self.w_root[i] + msg)
        return F.normalize(h, p=2, dim=1)


def _edges(g):
    n = len(g["nodes"])
    edges = torch.tensor(g["triples"], dtype=torch.long)
    both = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    loops = torch.stack(
        [torch.arange(n), torch.full((n,), g["n_rel"], dtype=torch.long), torch.arange(n)], dim=1
    )
    all_e = torch.cat([both, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0)
    return edges, src, rel, dst, deg


def _inductive_from_features(h, g, studies, qids, feat_rows, study_id_to_feat_row):
    """For query studies not in train graph: mix projected neighbors of feature nodes.
    We approximate by using the trained other-node embeddings via CTX/VO keys, and
    also the query's own tabular projection passed through feat_proj externally.
    Here `h` already includes train studies; for queries we rebuild from feature nodes only.
    """
    index = g["index"]
    meta = studies.set_index("nct_id")
    # h layout: studies first then others — extract other block by skipping n_study
    # Actually we pass full h numpy after encode; feature nodes live in index.
    out = []
    for q in qids:
        row = meta.loc[q]
        parts = []
        for key, w in (
            (str(row.arch_vo), 3),
            (f"DISEASE:{row.disease}", 2),
            (f"PLATFORM:{row.vo_platform}", 2),
            (f"PHASE:{row.phase}", 2),
            (f"PURPOSE:{row.primary_purpose}", 1),
            (f"CTX:{row.disease}|{row.vo_platform}|{row.phase}", 4),
            (f"ROUTE:{row.route}", 1),
            (f"SPONSOR:{row.sponsor_class}", 1),
        ):
            if key in index:
                parts.extend([h[index[key]]] * w)
        for tok in TOKEN.findall(str(row.disease_conditions).lower())[:15]:
            key = f"TOK:{tok}"
            if key in index:
                parts.append(h[index[key]])
        # also include the query's own tabular projection if provided
        if q in study_id_to_feat_row:
            parts.extend([feat_rows[study_id_to_feat_row[q]]] * 4)
        if not parts:
            out.append(np.zeros(h.shape[1]))
        else:
            v = np.mean(np.stack(parts), 0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out)


def train_feature_rgcn(fit, query, both, dim=128, epochs=500, lr=1e-2):
    torch.manual_seed(7)
    # build graph on fit; ensure study node order matches fit row order
    g = build_rich_graph(fit)
    # reorder check: study nodes are first in builder
    n_study = len(fit)
    assert g["nodes"][:n_study] == [f"STUDY:{n}" for n in fit["nct_id"]]

    X_fit, enc = _study_feature_matrix(fit, fit)
    X_query = enc.transform(query[list(INPUT_CATS)].fillna("unknown").astype(str)).astype(np.float32)
    X_both_fit_order = X_fit  # for inductive helper feat_proj outputs

    maps, inv = _maps(fit)
    n_classes = {t: len(m) for t, m in maps.items()}
    edges, src, rel, dst, deg = _edges(g)
    n = len(g["nodes"])
    model = FeatureRGCN(n, g["n_rel"], dim, n_classes, n_study, X_fit.shape[1], n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    study_feat = torch.tensor(X_fit, dtype=torch.float32)
    y = {t: torch.tensor([maps[t][str(v)] for v in fit[t]], dtype=torch.long) for t in COARSE}
    study_idx = torch.arange(n_study)

    print(f"FeatureRGCN nodes={n} triples={len(g['triples'])} feat_dim={X_fit.shape[1]}", flush=True)
    model.train()
    for ep in range(epochs):
        h = model.encode(study_feat, src, rel, dst, deg)
        logits = {t: model.heads[t](h[study_idx]) for t in COARSE}
        loss = sum(F.cross_entropy(logits[t], y[t]) for t in COARSE) / len(COARSE)
        perm = edges[torch.randperm(len(edges))][:768]
        pos = (h[perm[:, 0]] - h[perm[:, 2]]).pow(2).sum(1)
        neg = torch.randint(0, n, (len(perm),))
        loss = loss + 0.1 * F.relu(pos - (h[perm[:, 0]] - h[neg]).pow(2).sum(1) + 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = model.encode(study_feat, src, rel, dst, deg)
        # projected tabular vectors for queries (same feat_proj)
        q_proj = model.feat_proj(torch.tensor(X_query, dtype=torch.float32)).cpu().numpy()
        h_np = h.cpu().numpy()
        # map query id -> row in q_proj
        qids = list(query["nct_id"])
        qmap = {q: i for i, q in enumerate(qids)}
        qv = _inductive_from_features(h_np, g, both, qids, q_proj, qmap)
        # also fuse with pure tabular projection
        qv = F.normalize(torch.tensor(qv, dtype=torch.float32) + 0.5 * torch.tensor(q_proj), p=2, dim=1)
        preds = {q: {} for q in qids}
        train_vecs = {nct: h_np[i] for i, nct in enumerate(fit["nct_id"])}
        query_vecs = {q: qv[i].numpy() for i, q in enumerate(qids)}
        for t in COARSE:
            for q, i in zip(qids, model.heads[t](qv).argmax(1).tolist()):
                preds[q][t] = inv[t][int(i)]
    return preds, train_vecs, query_vecs, enc, model, g, h_np


def classical_plus_gnn(train, test, train_vecs, query_vecs, kind="gbdt", emb_dim=32):
    cats = list(INPUT_CATS)
    rng = np.random.default_rng(7)
    dim = next(iter(train_vecs.values())).shape[0]
    proj = rng.normal(size=(dim, emb_dim))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

    def pack(df, vecs):
        Xcat = df[cats].fillna("unknown").astype(str).reset_index(drop=True)
        E = np.stack([(vecs[n] @ proj) if n in vecs else np.zeros(emb_dim) for n in df["nct_id"]])
        for i in range(emb_dim):
            Xcat[f"g{i}"] = E[:, i]
        return Xcat

    Xtr = pack(train, train_vecs)
    Xte = pack(test, query_vecs)
    emb_cols = [f"g{i}" for i in range(emb_dim)]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("g", StandardScaler(), emb_cols),
        ]
    )
    out = {n: {} for n in test["nct_id"]}
    for t in COARSE:
        ytr = train[t].astype(str).to_numpy()
        if kind == "lr":
            clf = LogisticRegression(max_iter=500, solver="liblinear")
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.08, max_iter=180, random_state=7
            )
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        for nct, p in zip(test["nct_id"], pipe.predict(Xte)):
            out[nct][t] = str(p)
    return out


def agreement_ensemble(parts: dict[str, dict], prefer: str, targets=COARSE):
    """If all graph-based methods agree, take that; else prefer named method."""
    keys = list(parts)
    qids = list(parts[prefer])
    out = {}
    for q in qids:
        out[q] = {}
        for t in targets:
            votes = Counter(parts[k][q][t] for k in keys)
            top, c = votes.most_common(1)[0]
            out[q][t] = top if c >= 2 else parts[prefer][q][t]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = add_coarse(build_ctdr_table())
    table["start_date_parsed"] = pd.to_datetime(table["start_date"], errors="coerce")
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    fit = train[train["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train[train["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    print(f"fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    both_val = pd.concat([fit, val], ignore_index=True)
    val_preds = {
        "frequency_by_phase": frequency_predict(fit, val),
        "logistic": classical_predict(fit, val, "lr"),
        "hist_gbdt": classical_predict(fit, val, "gbdt"),
        "ontology_knn_k20": ontology_knn(fit, val, 20),
    }

    print("=== FeatureRGCN ===", flush=True)
    feat_pred, tr_vecs, qa_vecs, enc, model, g, h_np = train_feature_rgcn(
        fit, val, both_val, dim=128, epochs=500, lr=1e-2
    )
    val_preds["feature_rgcn"] = feat_pred
    val_preds["gbdt_plus_feature_rgcn"] = classical_plus_gnn(
        fit, val, tr_vecs, qa_vecs, kind="gbdt"
    )
    val_preds["logistic_plus_feature_rgcn"] = classical_plus_gnn(
        fit, val, tr_vecs, qa_vecs, kind="lr"
    )
    # graph-family ensemble preferring feature_rgcn
    val_preds["graph_family_ensemble"] = agreement_ensemble(
        {
            "feature_rgcn": val_preds["feature_rgcn"],
            "gbdt_plus_feature_rgcn": val_preds["gbdt_plus_feature_rgcn"],
            "ontology_knn_k20": val_preds["ontology_knn_k20"],
        },
        prefer="gbdt_plus_feature_rgcn",
    )

    val_scores = {}
    for name, pred in val_preds.items():
        rep = score_predictions(val, pred, COARSE)
        val_scores[name] = {
            "coarse4": _mean(rep, COARSE),
            "hard2": _mean(rep, HARD2),
        }
        print(
            f"  VAL {val_scores[name]['coarse4']['mean_accuracy']:.4f}  "
            f"hard2={val_scores[name]['hard2']['mean_accuracy']:.4f}  {name}",
            flush=True,
        )

    def is_graphy(name: str) -> bool:
        return any(x in name for x in ("rgcn", "graph_family", "plus_feature"))

    best = max(
        val_scores,
        key=lambda n: (val_scores[n]["coarse4"]["mean_accuracy"], val_scores[n]["hard2"]["mean_accuracy"]),
    )
    best_graph = max(
        (n for n in val_scores if is_graphy(n)),
        key=lambda n: (val_scores[n]["coarse4"]["mean_accuracy"], val_scores[n]["hard2"]["mean_accuracy"]),
    )
    print(f"LOCKED={best}  BEST_GRAPH={best_graph}", flush=True)

    # test once for locked + graph best + baselines
    both_te = pd.concat([train, test], ignore_index=True)
    test_preds = {
        "frequency_by_phase": frequency_predict(train, test),
        "logistic": classical_predict(train, test, "lr"),
        "hist_gbdt": classical_predict(train, test, "gbdt"),
        "ontology_knn_k20": ontology_knn(train, test, 20),
    }
    print("=== refit FeatureRGCN on full train ===", flush=True)
    feat_te, tr_v, q_v, *_ = train_feature_rgcn(train, test, both_te, dim=128, epochs=500, lr=1e-2)
    test_preds["feature_rgcn"] = feat_te
    test_preds["gbdt_plus_feature_rgcn"] = classical_plus_gnn(train, test, tr_v, q_v, kind="gbdt")
    test_preds["logistic_plus_feature_rgcn"] = classical_plus_gnn(train, test, tr_v, q_v, kind="lr")
    test_preds["graph_family_ensemble"] = agreement_ensemble(
        {
            "feature_rgcn": test_preds["feature_rgcn"],
            "gbdt_plus_feature_rgcn": test_preds["gbdt_plus_feature_rgcn"],
            "ontology_knn_k20": test_preds["ontology_knn_k20"],
        },
        prefer="gbdt_plus_feature_rgcn",
    )

    test_scores = {}
    for name, pred in test_preds.items():
        rep = score_predictions(test, pred, COARSE)
        test_scores[name] = {
            "coarse4": _mean(rep, COARSE),
            "hard2": _mean(rep, HARD2),
            "per_target": {t: rep["targets"][t] for t in COARSE},
        }

    payload = {
        "locked": best,
        "best_graph_method": best_graph,
        "graph_wins_on_val": is_graphy(best),
        "val_scores": val_scores,
        "test_scores": {
            n: {
                "coarse4": test_scores[n]["coarse4"]["mean_accuracy"],
                "hard2": test_scores[n]["hard2"]["mean_accuracy"],
                "per_target": {t: test_scores[n]["per_target"][t]["accuracy"] for t in COARSE},
            }
            for n in test_scores
        },
    }
    (OUT / "gnn_first.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"locked={best}", f"best_graph={best_graph}", f"graph_wins={payload['graph_wins_on_val']}", "", "## VAL"]
    for n, s in sorted(val_scores.items(), key=lambda kv: -kv[1]["coarse4"]["mean_accuracy"]):
        lines.append(f"  {s['coarse4']['mean_accuracy']:.4f}  {n}")
    lines += ["", "## TEST"]
    for n, s in sorted(test_scores.items(), key=lambda kv: -kv[1]["coarse4"]["mean_accuracy"]):
        mark = " <-- >=0.80" if s["coarse4"]["mean_accuracy"] >= 0.8 else ""
        lines.append(f"  {s['coarse4']['mean_accuracy']:.4f}  {n}{mark}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
