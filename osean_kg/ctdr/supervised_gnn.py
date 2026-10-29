"""Supervised CompGCN for CTDR design variables (validation-locked).

Unlike the unsupervised embed → kNN pipeline, this trains a classification
head on study vectors with cross-entropy on the four structure targets.
Test / val studies are encoded inductively from feature nodes only.

  python -u -m ctdr.supervised_gnn
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.graph import build_input_graph  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.regress import frequency_predict, regress_predict  # noqa: E402
from ctdr.recommend import ontology_knn_recommend  # noqa: E402

STRUCT = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
)
VAL_START = "2019-01-01"
OUT = cfg.CTDR_DIR / "optimize" / "supervised_gnn"


def _struct(rep: dict) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in STRUCT]
    f1s = [rep["targets"][t]["macro_f1"] for t in STRUCT]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "per_target": {t: rep["targets"][t] for t in STRUCT},
    }


def _label_maps(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for t in STRUCT:
        labs = sorted({str(x) for x in train[t].astype(str) if str(x) not in {"unknown", "nan", ""}})
        # keep rare labels; map unknowns later to majority
        maps[t] = {lab: i for i, lab in enumerate(labs)}
    return maps


def _y_tensor(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    out = {}
    for t, m in maps.items():
        counts = Counter(str(x) for x in df[t])
        maj = counts.most_common(1)[0][0]
        if maj not in m:
            maj = next(iter(m))
        idx = [m.get(str(v), m[maj]) for v in df[t].astype(str)]
        out[t] = torch.tensor(idx, dtype=torch.long)
    return out


class SupervisedCompGCN(torch.nn.Module):
    def __init__(self, n_nodes: int, n_rel: int, dim: int, n_classes: dict[str, int], n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.x = torch.nn.Parameter(torch.empty(n_nodes, dim))
        torch.nn.init.xavier_uniform_(self.x)
        self.rel_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(n_rel + 1, dim)) for _ in range(n_layers)]
        )
        for w in self.rel_w:
            torch.nn.init.xavier_uniform_(w)
        self.heads = torch.nn.ModuleDict(
            {t: torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, c)) for t, c in n_classes.items()}
        )

    def encode(self, src, rel, dst, deg) -> torch.Tensor:
        h = self.x
        for i in range(self.n_layers):
            w = self.rel_w[i]
            nbr = h[dst] * w[rel]
            nbr = nbr / deg[src].unsqueeze(1) / deg[dst].unsqueeze(1)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h + msg)
        return F.normalize(h, p=2, dim=1)

    def forward_studies(self, h: torch.Tensor, study_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        z = h[study_idx]
        return {t: head(z) for t, head in self.heads.items()}


def _edge_tensors(graph: dict, n: int):
    edges = torch.tensor(graph["triples"], dtype=torch.long)
    both = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    loops = torch.stack(
        [torch.arange(n), torch.full((n,), graph["n_rel"], dtype=torch.long), torch.arange(n)],
        dim=1,
    )
    all_e = torch.cat([both, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0).sqrt()
    return edges, src, rel, dst, deg


def _inductive_vectors(
    h: np.ndarray,
    graph: dict,
    studies: pd.DataFrame,
    query_ids: list[str],
) -> np.ndarray:
    index = graph["index"]
    meta = studies.set_index("nct_id")
    wmap = {
        "arch_vo": 3,
        "disease": 2,
        "vo_platform": 2,
        "phase": 2,
        "primary_purpose": 1,
        "route": 1,
        "dose": 1,
        "sex": 1,
        "older": 1,
        "child": 1,
        "sponsor_class": 1,
    }
    cols = [
        ("arch_vo", None),
        ("disease", "DISEASE"),
        ("phase", "PHASE"),
        ("primary_purpose", "PURPOSE"),
        ("vo_platform", "PLATFORM"),
        ("route", "ROUTE"),
        ("dose", "DOSE"),
        ("sex", "SEX"),
        ("older", "OLDER"),
        ("child", "CHILD"),
        ("sponsor_class", "SPONSOR"),
    ]
    out = []
    for q in query_ids:
        row = meta.loc[q]
        parts = []
        vo = str(row["arch_vo"])
        if vo in index:
            parts.extend([h[index[vo]]] * wmap["arch_vo"])
        for col, prefix in cols[1:]:
            key = f"{prefix}:{row[col]}"
            if key in index:
                parts.extend([h[index[key]]] * wmap.get(col, 1))
        if not parts:
            out.append(np.zeros(h.shape[1]))
        else:
            v = np.mean(np.stack(parts), axis=0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out)


def train_and_predict(
    fit: pd.DataFrame,
    query: pd.DataFrame,
    both: pd.DataFrame,
    *,
    dim: int = 128,
    epochs: int = 400,
    lr: float = 1e-2,
    lambda_link: float = 0.2,
    seed: int = 7,
) -> dict[str, dict[str, str]]:
    torch.manual_seed(seed)
    g = build_input_graph(fit)
    n = len(g["nodes"])
    maps = _label_maps(fit)
    n_classes = {t: len(m) for t, m in maps.items()}
    inv = {t: {i: lab for lab, i in m.items()} for t, m in maps.items()}

    model = SupervisedCompGCN(n, g["n_rel"], dim, n_classes, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    edges, src, rel, dst, deg = _edge_tensors(g, n)

    study_ids = list(fit["nct_id"])
    study_idx = torch.tensor([g["index"][f"STUDY:{s}"] for s in study_ids], dtype=torch.long)
    y = _y_tensor(fit, maps)

    model.train()
    for ep in range(epochs):
        h = model.encode(src, rel, dst, deg)
        logits = model.forward_studies(h, study_idx)
        loss_cls = sum(F.cross_entropy(logits[t], y[t]) for t in STRUCT) / len(STRUCT)
        # light link loss to keep graph geometry
        perm = edges[torch.randperm(len(edges))][:1024]
        hh, tt = perm[:, 0], perm[:, 2]
        neg = torch.randint(0, n, (len(perm),))
        pos = (h[hh] - h[tt]).pow(2).sum(1)
        neg_s = (h[hh] - h[neg]).pow(2).sum(1)
        loss_link = F.relu(pos - neg_s + 1.0).mean()
        loss = loss_cls + lambda_link * loss_link
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = model.encode(src, rel, dst, deg).cpu().numpy()
        # inductive query vectors → heads
        qids = list(query["nct_id"])
        qv = _inductive_vectors(h, g, both, qids)
        qt = torch.tensor(qv, dtype=torch.float32)
        preds: dict[str, dict[str, str]] = {q: {} for q in qids}
        for t, head in model.heads.items():
            logits = head(qt)
            pred_i = logits.argmax(dim=1).tolist()
            for q, i in zip(qids, pred_i):
                preds[q][t] = inv[t].get(int(i), "unknown")
    return preds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_ctdr_table()
    table["start_date_parsed"] = pd.to_datetime(table["start_date"], errors="coerce")
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    fit = train[train["start_date_parsed"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
    val = train[train["start_date_parsed"] >= pd.Timestamp(VAL_START)].reset_index(drop=True)
    print(f"fit={len(fit)} val={len(val)} test={len(test)}", flush=True)

    both_val = pd.concat([fit, val], ignore_index=True)
    classical = {
        "frequency_by_phase": frequency_predict(fit, val, STRUCT),
        "logistic": regress_predict(fit, val, "lr", STRUCT),
        "hist_gbdt": regress_predict(fit, val, "gbdt", STRUCT),
        "ontology_knn_k20": ontology_knn_recommend(
            both_val, list(fit["nct_id"]), list(val["nct_id"]), STRUCT, k=20
        ),
    }
    val_scores = {n: _struct(score_predictions(val, p, STRUCT)) for n, p in classical.items()}
    for n, s in sorted(val_scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        print(f"  VAL {s['mean_accuracy']:.4f}  {n}", flush=True)

    grid = []
    # Compact validation grid: supervised CompGCN is expensive.
    for dim, epochs, lam, lr in (
        (64, 300, 0.2, 1e-2),
        (128, 400, 0.1, 1e-2),
        (128, 400, 0.3, 5e-3),
        (128, 500, 0.2, 1e-2),
        (128, 500, 0.2, 5e-3),
    ):
        name = f"sup_compgcn_d{dim}_e{epochs}_l{lam}_lr{lr}"
        print(f"train {name}", flush=True)
        pred = train_and_predict(
            fit, val, both_val, dim=dim, epochs=epochs, lr=lr, lambda_link=lam
        )
        sc = _struct(score_predictions(val, pred, STRUCT))
        val_scores[name] = sc
        grid.append((name, sc, dict(dim=dim, epochs=epochs, lr=lr, lambda_link=lam)))
        print(
            f"  VAL {sc['mean_accuracy']:.4f} f1={sc['mean_macro_f1']:.4f}  {name}",
            flush=True,
        )

    best = max(val_scores, key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]))
    print(f"LOCKED={best} val={val_scores[best]['mean_accuracy']:.4f}", flush=True)

    # test once
    both_te = pd.concat([train, test], ignore_index=True)
    test_preds = {
        "frequency_by_phase": frequency_predict(train, test, STRUCT),
        "logistic": regress_predict(train, test, "lr", STRUCT),
        "hist_gbdt": regress_predict(train, test, "gbdt", STRUCT),
        "ontology_knn_k20": ontology_knn_recommend(
            both_te, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20
        ),
    }
    if best.startswith("sup_compgcn"):
        cfg_best = next(c for n, _, c in grid if n == best)
        test_preds[best] = train_and_predict(train, test, both_te, **cfg_best)
    else:
        test_preds[best] = test_preds.get(best) or regress_predict(train, test, "gbdt", STRUCT)

    # also always score best supervised on test for reporting even if not locked
    best_sup = max(
        (n for n in val_scores if n.startswith("sup_compgcn")),
        key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]),
        default=None,
    )
    if best_sup and best_sup not in test_preds:
        cfg_sup = next(c for n, _, c in grid if n == best_sup)
        test_preds[best_sup] = train_and_predict(train, test, both_te, **cfg_sup)

    test_scores = {n: _struct(score_predictions(test, p, STRUCT)) for n, p in test_preds.items()}
    payload = {
        "locked": best,
        "val_scores_top": sorted(
            ({"name": n, **{k: val_scores[n][k] for k in ("mean_accuracy", "mean_macro_f1")}} for n in val_scores),
            key=lambda d: (-d["mean_accuracy"], -d["mean_macro_f1"]),
        )[:20],
        "test_scores": {
            n: {k: test_scores[n][k] for k in ("mean_accuracy", "mean_macro_f1")}
            for n in test_scores
        },
        "gnn_locked": best.startswith("sup_compgcn"),
    }
    (OUT / "supervised_gnn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"locked={best}", f"gnn_locked={payload['gnn_locked']}", "", "## VAL TOP"]
    for row in payload["val_scores_top"][:15]:
        lines.append(f"  {row['mean_accuracy']:.4f}  {row['name']}")
    lines.append("")
    lines.append("## TEST")
    for n, s in sorted(test_scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        lines.append(f"  {s['mean_accuracy']:.4f}  {n}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
