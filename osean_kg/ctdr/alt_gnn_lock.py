"""Alternative GNNs on the OSEAN/CTDR input graph (validation-locked).

Models (all use build_input_graph triples; design targets never as edges):
  - RGCN: relation-specific linear transforms + aggregation
  - SAGE: mean-aggregation inductive encoder with relation gates
  - GAT: attention over neighbors (relation-aware bias)

Each has supervised multi-head classifiers for the four structure targets.
Inductive queries = weighted mean of feature-node embeddings.

  python -u -m ctdr.alt_gnn_lock
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
OUT = cfg.CTDR_DIR / "optimize" / "alt_gnn"


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
        maps[t] = {lab: i for i, lab in enumerate(labs)}
    return maps


def _y_tensor(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict[str, torch.Tensor]:
    out = {}
    for t, m in maps.items():
        counts = Counter(str(x) for x in df[t])
        maj = counts.most_common(1)[0][0]
        if maj not in m:
            maj = next(iter(m))
        out[t] = torch.tensor([m.get(str(v), m[maj]) for v in df[t].astype(str)], dtype=torch.long)
    return out


def _edges(graph: dict, n: int):
    edges = torch.tensor(graph["triples"], dtype=torch.long)
    both = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    loops = torch.stack(
        [torch.arange(n), torch.full((n,), graph["n_rel"], dtype=torch.long), torch.arange(n)],
        dim=1,
    )
    all_e = torch.cat([both, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0)
    return edges, src, rel, dst, deg


def _inductive(h: np.ndarray, graph: dict, studies: pd.DataFrame, qids: list[str]) -> np.ndarray:
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
        ("vo_platform", "PLATFORM"),
        ("phase", "PHASE"),
        ("primary_purpose", "PURPOSE"),
        ("route", "ROUTE"),
        ("dose", "DOSE"),
        ("sex", "SEX"),
        ("older", "OLDER"),
        ("child", "CHILD"),
        ("sponsor_class", "SPONSOR"),
    ]
    rows = []
    for q in qids:
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
            rows.append(np.zeros(h.shape[1]))
        else:
            v = np.mean(np.stack(parts), 0)
            rows.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(rows)


class _Heads(torch.nn.Module):
    def __init__(self, dim: int, n_classes: dict[str, int]):
        super().__init__()
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

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {t: h(z) for t, h in self.heads.items()}


class RGCN(torch.nn.Module):
    """R-GCN-style: per-relation diagonal gates + mean aggregation."""

    def __init__(self, n: int, n_rel: int, dim: int, n_classes: dict[str, int], n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.x = torch.nn.Parameter(torch.empty(n, dim))
        torch.nn.init.xavier_uniform_(self.x)
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
        self.clf = _Heads(dim, n_classes)

    def encode(self, src, rel, dst, deg):
        h = self.x
        for i in range(self.n_layers):
            nbr = h[dst] * self.w_rel[i][rel]
            nbr = nbr / deg[src].unsqueeze(1).clamp(min=1.0)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h @ self.w_root[i] + msg)
        return F.normalize(h, p=2, dim=1)


class SAGE(torch.nn.Module):
    """GraphSAGE-mean with relation gates; concat self || neigh then linear."""

    def __init__(self, n: int, n_rel: int, dim: int, n_classes: dict[str, int], n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.x = torch.nn.Parameter(torch.empty(n, dim))
        torch.nn.init.xavier_uniform_(self.x)
        self.w_rel = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(n_rel + 1, dim)) for _ in range(n_layers)]
        )
        self.w_self = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(2 * dim, dim)) for _ in range(n_layers)]
        )
        for w in self.w_rel:
            torch.nn.init.xavier_uniform_(w)
        for w in self.w_self:
            torch.nn.init.xavier_uniform_(w)
        self.clf = _Heads(dim, n_classes)

    def encode(self, src, rel, dst, deg):
        h = self.x
        for i in range(self.n_layers):
            nbr = h[dst] * self.w_rel[i][rel]
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            msg = msg / deg.unsqueeze(1).clamp(min=1.0)
            h = F.relu(torch.cat([h, msg], dim=1) @ self.w_self[i])
            h = F.normalize(h, p=2, dim=1)
        return h


class GAT(torch.nn.Module):
    """Single-head relation-biased attention aggregation."""

    def __init__(self, n: int, n_rel: int, dim: int, n_classes: dict[str, int], n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.x = torch.nn.Parameter(torch.empty(n, dim))
        torch.nn.init.xavier_uniform_(self.x)
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(dim, dim)) for _ in range(n_layers)]
        )
        self.att = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(2 * dim)) for _ in range(n_layers)]
        )
        self.rel_bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(n_rel + 1)) for _ in range(n_layers)]
        )
        for w in self.w:
            torch.nn.init.xavier_uniform_(w)
        for a in self.att:
            torch.nn.init.xavier_uniform_(a.unsqueeze(0))
        self.clf = _Heads(dim, n_classes)

    def encode(self, src, rel, dst, deg):
        # deg unused; attention replaces degree norm
        h = self.x
        for i in range(self.n_layers):
            hw = h @ self.w[i]
            # e_ij = LeakyReLU(a^T [h_i || h_j] + bias_r)
            a = self.att[i]
            score = (hw[src] * a[: hw.size(1)]).sum(1) + (hw[dst] * a[hw.size(1) :]).sum(1)
            score = F.leaky_relu(score + self.rel_bias[i][rel], 0.2)
            # stable softmax per source via scatter max/sum
            score_max = torch.full((h.size(0),), -1e9, device=h.device)
            score_max.scatter_reduce_(0, src, score, reduce="amax", include_self=True)
            score = score - score_max[src]
            exp = score.exp()
            denom = torch.zeros(h.size(0), device=h.device)
            denom.index_add_(0, src, exp)
            alpha = exp / (denom[src] + 1e-8)
            msg = torch.zeros_like(hw)
            msg.index_add_(0, src, hw[dst] * alpha.unsqueeze(1))
            h = F.elu(msg)
            h = F.normalize(h, p=2, dim=1)
        return h


def _train_one(
    kind: str,
    fit: pd.DataFrame,
    query: pd.DataFrame,
    both: pd.DataFrame,
    *,
    dim: int,
    epochs: int,
    lr: float,
    seed: int = 7,
) -> dict[str, dict[str, str]]:
    torch.manual_seed(seed)
    g = build_input_graph(fit)
    n = len(g["nodes"])
    maps = _label_maps(fit)
    n_classes = {t: len(m) for t, m in maps.items()}
    inv = {t: {i: lab for lab, i in m.items()} for t, m in maps.items()}
    edges, src, rel, dst, deg = _edges(g, n)

    if kind == "rgcn":
        model = RGCN(n, g["n_rel"], dim, n_classes)
    elif kind == "sage":
        model = SAGE(n, g["n_rel"], dim, n_classes)
    elif kind == "gat":
        model = GAT(n, g["n_rel"], dim, n_classes)
    else:
        raise ValueError(kind)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    study_ids = list(fit["nct_id"])
    study_idx = torch.tensor([g["index"][f"STUDY:{s}"] for s in study_ids], dtype=torch.long)
    y = _y_tensor(fit, maps)

    model.train()
    for _ in range(epochs):
        h = model.encode(src, rel, dst, deg)
        logits = model.clf(h[study_idx])
        loss = sum(F.cross_entropy(logits[t], y[t]) for t in STRUCT) / len(STRUCT)
        # tiny structure regularizer
        perm = edges[torch.randperm(len(edges))][:512]
        pos = (h[perm[:, 0]] - h[perm[:, 2]]).pow(2).sum(1)
        neg = torch.randint(0, n, (len(perm),))
        neg_s = (h[perm[:, 0]] - h[neg]).pow(2).sum(1)
        loss = loss + 0.1 * F.relu(pos - neg_s + 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = model.encode(src, rel, dst, deg).cpu().numpy()
        qids = list(query["nct_id"])
        qv = torch.tensor(_inductive(h, g, both, qids), dtype=torch.float32)
        logits = model.clf(qv)
        preds = {q: {} for q in qids}
        for t in STRUCT:
            for q, i in zip(qids, logits[t].argmax(1).tolist()):
                preds[q][t] = inv[t][int(i)]
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
    val_scores = {}
    classical = {
        "hist_gbdt": regress_predict(fit, val, "gbdt", STRUCT),
        "logistic": regress_predict(fit, val, "lr", STRUCT),
        "ontology_knn_k20": ontology_knn_recommend(
            both_val, list(fit["nct_id"]), list(val["nct_id"]), STRUCT, k=20
        ),
        "frequency_by_phase": frequency_predict(fit, val, STRUCT),
    }
    for n, p in classical.items():
        val_scores[n] = _struct(score_predictions(val, p, STRUCT))
        print(f"  VAL {val_scores[n]['mean_accuracy']:.4f}  {n}", flush=True)

    grid = []
    for kind in ("rgcn", "sage", "gat"):
        for dim, epochs, lr in (
            (64, 300, 1e-2),
            (128, 400, 1e-2),
            (128, 400, 5e-3),
        ):
            name = f"{kind}_d{dim}_e{epochs}_lr{lr}"
            print(f"train {name}", flush=True)
            pred = _train_one(kind, fit, val, both_val, dim=dim, epochs=epochs, lr=lr)
            sc = _struct(score_predictions(val, pred, STRUCT))
            val_scores[name] = sc
            grid.append((name, kind, dict(dim=dim, epochs=epochs, lr=lr)))
            print(f"  VAL {sc['mean_accuracy']:.4f} f1={sc['mean_macro_f1']:.4f}  {name}", flush=True)

    best = max(val_scores, key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]))
    best_gnn = max(
        (n for n in val_scores if n.split("_")[0] in {"rgcn", "sage", "gat"}),
        key=lambda n: (val_scores[n]["mean_accuracy"], val_scores[n]["mean_macro_f1"]),
    )
    print(f"LOCKED={best}  BEST_GNN={best_gnn}", flush=True)

    both_te = pd.concat([train, test], ignore_index=True)
    test_preds = {
        "hist_gbdt": regress_predict(train, test, "gbdt", STRUCT),
        "logistic": regress_predict(train, test, "lr", STRUCT),
        "ontology_knn_k20": ontology_knn_recommend(
            both_te, list(train["nct_id"]), list(test["nct_id"]), STRUCT, k=20
        ),
        "frequency_by_phase": frequency_predict(train, test, STRUCT),
    }
    # always score best GNN on test for reporting
    kind = best_gnn.split("_")[0]
    cfg_g = next(c for n, k, c in grid if n == best_gnn)
    test_preds[best_gnn] = _train_one(kind, train, test, both_te, **cfg_g)
    if best != best_gnn and best in classical:
        test_preds[best] = test_preds[best]

    test_scores = {n: _struct(score_predictions(test, p, STRUCT)) for n, p in test_preds.items()}
    payload = {
        "locked": best,
        "best_gnn_on_val": best_gnn,
        "val_top": sorted(
            [{"name": n, "mean_accuracy": val_scores[n]["mean_accuracy"], "mean_macro_f1": val_scores[n]["mean_macro_f1"]} for n in val_scores],
            key=lambda d: (-d["mean_accuracy"], -d["mean_macro_f1"]),
        )[:20],
        "test_scores": {
            n: {"mean_accuracy": s["mean_accuracy"], "mean_macro_f1": s["mean_macro_f1"]}
            for n, s in test_scores.items()
        },
        "gnn_locked": best.split("_")[0] in {"rgcn", "sage", "gat"},
        "graph": "OSEAN/CTDR input graph (SEA-CDM boxes + vo.owl + same_context); targets excluded",
    }
    (OUT / "alt_gnn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"locked={best}",
        f"best_gnn={best_gnn} val={val_scores[best_gnn]['mean_accuracy']:.4f}",
        f"gnn_locked={payload['gnn_locked']}",
        "",
        "## VAL TOP",
    ]
    for row in payload["val_top"][:15]:
        lines.append(f"  {row['mean_accuracy']:.4f}  {row['name']}")
    lines += ["", "## TEST"]
    for n, s in sorted(test_scores.items(), key=lambda kv: -kv[1]["mean_accuracy"]):
        lines.append(f"  {s['mean_accuracy']:.4f}  {n}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
