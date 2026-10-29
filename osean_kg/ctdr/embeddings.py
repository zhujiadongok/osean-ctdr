"""TransE and CompGCN-style encoders for the CTDR input graph."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ctdr import config as cfg


def train_transe(
    n: int,
    n_rel: int,
    triples: list[tuple[int, int, int]],
    *,
    dim: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    n_neg: int | None = None,
) -> np.ndarray:
    dim = dim or cfg.DIM
    epochs = epochs or cfg.EPOCHS
    lr = lr if lr is not None else cfg.TRANSE_LR
    n_neg = n_neg if n_neg is not None else cfg.N_NEG

    torch.manual_seed(cfg.SEED)
    ent = torch.nn.Embedding(n, dim)
    rel = torch.nn.Embedding(n_rel, dim)
    torch.nn.init.xavier_uniform_(ent.weight)
    torch.nn.init.xavier_uniform_(rel.weight)
    opt = torch.optim.Adam(list(ent.parameters()) + list(rel.parameters()), lr=lr)
    data = torch.tensor(triples, dtype=torch.long)
    for _ in range(epochs):
        perm = data[torch.randperm(len(data))]
        for start in range(0, len(perm), cfg.BATCH):
            batch = perm[start : start + cfg.BATCH]
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            bsz = len(batch)
            neg = torch.randint(0, n, (bsz, n_neg))
            pos = (ent(h) + rel(r) - ent(t)).norm(p=1, dim=1)
            # best (hardest) negative among n_neg
            h_exp = ent(h).unsqueeze(1).expand(-1, n_neg, -1)
            r_exp = rel(r).unsqueeze(1).expand(-1, n_neg, -1)
            neg_e = ent(neg)
            neg_d = (h_exp + r_exp - neg_e).norm(p=1, dim=2).min(dim=1).values
            loss = F.relu(pos - neg_d + cfg.MARGIN).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ent.weight.data = F.normalize(ent.weight.data, p=2, dim=1)
    return ent.weight.detach().numpy()


def train_compgcn(
    n: int,
    n_rel: int,
    triples: list[tuple[int, int, int]],
    *,
    dim: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> np.ndarray:
    """Backward-compatible CompGCN entry point."""
    return train_compgcn_v2(
        n, n_rel, triples, dim=dim, epochs=epochs, lr=lr, n_layers=2, n_neg=5
    )


def train_compgcn_v2(
    n: int,
    n_rel: int,
    triples: list[tuple[int, int, int]],
    *,
    dim: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    n_layers: int = 3,
    n_neg: int = 8,
) -> np.ndarray:
    """Stronger CompGCN-style encoder: self-loops, residual layers, hard negatives."""
    dim = dim or cfg.DIM
    epochs = epochs or max(cfg.EPOCHS, 500)
    lr = lr if lr is not None else min(cfg.COMPGCN_LR, 2e-2)

    torch.manual_seed(cfg.SEED)
    x = torch.nn.Parameter(torch.empty(n, dim))
    torch.nn.init.xavier_uniform_(x)
    # per-layer relation gates + a shared self-loop scale
    rel_w = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.ones(n_rel + 1, dim)) for _ in range(n_layers)]
    )
    for w in rel_w:
        torch.nn.init.xavier_uniform_(w)
    opt = torch.optim.Adam([x, *rel_w.parameters()], lr=lr)

    edges = torch.tensor(triples, dtype=torch.long)
    both = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    # self-loops use relation id = n_rel
    loops = torch.stack(
        [
            torch.arange(n, dtype=torch.long),
            torch.full((n,), n_rel, dtype=torch.long),
            torch.arange(n, dtype=torch.long),
        ],
        dim=1,
    )
    all_e = torch.cat([both, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0).sqrt()

    def encode() -> torch.Tensor:
        h = x
        for layer_i in range(n_layers):
            w = rel_w[layer_i]
            nbr = h[dst] * w[rel]
            nbr = nbr / deg[src].unsqueeze(1) / deg[dst].unsqueeze(1)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h + msg)
        return h

    for ep in range(epochs):
        h2 = encode()
        perm = edges[torch.randperm(len(edges))][: max(cfg.BATCH * 3, 1536)]
        hh, rr, tt = perm[:, 0], perm[:, 1], perm[:, 2]
        neg = torch.randint(0, n, (len(perm), n_neg))
        # relation-aware distance: ||h + gate(r) - t||
        gate = rel_w[-1][rr]
        pos = (h2[hh] * gate - h2[tt]).pow(2).sum(1)
        h_exp = (h2[hh] * gate).unsqueeze(1)
        neg_d = (h_exp - h2[neg]).pow(2).sum(2).min(dim=1).values
        loss = F.relu(pos - neg_d + cfg.MARGIN).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                x.data = F.normalize(x.data, p=2, dim=1)
    with torch.no_grad():
        h2 = F.normalize(encode(), p=2, dim=1)
    return h2.detach().numpy()


def study_matrix(emb: np.ndarray, graph: dict) -> tuple[np.ndarray, list[str]]:
    ids = graph["study_ids"]
    index = graph["index"]
    vecs = np.stack([emb[index[f"STUDY:{n}"]] for n in ids])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms, ids


def inductive_study_vectors(
    emb: np.ndarray,
    train_graph: dict,
    studies: "pd.DataFrame",
    query_ids: list[str],
) -> dict[str, np.ndarray]:
    """Embed a study as the mean of its known INPUT feature-node vectors."""
    import pandas as pd  # local to avoid hard cycle

    assert isinstance(studies, pd.DataFrame)
    index = train_graph["index"]
    meta = studies.set_index("nct_id")
    out: dict[str, np.ndarray] = {}
    feat_cols = [
        ("arch_vo", None),  # raw VO id
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
    for q in query_ids:
        if q not in meta.index:
            continue
        row = meta.loc[q]
        parts = []
        # Heavier weight on ontology/product and phase — same signal as ontology-kNN.
        weights = {
            "arch_vo": 3.0,
            "disease": 2.0,
            "vo_platform": 2.0,
            "phase": 2.0,
            "primary_purpose": 1.0,
            "route": 1.0,
            "dose": 0.5,
            "sex": 0.5,
            "older": 0.5,
            "child": 0.5,
            "sponsor_class": 0.5,
        }
        vo = str(row["arch_vo"])
        if vo in index:
            for _ in range(int(weights["arch_vo"])):
                parts.append(emb[index[vo]])
        for col, prefix in feat_cols[1:]:
            key = f"{prefix}:{row[col]}"
            if key in index:
                for _ in range(max(1, int(round(weights.get(col, 1.0))))):
                    parts.append(emb[index[key]])
        if not parts:
            out[q] = np.zeros(emb.shape[1], dtype=np.float64)
            continue
        v = np.mean(np.stack(parts), axis=0)
        nrm = np.linalg.norm(v) + 1e-8
        out[q] = v / nrm
    return out
