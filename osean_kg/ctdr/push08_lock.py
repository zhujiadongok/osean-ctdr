"""Setting B: coarsened design labels + enriched OSEAN graph + GNN.

Post-hoc sensitivity (not a success criterion of mean accuracy >= 0.80).
Fine-grained Setting A remains the primary analysis.

  allocation          : RANDOMIZED / NON_RANDOMIZED   (unchanged)
  masking_binary      : OPEN (NONE) / BLINDED (any known mask); unknown kept
  intervention_binary : PARALLEL / OTHER; unknown kept
  enrollment_binary   : SMALL (<150) / LARGE (>=150); unknown kept

Unknown gold labels are excluded from training loss and from evaluation
(n_eval reported per target via score_predictions).

Protocol: fit <2019, val 2019-2021, test >=2022. Pick on val only.
The 2022+ cohort was also used in earlier exploratory sweeps — do not
claim an untouched confirmatory test.

  python -u -m ctdr.push08_lock
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
from sklearn.preprocessing import OneHotEncoder

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr import config as cfg  # noqa: E402
from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.metrics import score_predictions  # noqa: E402
from ctdr.recommend import ontology_knn_recommend  # noqa: E402
from vo_architecture import load_classes  # noqa: E402

VAL_START = "2019-01-01"
OUT = cfg.CTDR_DIR / "optimize" / "push08"
TOKEN = re.compile(r"[a-z0-9]{3,}")

COARSE = (
    "allocation",
    "masking_binary",
    "intervention_binary",
    "enrollment_binary",
)
HARD2 = ("masking_binary", "enrollment_binary")
EASY3 = ("allocation", "masking_binary", "intervention_binary")

INPUT_CATS = (
    "disease",
    "arch_vo",
    "vo_platform",
    "phase",
    "primary_purpose",
    "route",
    "dose",
    "sex",
    "older",
    "child",
    "sponsor_class",
)

UNK = frozenset({"unknown", "nan", "", "None", "none"})


def _is_unk(s: pd.Series) -> pd.Series:
    return s.astype(str).isin(UNK)


def add_coarse(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine labels → coarse; preserve unknown (do not coerce to a class)."""
    out = df.copy()
    m = out["masking"].astype(str)
    out["masking_binary"] = np.where(
        _is_unk(m), "unknown", np.where(m.eq("NONE"), "OPEN", "BLINDED")
    )
    im = out["intervention_model"].astype(str)
    out["intervention_binary"] = np.where(
        _is_unk(im), "unknown", np.where(im.eq("PARALLEL"), "PARALLEL", "OTHER")
    )
    eb = out["enrollment_bucket"].astype(str)
    out["enrollment_binary"] = np.where(
        _is_unk(eb),
        "unknown",
        np.where(eb.isin(["lt_50", "50_149"]), "SMALL", "LARGE"),
    )
    return out


def _mean(rep: dict, targets: tuple[str, ...]) -> dict:
    accs = [rep["targets"][t]["accuracy"] for t in targets if rep["targets"][t].get("n_eval", 0) > 0]
    f1s = [rep["targets"][t]["macro_f1"] for t in targets if rep["targets"][t].get("n_eval", 0) > 0]
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_target": {t: rep["targets"][t] for t in targets},
    }


def frequency_predict(train: pd.DataFrame, test: pd.DataFrame, targets=COARSE):
    out = {n: {} for n in test["nct_id"]}
    for t in targets:
        known = train.loc[~_is_unk(train[t]), t].astype(str)
        gmaj = Counter(known).most_common(1)[0][0] if len(known) else "unknown"
        by = {}
        for ph, g in train.groupby("phase"):
            kk = g.loc[~_is_unk(g[t]), t].astype(str)
            if len(kk):
                by[ph] = Counter(kk).most_common(1)[0][0]
        for row in test.itertuples(index=False):
            out[row.nct_id][t] = by.get(row.phase, gmaj)
    return out


def classical_predict(train, test, kind="gbdt", targets=COARSE):
    out = {n: {} for n in test["nct_id"]}
    cats = list(INPUT_CATS)
    for t in targets:
        keep = ~_is_unk(train[t])
        if keep.sum() < 2:
            maj = "unknown"
            for nct in test["nct_id"]:
                out[nct][t] = maj
            continue
        Xtr = train.loc[keep, cats].fillna("unknown").astype(str)
        ytr = train.loc[keep, t].astype(str).to_numpy()
        Xte = test[cats].fillna("unknown").astype(str)
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)]
        )
        if kind == "lr":
            clf = LogisticRegression(max_iter=500, solver="liblinear")
        else:
            clf = HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.08, max_iter=150, random_state=7
            )
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        for nct, p in zip(test["nct_id"], pipe.predict(Xte)):
            out[nct][t] = str(p)
    return out


def ontology_knn(train, test, k=20, targets=COARSE, mode="exact"):
    """Default mode=exact preserves historical Setting B ontology_knn numbers.

    Pass mode='graph' for subclass+role OntSim with matched Material/platform weights.
    """
    both = pd.concat([train, test], ignore_index=True)
    return ontology_knn_recommend(
        both, list(train["nct_id"]), list(test["nct_id"]), targets=targets, k=k, mode=mode
    )


# ---- enriched graph ----
REL = {
    "typed_as": 0,
    "has_disease": 1,
    "has_phase": 2,
    "has_purpose": 3,
    "has_platform": 4,
    "has_route": 5,
    "has_dose": 6,
    "has_sex": 7,
    "has_older": 8,
    "has_child": 9,
    "has_sponsor": 10,
    "subClassOf": 11,
    "has_role": 12,
    "same_context": 13,
    "has_ctx": 14,
    "has_cond_tok": 15,
}


def build_rich_graph(studies: pd.DataFrame) -> dict:
    classes = load_classes()
    arch_ids = {v for v in studies["arch_vo"].astype(str) if v.startswith("VO_")}
    vo_nodes = set(arch_ids)
    changed = True
    while changed:
        changed = False
        for vid in list(vo_nodes):
            for parent in classes.get(vid, {}).get("parents", []):
                if parent in classes and not classes[parent].get("deprecated") and parent not in vo_nodes:
                    vo_nodes.add(parent)
                    changed = True
            for role in classes.get(vid, {}).get("roles", []):
                if role in classes and not classes[role].get("deprecated") and role not in vo_nodes:
                    vo_nodes.add(role)
                    changed = True

    box_nodes = []
    for col, prefix in (
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
    ):
        for val in sorted(studies[col].astype(str).unique()):
            if val and val != "nan":
                box_nodes.append(f"{prefix}:{val}")

    ctx_nodes = []
    tok_nodes = set()
    for row in studies.itertuples(index=False):
        ctx_nodes.append(f"CTX:{row.disease}|{row.vo_platform}|{row.phase}")
        for tok in TOKEN.findall(str(row.disease_conditions).lower()):
            if tok not in {"and", "the", "for", "with", "from"}:
                tok_nodes.add(f"TOK:{tok}")
    ctx_nodes = sorted(set(ctx_nodes))
    tok_nodes = sorted(tok_nodes)

    study_nodes = [f"STUDY:{n}" for n in studies["nct_id"]]
    nodes = study_nodes + sorted(vo_nodes) + box_nodes + ctx_nodes + tok_nodes
    index = {n: i for i, n in enumerate(nodes)}
    triples = []

    for row in studies.itertuples(index=False):
        s = index[f"STUDY:{row.nct_id}"]
        if str(row.arch_vo).startswith("VO_") and row.arch_vo in index:
            triples.append((s, REL["typed_as"], index[row.arch_vo]))
        for col, rel, prefix in (
            ("disease", "has_disease", "DISEASE"),
            ("phase", "has_phase", "PHASE"),
            ("primary_purpose", "has_purpose", "PURPOSE"),
            ("vo_platform", "has_platform", "PLATFORM"),
            ("route", "has_route", "ROUTE"),
            ("dose", "has_dose", "DOSE"),
            ("sex", "has_sex", "SEX"),
            ("older", "has_older", "OLDER"),
            ("child", "has_child", "CHILD"),
            ("sponsor_class", "has_sponsor", "SPONSOR"),
        ):
            key = f"{prefix}:{getattr(row, col)}"
            if key in index:
                triples.append((s, REL[rel], index[key]))
        ctx = f"CTX:{row.disease}|{row.vo_platform}|{row.phase}"
        if ctx in index:
            triples.append((s, REL["has_ctx"], index[ctx]))
        for tok in TOKEN.findall(str(row.disease_conditions).lower()):
            key = f"TOK:{tok}"
            if key in index:
                triples.append((s, REL["has_cond_tok"], index[key]))

    for vid in vo_nodes:
        for parent in classes.get(vid, {}).get("parents", []):
            if parent in index:
                triples.append((index[vid], REL["subClassOf"], index[parent]))
        for role in classes.get(vid, {}).get("roles", []):
            if role in index:
                triples.append((index[vid], REL["has_role"], index[role]))

    buckets = defaultdict(list)
    for row in studies.itertuples(index=False):
        buckets[(str(row.disease), str(row.vo_platform), str(row.phase))].append(str(row.nct_id))
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        hub = ids[0]
        hs = index[f"STUDY:{hub}"]
        for other in ids[1:]:
            os_ = index[f"STUDY:{other}"]
            triples.append((hs, REL["same_context"], os_))
            triples.append((os_, REL["same_context"], hs))
        for a, b in zip(ids, ids[1:]):
            ia, ib = index[f"STUDY:{a}"], index[f"STUDY:{b}"]
            triples.append((ia, REL["same_context"], ib))
            triples.append((ib, REL["same_context"], ia))

    return {
        "nodes": nodes,
        "index": index,
        "triples": triples,
        "n_rel": len(REL),
        "study_ids": list(studies["nct_id"]),
    }


class RGCNClf(torch.nn.Module):
    def __init__(self, n, n_rel, dim, n_classes, n_layers=3):
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
        self.heads = torch.nn.ModuleDict(
            {
                t: torch.nn.Sequential(
                    torch.nn.Linear(dim, dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.15),
                    torch.nn.Linear(dim, c),
                )
                for t, c in n_classes.items()
            }
        )

    def encode(self, src, rel, dst, deg):
        h = self.x
        for i in range(self.n_layers):
            nbr = h[dst] * self.w_rel[i][rel]
            nbr = nbr / deg[src].unsqueeze(1).clamp(min=1.0)
            msg = torch.zeros_like(h)
            msg.index_add_(0, src, nbr)
            h = F.relu(h @ self.w_root[i] + msg)
        return F.normalize(h, p=2, dim=1)


def _inductive(h, graph, studies, qids):
    index = graph["index"]
    meta = studies.set_index("nct_id")
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
            (f"CTX:{row.disease}|{row.vo_platform}|{row.phase}", 3),
            (f"ROUTE:{row.route}", 1),
            (f"SPONSOR:{row.sponsor_class}", 1),
        ):
            if key in index:
                parts.extend([h[index[key]]] * w)
        for tok in TOKEN.findall(str(row.disease_conditions).lower())[:12]:
            key = f"TOK:{tok}"
            if key in index:
                parts.append(h[index[key]])
        if not parts:
            out.append(np.zeros(h.shape[1]))
        else:
            v = np.mean(np.stack(parts), 0)
            out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out)


def train_rgcn(fit, query, both, dim=128, epochs=450, lr=1e-2, targets=COARSE):
    torch.manual_seed(7)
    g = build_rich_graph(fit)
    n = len(g["nodes"])
    print(f"rich graph nodes={n} triples={len(g['triples'])}", flush=True)
    maps = {}
    for t in targets:
        labs = sorted({str(v) for v in fit[t].astype(str) if str(v) not in UNK})
        if not labs:
            labs = ["unknown"]
        maps[t] = {lab: i for i, lab in enumerate(labs)}
    inv = {t: {i: lab for lab, i in m.items()} for t, m in maps.items()}
    n_classes = {t: len(m) for t, m in maps.items()}
    # Per-target rows with known gold (exclude unknown from CE loss).
    known_mask = {
        t: torch.tensor([str(v) not in UNK for v in fit[t].astype(str)], dtype=torch.bool)
        for t in targets
    }

    edges = torch.tensor(g["triples"], dtype=torch.long)
    both_e = torch.cat([edges, edges[:, [2, 1, 0]]], dim=0)
    loops = torch.stack(
        [torch.arange(n), torch.full((n,), g["n_rel"], dtype=torch.long), torch.arange(n)], dim=1
    )
    all_e = torch.cat([both_e, loops], dim=0)
    src, rel, dst = all_e[:, 0], all_e[:, 1], all_e[:, 2]
    deg = torch.bincount(src, minlength=n).float().clamp(min=1.0)

    model = RGCNClf(n, g["n_rel"], dim, n_classes, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    study_idx = torch.tensor([g["index"][f"STUDY:{s}"] for s in fit["nct_id"]], dtype=torch.long)
    y = {}
    for t in targets:
        vals = []
        for v in fit[t].astype(str):
            vals.append(maps[t][str(v)] if str(v) not in UNK else 0)
        y[t] = torch.tensor(vals, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        h = model.encode(src, rel, dst, deg)
        logits = {t: model.heads[t](h[study_idx]) for t in targets}
        losses = []
        for t in targets:
            m = known_mask[t]
            if int(m.sum()) == 0:
                continue
            losses.append(F.cross_entropy(logits[t][m], y[t][m]))
        loss = sum(losses) / max(len(losses), 1)
        perm = edges[torch.randperm(len(edges))][:768]
        pos = (h[perm[:, 0]] - h[perm[:, 2]]).pow(2).sum(1)
        neg = torch.randint(0, n, (len(perm),))
        loss = loss + 0.15 * F.relu(pos - (h[perm[:, 0]] - h[neg]).pow(2).sum(1) + 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = model.encode(src, rel, dst, deg).cpu().numpy()
        qids = list(query["nct_id"])
        qv = torch.tensor(_inductive(h, g, both, qids), dtype=torch.float32)
        preds = {q: {} for q in qids}
        for t in targets:
            for q, i in zip(qids, model.heads[t](qv).argmax(1).tolist()):
                preds[q][t] = inv[t][int(i)]
    return preds


def hybrid_onto_rgcn_vote(train, test, rgcn_pred, k=20, alpha=0.5, targets=COARSE):
    """Blend ontology neighbor vote with RGCN hard prediction via confidence proxy: equal blend by picking onto unless conflict then prefer RGCN if alpha>=0.5."""
    onto = ontology_knn(train, test, k=k, targets=targets)
    out = {}
    for q in test["nct_id"]:
        out[q] = {}
        for t in targets:
            # simple: if alpha>=0.5 use rgcn else onto; also try agreement
            o, r = onto[q][t], rgcn_pred[q][t]
            if o == r:
                out[q][t] = o
            else:
                out[q][t] = r if alpha >= 0.5 else o
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
    print("coarse label rates (train):", {t: train[t].value_counts(normalize=True).to_dict() for t in COARSE}, flush=True)

    both_val = pd.concat([fit, val], ignore_index=True)
    val_preds = {
        "frequency_by_phase": frequency_predict(fit, val),
        "logistic": classical_predict(fit, val, "lr"),
        "hist_gbdt": classical_predict(fit, val, "gbdt"),
        "ontology_knn_k20": ontology_knn(fit, val, 20),
    }
    print("train rich-graph RGCN...", flush=True)
    val_preds["rgcn_rich"] = train_rgcn(fit, val, both_val, dim=128, epochs=450, lr=1e-2)
    for alpha in (0.5, 0.7):
        val_preds[f"hybrid_onto_rgcn_a{alpha}"] = hybrid_onto_rgcn_vote(
            fit, val, val_preds["rgcn_rich"], alpha=alpha
        )

    val_scores = {}
    for name, pred in val_preds.items():
        rep = score_predictions(val, pred, COARSE)
        val_scores[name] = {
            "coarse4": _mean(rep, COARSE),
            "hard2": _mean(rep, HARD2),
            "easy3": _mean(rep, EASY3),
        }
        print(
            f"  VAL {name}: coarse4={val_scores[name]['coarse4']['mean_accuracy']:.4f} "
            f"hard2={val_scores[name]['hard2']['mean_accuracy']:.4f} "
            f"easy3={val_scores[name]['easy3']['mean_accuracy']:.4f}",
            flush=True,
        )

    # lock by coarse4 then hard2
    best = max(
        val_scores,
        key=lambda n: (
            val_scores[n]["coarse4"]["mean_accuracy"],
            val_scores[n]["hard2"]["mean_accuracy"],
        ),
    )
    best_gnn = max(
        (n for n in val_scores if "rgcn" in n),
        key=lambda n: (
            val_scores[n]["coarse4"]["mean_accuracy"],
            val_scores[n]["hard2"]["mean_accuracy"],
        ),
    )
    print(f"LOCKED={best}  BEST_GNN={best_gnn}", flush=True)

    # test once
    both_te = pd.concat([train, test], ignore_index=True)
    test_preds = {
        "frequency_by_phase": frequency_predict(train, test),
        "logistic": classical_predict(train, test, "lr"),
        "hist_gbdt": classical_predict(train, test, "gbdt"),
        "ontology_knn_k20": ontology_knn(train, test, 20),
        "rgcn_rich": train_rgcn(train, test, both_te, dim=128, epochs=450, lr=1e-2),
    }
    test_preds["hybrid_onto_rgcn_a0.7"] = hybrid_onto_rgcn_vote(
        train, test, test_preds["rgcn_rich"], alpha=0.7
    )

    test_scores = {}
    for name, pred in test_preds.items():
        rep = score_predictions(test, pred, COARSE)
        test_scores[name] = {
            "coarse4": _mean(rep, COARSE),
            "hard2": _mean(rep, HARD2),
            "easy3": _mean(rep, EASY3),
            "per_target": {t: rep["targets"][t] for t in COARSE},
        }

    unk_counts = {
        split_name: {
            t: int(_is_unk(df[t]).sum())
            for t in COARSE
        }
        for split_name, df in (("fit", fit), ("val", val), ("test", test), ("train", train))
    }
    payload = {
        "note": (
            "Post-hoc coarsened labels + enriched OSEAN graph. "
            "Unknown gold preserved and excluded from train loss / eval. "
            "Not a confirmatory untouched test; 2022+ used in earlier exploration."
        ),
        "targets_coarse4": list(COARSE),
        "unknown_label_counts": unk_counts,
        "locked_on_val": best,
        "best_gnn_on_val": best_gnn,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "gnn_is_best_on_val_coarse4": best == best_gnn
        or val_scores[best_gnn]["coarse4"]["mean_accuracy"]
        >= val_scores[best]["coarse4"]["mean_accuracy"] - 1e-12,
    }
    (OUT / "push08.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "PUSH08 REDESIGN",
        f"locked={best} best_gnn={best_gnn}",
        "",
        "## VAL coarse4",
    ]
    for n, s in sorted(val_scores.items(), key=lambda kv: -kv[1]["coarse4"]["mean_accuracy"]):
        lines.append(
            f"  {s['coarse4']['mean_accuracy']:.4f}  hard2={s['hard2']['mean_accuracy']:.4f}  {n}"
        )
    lines += ["", "## TEST coarse4"]
    for n, s in sorted(test_scores.items(), key=lambda kv: -kv[1]["coarse4"]["mean_accuracy"]):
        lines.append(
            f"  {s['coarse4']['mean_accuracy']:.4f}  hard2={s['hard2']['mean_accuracy']:.4f}  {n}"
        )
        for t in COARSE:
            lines.append(f"      {t}: {s['per_target'][t]['accuracy']:.3f}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
