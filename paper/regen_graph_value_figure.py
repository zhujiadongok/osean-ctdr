"""Figure: ontology-connected unseen Material — pooled methods + LOMO per-class Δ."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vo_eligible_labels import VO_ELIGIBLE_LABELS_SHORT

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figures"
GV = (
    ROOT.parent
    / "osean_kg"
    / "ckpts"
    / "sea_ml_v14"
    / "ctdr"
    / "optimize"
    / "graph_value"
    / "graph_value.json"
)

METHOD_LABELS = {
    "ontsim_subclass_role_k20": "OntSim+role",
    "frequency_by_phase": "Phase-frequency",
    "hist_gbdt_no_vo": "HistGBDT no VO",
    "ontsim_subclass_k20": "OntSim subclass",
    "hist_gbdt_full": "HistGBDT",
    "hist_gbdt_no_material": "HistGBDT no Material",
    "exact_knn_k20": "Exact-match kNN",
}


def main() -> None:
    data = json.loads(GV.read_text(encoding="utf-8"))
    um = data["unseen_material"]
    pooled = um["pooled_holdout"]["scores"]
    per = um.get("lomo", {}).get("per_class") or um["lomo_per_class"]
    order = um["protocol"]["eligible_materials"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel A: pooled methods, ordered by trial-mean (Table 2 order)
    ax = axes[0]
    ranked = sorted(
        ((k, METHOD_LABELS[k], pooled[k]["trial_mean_accuracy"]) for k in METHOD_LABELS if k in pooled),
        key=lambda x: -x[2],
    )
    names = [lab for _, lab, _ in ranked]
    vals = [v for _, _, v in ranked]
    y = np.arange(len(names))
    colors = ["#1D4ED8" if n == "OntSim+role" else "#93C5FD" for n in names]
    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Trial-mean accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_title("A. Pooled holdout (9 classes unseen)")
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.01, 0.92), yi, f"{v:.3f}", va="center", fontsize=8)

    # Panel B: LOMO per-class OntSim+role − exact (VO class labels, not product names)
    ax = axes[1]
    deltas = [per[m]["delta_ontsim_role_minus_exact_trial"] for m in order]
    labs = [VO_ELIGIBLE_LABELS_SHORT.get(m, m) for m in order]
    y = np.arange(len(labs))
    colors = ["#15803D" if d >= 0 else "#B91C1C" for d in deltas]
    ax.barh(y, deltas, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ trial-mean (OntSim+role − exact)")
    ax.set_title("B. LOMO per-class transfer")
    xmax = max(abs(min(deltas)), abs(max(deltas)), 0.02) * 1.45
    ax.set_xlim(-xmax, xmax)
    for yi, d in zip(y, deltas):
        ax.text(
            d + (0.003 if d >= 0 else -0.003),
            yi,
            f"{d:+.3f}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=7,
            color="#0F172A",
        )

    fig.suptitle(
        "Figure 6. Ontology relations under ontology-connected unseen Materials",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = FIG / "fig6_graph_value_unseen.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
