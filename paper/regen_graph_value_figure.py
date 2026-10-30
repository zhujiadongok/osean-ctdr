"""Figure 6: post-remediation pooled ranks + LOMO primary OntSim−exact Δ."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vo_eligible_labels import VO_ELIGIBLE_LABELS_SHORT

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "paper_figures"
EVAL = (
    ROOT.parent
    / "osean_kg"
    / "ckpts"
    / "sea_ml_v14"
    / "ctdr"
    / "optimize"
    / "mapping_audit"
    / "after_vo_fix_eval.json"
)

METHOD_LABELS = {
    "hist_gbdt": "HistGBDT",
    "frequency_by_phase": "Phase-frequency",
    "exact": "Exact-match kNN",
    "ontsim_role": "OntSim+role (primary)",
    "gated_ic_sib": "IC-sib (exploratory)",
}


def main() -> None:
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    pooled = data["pooled"]
    per = data["lomo_per_class"]
    order = data["eligible_classes"]
    by_vo = {r["vo"]: r for r in per}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [1.05, 1.35]})

    ax = axes[0]
    ranked = sorted(
        (
            (k, METHOD_LABELS[k], pooled[k]["trial_mean_accuracy"])
            for k in METHOD_LABELS
            if k in pooled
        ),
        key=lambda x: -x[2],
    )
    names = [lab for _, lab, _ in ranked]
    vals = [v for _, _, v in ranked]
    y = np.arange(len(names))
    colors = ["#1D4ED8" if "primary" in n else "#93C5FD" for n in names]
    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Trial-mean accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"A. Pooled holdout ({len(order)} classes)")
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.01, 0.92), yi, f"{v:.3f}", va="center", fontsize=8)

    ax = axes[1]
    deltas = [by_vo[m]["ontsim_role"] - by_vo[m]["exact"] for m in order]
    labs = [VO_ELIGIBLE_LABELS_SHORT.get(m, m) for m in order]
    y = np.arange(len(labs))
    colors = ["#15803D" if d >= 0 else "#B91C1C" for d in deltas]
    ax.barh(y, deltas, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ trial-mean (OntSim+role − exact)")
    ax.set_title("B. LOMO per-class (primary OntSim)")
    xmax = max(abs(min(deltas)), abs(max(deltas)), 0.02) * 1.45
    ax.set_xlim(-xmax, xmax)
    for yi, d in zip(y, deltas):
        ax.text(
            d + (0.002 if d >= 0 else -0.002),
            yi,
            f"{d:+.3f}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=6.5,
            color="#0F172A",
        )

    fig.suptitle(
        "Figure 6. Ontology-connected unseen Materials after primary-investigational mapping remediation",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out = FIG / "fig6_graph_value_unseen.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
