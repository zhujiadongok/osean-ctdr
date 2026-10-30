"""Export CTDR input graph triples for human viewing.

  python -u -m ctdr.export_input_graph
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG))

from ctdr.build_table import build_ctdr_table  # noqa: E402
from ctdr.graph import REL, build_input_graph  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "ckpts" / "sea_ml_v14" / "ctdr" / "optimize" / "graph_export"
EXAMPLE_NCT = "NCT04649151"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_ctdr_table()
    g = build_input_graph(df)
    nodes = g["nodes"]
    id2rel = {v: k for k, v in REL.items()}

    trip_path = OUT / "ctdr_input_graph_triples.tsv"
    with trip_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["head", "relation", "tail"])
        for h, r, t in g["triples"]:
            w.writerow([nodes[h], id2rel[r], nodes[t]])

    sid = g["index"][f"STUDY:{EXAMPLE_NCT}"]
    ego = [
        (nodes[h], id2rel[r], nodes[t])
        for h, r, t in g["triples"]
        if h == sid or t == sid
    ]
    ego_path = OUT / f"ego_{EXAMPLE_NCT}_triples.tsv"
    with ego_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["head", "relation", "tail"])
        for row in ego:
            w.writerow(row)

    outbound = [row for row in ego if row[0].startswith("STUDY:")]
    lines = ["flowchart LR", f'  S["{EXAMPLE_NCT}"]']
    for i, (_h, rel, tail) in enumerate(outbound[:20]):
        tid = f"T{i}"
        safe = tail.replace('"', "")
        lines.append(f'  {tid}["{safe}"]')
        lines.append(f"  S -->|{rel}| {tid}")
    (OUT / f"ego_{EXAMPLE_NCT}_snippet.mmd").write_text("\n".join(lines), encoding="utf-8")

    # Material VO edges for the paper example class
    mat = "VO_0005198"
    vo_edges = [
        (nodes[h], id2rel[r], nodes[t])
        for h, r, t in g["triples"]
        if nodes[h] == mat or nodes[t] == mat
    ]
    with (OUT / f"vo_{mat}_edges.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["head", "relation", "tail"])
        for row in vo_edges:
            w.writerow(row)

    print(f"nodes={len(nodes)} triples={len(g['triples'])}")
    print(f"full graph TSV: {trip_path}")
    print(f"ego subgraph ({len(ego)} edges): {ego_path}")
    print(f"mermaid snippet: {OUT / f'ego_{EXAMPLE_NCT}_snippet.mmd'}")
    print("sample study outbound edges:")
    for row in outbound[:12]:
        print(" ", row)
    print(f"VO edges for {mat}: {len(vo_edges)}")
    for row in vo_edges[:10]:
        print(" ", row)


if __name__ == "__main__":
    main()
