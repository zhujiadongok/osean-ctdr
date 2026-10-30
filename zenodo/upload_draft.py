"""Update reserved Zenodo deposition as Save draft (do NOT publish).

Requires env ZENODO_TOKEN with deposit:write scope.
Deposition: https://doi.org/10.5281/zenodo.23773934
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

DEPOSITION_ID = 23773934
ZIP = Path(__file__).resolve().parent / "OSEAN-CTDR_supporting_data.zip"
META_PATH = Path(__file__).resolve().parent / "package" / "ZENODO_METADATA.txt"
BASE = "https://zenodo.org/api"


def metadata() -> dict:
    return {
        "title": (
            "OSEAN-CTDR supporting data: Vaccine Ontology, CTDR feature "
            "sidecars, and frozen evaluation artifacts"
        ),
        "upload_type": "dataset",
        "description": (
            "<p>Supporting data for OSEAN-CTDR (SEA-CDM–based vaccine trial "
            "evidence integration and ontology-guided design-field prediction). "
            "Includes the Vaccine Ontology OWL file used in analyses "
            "(release 2026-06-12), CTDR feature sidecars, frozen result JSON "
            "artifacts (Table 1 / Setting B / post-remediation LOMO), mapping "
            "audit overlays, and class-label helpers. Companion code: "
            "<a href=\"https://github.com/zhujiadongok/osean-ctdr\">"
            "https://github.com/zhujiadongok/osean-ctdr</a>. Full AACT dump is "
            "not redistributed; local snapshot identifiers and checksums are "
            "documented in README.md and the manuscript Table S8.</p>"
        ),
        "creators": [
            {"name": "Zhu, Jiadong", "affiliation": "University of Michigan"},
            {"name": "Yeh, Feng-Yu", "affiliation": "University of Michigan"},
            {"name": "Zheng, Jie", "affiliation": "University of Michigan"},
            {"name": "He, Yongqun", "affiliation": "University of Michigan"},
        ],
        "keywords": [
            "clinical trials",
            "SEA-CDM",
            "Vaccine Ontology",
            "ClinicalTrials.gov",
            "knowledge graph",
            "OSEAN-CTDR",
        ],
        "access_right": "open",
        "license": "cc-by-4.0",
        "related_identifiers": [
            {
                "identifier": "https://github.com/zhujiadongok/osean-ctdr",
                "relation": "isSupplementTo",
                "resource_type": "software",
            },
            {
                "identifier": "http://purl.obolibrary.org/obo/vo/releases/2026-06-12/vo.owl",
                "relation": "isDerivedFrom",
                "resource_type": "dataset",
            },
        ],
    }


def main() -> None:
    token = os.environ.get("ZENODO_TOKEN") or os.environ.get("ZENODO_ACCESS_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN (deposit:write). Not publishing.", file=sys.stderr)
        sys.exit(2)
    if not ZIP.is_file():
        print(f"ERROR: missing {ZIP}", file=sys.stderr)
        sys.exit(2)

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE}/deposit/depositions/{DEPOSITION_ID}", headers=headers, timeout=60)
    r.raise_for_status()
    dep = r.json()
    print("state:", dep.get("state"), "submitted:", dep.get("submitted"), "doi:", dep.get("doi"))
    if dep.get("submitted"):
        print("ERROR: deposition already published; refusing to mutate without newversion.", file=sys.stderr)
        sys.exit(3)

    # Replace files: delete existing then upload zip via bucket
    for f in dep.get("files") or []:
        fid = f.get("id")
        if not fid:
            continue
        dr = requests.delete(
            f"{BASE}/deposit/depositions/{DEPOSITION_ID}/files/{fid}",
            headers=headers,
            timeout=60,
        )
        print("delete", f.get("filename"), dr.status_code)

    bucket = dep["links"]["bucket"]
    with ZIP.open("rb") as fh:
        put = requests.put(
            f"{bucket}/{ZIP.name}",
            data=fh,
            headers={**headers, "Content-Type": "application/octet-stream"},
            timeout=600,
        )
    print("upload", put.status_code, put.text[:300])
    put.raise_for_status()

    # Save draft metadata (PUT deposition) — does not publish
    pr = requests.put(
        f"{BASE}/deposit/depositions/{DEPOSITION_ID}",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps({"metadata": metadata()}),
        timeout=60,
    )
    print("metadata", pr.status_code)
    if pr.status_code >= 400:
        print(pr.text[:1000], file=sys.stderr)
        pr.raise_for_status()

    out = pr.json()
    print(
        json.dumps(
            {
                "id": out.get("id"),
                "state": out.get("state"),
                "submitted": out.get("submitted"),
                "doi": out.get("doi") or out.get("metadata", {}).get("prereserve_doi"),
                "html": out.get("links", {}).get("html"),
                "note": "Saved as draft only; not published.",
            },
            indent=2,
        )
    )
    META_PATH.write_text(
        META_PATH.read_text(encoding="utf-8")
        if META_PATH.exists()
        else "",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
