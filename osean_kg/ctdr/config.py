"""CTDR experiment configuration (single source of truth)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OSEAN_KG = Path(__file__).resolve().parents[1]
V14 = ROOT / "SEA-CDM AACT vaccine remapped" / "v1.4"
CKPT = OSEAN_KG / "ckpts" / "sea_ml_v14"
CTDR_DIR = CKPT / "ctdr"

PROFILE = V14 / "study_context_profile.csv"
DISEASE_VO_MAP = CKPT / "disease_vo_map.csv"
# Primary-investigational mapping overlay (Material + platform fixes from mapping audit)
VO_CORRECTIONS = (
    CKPT / "ctdr" / "optimize" / "mapping_audit" / "vo_corrections_primary.csv"
)

# --- cohort / split ---
SPLIT_DATE = "2022-01-01"  # train: start_date < SPLIT_DATE; test: >=
REQUIRE_INTERVENTIONAL = True
REQUIRE_PHASE = True
REQUIRE_ALLOCATION = True  # target must be labeled
REQUIRE_ARCH_VO = True  # ontology / KG arms need Material

# --- targets (hidden at test; never used as graph edges) ---
TARGETS = (
    "allocation",
    "masking",
    "intervention_model",
    "enrollment_bucket",
    "primary_outcome_family",
)

# --- inputs (allowed in features / graph) ---
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

# --- embedding (defaults; optimize.py may override) ---
DIM = 64
EPOCHS = 400
TRANSE_LR = 1e-2
COMPGCN_LR = 5e-2
BATCH = 512
SEED = 7
MARGIN = 1.0
N_NEG = 5  # TransE negatives per positive

# --- kNN recommendation ---
K_LIST = (5, 10, 15, 20)
DEFAULT_K = 10
HYBRID_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

# --- regression ---
REG_MAX_ITER = 500
GBDT_MAX_DEPTH = 4
GBDT_LR = 0.08
GBDT_ITER = 150

ENROLL_EDGES = (50, 150, 500)  # -> 4 buckets
