"""
Central Configuration — Sensor-Graph + Domain Adaptation DAGHAR Project
========================================================================
Single source of truth for all paths, hyperparameters, and SO settings.

This is Project 5 (sensor-graph & DA SO-DAGHAR), building on the learnings
from Projects 1-4. Key differences from the freq/hybrid project:
  - Graph construction: 6 sensor nodes instead of 60 timestep nodes
  - Models: SensorGraphModel (CNN encoder + GNN), CNN-only baseline
  - Domain adaptation: DANN with gradient reversal (Phase 2)
  - SO: sensor-level ablation instead of arbitrary node coarsening (Phase 3)
"""

import os
import torch


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Project root = the folder that contains this `utils/` package (i.e. the
# uploaded Project_SensorGraph_P11 folder, wherever it lives on Drive).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA_ROOT is an EXTERNAL dataset path — edit it to wherever your DAGHAR
# standardized_view folders live (it does not move with this project).
DATA_ROOT = os.environ.get("DATA_PATH",
                           "/content/drive/MyDrive/HAR-Datasets/DAGHAR/standardized_view")

# RESULTS_ROOT is derived from the project location, so results ALWAYS land in
# <this project>/results no matter where you upload the folder. (Previously this
# was a hardcoded absolute path, which drifted from PROJECT_PATH and wrote results
# to the wrong place.)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# Local development override (uncomment when running outside Colab):
# DATA_ROOT = "./data/standardized_view"


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER NAME MAPPING
# ─────────────────────────────────────────────────────────────────────────────

DATASET_FOLDER_NAMES = {
    "KuHar":           "KuHar",
    "MotionSense":     "MotionSense",
    "RealWorld-Thigh": "RealWorld-Thigh",   # match actual Drive folder (capital W)
    "RealWorld-Waist": "RealWorld-Waist",   # match actual Drive folder (capital W)
    "UCI":             "UCI",
    "WISDM":           "WISDM",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sensor-graph topology definitions
# ─────────────────────────────────────────────────────────────────────────────

# Node indices: 0=accel-x, 1=accel-y, 2=accel-z, 3=gyro-x, 4=gyro-y, 5=gyro-z

# Fully connected (15 undirected = 30 directed edges)
EDGES_FULL = [
    (0,1),(0,2),(0,3),(0,4),(0,5),
    (1,2),(1,3),(1,4),(1,5),
    (2,3),(2,4),(2,5),
    (3,4),(3,5),
    (4,5),
]

# Physically structured: intra-modality + axis-aligned cross-modal (9 undirected = 18 directed)
EDGES_PHYSICAL = [
    # Intra-accelerometer
    (0,1),(0,2),(1,2),
    # Intra-gyroscope
    (3,4),(3,5),(4,5),
    # Cross-modal (axis-aligned)
    (0,3),(1,4),(2,5),
]

# Minimal: only axis-aligned cross-modal (3 undirected = 6 directed)
EDGES_MINIMAL = [
    (0,3),(1,4),(2,5),
]

EDGE_TOPOLOGIES = {
    "full": EDGES_FULL,
    "physical": EDGES_PHYSICAL,
    "minimal": EDGES_MINIMAL,
}


def get_config():
    """Returns the full configuration dict."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return {
        # ───────────────────────── Paths ─────────────────────────────────────
        "data_root":            DATA_ROOT,
        "results_root":         RESULTS_ROOT,
        "dataset_folder_names": dict(DATASET_FOLDER_NAMES),

        # ───────────────────────── Data constants ────────────────────────────
        "daghar_timesteps": 60,
        "daghar_channels":  6,
        "num_classes":      6,
        "num_sensor_nodes": 6,

        # ───────────────────────── Sensor-graph model ────────────────────────
        # CNN encoder settings
        "cnn_channels":     [32, 64],       # Conv1d channel progression
        "cnn_kernel_size":  5,              # kernel size for Conv1d
        "cnn_embed_dim":    64,             # output embedding per node
        "cnn_weight_sharing": "per_modality",  # "shared", "per_modality", "independent"
        "cnn_temporal_pool": "avg",         # P11: "avg" (mean, =P1–P10), "bigru", "attention"

        # P12 — dual-view encoder + normalization upgrade
        "encoder_mode":     "single",       # "single" (P9/P11) | "dualview" (P12: split time/freq + gate)
        "dualview_time_len": 60,            # split point for hybrid: time[60] | freq[remainder]
        "normalize_mode":   "per_channel",  # "per_channel" (P9/P11) | "per_bin_channel" (P12)

        # P13 — source-domain augmentation (TRAIN ONLY; tune strength on source-val)
        "augment":          False,          # master switch (False = P9/P11/P12 behaviour)
        "aug_copies":       2,              # # augmented passes appended to the originals
        "aug_jitter":       0.05,           # additive Gaussian noise sigma (signal is z-normed)
        "aug_scaling":      0.10,           # per-channel gain sigma
        "aug_mag_warp":     0.20,           # magnitude-warp curve sigma
        "aug_time_warp":    0.20,           # time-warp curve sigma
        "aug_time_shift":   4,              # max circular time shift (of 60 samples)
        "aug_channel_mask": 0.0,            # prob of zeroing a channel (off by default)
        "aug_knots":        4,              # knots for the warp curves

        # Graph topology
        "edge_topology":    "full",         # "full", "physical", "minimal"
        "use_edge_type":    True,           # encode edge type as attribute

        # GNN settings
        "gnn_type":         "sage",         # "sage", "gatv2", "gin"
        "gnn_layers":       2,
        "gnn_hidden":       64,
        "gnn_dropout":      0.2,

        # ───────────────────────── Training ──────────────────────────────────
        "batch_size":       64,             # larger batch OK — graphs are tiny (6 nodes)
        "epochs":           100,
        "learning_rate":    0.001,
        "weight_decay":     5e-4,
        "log_interval":     5,
        "patience":         20,

        # ───────────────────────── DANN (Phase 2) ────────────────────────────
        "dann_enabled":     False,          # set True for Phase 2
        "dann_lambda_max":  1.0,            # max adversarial weight
        "dann_domain_hidden": 64,           # domain discriminator hidden size
        "dann_schedule":    "ganin",        # "linear" or "ganin"

        # ───────────────────────── P17 — transductive LODO-DA ────────────────
        # When True, the loader also returns the TARGET train split as UNLABELED
        # adaptation data (target_unlabeled_graphs). The P17 source-only baseline
        # leaves this False/unused — it is the clean DG-equivalent lower bound that
        # P18+ UDA methods (DANN target-align / CORAL / MMD) are measured against.
        "load_target_unlabeled": False,
        "normalize_include_target": False,  # P18a: fit z-norm on source-train ∪ target-unlabeled
        "log_every":        5,              # training-log cadence (epochs)

        # ───────────────────────── SO (Phase 3) ──────────────────────────────
        # These will be filled in during Phase 3
        "so_type":          "baseline",

        # ───────────────────────── Device & reproducibility ──────────────────
        "device": device,
        "seed":   42,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_results_dirs(results_root=None):
    """Create results subdirectory tree."""
    root = results_root or RESULTS_ROOT
    subdirs = [
        "lodo/folds", "lodo/summary", "lodo/checkpoints",
        "lodo/training_curves", "tables", "figures", "stats",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def print_config_summary(config=None):
    """Pretty-print active config."""
    cfg = config or get_config()
    print("=" * 60)
    print("  Active configuration")
    print("=" * 60)
    print(f"  data_root      : {cfg['data_root']}")
    print(f"  results_root   : {cfg['results_root']}")
    print(f"  device         : {cfg['device']}")
    print(f"  seed           : {cfg['seed']}")
    print(f"  batch_size     : {cfg['batch_size']}")
    print(f"  epochs         : {cfg['epochs']}")
    print(f"  patience       : {cfg['patience']}")
    print(f"  CNN channels   : {cfg['cnn_channels']} → embed {cfg['cnn_embed_dim']}")
    print(f"  CNN weights    : {cfg['cnn_weight_sharing']}")
    print(f"  Edge topology  : {cfg['edge_topology']}")
    print(f"  GNN            : {cfg['gnn_type']} × {cfg['gnn_layers']} layers")
    print(f"  DANN enabled   : {cfg['dann_enabled']}")
    print("=" * 60)
