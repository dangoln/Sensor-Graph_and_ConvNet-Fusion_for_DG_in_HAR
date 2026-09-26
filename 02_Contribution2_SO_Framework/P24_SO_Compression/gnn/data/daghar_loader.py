"""
DAGHAR Sensor-Graph Data Loader
================================
Loads DAGHAR standardized_view CSVs and builds sensor-channel graphs.

Each 3-second HAR window [60 timesteps × 6 channels] becomes a graph with:
  - 6 nodes (one per sensor channel: accel-x/y/z, gyro-x/y/z)
  - Each node's features = the 60-timestep raw signal for that channel
  - Fixed edges based on physical sensor relationships
  - Optional edge type attributes (intra-accel, intra-gyro, cross-modal)

This replaces the timestep-as-node approach from Projects 2-4.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# DAGHAR constants
# ─────────────────────────────────────────────────────────────────────────────

DAGHAR_DATASETS = [
    "KuHar", "MotionSense", "RealWorld-Thigh",
    "RealWorld-Waist", "UCI", "WISDM",
]

DAGHAR_NUM_CLASSES = 6
DAGHAR_ACTIVITY_NAMES = {
    0: "sit", 1: "stand", 2: "walk",
    3: "stair up", 4: "stair down", 5: "run",
}

DAGHAR_DATASET_CLASSES = {
    "KuHar":           {0, 1, 2, 3, 4, 5},
    "MotionSense":     {0, 1, 2, 3, 4, 5},
    "RealWorld-Thigh": {0, 1, 2, 3, 4, 5},
    "RealWorld-Waist": {0, 1, 2, 3, 4, 5},
    "UCI":             {0, 1, 2, 3, 4},
    "WISDM":           {0, 1, 2, 5},
}

DAGHAR_TIMESTEPS = 60
DAGHAR_CHANNELS  = 6
DAGHAR_AXES      = ["accel-x", "accel-y", "accel-z",
                    "gyro-x",  "gyro-y",  "gyro-z"]
DAGHAR_LABEL_COLUMN = "standard activity code"


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading (same as before)
# ─────────────────────────────────────────────────────────────────────────────

def _build_signal_columns(timesteps=DAGHAR_TIMESTEPS, axes=DAGHAR_AXES):
    cols = []
    for axis in axes:
        for t in range(timesteps):
            cols.append(f"{axis}-{t}")
    return cols


def _resolve_folder(data_root, folder):
    """
    Resolve the actual on-disk subfolder of ``data_root`` matching ``folder``,
    tolerant of case differences (e.g. ``RealWorld-Thigh`` vs ``Realworld-thigh``)
    and of a leading/trailing-case mismatch on any dataset name.

    Returns the real folder name if a (case-insensitive) match exists, otherwise
    returns ``folder`` unchanged so the caller raises an informative error.
    """
    if os.path.isdir(os.path.join(data_root, folder)):
        return folder
    if os.path.isdir(data_root):
        target = folder.lower()
        for name in os.listdir(data_root):
            if name.lower() == target and os.path.isdir(os.path.join(data_root, name)):
                return name
    return folder


def load_daghar_dataset(data_root, dataset_name, split="train", folder_map=None):
    """
    Load a single DAGHAR dataset split.

    Returns:
        X:     [N, T, C] float32 tensor (T=60, C=6)
        y:     [N] int64 tensor
        users: numpy array of user IDs
    """
    if dataset_name not in DAGHAR_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split '{split}'")

    folder = folder_map[dataset_name] if folder_map else dataset_name
    # Tolerate folder-name case differences on disk (e.g. RealWorld-Thigh)
    folder = _resolve_folder(data_root, folder)
    csv_path = os.path.join(data_root, folder, f"{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"DAGHAR CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    signal_cols = _build_signal_columns()

    N = len(df)
    flat = df[signal_cols].to_numpy(dtype=np.float32)
    X_ct = flat.reshape(N, DAGHAR_CHANNELS, DAGHAR_TIMESTEPS)  # [N, C, T]
    X    = np.transpose(X_ct, (0, 2, 1))                        # [N, T, C]

    y = df[DAGHAR_LABEL_COLUMN].to_numpy(dtype=np.int64)
    users = df["user"].to_numpy()
    if np.issubdtype(users.dtype, np.floating):
        users = users.astype(np.int64)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        users,
    )


def load_daghar_aggregated(data_root, dataset_names, split="train",
                           folder_map=None):
    """
    Load and concatenate multiple DAGHAR datasets.

    Returns:
        X, y, users, sources (dataset-of-origin string per sample)
    """
    X_list, y_list, user_list, src_list = [], [], [], []
    for name in dataset_names:
        X_d, y_d, u_d = load_daghar_dataset(
            data_root, name, split=split, folder_map=folder_map
        )
        X_list.append(X_d)
        y_list.append(y_d)
        user_list.append(u_d)
        src_list.append(np.array([name] * len(y_d), dtype=object))

    return (
        torch.cat(X_list, dim=0),
        torch.cat(y_list, dim=0),
        np.concatenate(user_list),
        np.concatenate(src_list),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sensor-graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _get_edge_type(src, dst):
    """
    Classify an edge between two sensor nodes.
    Nodes 0-2 = accel, nodes 3-5 = gyro.

    Returns:
        0 = intra-accelerometer
        1 = intra-gyroscope
        2 = cross-modal
    """
    src_is_accel = src < 3
    dst_is_accel = dst < 3
    if src_is_accel and dst_is_accel:
        return 0
    elif not src_is_accel and not dst_is_accel:
        return 1
    else:
        return 2


def build_edge_index(edge_list, use_edge_type=True):
    """
    Build bidirectional edge_index and optional edge_type from an edge list.

    Args:
        edge_list:     list of (src, dst) tuples (undirected edges)
        use_edge_type: if True, also returns edge_type tensor

    Returns:
        edge_index: [2, num_edges] LongTensor (bidirectional)
        edge_type:  [num_edges] LongTensor or None
    """
    src_list, dst_list, type_list = [], [], []

    for (s, d) in edge_list:
        # Forward
        src_list.append(s)
        dst_list.append(d)
        type_list.append(_get_edge_type(s, d))
        # Backward (undirected)
        src_list.append(d)
        dst_list.append(s)
        type_list.append(_get_edge_type(d, s))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long) if use_edge_type else None

    return edge_index, edge_type


def _compute_node_features(X, feature_mode="time"):
    """
    Transform raw sensor data into node features based on feature mode.

    Args:
        X: [N, T, C] numpy array or tensor (T=60, C=6)
        feature_mode: "time" | "freq" | "hybrid"

    Returns:
        X_feat: [N, F, C] numpy array where F depends on mode
        feat_dim: int, the feature dimension F
    """
    X_np = X.numpy() if torch.is_tensor(X) else X
    N, T, C = X_np.shape

    if feature_mode == "time":
        return X_np, T  # [N, 60, 6]

    elif feature_mode == "freq":
        # FFT magnitude spectrum per channel, drop DC component
        # rfft gives T//2 + 1 = 31 values, drop index 0 (DC) → 30 values
        X_freq = np.zeros((N, T // 2, C), dtype=np.float32)
        for c in range(C):
            fft_mag = np.abs(np.fft.rfft(X_np[:, :, c], axis=1))  # [N, 31]
            X_freq[:, :, c] = fft_mag[:, 1:]  # drop DC, keep 30 bins
        return X_freq, T // 2  # [N, 30, 6]

    elif feature_mode == "hybrid":
        # Concatenate raw time + FFT magnitude
        X_freq = np.zeros((N, T // 2, C), dtype=np.float32)
        for c in range(C):
            fft_mag = np.abs(np.fft.rfft(X_np[:, :, c], axis=1))
            X_freq[:, :, c] = fft_mag[:, 1:]
        X_hybrid = np.concatenate([X_np, X_freq], axis=1)  # [N, 90, 6]
        return X_hybrid, T + T // 2  # [N, 90, 6]

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_sensor_graphs(X, y, edge_list, use_edge_type=True,
                        sources=None, domain_labels=None,
                        normalize=True, stats=None, verbose=True,
                        feature_mode="time", normalize_mode="per_channel",
                        edge_sets=None):
    """
    Convert raw sensor data into sensor-channel graphs.

    Args:
        X:              [N, T, C] tensor (T=60 timesteps, C=6 channels)
        y:              [N] label tensor
        edge_list:      list of (src, dst) undirected edge tuples
        edge_sets:      P21 — optional list (len N) of PER-WINDOW undirected edge
                        lists (from data.pearson). When given, each graph gets its
                        own data-driven edge_index and `edge_list` is ignored.
        use_edge_type:  include edge type attributes
        sources:        [N] array of source dataset names (for DANN)
        domain_labels:  [N] array of integer domain labels (for DANN)
        normalize:      whether to z-normalize each feature dimension
        stats:          (mean, std) from training set; if None, compute from X
        verbose:        print progress
        feature_mode:   "time" | "freq" | "hybrid"
        normalize_mode: "per_channel"      → one mean/std per channel (P9/P11), or
                        "per_bin_channel"  → one mean/std per (feature-bin, channel)
                                             (P12). Both are fit on source-train only
                                             (stats are passed through to val/test).

    Returns:
        graphs:  list of PyG Data objects
        stats:   (feat_means, feat_stds) for reuse on val/test
    """
    N, T, C = X.shape
    # P24 NC: C may be < 6 when channels were coarsened away upstream.
    assert 1 <= C <= 6, f"Expected 1-6 channels, got {C}"

    # Step 1: Compute features based on mode
    X_feat, feat_dim = _compute_node_features(X, feature_mode)
    # X_feat: [N, F, C] where F = feat_dim

    if verbose:
        print(f"    feature_mode={feature_mode}, feat_dim={feat_dim}, "
              f"normalize_mode={normalize_mode}")

    # Edge structure: shared (fixed topology) or per-window (P21 Pearson)
    if edge_sets is not None:
        assert len(edge_sets) == N, \
            f"edge_sets length {len(edge_sets)} != N {N}"
        per_window_edges = [build_edge_index(es, use_edge_type) for es in edge_sets]
        edge_index, edge_type = per_window_edges[0]   # for the log line only
    else:
        per_window_edges = None
        # Pre-compute shared edge structure (same for all graphs)
        edge_index, edge_type = build_edge_index(edge_list, use_edge_type)

    # Normalization (z-score). stats are fit on the source-train split and reused
    # for val/test, so there is no target leakage in either mode.
    if normalize:
        if stats is None:
            if normalize_mode == "per_bin_channel":
                # mean/std over samples only → keeps a separate stat per
                # (feature-bin, channel): shape [F, C]
                means = X_feat.mean(axis=0)            # [F, C]
                stds  = X_feat.std(axis=0)             # [F, C]
            else:  # "per_channel" (default, P9/P11)
                means = X_feat.mean(axis=(0, 1))       # [C]
                stds  = X_feat.std(axis=(0, 1))        # [C]
            stds[stds < 1e-8] = 1.0
            stats = (means, stds)
        else:
            means, stds = stats

        if means.ndim == 2:        # per-(bin, channel): [F, C]
            X_norm = (X_feat - means[None, :, :]) / stds[None, :, :]
        else:                       # per-channel: [C]
            X_norm = (X_feat - means[None, None, :]) / stds[None, None, :]
        X_norm = torch.tensor(X_norm, dtype=torch.float32)
    else:
        X_norm = torch.tensor(X_feat, dtype=torch.float32)
        stats = (np.zeros(C), np.ones(C))

    if verbose:
        print(f"    building {N} sensor graphs ({C} nodes, "
              f"{edge_index.shape[1]} edges, {feat_dim} features each)...")

    graphs = []
    for i in range(N):
        # Node features: each node gets its feature vector
        # X_norm[i] is [F, C], we want [C, F] so each row is a node
        node_features = X_norm[i].T  # [C, F] = [6, feat_dim]

        if per_window_edges is not None:                       # P21 Pearson mode
            ei_i, et_i = per_window_edges[i]
        else:
            ei_i, et_i = edge_index, edge_type                 # shared reference

        data = Data(
            x=node_features,           # [6, 60]
            edge_index=ei_i,           # [2, num_edges]
            y=y[i].unsqueeze(0),       # [1]
        )

        if et_i is not None:
            data.edge_type = et_i

        # Store domain info for DANN
        if domain_labels is not None:
            data.domain = torch.tensor([domain_labels[i]], dtype=torch.long)

        graphs.append(data)

        if verbose and (i + 1) % 5000 == 0:
            print(f"    built {i + 1}/{N} graphs")

    if verbose:
        print(f"    done: {N} graphs built")

    return graphs, stats


# ─────────────────────────────────────────────────────────────────────────────
# LODO graph building
# ─────────────────────────────────────────────────────────────────────────────

def _fit_norm_stats(X, feature_mode="time", normalize_mode="per_channel"):
    """P18a — fit z-norm (mean, std) stats from raw windows X [N, T, C].

    Used to compute TRANSDUCTIVE normalization stats over source-train UNION
    target-unlabeled. Mirrors the stat-fitting branch inside build_sensor_graphs so
    the two are guaranteed consistent.
    """
    X_feat, _ = _compute_node_features(X, feature_mode)   # [N, F, C]
    if normalize_mode == "per_bin_channel":
        means = X_feat.mean(axis=0)            # [F, C]
        stds  = X_feat.std(axis=0)             # [F, C]
    else:                                       # "per_channel" (P9 default)
        means = X_feat.mean(axis=(0, 1))       # [C]
        stds  = X_feat.std(axis=(0, 1))        # [C]
    stds[stds < 1e-8] = 1.0
    return (means, stds)


def _make_domain_labels(sources):
    """Convert source dataset names to integer domain labels."""
    unique_sources = sorted(set(sources))
    src_to_idx = {s: i for i, s in enumerate(unique_sources)}
    return np.array([src_to_idx[s] for s in sources], dtype=np.int64), src_to_idx


def build_lodo_sensor_graphs(data_root, target_dataset, config=None,
                              folder_map=None):
    """
    Build sensor-channel graphs for one LODO fold.

    Returns dict with train_graphs, val_graphs, test_graphs, metadata.
    """
    from utils.config import EDGE_TOPOLOGIES

    if target_dataset not in DAGHAR_DATASETS:
        raise ValueError(f"Unknown target '{target_dataset}'")

    cfg = config or {}
    edge_topology = cfg.get("edge_topology", "full")
    use_edge_type = cfg.get("use_edge_type", True)
    feature_mode = cfg.get("feature_mode", "time")
    normalize_mode = cfg.get("normalize_mode", "per_channel")  # P12
    # P17 — transductive LODO-DA plumbing. When True, also load the TARGET's own
    # `train` split as UNLABELED adaptation data (consumed by P18 UDA: DANN/CORAL/
    # MMD). It is normalized with SOURCE-train stats so the P17 source-only baseline
    # stays numerically identical to the DG setting (a clean lower bound). Target
    # labels are loaded for optional diagnostics ONLY and are never used for training
    # or model selection. See PROTOCOL_DA_LODO.md.
    load_target_unlabeled = cfg.get("load_target_unlabeled", False)
    # P18a — TRANSDUCTIVE normalization. When True, the z-norm stats are fit on
    # source-train UNION target-unlabeled (instead of source-train only). This is the
    # ONE change vs the P17 baseline: a principled, train-time version of AdaBN that
    # aligns input statistics to the target domain. Still no target LABELS used.
    normalize_include_target = cfg.get("normalize_include_target", False)
    need_target = load_target_unlabeled or normalize_include_target

    edge_list = EDGE_TOPOLOGIES[edge_topology]
    sources = [d for d in DAGHAR_DATASETS if d != target_dataset]

    print(f"[LODO] target = {target_dataset}")
    print(f"[LODO] sources = {sources}")
    print(f"[LODO] feature_mode = {feature_mode}  normalize_mode = {normalize_mode}")

    # Load data
    X_tr, y_tr, _, src_tr = load_daghar_aggregated(
        data_root, sources, split="train", folder_map=folder_map
    )
    X_val, y_val, _, _ = load_daghar_aggregated(
        data_root, sources, split="validation", folder_map=folder_map
    )
    X_te, y_te, _ = load_daghar_dataset(
        data_root, target_dataset, split="test", folder_map=folder_map
    )

    print(f"[LODO] samples: train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")

    # Domain labels for DANN
    domain_labels_tr, src_to_idx = _make_domain_labels(src_tr)
    num_domains = len(src_to_idx)

    # P13 — source-domain augmentation (TRAIN ONLY; val/test untouched → no leakage)
    if cfg.get("augment", False):
        from data.augment import augment_training_set
        n_before = len(y_tr)
        X_tr, y_tr, src_tr, domain_labels_tr = augment_training_set(
            X_tr, y_tr, src_tr, domain_labels_tr, cfg, seed=cfg.get("seed", 42),
        )
        print(f"[LODO] augment: train {n_before} -> {len(y_tr)} "
              f"(x{cfg.get('aug_copies', 1) + 1})  [source-train only]")

    # P24 — classic SO input reductions, applied identically to ALL splits.
    #   NC (node coarsening):   keep a channel subset (cfg["nc_keep"], original ids)
    #   AW (adaptive windowing): downsample to cfg["aw_timesteps"] via even striding
    nc_keep = cfg.get("nc_keep")
    aw_timesteps = cfg.get("aw_timesteps")
    if nc_keep:
        nc_keep = sorted(int(c) for c in nc_keep)
        X_tr  = X_tr[:, :, nc_keep]
        X_val = X_val[:, :, nc_keep]
        X_te  = X_te[:, :, nc_keep]
        print(f"[LODO] P24 NC: keeping channels {nc_keep} -> {len(nc_keep)} nodes/graph")
    if aw_timesteps:
        T0 = X_tr.shape[1]
        aw_idx = torch.linspace(0, T0 - 1, int(aw_timesteps)).round().long()
        X_tr  = X_tr[:, aw_idx, :]
        X_val = X_val[:, aw_idx, :]
        X_te  = X_te[:, aw_idx, :]
        print(f"[LODO] P24 AW: {T0} -> {int(aw_timesteps)} timesteps/window (strided)")

    # Load the target-unlabeled pool early if any DA option needs it.
    X_tu = y_tu = None
    if need_target:
        X_tu, y_tu, _ = load_daghar_dataset(
            data_root, target_dataset, split="train", folder_map=folder_map
        )

    # P18a — precompute TRANSDUCTIVE stats (source-train ∪ target-unlabeled). When off,
    # leave None so build_sensor_graphs fits source-train-only stats (= P17 baseline).
    precomputed_stats = None
    norm_desc = "source-train stats"
    if normalize_include_target:
        X_comb = torch.cat([X_tr, X_tu], dim=0)
        precomputed_stats = _fit_norm_stats(X_comb, feature_mode, normalize_mode)
        norm_desc = "TRANSDUCTIVE stats (source-train ∪ target-unlabeled)"
        print(f"[LODO-DA] transductive normalization: stats fit on "
              f"source-train ∪ target-unlabeled ({len(y_tr)}+{len(y_tu)} windows)")

    # P21 — Pearson-correlation graph construction (data-driven, per window).
    # Each window's adjacency is computed from its OWN raw signal only, so there is
    # no cross-sample or cross-split statistic and no leakage of any kind.
    edge_mode = cfg.get("edge_mode", "fixed")          # "fixed" | "pearson"
    es_tr = es_val = es_te = es_tu = None
    if edge_mode == "pearson":
        from data.pearson import pearson_abs_matrix, select_edges, describe
        p_mode  = cfg.get("pearson_select", "topk")    # "topk" | "tau"
        p_topk  = int(cfg.get("pearson_topk", 9))
        p_tau   = float(cfg.get("pearson_tau", 0.3))
        p_alpha = float(cfg.get("pearson_alpha", 1.0))
        print(f"[LODO] P21 edge_mode=pearson  select={p_mode}  topk={p_topk}  "
              f"tau={p_tau}  alpha={p_alpha}")

        def _edges(X_raw, label):
            R = pearson_abs_matrix(X_raw)
            es = select_edges(R, mode=p_mode, topk=p_topk, tau=p_tau, alpha=p_alpha,
                              channels=nc_keep)
            describe(es, label)
            return es

        es_tr  = _edges(X_tr, "train")
        es_val = _edges(X_val, "val")
        es_te  = _edges(X_te, "test")
        if load_target_unlabeled and X_tu is not None:
            es_tu = _edges(X_tu, "target-unlabeled")

    # Build graphs. stats: transductive if precomputed, else source-train-only (P17).
    train_graphs, stats = build_sensor_graphs(
        X_tr, y_tr, edge_list, use_edge_type,
        sources=src_tr, domain_labels=domain_labels_tr,
        normalize=True, stats=precomputed_stats, verbose=True,
        feature_mode=feature_mode, normalize_mode=normalize_mode,
        edge_sets=es_tr,
    )

    val_graphs, _ = build_sensor_graphs(
        X_val, y_val, edge_list, use_edge_type,
        normalize=True, stats=stats, verbose=False,
        feature_mode=feature_mode, normalize_mode=normalize_mode,
        edge_sets=es_val,
    )

    test_graphs, _ = build_sensor_graphs(
        X_te, y_te, edge_list, use_edge_type,
        normalize=True, stats=stats, verbose=False,
        feature_mode=feature_mode, normalize_mode=normalize_mode,
        edge_sets=es_te,
    )

    # TARGET-UNLABELED adaptation pool graphs (built only if explicitly requested).
    # Normalized with the SAME stats as everything else (transductive or source-only).
    target_unlabeled_graphs = None
    n_target_unlabeled = 0
    if load_target_unlabeled:
        target_unlabeled_graphs, _ = build_sensor_graphs(
            X_tu, y_tu, edge_list, use_edge_type,
            normalize=True, stats=stats, verbose=False,
            feature_mode=feature_mode, normalize_mode=normalize_mode,
            edge_sets=es_tu,
        )
        n_target_unlabeled = len(y_tu)
        print(f"[LODO-DA] target-unlabeled (target train, labels withheld): "
              f"{n_target_unlabeled} graphs  [normalized with {norm_desc}]")

    # Detect actual feature dimension from built graphs
    actual_feat_dim = train_graphs[0].x.shape[1]

    return {
        "train_graphs":   train_graphs,
        "val_graphs":     val_graphs,
        "test_graphs":    test_graphs,
        "target_unlabeled_graphs": target_unlabeled_graphs,  # P17 DA plumbing (None if off)
        "train_y":        y_tr.numpy(),
        "val_y":          y_val.numpy(),
        "test_y":         y_te.numpy(),
        "train_sources":  src_tr,
        "target_dataset": target_dataset,
        "sources":        sources,
        "n_train":        len(y_tr),
        "n_val":          len(y_val),
        "n_test":         len(y_te),
        "n_target_unlabeled": n_target_unlabeled,
        "num_domains":    num_domains,
        "domain_map":     src_to_idx,
        "norm_stats":     stats,
        "feature_mode":   feature_mode,
        "node_feat_dim":  actual_feat_dim,
    }
