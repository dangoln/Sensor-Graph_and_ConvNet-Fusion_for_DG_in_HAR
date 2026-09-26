"""
P21 — Pearson-correlation graph construction (Yang et al. 2024, GDA-inspired).

Builds a DATA-DRIVEN adjacency per window from the Pearson correlation between
the 6 sensor channels, replacing the fixed hand-designed topologies of P6-P18.

Key design notes
----------------
* GraphSAGE ignores edge *weights*, so in P21 (SAGE control backbone) the
  Pearson matrix acts through TOPOLOGY SELECTION: which of the 15 possible
  channel pairs become edges in each window.  Weighted adjacency is deferred
  to P22 (ChebNet consumes edge_weight natively via the Laplacian).
* Correlation is computed on the RAW 60-step time signal per channel
  (before normalization — Pearson is invariant to per-channel z-scoring,
  and the hybrid FFT bins are not a physically meaningful correlation space).
* Selection modes (the P21 sweep grid):
    - topk:  keep the k strongest pairs by blended score (k in {6, 9, 12})
    - tau:   keep pairs with blended score >= tau, padded to >= min_edges
* Blending with the physical prior (alpha < 1) interpolates between pure
  data-driven (alpha=1) and the P6 "physical" topology:
        score_ij = alpha * |rho_ij| + (1 - alpha) * 1[(i,j) in physical]
* Leakage: none possible — each window's adjacency is computed from that
  window's own signal only (no statistics shared across samples or splits).

Pure numpy + torch; no model code.
"""
import numpy as np
import torch

NUM_NODES = 6
ALL_PAIRS = [(i, j) for i in range(NUM_NODES) for j in range(i + 1, NUM_NODES)]  # 15

# P6 "physical" prior: intra-accel, intra-gyro, axis-aligned cross-modal
PHYSICAL_PRIOR = {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3), (1, 4), (2, 5)}


def pearson_abs_matrix(X):
    """
    |Pearson rho| between channels, per window. Vectorized over N.

    Args:
        X: [N, T, C] tensor or ndarray of RAW time signals (T=60, C=6)
    Returns:
        R: [N, C, C] float32 ndarray, R[n,i,j] = |rho(channel_i, channel_j)| in window n
    """
    Xn = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    Xn = Xn.astype(np.float64)
    N, T, C = Xn.shape
    Xc = Xn - Xn.mean(axis=1, keepdims=True)            # center per channel
    sd = Xn.std(axis=1)                                  # [N, C]
    sd[sd < 1e-8] = 1e-8                                 # constant channel -> rho ~ 0
    cov = np.einsum("ntc,ntd->ncd", Xc, Xc) / T          # [N, C, C]
    rho = cov / (sd[:, :, None] * sd[:, None, :])
    rho = np.clip(rho, -1.0, 1.0)
    return np.abs(rho).astype(np.float32)


def select_edges(R, mode="topk", topk=9, tau=0.3, alpha=1.0, min_edges=3,
                 channels=None):
    """
    Turn per-window |rho| matrices into per-window undirected edge lists.

    Args:
        R:         [N, C, C] |rho| from pearson_abs_matrix
        mode:      "topk" | "tau"
        topk:      number of undirected pairs to keep (mode=topk)
        tau:       score threshold (mode=tau)
        alpha:     blend weight: alpha*|rho| + (1-alpha)*physical_prior
        min_edges: floor on edges per window (pad with strongest pairs) so no
                   window degenerates to an edgeless graph
        channels:  P24 NC — ORIGINAL channel ids of R's columns (e.g. [0,1,2]
                   for accel_only). Pairs are emitted in LOCAL indices (0..C-1);
                   physical-prior membership is checked on the original ids.
                   Default None = all 6 channels.
    Returns:
        edge_sets: list (len N) of lists of (src, dst) undirected pairs
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    if mode not in ("topk", "tau"):
        raise ValueError(f"unknown selection mode '{mode}'")

    chans = list(range(R.shape[1])) if channels is None else list(channels)
    C = len(chans)
    assert R.shape[1] == C, f"R has {R.shape[1]} channels, channels says {C}"
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]   # local indices
    topk = min(topk, len(pairs))
    min_edges = min(min_edges, len(pairs))

    N = R.shape[0]
    prior_vec = np.array(
        [1.0 if (chans[i], chans[j]) in PHYSICAL_PRIOR else 0.0 for (i, j) in pairs],
        dtype=np.float32)
    pair_idx = np.array(pairs)
    rho_vec = R[:, pair_idx[:, 0], pair_idx[:, 1]]               # [N, P]
    scores = alpha * rho_vec + (1.0 - alpha) * prior_vec[None, :]  # [N, P]

    order = np.argsort(-scores, axis=1)                          # strongest first
    edge_sets = []
    for n in range(N):
        if mode == "topk":
            keep = order[n, :topk]
        else:  # tau
            keep = np.where(scores[n] >= tau)[0]
            if len(keep) < min_edges:                            # pad with strongest
                keep = order[n, :min_edges]
        edge_sets.append([pairs[i] for i in keep])
    return edge_sets


def describe(edge_sets, label=""):
    """One-line stats so every run logs what the graphs actually look like."""
    counts = np.array([len(e) for e in edge_sets])
    flat = {}
    for es in edge_sets:
        for p in es:
            flat[p] = flat.get(p, 0) + 1
    top = sorted(flat.items(), key=lambda kv: -kv[1])[:5]
    top_str = ", ".join(f"{p}:{c * 100.0 / len(edge_sets):.0f}%" for p, c in top)
    print(f"    [pearson{(' ' + label) if label else ''}] edges/window: "
          f"mean {counts.mean():.1f}  min {counts.min()}  max {counts.max()}  "
          f"| most-kept pairs: {top_str}")
