"""
Source-Domain Augmentation (P13)
================================
Leakage-free time-series augmentation for HAR domain generalisation. These are
applied to the FIVE source training windows ONLY (never validation or the
held-out target), to broaden the source distribution so the model transfers
better to unseen domains — especially KuHar, whose shift is sampling-rate /
demographic rather than placement.

Augmentations (per the plan; rotation deliberately excluded):
  - jitter         : additive Gaussian noise
  - scaling        : per-channel multiplicative gain ~ N(1, sigma)
  - magnitude_warp : smooth per-channel gain curve along time (knot spline)
  - time_warp      : smooth distortion of the time axis (knot spline)
  - time_shift     : small circular shift along time
  - channel_mask   : zero a random channel (off by default)

Operates on raw windows X of shape [N, T, C] BEFORE feature extraction (FFT),
so the frequency view reflects the augmented signal. All randomness comes from
a seeded numpy Generator for reproducibility.

Reference: Um et al. 2017 (wearable sensor data augmentation); Napoli et al.
2024 ("maintain a broad distribution"), cited in the P11-17 plan.
"""

import numpy as np


def _random_curve(T, n_knots, sigma, rng):
    """Smooth length-T curve through `n_knots` random values ~ N(1, sigma)."""
    n_knots = max(2, int(n_knots))
    xs = np.linspace(0, T - 1, n_knots)
    ys = rng.normal(1.0, sigma, n_knots)
    return np.interp(np.arange(T), xs, ys)


def jitter(x, sigma, rng):
    """x: [T, C] — add Gaussian noise."""
    return x + rng.normal(0.0, sigma, x.shape)


def scaling(x, sigma, rng):
    """Per-channel multiplicative gain ~ N(1, sigma)."""
    factor = rng.normal(1.0, sigma, (1, x.shape[1]))
    return x * factor


def magnitude_warp(x, sigma, knots, rng):
    """Multiply each channel by its own smooth gain curve along time."""
    T, C = x.shape
    curves = np.stack([_random_curve(T, knots, sigma, rng) for _ in range(C)], axis=1)
    return x * curves


def time_warp(x, sigma, knots, rng):
    """Warp the time axis with a smooth monotonic distortion, then resample."""
    T, C = x.shape
    curve = _random_curve(T, knots, sigma, rng)
    curve = np.clip(curve, 1e-3, None)          # keep strictly increasing
    cum = np.cumsum(curve)
    cum = (cum - cum[0]) / (cum[-1] - cum[0]) * (T - 1)   # normalise to [0, T-1]
    grid = np.arange(T)
    out = np.empty_like(x)
    for c in range(C):
        out[:, c] = np.interp(grid, cum, x[:, c])
    return out


def time_shift(x, max_shift, rng):
    """Circular shift along time by a random amount in [-max_shift, max_shift]."""
    if max_shift <= 0:
        return x
    s = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(x, s, axis=0)


def channel_mask(x, p, rng):
    """With probability p, zero out one random channel."""
    if p > 0 and rng.random() < p:
        x = x.copy()
        c = int(rng.integers(0, x.shape[1]))
        x[:, c] = 0.0
    return x


def augment_sample(x, cfg, rng):
    """
    Apply the enabled augmentations to one window x [T, C].
    Strength of each is read from cfg (0 / absent → disabled).
    """
    knots = cfg.get("aug_knots", 4)
    if cfg.get("aug_jitter", 0):       x = jitter(x, cfg["aug_jitter"], rng)
    if cfg.get("aug_scaling", 0):      x = scaling(x, cfg["aug_scaling"], rng)
    if cfg.get("aug_mag_warp", 0):     x = magnitude_warp(x, cfg["aug_mag_warp"], knots, rng)
    if cfg.get("aug_time_warp", 0):    x = time_warp(x, cfg["aug_time_warp"], knots, rng)
    if cfg.get("aug_time_shift", 0):   x = time_shift(x, int(cfg["aug_time_shift"]), rng)
    if cfg.get("aug_channel_mask", 0): x = channel_mask(x, cfg["aug_channel_mask"], rng)
    return x.astype(np.float32)


def augment_training_set(X, y, sources, domain_labels, cfg, seed=42):
    """
    Expand the SOURCE-TRAIN set with augmented copies.

    Args:
        X:             [N, T, C] tensor or ndarray of raw windows (source-train)
        y:             [N] label tensor
        sources:       [N] ndarray of source dataset names
        domain_labels: [N] ndarray of int domain labels
        cfg:           dict with aug_* strengths and aug_copies (# augmented passes)
        seed:          RNG seed

    Returns:
        X_aug:             [N*(1+aug_copies), T, C] float32 ndarray
        y_aug:             [N*(1+aug_copies)] tensor (same dtype as y)
        sources_aug:       ndarray
        domain_labels_aug: ndarray

    The original (un-augmented) windows are always kept; `aug_copies` additional
    randomly-augmented versions are appended.
    """
    n_copies = int(cfg.get("aug_copies", 1))
    X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X, dtype=np.float32)
    N = X_np.shape[0]
    rng = np.random.default_rng(seed)

    X_parts = [X_np.astype(np.float32)]
    for _ in range(n_copies):
        X_aug = np.empty_like(X_np, dtype=np.float32)
        for i in range(N):
            X_aug[i] = augment_sample(X_np[i], cfg, rng)
        X_parts.append(X_aug)

    X_out = np.concatenate(X_parts, axis=0)
    reps = 1 + n_copies

    # Tile labels / sources / domain-labels to match. Keep y's original type
    # (torch tensor in production, ndarray in unit tests).
    if hasattr(y, "numpy"):                       # torch tensor
        import torch
        y_out = torch.cat([y for _ in range(reps)], dim=0)
    else:                                          # ndarray / list
        y_out = np.tile(np.asarray(y), reps)
    sources_out = np.concatenate([np.asarray(sources)] * reps, axis=0)
    domain_out = np.concatenate([np.asarray(domain_labels)] * reps, axis=0)

    return X_out, y_out, sources_out, domain_out
