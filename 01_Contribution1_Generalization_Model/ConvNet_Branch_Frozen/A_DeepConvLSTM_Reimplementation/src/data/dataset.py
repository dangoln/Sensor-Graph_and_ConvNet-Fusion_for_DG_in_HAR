"""
DAGHAR dataset handling.

The DAGHAR "standardized_view" stores each fixed-length window as one CSV row.
Columns look like:

    accel-x-0 ... accel-x-59, accel-y-0 ... accel-y-59, ... gyro-z-0 ... gyro-z-59,
    <metadata columns>, standard activity code

This module turns those rows into model-ready tensors of shape
(N, 1, window_size, n_channels) plus integer labels (the DAGHAR
"standard activity code", 0-5).

It is deliberately model-agnostic so it can be reused by future pipelines.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# torch is imported lazily inside DAGHARWindows so that the pure-numpy data
# utilities (CSV parsing, channel selection, standardisation) can be used in
# environments where torch is not installed.
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch optional for data utils
    torch = None

    class Dataset:  # minimal stand-in so the class definition still imports
        pass


# --------------------------------------------------------------------------- #
#  Low-level CSV -> ndarray
# --------------------------------------------------------------------------- #
def build_channel_columns(channels: List[str], window_size: int) -> List[List[str]]:
    """Return, for each channel, the ordered list of its per-timestep column names."""
    return [[f"{ch}-{t}" for t in range(window_size)] for ch in channels]


def load_csv_windows(
    csv_path: str,
    channels: List[str],
    window_size: int,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one DAGHAR CSV into:
        X : float32 array (N, window_size, n_channels)
        y : int64   array (N,)
    Only the requested common channels are kept, so datasets with extra
    columns (e.g. MotionSense attitude/gravity) are handled uniformly.
    """
    df = pd.read_csv(csv_path)

    channel_cols = build_channel_columns(channels, window_size)
    missing = [c for cols in channel_cols for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {len(missing)} expected channel columns, "
            f"e.g. {missing[:3]}"
        )
    if label_column not in df.columns:
        raise ValueError(f"{csv_path} has no label column '{label_column}'")

    # Stack channels -> (N, window_size, n_channels)
    per_channel = [df[cols].to_numpy(dtype=np.float32) for cols in channel_cols]
    X = np.stack(per_channel, axis=-1)            # (N, T, C)
    y = df[label_column].to_numpy()

    # Labels may arrive as float (e.g. 3.0); coerce to int.
    y = np.rint(y.astype(np.float64)).astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- #
#  Input-view transform: time -> frequency (per-window FFT magnitude)
# --------------------------------------------------------------------------- #
def to_frequency_view(
    X: np.ndarray,
    drop_dc: bool = False,
    magnitude: str = "amplitude",
    window: str = "none",
) -> np.ndarray:
    """
    Convert time-domain windows to their frequency-domain view.

    X : (N, T, C) real time-domain windows
    -> (N, T//2 + 1, C) spectra (rfft along the time axis).

    This is a per-window preprocessing step, NOT a model change: the same
    DeepConvLSTM consumes the result. Using rfft (instead of full fft) keeps only
    the non-redundant half of the spectrum, the standard practice for real
    signals in HAR.

    window    : "none" | "hann"  -- taper applied before the FFT (reduces leakage).
    magnitude : "amplitude" |FFT| , "power" |FFT|^2 , or "log" log1p(|FFT|).
    """
    T = X.shape[1]
    if window == "hann":
        w = np.hanning(T).astype(np.float32)   # zero-edged taper, length T
        Xw = X * w[None, :, None]
    elif window == "none":
        Xw = X
    else:
        raise ValueError(f"Unknown fft_window '{window}'")

    spec = np.fft.rfft(Xw, axis=1)             # (N, T//2+1, C), complex
    mag = np.abs(spec).astype(np.float32)      # magnitude spectrum

    if magnitude == "amplitude":
        out = mag
    elif magnitude == "power":
        out = (mag ** 2).astype(np.float32)
    elif magnitude == "log":
        out = np.log1p(mag).astype(np.float32)  # log(1+|FFT|): non-negative, tames tails
    else:
        raise ValueError(f"Unknown fft_magnitude '{magnitude}'")

    if drop_dc:
        out = out[:, 1:, :]                     # drop bin 0 (window mean / DC)
    return np.ascontiguousarray(out)


def apply_input_view(
    X: np.ndarray,
    input_view: str,
    fft_drop_dc: bool,
    fft_magnitude: str = "amplitude",
    fft_window: str = "none",
) -> np.ndarray:
    """Dispatch on the configured input view."""
    if input_view == "time":
        return X
    if input_view == "frequency":
        return to_frequency_view(
            X, drop_dc=fft_drop_dc, magnitude=fft_magnitude, window=fft_window
        )
    raise ValueError(f"Unknown input_view '{input_view}'")


# --------------------------------------------------------------------------- #
#  Normalisation helper (fit on training data only)
# --------------------------------------------------------------------------- #
class Standardizer:
    """
    Z-score standardiser fitted on a (training) set.

    mode:
      "zscore_channel" -> stats per channel, pooled over time/bins   (axis 0,1)
      "zscore_feature" -> stats per (timestep-or-bin, channel)        (axis 0)
    """

    def __init__(self, mode: str = "zscore_channel") -> None:
        self.mode = mode
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        # X: (N, T, C)
        if self.mode == "zscore_feature":
            axis = (0,)
        elif self.mode == "zscore_channel":
            axis = (0, 1)
        else:
            raise ValueError(f"Unknown standardizer mode '{self.mode}'")
        self.mean = X.mean(axis=axis, keepdims=True)
        self.std = X.std(axis=axis, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return X
        return (X - self.mean) / self.std


# Backwards-compatible alias (per-channel standardiser).
class ChannelStandardizer(Standardizer):
    def __init__(self) -> None:
        super().__init__(mode="zscore_channel")


# --------------------------------------------------------------------------- #
#  Torch Dataset
# --------------------------------------------------------------------------- #
class DAGHARWindows(Dataset):
    """
    Wraps (X, y) arrays as a torch Dataset that yields:
        x : float32 tensor (1, window_size, n_channels)   <- 4D conv input
        y : int64   scalar tensor
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ImportError("PyTorch is required to build DAGHARWindows.")
        # add the singleton "channel" dim expected by the Conv2d front-end:
        # (N, T, C) -> (N, 1, T, C)
        self.X = torch.from_numpy(np.ascontiguousarray(X[:, None, :, :])).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------- #
#  Dataset discovery
# --------------------------------------------------------------------------- #
def dataset_split_path(
    data_path: str, folder_name: str, split: str
) -> str:
    """Path to <data_path>/<folder_name>/<split>.csv."""
    return os.path.join(data_path, folder_name, f"{split}.csv")


def load_dataset_split(
    data_path: str,
    folder_name: str,
    split: str,
    channels: List[str],
    window_size: int,
    label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one (dataset, split) pair, e.g. ('UCI', 'test')."""
    path = dataset_split_path(data_path, folder_name, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected DAGHAR split not found: {path}")
    return load_csv_windows(path, channels, window_size, label_column)
