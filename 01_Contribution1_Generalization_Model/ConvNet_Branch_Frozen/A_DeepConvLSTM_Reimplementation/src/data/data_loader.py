"""
Builds the LODO (Leave-One-Dataset-Out) splits and the corresponding
PyTorch DataLoaders.

For a target dataset D:
  * SOURCES   = all datasets except D
  * train set = concat of SOURCES' `train` split
  * val set   = concat of SOURCES' `validation` split   (model selection)
  * test set  = D's `test` split                         (cross-dataset eval)

The module returns ready-to-use DataLoaders plus a small metadata dict so the
training code never needs to know about CSV layouts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from configs.config import ExperimentConfig
from src.data.dataset import (
    DAGHARWindows,
    Standardizer,
    apply_input_view,
    load_dataset_split,
)


def _concat_splits(
    cfg: ExperimentConfig, dataset_keys: List[str], split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate one split across several datasets."""
    xs, ys = [], []
    for key in dataset_keys:
        folder = cfg.data.folder_name_map[key]
        X, y = load_dataset_split(
            data_path=cfg.paths.data_path,
            folder_name=folder,
            split=split,
            channels=cfg.data.channels,
            window_size=cfg.data.window_size,
            label_column=cfg.data.label_column,
        )
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _concat_target_eval(
    cfg: ExperimentConfig, target_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the held-out dataset's evaluation split(s)."""
    folder = cfg.data.folder_name_map[target_key]
    xs, ys = [], []
    for split in cfg.lodo.target_eval_splits:
        X, y = load_dataset_split(
            data_path=cfg.paths.data_path,
            folder_name=folder,
            split=split,
            channels=cfg.data.channels,
            window_size=cfg.data.window_size,
            label_column=cfg.data.label_column,
        )
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def build_lodo_fold(
    cfg: ExperimentConfig, target_key: str
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build (train_loader, val_loader, test_loader, meta) for one LODO fold whose
    held-out target dataset is `target_key`.
    """
    source_keys = [d for d in cfg.data.datasets if d != target_key]

    X_tr, y_tr = _concat_splits(cfg, source_keys, cfg.lodo.source_train_split)
    X_va, y_va = _concat_splits(cfg, source_keys, cfg.lodo.source_val_split)
    X_te, y_te = _concat_target_eval(cfg, target_key)

    # 1) Input-view transform (time -> raw, frequency -> FFT view).
    view = cfg.data.input_view
    fft_kw = dict(
        fft_magnitude=cfg.data.fft_magnitude,
        fft_window=cfg.data.fft_window,
    )
    X_tr = apply_input_view(X_tr, view, cfg.data.fft_drop_dc, **fft_kw)
    X_va = apply_input_view(X_va, view, cfg.data.fft_drop_dc, **fft_kw)
    X_te = apply_input_view(X_te, view, cfg.data.fft_drop_dc, **fft_kw)

    # 2) Optional normalisation, fit on source TRAIN only (no target leakage).
    norm = cfg.data.resolved_normalize()
    if norm != "none":
        std = Standardizer(mode=norm).fit(X_tr)
        X_tr, X_va, X_te = std.transform(X_tr), std.transform(X_va), std.transform(X_te)

    train_ds = DAGHARWindows(X_tr, y_tr)
    val_ds = DAGHARWindows(X_va, y_va)
    test_ds = DAGHARWindows(X_te, y_te)

    common = dict(num_workers=cfg.train.num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        drop_last=False, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, **common
    )

    meta = {
        "target": target_key,
        "sources": source_keys,
        "input_view": view,
        "seq_len": int(X_tr.shape[1]),
        "normalize": norm,
        "fft_magnitude": cfg.data.fft_magnitude if view == "frequency" else None,
        "fft_window": cfg.data.fft_window if view == "frequency" else None,
        "fft_drop_dc": cfg.data.fft_drop_dc if view == "frequency" else None,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "train_class_counts": _class_counts(y_tr, cfg.data.num_classes),
        "test_class_counts": _class_counts(y_te, cfg.data.num_classes),
    }
    return train_loader, val_loader, test_loader, meta


def _class_counts(y: np.ndarray, num_classes: int) -> List[int]:
    return [int((y == c).sum()) for c in range(num_classes)]


def iter_lodo_folds(cfg: ExperimentConfig):
    """Yield target_key for each LODO fold (one per dataset)."""
    for target_key in cfg.data.datasets:
        yield target_key
