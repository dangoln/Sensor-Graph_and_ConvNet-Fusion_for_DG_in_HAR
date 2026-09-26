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
    cfg: ExperimentConfig, target_key: str, with_target_unlabeled: bool = False,
    transductive_norm: bool = False,
) -> Tuple:
    """
    Build (train_loader, val_loader, test_loader, meta) for one LODO fold whose
    held-out target dataset is `target_key`.

    P17 transductive LODO-DA: if ``with_target_unlabeled=True``, also build and
    return a loader over the TARGET's own train split as UNLABELED adaptation data
    (consumed by P18 UDA only). Target labels are kept for diagnostics only — never
    used for training or model selection. Return becomes a 5-tuple:
    (train, val, test, target_unlabeled, meta). See PROTOCOL_DA_LODO.md.

    P18a — ``transductive_norm=True`` fits the input Standardizer on source-train
    UNION target-unlabeled (instead of source-train only). This is the ONE change
    vs the P17 baseline: a principled, fit-time version of AdaBN. Still no target
    LABELS used. Implies ``with_target_unlabeled``.
    """
    if transductive_norm:
        with_target_unlabeled = True
    source_keys = [d for d in cfg.data.datasets if d != target_key]

    X_tr, y_tr = _concat_splits(cfg, source_keys, cfg.lodo.source_train_split)
    X_va, y_va = _concat_splits(cfg, source_keys, cfg.lodo.source_val_split)
    X_te, y_te = _concat_target_eval(cfg, target_key)

    # P17 — target's own train split as the unlabeled adaptation pool.
    X_tu = y_tu = None
    if with_target_unlabeled:
        folder = cfg.data.folder_name_map[target_key]
        X_tu, y_tu = load_dataset_split(
            data_path=cfg.paths.data_path,
            folder_name=folder,
            split=cfg.lodo.source_train_split,   # the TARGET's train split
            channels=cfg.data.channels,
            window_size=cfg.data.window_size,
            label_column=cfg.data.label_column,
        )

    # 1) Input-view transform (time -> raw, frequency -> FFT view).
    view = cfg.data.input_view
    fft_kw = dict(
        fft_magnitude=cfg.data.fft_magnitude,
        fft_window=cfg.data.fft_window,
    )
    X_tr = apply_input_view(X_tr, view, cfg.data.fft_drop_dc, **fft_kw)
    X_va = apply_input_view(X_va, view, cfg.data.fft_drop_dc, **fft_kw)
    X_te = apply_input_view(X_te, view, cfg.data.fft_drop_dc, **fft_kw)
    if with_target_unlabeled:
        X_tu = apply_input_view(X_tu, view, cfg.data.fft_drop_dc, **fft_kw)

    # 2) Normalisation. P17 baseline: fit on source-train only. P18a transductive:
    #    fit on source-train ∪ target-unlabeled (no target labels used either way).
    norm = cfg.data.resolved_normalize()
    if norm != "none":
        if transductive_norm:
            import numpy as _np
            fit_X = _np.concatenate([X_tr, X_tu], axis=0)
            print(f"  [cnn] transductive normalization: Standardizer fit on "
                  f"source-train ∪ target-unlabeled ({len(X_tr)}+{len(X_tu)} windows)")
        else:
            fit_X = X_tr
        std = Standardizer(mode=norm).fit(fit_X)
        X_tr, X_va, X_te = std.transform(X_tr), std.transform(X_va), std.transform(X_te)
        if with_target_unlabeled:
            X_tu = std.transform(X_tu)

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

    if with_target_unlabeled:
        target_unlabeled_ds = DAGHARWindows(X_tu, y_tu)
        target_unlabeled_loader = DataLoader(
            target_unlabeled_ds, batch_size=cfg.train.batch_size, shuffle=True,
            drop_last=False, **common,
        )
        meta["n_target_unlabeled"] = int(len(target_unlabeled_ds))
        return train_loader, val_loader, test_loader, target_unlabeled_loader, meta

    return train_loader, val_loader, test_loader, meta


def _class_counts(y: np.ndarray, num_classes: int) -> List[int]:
    return [int((y == c).sum()) for c in range(num_classes)]


def iter_lodo_folds(cfg: ExperimentConfig):
    """Yield target_key for each LODO fold (one per dataset)."""
    for target_key in cfg.data.datasets:
        yield target_key
