"""
Results Logger — DAGHAR LODO experiments
=========================================
Saves per-(model, SO, target) JSON results, training curves, and
aggregated summaries.

Schema mirrors the UCI-HAR project's logger but replaces fold_idx
with target_dataset and adds per-class F1 + confusion matrices.
"""

import os
import json
import numpy as np

from utils.config import ensure_results_dirs


def _safe_json(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _sanitize(d):
    """Recursively convert a dict's values to JSON-safe types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _sanitize(v)
        elif isinstance(v, list):
            out[k] = [_safe_json(x) for x in v]
        else:
            out[k] = _safe_json(v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold (per-target) results
# ─────────────────────────────────────────────────────────────────────────────

def save_lodo_fold_result(results_root, experiment_name, target_dataset,
                          metrics, storage_stats, training_time_seconds):
    """
    Save one LODO fold result to:
        results_root/lodo/folds/{experiment_name}_{target_dataset}.json

    Args:
        results_root:          base results directory
        experiment_name:       e.g. "gcn_baseline", "gat_nc"
        target_dataset:        e.g. "UCI", "KuHar"
        metrics:               dict from evaluator (accuracy, f1_macro,
                               per_class_accuracy, confusion_matrix, ...)
        storage_stats:         dict with gsp, srr, tsc, deployment_mb, etc.
        training_time_seconds: wall-clock training time for this fold
    """
    ensure_results_dirs(results_root)

    result = {
        "experiment_name":        experiment_name,
        "target_dataset":         target_dataset,
        "accuracy":               metrics.get("accuracy"),
        "f1_macro":               metrics.get("f1_macro"),
        "per_class_accuracy":     metrics.get("per_class_accuracy"),
        "confusion_matrix":       metrics.get("confusion_matrix"),
        "training_time_seconds":  training_time_seconds,
    }
    result.update(storage_stats)
    result = _sanitize(result)

    folds_dir = os.path.join(results_root, "lodo", "folds")
    filename = f"{experiment_name}_{target_dataset.lower().replace('-', '_')}.json"
    path = os.path.join(folds_dir, filename)

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → Saved fold result: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(results_root, experiment_name, target_dataset,
                         train_losses, train_accs, val_losses, val_accs):
    """
    Save per-epoch training curves for one LODO fold.

    Args:
        results_root:   base results directory
        experiment_name: e.g. "gat_nc"
        target_dataset:  e.g. "UCI"
        train_losses, train_accs, val_losses, val_accs: lists of floats
    """
    ensure_results_dirs(results_root)

    curves = {
        "experiment_name": experiment_name,
        "target_dataset":  target_dataset,
        "train_loss":  [float(x) for x in train_losses],
        "train_acc":   [float(x) for x in train_accs],
        "val_loss":    [float(x) for x in val_losses],
        "val_acc":     [float(x) for x in val_accs],
    }

    curves_dir = os.path.join(results_root, "lodo", "training_curves")
    filename = f"{experiment_name}_{target_dataset.lower().replace('-', '_')}_curves.json"
    path = os.path.join(curves_dir, filename)

    with open(path, "w") as f:
        json.dump(curves, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment summary (across all 6 LODO targets)
# ─────────────────────────────────────────────────────────────────────────────

def save_lodo_summary(results_root, experiment_name, fold_results,
                      storage_stats, total_time_seconds):
    """
    Compute and save mean ± std across the 6 LODO targets.

    Args:
        results_root:      base results directory
        experiment_name:   e.g. "gat_nc"
        fold_results:      list of 6 dicts (one per target), each containing
                           at least 'accuracy', 'f1_macro', 'target_dataset'
        storage_stats:     dict with shared storage metrics (gsp, srr, etc.)
        total_time_seconds: total wall-clock time for all 6 folds

    Returns:
        summary dict
    """
    ensure_results_dirs(results_root)

    accs = np.array([r["accuracy"] for r in fold_results])
    f1s  = np.array([r["f1_macro"] for r in fold_results])

    summary = {
        "experiment_name":    experiment_name,
        "n_targets":          len(fold_results),
        "targets":            [r["target_dataset"] for r in fold_results],
        "accuracy_mean":      float(accs.mean()),
        "accuracy_std":       float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "accuracy_per_target": {r["target_dataset"]: float(r["accuracy"])
                                for r in fold_results},
        "f1_macro_mean":      float(f1s.mean()),
        "f1_macro_std":       float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "f1_per_target":      {r["target_dataset"]: float(r["f1_macro"])
                               for r in fold_results},
        "total_time_seconds": float(total_time_seconds),
    }
    summary.update(_sanitize(storage_stats))

    summary_dir = os.path.join(results_root, "lodo", "summary")
    filename = f"{experiment_name}_lodo_summary.json"
    path = os.path.join(summary_dir, filename)

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Saved LODO summary: {path}")
    return summary


def append_to_master(results_root, experiment_name, summary):
    """
    Append one experiment's summary to the master results file
    (all_lodo_results.json). Creates or updates the file.
    """
    ensure_results_dirs(results_root)
    master_path = os.path.join(results_root, "lodo", "summary",
                               "all_lodo_results.json")

    if os.path.isfile(master_path):
        with open(master_path) as f:
            master = json.load(f)
    else:
        master = {}

    master[experiment_name] = _sanitize(summary)

    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)
    return master_path
