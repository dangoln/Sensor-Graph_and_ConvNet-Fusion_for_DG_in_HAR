"""
LODO (Leave-One-Dataset-Out) driver for DeepConvLSTM on DAGHAR.

For each of the 6 DAGHAR datasets, hold it out as the target domain, train on the
other 5 (train split), select on their validation split, and evaluate on the
held-out dataset's test split. Saves per-fold metrics, a summary, and the best
checkpoint per fold.

Run from the project root:
    python -m experiments.run_lodo
or
    python experiments/run_lodo.py

Optionally override paths via environment variables:
    PROJECT_PATH, DATA_PATH, RESULTS_DIR
or via CLI flags (see --help).
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable whether run as module or script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from configs.config import get_config
from src.data.data_loader import build_lodo_fold, iter_lodo_folds
from src.models.deepconvlstm import build_model
from src.training.trainer import train_model, evaluate
from src.training.metrics import summarize_folds
from src.utils.common import set_seed, get_device, ensure_dir, save_json


def parse_args():
    p = argparse.ArgumentParser(description="DeepConvLSTM LODO on DAGHAR")
    p.add_argument("--data_path", type=str, default=None,
                   help="Override DATA_PATH (standardized_view folder).")
    p.add_argument("--project_path", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--input_view", type=str, default=None,
                   choices=["time", "frequency"],
                   help="Input representation: raw time signal or FFT view.")
    p.add_argument("--normalize", type=str, default=None,
                   choices=["auto", "none", "zscore_channel", "zscore_feature"])
    p.add_argument("--run_name", type=str, default=None,
                   help="Group results under results/<run_name>/<view>/ .")
    p.add_argument("--fft_magnitude", type=str, default=None,
                   choices=["amplitude", "power", "log"])
    p.add_argument("--fft_window", type=str, default=None,
                   choices=["none", "hann"])
    p.add_argument("--fft_drop_dc", action="store_true")
    p.add_argument("--model_selection_metric", type=str, default=None,
                   choices=["val_loss", "val_acc", "val_f1_macro"])
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--targets", type=str, default=None,
                   help="Comma-separated subset of datasets to hold out "
                        "(default: all 6).")
    p.add_argument("--quiet", action="store_true", help="Less per-epoch logging.")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.data_path:    cfg.paths.data_path = args.data_path
    if args.project_path: cfg.paths.project_path = args.project_path
    if args.results_dir:  cfg.paths.results_dir = args.results_dir
    if args.max_epochs:   cfg.train.max_epochs = args.max_epochs
    if args.batch_size:   cfg.train.batch_size = args.batch_size
    if args.lr:           cfg.train.learning_rate = args.lr
    if args.seed is not None: cfg.train.seed = args.seed
    if args.input_view:   cfg.data.input_view = args.input_view
    if args.normalize:    cfg.data.normalize = args.normalize
    if args.run_name:     cfg.paths.run_name = args.run_name
    if args.fft_magnitude: cfg.data.fft_magnitude = args.fft_magnitude
    if args.fft_window:   cfg.data.fft_window = args.fft_window
    if args.fft_drop_dc:  cfg.data.fft_drop_dc = True
    if args.model_selection_metric:
        cfg.train.model_selection_metric = args.model_selection_metric
    if args.patience is not None:
        cfg.train.early_stopping_patience = args.patience
    return cfg.sync()


def main():
    args = parse_args()
    cfg = apply_overrides(get_config(), args)
    verbose = not args.quiet

    device = get_device(cfg.train.device)
    results_base = (
        os.path.join(cfg.paths.project_path, cfg.paths.results_dir)
        if not os.path.isabs(cfg.paths.results_dir) else cfg.paths.results_dir
    )
    # Group by run_name then input view so runs never overwrite each other.
    results_root = ensure_dir(
        os.path.join(results_base, cfg.paths.run_name, cfg.data.input_view)
    )

    targets = (
        [t.strip() for t in args.targets.split(",")]
        if args.targets else list(iter_lodo_folds(cfg))
    )

    print("=" * 70)
    print("DeepConvLSTM  |  LODO on DAGHAR (standardized_view)")
    print(f"device       : {device}")
    print(f"data_path    : {cfg.paths.data_path}")
    print(f"results_dir  : {results_root}")
    print(f"run_name     : {cfg.paths.run_name}")
    print(f"input_view   : {cfg.data.input_view}  "
          f"(seq_len={cfg.effective_seq_len()}, "
          f"normalize={cfg.data.resolved_normalize()})")
    if cfg.data.input_view == "frequency":
        print(f"fft          : magnitude={cfg.data.fft_magnitude}, "
              f"window={cfg.data.fft_window}, drop_dc={cfg.data.fft_drop_dc}")
    print(f"selection    : {cfg.train.model_selection_metric} "
          f"(max_epochs={cfg.train.max_epochs}, "
          f"patience={cfg.train.early_stopping_patience})")
    print(f"channels({len(cfg.data.channels)}): {cfg.data.channels}")
    print(f"classes ({cfg.data.num_classes}) : {cfg.data.class_names}")
    print(f"folds        : {targets}")
    print("=" * 70)

    fold_metrics = {}
    for target in targets:
        set_seed(cfg.train.seed)  # same init per fold -> comparable folds
        print(f"\n#### LODO fold | held-out target = {target} ####")

        train_loader, val_loader, test_loader, meta = build_lodo_fold(cfg, target)
        print(f"  sources : {meta['sources']}")
        print(f"  sizes   : train={meta['n_train']} val={meta['n_val']} "
              f"test={meta['n_test']}")
        print(f"  train class counts: {meta['train_class_counts']}")
        print(f"  test  class counts: {meta['test_class_counts']}")

        model = build_model(cfg.model)
        history = train_model(
            model, train_loader, val_loader, device,
            cfg.train, cfg.data.num_classes,
            log_prefix=f"  [{target}] ", verbose=verbose,
        )

        test_metrics, _ = evaluate(model, test_loader, device, cfg.data.num_classes)
        test_metrics["meta"] = meta
        test_metrics["best_epoch"] = history["best_epoch"]
        fold_metrics[target] = test_metrics

        print(f"  >> TARGET={target}  "
              f"acc={test_metrics['accuracy']:.4f}  "
              f"f1_macro={test_metrics['f1_macro']:.4f}  "
              f"f1_weighted={test_metrics['f1_weighted']:.4f}")

        # Persist per-fold artefacts.
        fold_dir = ensure_dir(os.path.join(results_root, f"fold_{target}"))
        save_json({"metrics": test_metrics, "history": history},
                  os.path.join(fold_dir, "results.json"))
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

    # Aggregate summary across folds.
    summary = summarize_folds(fold_metrics)
    print("\n" + "=" * 70)
    print("LODO SUMMARY (mean +/- std across folds)")
    for k, v in summary.items():
        print(f"  {k:12s}: {v['mean']:.4f} +/- {v['std']:.4f} "
              f"(min {v['min']:.4f}, max {v['max']:.4f})")
    print("=" * 70)

    save_json(
        {
            "config": cfg.to_dict(),
            "per_fold": {
                t: {kk: m[kk] for kk in
                    ["accuracy", "f1_weighted", "f1_macro", "f1_per_class",
                     "confusion_matrix", "best_epoch", "meta"]}
                for t, m in fold_metrics.items()
            },
            "summary": summary,
        },
        os.path.join(results_root, "lodo_summary.json"),
    )
    print(f"\nSaved: {os.path.join(results_root, 'lodo_summary.json')}")


if __name__ == "__main__":
    main()
