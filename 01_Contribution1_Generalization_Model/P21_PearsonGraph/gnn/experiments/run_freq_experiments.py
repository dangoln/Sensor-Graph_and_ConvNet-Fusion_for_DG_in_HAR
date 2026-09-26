"""
Phase 2C: Frequency Feature Experiments for DANN+SAGE
=====================================================
Tests frequency-domain and hybrid node features on the best model
(DANN+SAGE with original small backbone).

Experiments:
  1. freq:   FFT magnitude spectrum [30 bins] as node features
  2. hybrid: Raw time [60] + FFT magnitude [30] = [90] per node
  3. freq_tuned: Freq features + lambda_max=0.5 + lr=0.0005

All use the original small backbone (51K params) since Phase 2B proved
bigger doesn't help.

Usage:
    python experiments/run_freq_experiments.py
    python experiments/run_freq_experiments.py --config freq
    python experiments/run_freq_experiments.py --config hybrid
    python experiments/run_freq_experiments.py --config freq_tuned
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader

from utils.config import get_config, ensure_results_dirs, print_config_summary
from utils.seed import set_seed
from utils.results_logger import (
    save_lodo_fold_result, save_training_curves,
    save_lodo_summary, append_to_master,
)
from utils.checkpoints import save_checkpoint
from data.daghar_loader import DAGHAR_DATASETS, build_lodo_sensor_graphs
from training.dann_trainer import DANNTrainer
from models.sensor_graph import SensorGraphModel
from models.da_sensor_graph import DomainAdversarialWrapper

import tempfile


FREQ_CONFIGS = {
    "freq": {
        "name": "Freq features [30] + DANN+SAGE",
        "feature_mode": "freq",
        "daghar_timesteps": 30,  # CNN input_length = 30 freq bins
        "learning_rate": 0.001,
        "dann_lambda_max": 1.0,
    },
    "hybrid": {
        "name": "Hybrid features [90] + DANN+SAGE",
        "feature_mode": "hybrid",
        "daghar_timesteps": 90,  # CNN input_length = 60 time + 30 freq
        "learning_rate": 0.001,
        "dann_lambda_max": 1.0,
    },
    "freq_tuned": {
        "name": "Freq [30] + DANN+SAGE (tuned: lambda=0.5, lr=5e-4)",
        "feature_mode": "freq",
        "daghar_timesteps": 30,
        "learning_rate": 0.0005,
        "dann_lambda_max": 0.5,
    },
    # ── P13: Source-domain augmentation ──────────────────────────────────────
    # Baseline = P9 (DANN+SAGE+hybrid, 73.05%); P11/P12 architecture changes
    # reverted. The ONLY change here is augmenting the FIVE source-train sets
    # (never the target), to broaden the source distribution and help the hardest
    # fold (KuHar). Encoder/normalisation stay at the P9 defaults.
    #   p13_aug        = full set (jitter+scaling+mag/time-warp+shift), x2 copies   [headline]
    #   p13_aug_warp   = warp + jitter only (the plan's predicted KuHar helpers)
    #   p13_aug_strong = larger strengths, x3 copies (probe if more helps)
    "p13_aug": {
        "name": "P13: Hybrid + DANN+SAGE + source augmentation (full, x2)",
        "feature_mode": "hybrid",
        "daghar_timesteps": 90,
        "learning_rate": 0.001,
        "dann_lambda_max": 1.0,
        "augment": True,
        "aug_copies": 2,
        "aug_jitter": 0.05, "aug_scaling": 0.10,
        "aug_mag_warp": 0.20, "aug_time_warp": 0.20,
        "aug_time_shift": 4, "aug_channel_mask": 0.0,
    },
    "p13_aug_warp": {
        "name": "P13: warp+jitter only (predicted KuHar helpers), x2",
        "feature_mode": "hybrid",
        "daghar_timesteps": 90,
        "learning_rate": 0.001,
        "dann_lambda_max": 1.0,
        "augment": True,
        "aug_copies": 2,
        "aug_jitter": 0.05, "aug_scaling": 0.0,
        "aug_mag_warp": 0.30, "aug_time_warp": 0.30,
        "aug_time_shift": 0, "aug_channel_mask": 0.0,
    },
    "p13_aug_strong": {
        "name": "P13: stronger augmentation, x3 copies",
        "feature_mode": "hybrid",
        "daghar_timesteps": 90,
        "learning_rate": 0.001,
        "dann_lambda_max": 1.0,
        "augment": True,
        "aug_copies": 3,
        "aug_jitter": 0.08, "aug_scaling": 0.15,
        "aug_mag_warp": 0.30, "aug_time_warp": 0.30,
        "aug_time_shift": 6, "aug_channel_mask": 0.0,
    },
}


def get_model_size_mb(model):
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    tmp.close()
    try:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    finally:
        os.unlink(tmp.name)
    return size_mb


def run_one_freq_fold(target_dataset, config, experiment_name):
    """Run one LODO fold with frequency features + DANN+SAGE."""
    device = torch.device(config["device"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = config["patience"]
    results_root = config["results_root"]
    fmap = config["dataset_folder_names"]

    print(f"\n{'='*70}")
    print(f"  {experiment_name.upper()} -> target={target_dataset}")
    print(f"{'='*70}")

    fold_start = time.time()

    # Build graphs with feature_mode
    print(f"\n  Step 1/3 -- Building sensor graphs...")
    lodo_data = build_lodo_sensor_graphs(
        config["data_root"], target_dataset,
        config=config, folder_map=fmap,
    )

    train_graphs = lodo_data["train_graphs"]
    val_graphs = lodo_data["val_graphs"]
    test_graphs = lodo_data["test_graphs"]
    num_domains = lodo_data["num_domains"]
    actual_feat_dim = lodo_data["node_feat_dim"]

    print(f"    Node feature dim: {actual_feat_dim}")
    print(f"    Domains: {num_domains}")

    # Update config with actual feature dimension for CNN
    config["daghar_timesteps"] = actual_feat_dim

    # Create model
    print(f"\n  Step 2/3 -- Creating DANN+SAGE model...")
    set_seed(config["seed"])
    backbone = SensorGraphModel(config)
    model = DomainAdversarialWrapper(backbone, num_domains, config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {num_params:,}")
    print(f"    CNN input_length: {actual_feat_dim}")

    # Train
    print(f"\n  Step 3/3 -- Training with DANN...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5,
    )

    trainer = DANNTrainer(
        model, optimizer, device,
        lambda_max=config.get("dann_lambda_max", 1.0),
        schedule=config.get("dann_schedule", "ganin"),
        max_grad_norm=1.0,
        scheduler=scheduler,
        patience=patience,
    )
    trainer.total_epochs = epochs

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    log_interval = config.get("log_interval", 5)
    print(f"    LR={config['learning_rate']}  lambda_max={config.get('dann_lambda_max', 1.0)}")
    print(f"    train={len(train_graphs)}  val={len(val_graphs)}  test={len(test_graphs)}")
    print(f"    {'Ep':>4}  {'TrLoss':>8}  {'ActL':>7}  {'DomL':>7}  "
          f"{'TrAcc':>7}  {'ValAcc':>7}  {'Lam':>6}")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, epochs + 1):
        tr = trainer.train_epoch(train_loader)
        vl = trainer.validate(val_loader)
        is_best = trainer.update_checkpoint(vl["accuracy"], epoch)

        train_losses.append(tr["loss"])
        train_accs.append(tr["accuracy"])
        val_losses.append(vl["loss"])
        val_accs.append(vl["accuracy"])

        if epoch % log_interval == 0 or epoch == epochs:
            bm = " *" if is_best else ""
            print(f"    {epoch:>4}  {tr['loss']:>8.4f}  "
                  f"{tr['activity_loss']:>7.4f}  "
                  f"{tr['domain_loss']:>7.4f}  "
                  f"{tr['accuracy']*100:>6.1f}%  "
                  f"{vl['accuracy']*100:>6.1f}%  "
                  f"{tr['lambda']:>6.3f}{bm}")

        if trainer.stopped_early:
            print(f"    >> Early stopping at epoch {epoch}")
            break

    trainer.restore_best()
    train_time = time.time() - fold_start

    print(f"    Best: epoch {trainer.best_epoch} "
          f"(val acc {trainer.best_val_acc*100:.2f}%)  Time: {train_time:.0f}s")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model.predict(batch)
            all_preds.append(out.argmax(dim=1).cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1_macro = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')

    print(f"    Test acc: {accuracy*100:.2f}%  F1: {f1_macro*100:.2f}%")

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "training_time_seconds": train_time,
        "target_dataset": target_dataset,
        "num_params": num_params,
        "gsp": len(train_graphs) * train_graphs[0].edge_index.shape[1],
        "srr": 0.0,
        "tsc": 0.0,
    }

    # Save
    storage_stats = {
        "gsp": metrics["gsp"],
        "avg_edges_per_graph": float(train_graphs[0].edge_index.shape[1]),
        "avg_nodes_per_graph": 6.0,
        "model_size_mb": float(get_model_size_mb(model)),
        "srr": 0.0,
    }

    save_lodo_fold_result(
        results_root, experiment_name, target_dataset,
        metrics, storage_stats, train_time,
    )
    save_training_curves(
        results_root, experiment_name, target_dataset,
        train_losses, train_accs, val_losses, val_accs,
    )

    return metrics


def run_freq_experiments(configs_to_run=None, targets=None):
    """Run frequency feature experiments."""
    if configs_to_run is None:
        configs_to_run = list(FREQ_CONFIGS.keys())

    print("\n" + "=" * 70)
    print("  PHASE 2C: FREQUENCY FEATURE EXPERIMENTS FOR DANN+SAGE")
    print(f"  Configs: {configs_to_run}")
    print("=" * 70)
    print("\n  Reference: DANN+SAGE time-domain = 70.73% mean acc")
    print("  Target: >= 77% (DAGHAR ConvNet SOTA)")

    all_summaries = {}

    for cfg_key in configs_to_run:
        cfg = FREQ_CONFIGS[cfg_key]
        experiment_name = f"dann_sage_{cfg_key}"

        print(f"\n{'#'*70}")
        print(f"  Config: {cfg['name']}")
        print(f"{'#'*70}")

        base_config = get_config()
        base_config["dann_enabled"] = True
        base_config["gnn_type"] = "sage"
        base_config["feature_mode"] = cfg["feature_mode"]
        base_config["daghar_timesteps"] = cfg["daghar_timesteps"]
        base_config["learning_rate"] = cfg["learning_rate"]
        base_config["dann_lambda_max"] = cfg["dann_lambda_max"]
        base_config["dann_schedule"] = "ganin"
        base_config["cnn_temporal_pool"] = cfg.get("cnn_temporal_pool", "avg")  # P11
        base_config["encoder_mode"]    = cfg.get("encoder_mode", "single")        # P12
        base_config["dualview_time_len"] = cfg.get("dualview_time_len", 60)       # P12
        base_config["normalize_mode"]  = cfg.get("normalize_mode", "per_channel") # P12
        # P13 — source-domain augmentation flags (pass through any that are set)
        for _k in ("augment", "aug_copies", "aug_jitter", "aug_scaling",
                   "aug_mag_warp", "aug_time_warp", "aug_time_shift",
                   "aug_channel_mask", "aug_knots"):
            if _k in cfg:
                base_config[_k] = cfg[_k]

        ensure_results_dirs(base_config["results_root"])

        fold_targets = targets or list(DAGHAR_DATASETS)
        fold_results = []
        total_start = time.time()

        print_config_summary(base_config)

        for target in fold_targets:
            result = run_one_freq_fold(target, base_config.copy(),
                                       experiment_name)
            if result is not None:
                fold_results.append(result)

        total_time = time.time() - total_start

        if fold_results:
            storage_stats = {"srr": 0.0, "gsp": fold_results[0].get("gsp", 0)}
            summary = save_lodo_summary(
                base_config["results_root"], experiment_name,
                fold_results, storage_stats, total_time,
            )
            append_to_master(base_config["results_root"], experiment_name, summary)
            all_summaries[cfg_key] = {
                "mean_acc": np.mean([r["accuracy"] for r in fold_results]),
                "mean_f1": np.mean([r["f1_macro"] for r in fold_results]),
                "std_acc": np.std([r["accuracy"] for r in fold_results], ddof=1),
                "results": fold_results,
            }

            accs = [r["accuracy"] for r in fold_results]
            f1s = [r["f1_macro"] for r in fold_results]
            print(f"\n  {experiment_name}: mean_acc={np.mean(accs)*100:.2f}% "
                  f"+/- {np.std(accs, ddof=1)*100:.2f}%  "
                  f"mean_f1={np.mean(f1s)*100:.2f}%")
            for r in fold_results:
                print(f"    {r['target_dataset']:18s}  "
                      f"acc={r['accuracy']*100:.2f}%  f1={r['f1_macro']*100:.2f}%")

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FREQUENCY FEATURE EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Config':15s}  {'Description':50s}  {'Mean Acc':>10s}  {'Mean F1':>10s}")
    print(f"  {'-'*15}  {'-'*50}  {'-'*10}  {'-'*10}")
    print(f"  {'time (base)':15s}  {'DANN+SAGE time-domain [60] (Phase 2)':50s}  {'70.73%':>10s}  {'64.73%':>10s}")

    for cfg_key in configs_to_run:
        if cfg_key in all_summaries:
            s = all_summaries[cfg_key]
            name = FREQ_CONFIGS[cfg_key]["name"]
            print(f"  {cfg_key:15s}  {name:50s}  {s['mean_acc']*100:>9.2f}%  {s['mean_f1']*100:>9.2f}%")

    print(f"\n  DAGHAR ConvNet SOTA: 77.1% mean acc")
    print(f"{'='*70}\n")

    return all_summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Frequency feature experiments for DANN+SAGE"
    )
    parser.add_argument("--config", type=str, nargs="+", default=None,
                        choices=list(FREQ_CONFIGS.keys()),
                        help="Configs to run (default: all)")
    parser.add_argument("--target", type=str, default=None,
                        help="Single LODO target for quick testing")
    args = parser.parse_args()

    targets = [args.target] if args.target else None
    run_freq_experiments(
        configs_to_run=args.config,
        targets=targets,
    )
