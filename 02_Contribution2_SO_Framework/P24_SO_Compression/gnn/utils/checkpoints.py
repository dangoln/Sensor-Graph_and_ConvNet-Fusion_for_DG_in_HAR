"""
Checkpoint Manager — DAGHAR LODO experiments
==============================================
Saves and loads model checkpoints with full reproducibility metadata.

Each checkpoint is a single .pt file containing:
  - model_state_dict
  - model_type, model_config (architecture params)
  - so_technique, so_config
  - lodo_target, lodo_sources
  - seed, best_epoch, best_val_acc
  - test_metrics
  - library versions
"""

import os
import torch


def save_checkpoint(results_root, experiment_name, target_dataset,
                    model, model_type, config,
                    so_technique, best_epoch, best_val_acc,
                    test_metrics=None, sources=None):
    """
    Save a reproducibility-complete checkpoint.

    File:  results_root/lodo/checkpoints/{experiment_name}_{target}.pt

    Args:
        results_root:    base results directory
        experiment_name: e.g. "gat_nc"
        target_dataset:  e.g. "UCI"
        model:           the trained model (after restore_best)
        model_type:      "gcn" or "gat"
        config:          full config dict
        so_technique:    "baseline", "nc", "gs", etc.
        best_epoch:      epoch of the best validation accuracy
        best_val_acc:    best validation accuracy
        test_metrics:    optional dict of test-set metrics
        sources:         optional list of source dataset names

    Returns:
        path to the saved checkpoint
    """
    ckpt_dir = os.path.join(results_root, "lodo", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    target_slug = target_dataset.lower().replace("-", "_")
    filename = f"{experiment_name}_{target_slug}.pt"
    path = os.path.join(ckpt_dir, filename)

    bundle = {
        # Weights
        "model_state_dict": model.state_dict(),

        # Architecture
        "model_type":   model_type,
        "model_config": {
            "in_channels":     config.get("daghar_channels", 6) * 7,  # 42
            "hidden_channels": config.get("hidden_channels", 128),
            "num_classes":     config.get("num_classes", 6),
        },

        # SO
        "so_technique": so_technique,
        "so_config":    {
            "nc_ratio":  config.get("nc_ratio"),
            "gs_ratio":  config.get("gs_ratio"),
            "red_drop_ratio": config.get("red_drop_ratio"),
        },

        # LODO
        "lodo_target":  target_dataset,
        "lodo_sources": sources or [],

        # Training
        "seed":         config.get("seed", 42),
        "best_epoch":   best_epoch,
        "best_val_acc": best_val_acc,
        "test_metrics": test_metrics or {},

        # Versions (for reproducibility audits)
        "torch_version": torch.__version__,
        "daghar_view":   "standardized_view_v1",
    }

    try:
        import torch_geometric
        bundle["pyg_version"] = torch_geometric.__version__
    except Exception:
        bundle["pyg_version"] = "unknown"

    torch.save(bundle, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  → Saved checkpoint ({size_kb:.0f} KB): {path}")
    return path


def load_checkpoint(path, device="cpu"):
    """
    Load a checkpoint and reconstruct the model.

    Args:
        path:   path to .pt checkpoint file
        device: device to map tensors to

    Returns:
        dict with keys: 'model' (loaded nn.Module), plus all metadata
    """
    from models.gcn import GCNModel
    from models.gat import GATModel

    bundle = torch.load(path, map_location=device, weights_only=False)

    mc = bundle["model_config"]
    if bundle["model_type"] == "gcn":
        model = GCNModel(mc["in_channels"], mc["hidden_channels"],
                         mc["num_classes"])
    elif bundle["model_type"] == "gat":
        model = GATModel(mc["in_channels"], mc["hidden_channels"],
                         mc["num_classes"])
    else:
        raise ValueError(f"Unknown model_type: {bundle['model_type']}")

    model.load_state_dict(bundle["model_state_dict"])
    model = model.to(device)
    model.eval()

    bundle["model"] = model
    return bundle
