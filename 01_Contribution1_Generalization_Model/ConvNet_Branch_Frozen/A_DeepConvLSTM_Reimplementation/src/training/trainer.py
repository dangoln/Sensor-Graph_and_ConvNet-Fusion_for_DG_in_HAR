"""
Generic training / validation loop.

Reusable for any (model, dataloaders) pair. Implements the paper's training
recipe: cross-entropy loss, RMSProp (lr=1e-3, rho=0.9), with early stopping on
the source-domain validation loss and restoration of the best weights.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics


def build_optimizer(model: nn.Module, train_cfg) -> torch.optim.Optimizer:
    if train_cfg.optimizer.lower() == "rmsprop":
        # alpha is RMSProp's smoothing constant == paper's decay factor rho.
        return torch.optim.RMSprop(
            model.parameters(),
            lr=train_cfg.learning_rate,
            alpha=train_cfg.rho,
        )
    if train_cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    raise ValueError(f"Unknown optimizer '{train_cfg.optimizer}'")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[nn.Module] = None,
) -> Tuple[Dict, float]:
    """Return (metrics_dict, mean_loss) over a loader."""
    model.eval()
    all_true, all_pred = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        pred = logits.argmax(dim=1)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    mean_loss = (total_loss / n) if n else float("nan")
    return metrics, mean_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg,
    num_classes: int,
    log_prefix: str = "",
    verbose: bool = True,
) -> Dict:
    """
    Train with early stopping on validation loss; restore best weights.
    Returns a history dict.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)

    # Which validation metric selects the best checkpoint / drives early stopping.
    monitor = getattr(train_cfg, "model_selection_metric", "val_loss")
    higher_is_better = monitor != "val_loss"   # loss: lower better; acc/f1: higher
    best_score = -float("inf") if higher_is_better else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    patience = train_cfg.early_stopping_patience
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)

        val_metrics, val_loss = evaluate(
            model, val_loader, device, num_classes, criterion
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        score = {
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }[monitor]
        improved = (
            score > best_score + 1e-5 if higher_is_better
            else score < best_score - 1e-5
        )
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"{log_prefix}epoch {epoch:3d} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"val_acc {val_metrics['accuracy']:.4f} | "
                f"val_f1m {val_metrics['f1_macro']:.4f} | "
                f"{time.time()-t0:.1f}s"
                + ("  *" if improved else "")
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"{log_prefix}early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, best {monitor} {best_score:.4f})")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["monitor"] = monitor
    history["best_score"] = best_score
    return history
