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
from src.training.alignment import alignment_loss


def _cycle(loader):
    """Infinite iterator over a DataLoader (re-iterates when exhausted)."""
    while True:
        for b in loader:
            yield b


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
    log_every: int = 1,
    target_loader: Optional[DataLoader] = None,
    align_method: Optional[str] = None,
    align_lambda: float = 0.0,
) -> Dict:
    """
    Train with early stopping on validation loss; restore best weights.
    Returns a history dict.

    P18b — if ``target_loader`` is given and ``align_method`` is set, add
    ``align_lambda * align(feat_source, feat_target)`` (CORAL/MMD) on the model's
    last-timestep representation. Target batches are cycled. No target labels used;
    selection still on source-val.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    use_align = (align_method not in (None, "none", "")) and target_loader is not None \
        and align_lambda > 0.0
    target_iter = _cycle(target_loader) if use_align else None

    # Which validation metric selects the best checkpoint / drives early stopping.
    monitor = getattr(train_cfg, "model_selection_metric", "val_loss")
    higher_is_better = monitor != "val_loss"   # loss: lower better; acc/f1: higher
    best_score = -float("inf") if higher_is_better else float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    patience = train_cfg.early_stopping_patience
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1_macro": [],
               "align_loss": []}

    for epoch in range(1, train_cfg.max_epochs + 1):
        model.train()
        running, seen, align_running = 0.0, 0, 0.0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_align:
                logits, feat_s = model.forward_features(x)
                loss = criterion(logits, y)
                xt, _ = next(target_iter)
                _, feat_t = model.forward_features(xt.to(device))
                align = alignment_loss(align_method, feat_s, feat_t)
                loss = loss + align_lambda * align
                align_running += float(align.item()) * x.size(0)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)
        align_epoch = align_running / max(seen, 1)

        val_metrics, val_loss = evaluate(
            model, val_loader, device, num_classes, criterion
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["align_loss"].append(align_epoch)

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

        # Log on a cadence (every `log_every` epochs), plus the first epoch and any
        # epoch that improves the monitored metric, so progress is always visible.
        if verbose and (
            log_every <= 1 or epoch == 1 or improved or epoch % log_every == 0
            or epoch == train_cfg.max_epochs
        ):
            amsg = f"align {align_epoch:.4f} | " if use_align else ""
            print(
                f"{log_prefix}epoch {epoch:3d} | "
                f"train_loss {train_loss:.4f} | {amsg}val_loss {val_loss:.4f} | "
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
