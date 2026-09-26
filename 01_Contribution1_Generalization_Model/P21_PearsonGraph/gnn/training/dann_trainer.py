"""
DANN Trainer (Phase 2)
=======================
Training loop for Domain-Adversarial Neural Networks.

Key differences from standard Trainer:
  - Two losses: activity classification + domain discrimination
  - Lambda scheduling: GRL weight increases during training
  - Each batch needs both activity labels AND domain labels
  - Evaluation uses only the activity head
"""

import copy
import math
import torch
import torch.nn.functional as F

from utils.gradient_reversal import ganin_lambda_schedule, linear_lambda_schedule
from training.alignment import alignment_loss


def _cycle(loader):
    """Infinite iterator over a DataLoader (re-iterates when exhausted)."""
    while True:
        for b in loader:
            yield b


class DANNTrainer:
    """
    Trainer for Domain-Adversarial Sensor-Graph models.

    P18b — optional source↔target feature alignment (CORAL/MMD). When a
    target-unlabeled loader is passed to train_epoch and ``align_method`` is set, a
    ``align_lambda * alignment_loss(h_source, h_target)`` term is added on the graph
    embedding. The existing DANN-over-sources adversarial term is unchanged (part of
    the baseline recipe); alignment is the single new lever.
    """

    def __init__(self, model, optimizer, device,
                 lambda_max=1.0, schedule="ganin",
                 max_grad_norm=1.0, scheduler=None, patience=20,
                 align_method=None, align_lambda=0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_max = lambda_max
        self.schedule = schedule
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler = scheduler
        self.patience = patience
        self.align_method = align_method if align_method not in ("none", "") else None
        self.align_lambda = align_lambda

        self.activity_criterion = torch.nn.CrossEntropyLoss()
        self.domain_criterion = torch.nn.CrossEntropyLoss()

        # Checkpointing state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_weights = None
        self.epochs_no_improve = 0
        self.stopped_early = False

        # Training progress tracking
        self.total_epochs = 1  # set externally before training
        self.current_epoch = 0

    def _get_lambda(self):
        """Compute current lambda value based on training progress."""
        if self.total_epochs <= 1:
            progress = 1.0
        else:
            progress = self.current_epoch / self.total_epochs

        if self.schedule == "ganin":
            return self.lambda_max * ganin_lambda_schedule(progress)
        elif self.schedule == "linear":
            return linear_lambda_schedule(progress, self.lambda_max)
        else:
            return self.lambda_max

    def train_epoch(self, train_loader, target_loader=None):
        """
        Train one epoch with combined activity + domain (+ P18b alignment) loss.

        Each source batch must have:
          - batch.y:      activity labels
          - batch.domain: domain (source dataset) labels
        If ``target_loader`` is given and ``self.align_method`` is set, a
        ``align_lambda * align(h_source, h_target)`` term is added; target batches
        are cycled to match the (longer) source loader. No target labels used.
        """
        self.model.train()
        self.current_epoch += 1

        lambda_val = self._get_lambda()
        use_align = self.align_method is not None and target_loader is not None \
            and self.align_lambda > 0.0
        target_iter = _cycle(target_loader) if use_align else None

        total_act_loss = 0
        total_dom_loss = 0
        total_align_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # P18b: forward_features also returns the graph embedding h_source.
            activity_logits, domain_logits, h_src = self.model.forward_features(
                batch, lambda_val)

            # Activity loss
            act_loss = self.activity_criterion(activity_logits, batch.y)

            # Domain loss (only if domain labels available)
            if hasattr(batch, 'domain') and batch.domain is not None:
                dom_loss = self.domain_criterion(domain_logits, batch.domain)
                loss = act_loss + lambda_val * dom_loss
            else:
                dom_loss = torch.tensor(0.0, device=self.device)
                loss = act_loss

            # P18b — source↔target feature alignment (CORAL/MMD) on the embedding.
            if use_align:
                tgt = next(target_iter).to(self.device)
                h_tgt = self.model.embed(tgt)
                align = alignment_loss(self.align_method, h_src, h_tgt)
                loss = loss + self.align_lambda * align
                total_align_loss += float(align.item()) * batch.num_graphs

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            n = batch.num_graphs
            total_act_loss += act_loss.item() * n
            total_dom_loss += dom_loss.item() * n
            pred = activity_logits.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += n

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": (total_act_loss + total_dom_loss) / total_samples,
            "activity_loss": total_act_loss / total_samples,
            "domain_loss": total_dom_loss / total_samples,
            "align_loss": total_align_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "lambda": lambda_val,
        }

    def validate(self, val_loader):
        """Validate using activity head only."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                # Use predict() which only returns activity logits
                out = self.model.predict(batch)
                loss = self.activity_criterion(out, batch.y)

                total_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                total_correct += (pred == batch.y).sum().item()
                total_samples += batch.num_graphs

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def update_checkpoint(self, val_acc, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.epochs_no_improve = 0
            return True
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.stopped_early = True
            return False

    def restore_best(self):
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
