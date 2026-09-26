"""
Tiny end-to-end smoke test (no real training) to confirm the pipeline wires up:
data loading -> model forward/backward -> metrics.

Point --data_path at a folder that contains the DAGHAR dataset sub-folders.
For a quick check you can build a folder of just a couple of datasets with
train/validation/test CSVs.

    python -m experiments.smoke_test --data_path /path/to/standardized_view
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from configs.config import get_config
from src.data.data_loader import build_lodo_fold
from src.models.deepconvlstm import build_model
from src.training.trainer import train_model, evaluate
from src.utils.common import set_seed, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--target", default=None,
                   help="Held-out dataset key (default: first available).")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated subset of dataset keys to use.")
    p.add_argument("--input_view", default="time",
                   choices=["time", "frequency"])
    args = p.parse_args()

    cfg = get_config()
    cfg.paths.data_path = args.data_path
    cfg.data.input_view = args.input_view
    if args.datasets:
        keys = [k.strip() for k in args.datasets.split(",")]
        cfg.data.datasets = keys
    cfg.train.max_epochs = 2
    cfg.train.early_stopping_patience = 2
    cfg.train.batch_size = 32
    cfg.train.num_workers = 0
    cfg.sync()

    target = args.target or cfg.data.datasets[0]
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"device={device}  target={target}  datasets={cfg.data.datasets}")

    train_loader, val_loader, test_loader, meta = build_lodo_fold(cfg, target)
    print("meta:", meta)

    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}  seq_len_out={model.seq_len_out}  "
          f"lstm_input={model.lstm_input_size}")

    # one forward sanity check
    xb, yb = next(iter(train_loader))
    with torch.no_grad():
        out = model(xb.to(device))
    print(f"forward ok: input {tuple(xb.shape)} -> logits {tuple(out.shape)}")
    assert out.shape[1] == cfg.data.num_classes

    history = train_model(
        model, train_loader, val_loader, device,
        cfg.train, cfg.data.num_classes, log_prefix="  [smoke] ",
    )
    metrics, _ = evaluate(model, test_loader, device, cfg.data.num_classes)
    print(f"SMOKE OK | best_epoch={history['best_epoch']} "
          f"test_acc={metrics['accuracy']:.4f} "
          f"test_f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
