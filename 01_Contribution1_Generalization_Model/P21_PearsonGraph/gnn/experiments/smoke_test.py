"""
P11 Smoke Test — fast end-to-end check
======================================
Runs ONE LODO fold for a few epochs on a subsample, exercising the whole P11
path: DAGHAR CSV loading (folder-name resolution), sensor-graph construction,
the BiGRU temporal node encoder, the DANN wrapper, and training/eval.

Use this BEFORE the full 6-fold run to catch data-path / shape errors in ~1-2 min.

Usage:
    python -m experiments.smoke_test                      # target WISDM, 2 epochs, bigru
    python -m experiments.smoke_test --target UCI --epochs 3
    python -m experiments.smoke_test --temporal_pool attention
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from utils.config import get_config
from utils.seed import set_seed
from data.daghar_loader import build_lodo_sensor_graphs, DAGHAR_DATASETS
from models.sensor_graph import SensorGraphModel
from models.da_sensor_graph import DomainAdversarialWrapper
from training.dann_trainer import DANNTrainer


def main():
    ap = argparse.ArgumentParser(description="P13 smoke test")
    ap.add_argument("--target", default="WISDM", choices=DAGHAR_DATASETS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_train", type=int, default=4000,
                    help="subsample this many train graphs to keep it fast")
    ap.add_argument("--encoder_mode", default="single",
                    choices=["single", "dualview"])
    ap.add_argument("--normalize_mode", default="per_channel",
                    choices=["per_channel", "per_bin_channel"])
    ap.add_argument("--temporal_pool", default="avg",
                    choices=["avg", "bigru", "attention"])
    ap.add_argument("--augment", action="store_true", default=True,
                    help="apply P13 source augmentation (on by default)")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    args = ap.parse_args()

    cfg = get_config()
    # P13 recipe = P9 baseline (single encoder, per-channel norm) + source augmentation
    cfg["dann_enabled"] = True
    cfg["gnn_type"] = "sage"
    cfg["feature_mode"] = "hybrid"
    cfg["daghar_timesteps"] = 90
    cfg["cnn_temporal_pool"] = args.temporal_pool
    cfg["encoder_mode"] = args.encoder_mode
    cfg["normalize_mode"] = args.normalize_mode
    cfg["augment"] = args.augment
    cfg["aug_copies"] = 1   # keep the smoke test fast (1 augmented pass)
    cfg["epochs"] = args.epochs
    cfg["patience"] = args.epochs

    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    fmap = cfg["dataset_folder_names"]

    print("=" * 64)
    print(f"  P13 SMOKE TEST  target={args.target}  augment={args.augment}  "
          f"encoder={args.encoder_mode}  "
          f"norm={args.normalize_mode}  epochs={args.epochs}  device={device}")
    print("=" * 64)

    # 1) Data — this is what fails first if folder names are wrong
    print("\n[1/3] Loading data + building sensor graphs ...")
    lodo = build_lodo_sensor_graphs(
        cfg["data_root"], args.target, config=cfg, folder_map=fmap,
    )
    cfg["daghar_timesteps"] = lodo["node_feat_dim"]   # actual feature length
    num_domains = lodo["num_domains"]
    print(f"    node_feat_dim={lodo['node_feat_dim']}  num_domains={num_domains}")

    # 2) Model
    print(f"\n[2/3] Building DANN + SAGE model (encoder_mode={cfg['encoder_mode']}) ...")
    backbone = SensorGraphModel(cfg)
    model = DomainAdversarialWrapper(backbone, num_domains, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    params={n_params:,}")

    # 3) Short training loop
    print("\n[3/3] Training a few epochs ...")
    train_graphs = lodo["train_graphs"][: args.max_train]
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    trainer = DANNTrainer(
        model, optimizer, device,
        lambda_max=cfg["dann_lambda_max"], schedule="ganin",
        max_grad_norm=1.0, patience=args.epochs,
    )
    trainer.total_epochs = args.epochs

    train_loader = DataLoader(train_graphs, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(lodo["val_graphs"], batch_size=cfg["batch_size"])
    test_loader = DataLoader(lodo["test_graphs"], batch_size=cfg["batch_size"])

    for ep in range(1, args.epochs + 1):
        tr = trainer.train_epoch(train_loader)
        vl = trainer.validate(val_loader)
        print(f"    epoch {ep:>2}  train_acc={tr['accuracy']*100:5.1f}%  "
              f"val_acc={vl['accuracy']*100:5.1f}%  lambda={tr['lambda']:.3f}")

    # quick test eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model.predict(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    print(f"\n    held-out {args.target} test acc (under-trained, sanity only) = "
          f"{correct/total*100:.1f}%")

    print("\n✅ SMOKE TEST PASSED — data loading, graph build, "
          f"{args.encoder_mode} encoder ({args.normalize_mode} norm), and DANN "
          "training all run end-to-end.")
    print("   Now run the full 6-fold experiment.")


if __name__ == "__main__":
    main()
