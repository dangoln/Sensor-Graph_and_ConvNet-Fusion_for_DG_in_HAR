"""
P24 — train the locked P21-winner GNN per LODO fold and SAVE CHECKPOINTS,
so compression variants can be evaluated post-hoc without retraining.

Variants (--variant):
  full     : the locked P21 winner (embed/hidden 64, cnn [32,64]) — the 77.96% (5-seed) system
  compact  : retrained small model (embed/hidden 32, cnn [16,32]) — the
             architecture-level SO point on the storage-accuracy curve
  tiny     : embed/hidden 16, cnn [8,16] — the aggressive SO point

Classic SO retrain arms (P1/P6 suite, orthogonal to --variant):
  --nc accel_only|gyro_only|no_gz : Node Coarsening — drop sensor channels;
       graphs shrink to 3/3/5 nodes, Pearson prior remaps, edge-type embedding
       is disabled for these arms (its accel/gyro typing assumes all 6 nodes).
  --aw 40|30 : Adaptive Windowing — strided downsample 60 -> N timesteps/window
       (hybrid features recompute on the shorter signal).
RED / GS need NO retraining — see p24/edge_budget.py (inference-only on ckpts).

Output per fold (results/checkpoints/<variant>/):
    ckpt_<target>.pt   {state_dict, cfg-subset}    gnn_<target>.npz   test probs
    val_summary.json   {target: best source-val acc}

DG protocol, seed 42 (compression deltas vs the same checkpoint are deterministic;
the retrained arms note single-seed in the report).
"""
import os, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P24 = os.path.dirname(HERE)
import sys
GNN_DIR = os.path.join(P24, "gnn")
sys.path.insert(0, GNN_DIR)

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils.config import get_config
from utils.seed import set_seed
from data.daghar_loader import build_lodo_sensor_graphs, DAGHAR_DATASETS
from models.sensor_graph import SensorGraphModel
from models.da_sensor_graph import DomainAdversarialWrapper
from training.dann_trainer import DANNTrainer

TARGETS = list(DAGHAR_DATASETS)

VARIANTS = {
    "full":    {"cnn_embed_dim": 64, "gnn_hidden": 64, "cnn_channels": [32, 64]},
    "compact": {"cnn_embed_dim": 32, "gnn_hidden": 32, "cnn_channels": [16, 32]},
    "tiny":    {"cnn_embed_dim": 16, "gnn_hidden": 16, "cnn_channels": [8, 16]},
}

# P1/P6 Node-Coarsening configurations (original channel ids: 0-2 accel, 3-5 gyro)
NC_SETS = {
    "accel_only": [0, 1, 2],
    "gyro_only":  [3, 4, 5],
    "no_gz":      [0, 1, 2, 3, 4],
}


def p24_config(data_path, epochs, log_every, seed, variant, nc=None, aw=0):
    cfg = get_config()
    if data_path:
        cfg["data_root"] = data_path
    cfg["seed"] = seed
    cfg["dann_enabled"] = True
    cfg["gnn_type"] = "sage"
    cfg["feature_mode"] = "hybrid"
    cfg["daghar_timesteps"] = 90
    cfg["learning_rate"] = 0.001
    cfg["dann_lambda_max"] = 1.0
    cfg["dann_schedule"] = "ganin"
    cfg["encoder_mode"] = "single"
    cfg["cnn_temporal_pool"] = "avg"
    cfg["normalize_mode"] = "per_channel"
    cfg["augment"] = False
    cfg["epochs"] = epochs
    cfg["log_every"] = log_every
    cfg["load_target_unlabeled"] = False
    cfg["normalize_include_target"] = False
    cfg["align_method"] = "none"
    cfg["align_lambda"] = 0.0
    # P21 winner graph construction, frozen
    cfg["edge_mode"] = "pearson"
    cfg["pearson_select"] = "topk"
    cfg["pearson_topk"] = 9
    cfg["pearson_alpha"] = 0.5
    cfg["pearson_tau"] = 0.3
    # P24 variant capacity
    cfg.update(VARIANTS[variant])
    # P24 classic SO arms (NC / AW)
    if nc:
        keep = NC_SETS[nc]
        cfg["nc_keep"] = keep
        cfg["node_channel_ids"] = keep
        cfg["num_sensor_nodes"] = len(keep)
        cfg["pearson_topk"] = min(cfg["pearson_topk"],
                                  len(keep) * (len(keep) - 1) // 2)
        cfg["use_edge_type"] = False    # accel/gyro edge typing assumes 6 nodes
    if aw:
        cfg["aw_timesteps"] = int(aw)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--targets", default="all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--nc", choices=list(NC_SETS), default=None,
                    help="Node Coarsening: drop sensor channels (retrain arm)")
    ap.add_argument("--aw", type=int, default=0,
                    help="Adaptive Windowing: timesteps/window, e.g. 40 or 30")
    ap.add_argument("--out_root", default=os.path.join(P24, "results", "checkpoints"),
                    help="parent folder for checkpoints (default results/checkpoints; "
                         "smoke tests use a separate folder so real runs are never touched)")
    args = ap.parse_args()

    arm = args.variant + (f"_nc-{args.nc}" if args.nc else "") \
                       + (f"_aw{args.aw}" if args.aw else "") \
                       + (f"_s{args.seed}" if args.seed != 42 else "")
    out_dir = os.path.join(args.out_root, arm)
    os.makedirs(out_dir, exist_ok=True)
    targets = TARGETS if args.targets == "all" else args.targets.split(",")
    print(f"[P24] train+save | arm={arm} {VARIANTS[args.variant]} | "
          f"nc={args.nc} aw={args.aw or '-'} | seed={args.seed} | "
          f"targets={targets} | out={out_dir}", flush=True)

    val_path = os.path.join(out_dir, "val_summary.json")
    val_summary = json.load(open(val_path)) if os.path.isfile(val_path) else {}
    for t in targets:
        ck_out = os.path.join(out_dir, f"ckpt_{t}.pt")
        npz_out = os.path.join(out_dir, f"gnn_{t}.npz")
        if os.path.isfile(ck_out) and os.path.isfile(npz_out) and t in val_summary:
            print(f"  [{arm}:{t}] already done — skipping (resumable)", flush=True)
            continue

        cfg = p24_config(args.data_path, args.epochs, args.log_every, args.seed,
                         args.variant, nc=args.nc, aw=args.aw)
        device = torch.device(cfg["device"])
        set_seed(cfg["seed"])
        lodo = build_lodo_sensor_graphs(cfg["data_root"], t, config=cfg,
                                        folder_map=cfg["dataset_folder_names"])
        cfg["daghar_timesteps"] = lodo["node_feat_dim"]

        backbone = SensorGraphModel(cfg)
        model = DomainAdversarialWrapper(backbone, lodo["num_domains"], cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{arm}:{t}] trainable params: {n_params:,}", flush=True)

        epochs = cfg["epochs"]
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        trainer = DANNTrainer(model, optimizer, device,
                              lambda_max=cfg["dann_lambda_max"], schedule="ganin",
                              max_grad_norm=1.0, scheduler=scheduler, patience=cfg["patience"])
        trainer.total_epochs = epochs

        bs = cfg["batch_size"]
        train_loader = DataLoader(lodo["train_graphs"], batch_size=bs, shuffle=True)
        val_loader   = DataLoader(lodo["val_graphs"],   batch_size=bs, shuffle=False)
        test_loader  = DataLoader(lodo["test_graphs"],  batch_size=bs, shuffle=False)

        for epoch in range(1, epochs + 1):
            tr = trainer.train_epoch(train_loader)
            vl = trainer.validate(val_loader)
            improved = trainer.update_checkpoint(vl["accuracy"], epoch)
            if epoch == 1 or improved or epoch % args.log_every == 0 or epoch == epochs:
                print(f"    [{arm}:{t}] epoch {epoch:3d} | "
                      f"act_loss {tr['activity_loss']:.4f} | dom_loss {tr['domain_loss']:.4f} | "
                      f"lambda {tr['lambda']:.3f} | tr_acc {tr['accuracy']*100:5.1f}% | "
                      f"val_loss {vl['loss']:.4f} | val_acc {vl['accuracy']*100:5.1f}%"
                      + ("  *" if improved else ""), flush=True)
            if trainer.stopped_early:
                print(f"    [{arm}:{t}] early stop at epoch {epoch}", flush=True)
                break
        trainer.restore_best()

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                probs.append(F.softmax(model.predict(batch), dim=1).cpu().numpy())
                labels.append(batch.y.cpu().numpy())
        P = np.concatenate(probs).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        acc = float((P.argmax(1) == y).mean())

        np.savez(npz_out, probs=P, labels=y)
        torch.save({"state_dict": model.state_dict(),
                    "num_domains": lodo["num_domains"],
                    "node_feat_dim": lodo["node_feat_dim"],
                    "variant": args.variant,
                    "variant_cfg": VARIANTS[args.variant],
                    "nc": args.nc, "aw": args.aw,
                    "n_params": n_params}, ck_out)
        val_summary[t] = float(trainer.best_val_acc)
        json.dump(val_summary, open(val_path, "w"), indent=2)
        print(f"  [{arm}:{t}] src-val {trainer.best_val_acc*100:.2f}%  |  "
              f"TARGET-TEST {acc*100:.2f}%  |  saved ckpt + npz", flush=True)


if __name__ == "__main__":
    main()
