"""
P21 — GNN side: P9 recipe (DANN-over-sources + GraphSAGE + hybrid features) with
the graph construction switched between:

  --edge_mode fixed    : the P6 'full' fixed topology  (= Arm A control)
  --edge_mode pearson  : per-window Pearson-correlation adjacency (Arm B, NEW)
                         --select topk --topk K     keep K strongest pairs, or
                         --select tau  --tau T      keep pairs with score >= T
                         --alpha A                  blend with physical prior

Everything else is byte-identical to the P17/P18b source-only baseline: source-only
training (NO target data of any kind is loaded in P21 — DG protocol, numerically
comparable to the P18b 'base' cells), selection on source-val, per-epoch logs.

Output: <out_dir>/gnn_<target>.npz   probs [N,6], labels [N]
        <out_dir>/val_summary.json   {target: best source-val acc}  <- used for
        protocol-clean CONFIG selection in the sweep (never target-test).
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P21 = os.path.dirname(HERE)
GNN_DIR = os.path.join(P21, "gnn")
sys.path.insert(0, GNN_DIR)          # only the GNN codebase is importable here

import torch
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


def p21_config(args):
    cfg = get_config()
    if args.data_path:
        cfg["data_root"] = args.data_path
    cfg["seed"] = args.seed
    # ── P9 recipe, frozen (identical to P17/P18b base) ───────────────────────
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
    cfg["epochs"] = args.epochs
    cfg["log_every"] = args.log_every
    # ── P21: pure DG protocol — no target data loaded at all ─────────────────
    cfg["load_target_unlabeled"] = False
    cfg["normalize_include_target"] = False
    cfg["align_method"] = "none"
    cfg["align_lambda"] = 0.0
    # ── P21: the ONE lever — graph construction ──────────────────────────────
    cfg["edge_mode"] = args.edge_mode          # "fixed" | "pearson"
    cfg["edge_topology"] = "full"              # Arm A control topology
    cfg["pearson_select"] = args.select        # "topk" | "tau"
    cfg["pearson_topk"] = args.topk
    cfg["pearson_tau"] = args.tau
    cfg["pearson_alpha"] = args.alpha
    return cfg


def train_and_predict(target, cfg):
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    fmap = cfg["dataset_folder_names"]
    log_every = max(1, int(cfg.get("log_every", 5)))

    lodo = build_lodo_sensor_graphs(cfg["data_root"], target, config=cfg, folder_map=fmap)
    cfg["daghar_timesteps"] = lodo["node_feat_dim"]
    num_domains = lodo["num_domains"]
    print(f"  [gnn:{target}] source-train={lodo['n_train']}  source-val={lodo['n_val']}  "
          f"target-test={lodo['n_test']}  edge_mode={cfg['edge_mode']}", flush=True)

    backbone = SensorGraphModel(cfg)
    model = DomainAdversarialWrapper(backbone, num_domains, cfg).to(device)

    epochs = cfg["epochs"]
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    trainer = DANNTrainer(model, optimizer, device,
                          lambda_max=cfg["dann_lambda_max"], schedule="ganin",
                          max_grad_norm=1.0, scheduler=scheduler, patience=cfg["patience"])
    trainer.total_epochs = epochs

    bs = cfg["batch_size"]
    train_loader = DataLoader(lodo["train_graphs"], batch_size=bs, shuffle=True, drop_last=False)
    val_loader   = DataLoader(lodo["val_graphs"],   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(lodo["test_graphs"],  batch_size=bs, shuffle=False)  # ORDER preserved

    for epoch in range(1, epochs + 1):
        tr = trainer.train_epoch(train_loader)
        vl = trainer.validate(val_loader)
        improved = trainer.update_checkpoint(vl["accuracy"], epoch)
        if epoch == 1 or improved or epoch % log_every == 0 or epoch == epochs:
            print(f"    [gnn:{target}] epoch {epoch:3d} | "
                  f"act_loss {tr['activity_loss']:.4f} | dom_loss {tr['domain_loss']:.4f} | "
                  f"lambda {tr['lambda']:.3f} | tr_acc {tr['accuracy']*100:5.1f}% | "
                  f"val_loss {vl['loss']:.4f} | val_acc {vl['accuracy']*100:5.1f}%"
                  + ("  *" if improved else ""), flush=True)
        if trainer.stopped_early:
            print(f"    [gnn:{target}] early stop at epoch {epoch} "
                  f"(best epoch {trainer.best_epoch}, val_acc {trainer.best_val_acc*100:.1f}%)",
                  flush=True)
            break
    trainer.restore_best()
    best_val = float(trainer.best_val_acc)

    # Per-sample probabilities on the target test set, in test.csv order.
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model.predict(batch)              # [B, 6]
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    P = np.concatenate(probs, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    acc = float((P.argmax(1) == y).mean())
    print(f"  [gnn:{target}] best source-val acc = {best_val*100:.2f}%  |  "
          f"TARGET-TEST acc = {acc*100:.2f}%  (N={len(y)})", flush=True)
    return P, y, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--targets", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edge_mode", choices=["fixed", "pearson"], default="pearson")
    ap.add_argument("--select", choices=["topk", "tau"], default="topk")
    ap.add_argument("--topk", type=int, default=9)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = list(DAGHAR_DATASETS) if args.targets == "all" else args.targets.split(",")
    desc = (f"pearson select={args.select} topk={args.topk} tau={args.tau} "
            f"alpha={args.alpha}" if args.edge_mode == "pearson" else "fixed(full)")
    print(f"[GNN] P21 | edges={desc} | seed={args.seed} | P9 recipe, source-only (DG) | "
          f"targets={targets} | out={args.out_dir}", flush=True)

    val_path = os.path.join(args.out_dir, "val_summary.json")
    val_summary = json.load(open(val_path)) if os.path.isfile(val_path) else {}
    for t in targets:
        out = os.path.join(args.out_dir, f"gnn_{t}.npz")
        if os.path.isfile(out) and t in val_summary:
            print(f"  [gnn:{t}] already done — skipping (resumable)", flush=True)
            continue
        cfg_t = p21_config(args)                       # fresh per fold
        P, y, best_val = train_and_predict(t, cfg_t)
        np.savez(out, probs=P, labels=y)
        val_summary[t] = best_val
        json.dump(val_summary, open(val_path, "w"), indent=2)
        print(f"  saved {out}", flush=True)


if __name__ == "__main__":
    main()
